# src/train.py
from losses import FocalLoss
import argparse, os, json, csv
import numpy as np
import pandas as pd
import torch, torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from tqdm import tqdm
from copy import deepcopy

from dataset import RetinaDataset
from transforms import build_transforms
from model_densenet121 import build_densenet121
from utils import seed_everything, get_device, save_json

# ----------------------------
# Dataloaders with class-imbalance aware sampling
# ----------------------------
def build_loaders(train_csv, val_csv, batch_size, img_size, num_workers=2):
    # Albumentations transforms: strong aug for train, deterministic for val
    train_ds = RetinaDataset(train_csv, tfm=build_transforms(img_size, is_train=True))
    val_ds   = RetinaDataset(val_csv,   tfm=build_transforms(img_size, is_train=False))

    # WeightedRandomSampler: upsample rare classes by inverse frequency
    class_counts = (
        train_ds.df["label"].value_counts()
        .reindex(train_ds.classes)
        .fillna(0)
        .to_numpy()
    )
    inv = 1.0 / np.maximum(class_counts, 1)
    sample_weights = train_ds.df["label"].map({c: w for c, w in zip(train_ds.classes, inv)}).to_numpy()
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,  num_workers=num_workers)
    return train_loader, val_loader, train_ds.classes

# ----------------------------
# CSV logger (history.csv)
# ----------------------------
def _append_history_row(out_dir, row_dict):
    """Append per-epoch metrics (incl. LR) into runs/.../history.csv"""
    path = os.path.join(out_dir, "history.csv")
    is_new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row_dict.keys()))
        if is_new:
            writer.writeheader()
        writer.writerow(row_dict)

# ----------------------------
# Class weights via Effective Number of Samples (more stable than 1/freq)
# ----------------------------
def effective_num_weights(labels, beta=0.9999, num_classes=None):
    counts = labels.value_counts().sort_index()
    if num_classes is not None:
        counts = counts.reindex(range(num_classes)).fillna(0)
    counts = counts.to_numpy().astype(float)
    eff_num = 1.0 - np.power(beta, counts)
    weights = (1.0 - beta) / np.maximum(eff_num, 1e-8)
    weights = weights / weights.mean()  # normalize for scale stability
    return weights

# ----------------------------
# MixUp + EMA (stability + generalization)
# ----------------------------
def mixup_batch(x, y, alpha=0.2):
    """Classic MixUp on a batch: convex-combine pairs of images and labels"""
    if alpha <= 0:
        return x, y, None
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    x_mix = lam * x + (1 - lam) * x[idx]
    return x_mix, (y, y[idx]), lam

class EMA:
    """Exponential Moving Average of weights to smooth validation performance."""
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}
    def update(self, model):
        with torch.no_grad():
            for k, v in model.state_dict().items():
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
    def load_shadow(self, model):
        model.load_state_dict(self.shadow, strict=True)

# ----------------------------
# Train / Eval loops
# ----------------------------
def train_epoch(model, loader, criterion, optimizer, device, grad_clip=None, mixup_alpha=0.0, ema=None):
    model.train()
    all_preds, all_gts = [], []
    running_loss = 0.0

    for x, y in tqdm(loader, desc="Train", leave=False):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)

        # Optionally apply MixUp (loss is a convex mix of two CE/Focal losses)
        if mixup_alpha > 0:
            x, (y_a, y_b), lam = mixup_batch(x, y, alpha=mixup_alpha)
            out = model(x)
            loss = lam * criterion(out, y_a) + (1 - lam) * criterion(out, y_b)
            preds = out.argmax(1)
            all_preds.extend(preds.detach().cpu().tolist())
            all_gts.extend(y.detach().cpu().tolist())  # metrics vs original labels
        else:
            out = model(x)
            loss = criterion(out, y)
            all_preds.extend(out.argmax(1).detach().cpu().tolist())
            all_gts.extend(y.detach().cpu().tolist())

        # Backprop
        loss.backward()
        if grad_clip is not None and grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        # Update EMA shadow weights after each step
        if ema is not None:
            ema.update(model)

        running_loss += loss.item() * x.size(0)

    acc = accuracy_score(all_gts, all_preds)
    f1  = f1_score(all_gts, all_preds, average="macro")
    return running_loss / len(loader.dataset), acc, f1

@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    all_preds, all_gts = [], []
    running_loss = 0.0

    for x, y in tqdm(loader, desc="Val  ", leave=False):
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = criterion(out, y)
        running_loss += loss.item() * x.size(0)
        all_preds.extend(out.argmax(1).cpu().tolist())
        all_gts.extend(y.cpu().tolist())

    acc = accuracy_score(all_gts, all_preds)
    f1  = f1_score(all_gts, all_preds, average="macro")

    # Human-readable sanity printouts
    print(classification_report(all_gts, all_preds, digits=4, zero_division=0))
    unique, cnt = np.unique(all_preds, return_counts=True)
    print("Val prediction distribution:", dict(zip(unique.tolist(), cnt.tolist())))

    return running_loss / len(loader.dataset), acc, f1, all_preds, all_gts

# ----------------------------
# Main
# ----------------------------
def main(args):
    seed_everything(args.seed)
    device = get_device()
    print("Device:", device)

    # Data + classes
    train_loader, val_loader, classes = build_loaders(
        args.train_csv, args.val_csv, args.batch_size, args.img_size, args.num_workers
    )
    num_classes = len(classes)

    # Model
    model = build_densenet121(num_classes, pretrained=args.pretrained).to(device)

    # Freeze backbone for warm-up if pretrained
    backbone_params = list(model.features.parameters())
    head_params     = list(model.classifier.parameters())
    if args.pretrained:
        for p in backbone_params:
            p.requires_grad = False
        print(f">> Frozen backbone, training head for {args.freeze_epochs} epoch(s)")

    os.makedirs(args.out_dir, exist_ok=True)
    class_to_idx = {str(c): int(i) for i, c in enumerate(classes)}
    save_json(class_to_idx, os.path.join(args.out_dir, "class_to_idx.json"))

    # Class weights via Effective Number
    train_df = pd.read_csv(args.train_csv)
    eff_w = effective_num_weights(train_df["label"], beta=0.9999, num_classes=num_classes)
    class_weights = torch.tensor(eff_w, dtype=torch.float32, device=device)
    print("Loss class weights (effective-num):", class_weights.detach().cpu().numpy())

    # Loss: Focal (recommended) or CE; both support label smoothing and weights
    if args.focal:
        criterion = FocalLoss(
            gamma=2.0,
            weight=class_weights if args.weighted_loss else None,
            label_smoothing=args.label_smoothing,
            reduction="mean",
        )
    else:
        def ce_loss(logits, target):
            return nn.functional.cross_entropy(
                logits, target,
                weight=class_weights if args.weighted_loss else None,
                label_smoothing=args.label_smoothing
            )
        criterion = ce_loss

    # Optimizer: two LR groups during warm-up (head higher, backbone lower)
    optimizer = AdamW([
        {"params": head_params, "lr": args.head_lr},
        {"params": backbone_params, "lr": args.backbone_lr}
    ], weight_decay=args.weight_decay)

    # CosineAnnealingWarmRestarts: smooth cyclical LR; pairs well with unfreezing
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=max(args.freeze_epochs + 2, 4),
        T_mult=2
    )

    # Optional Exponential Moving Average of weights
    ema = EMA(model, decay=0.999) if args.ema else None

    best_f1, no_improve = -1.0, 0
    for epoch in range(1, args.epochs+1):
        # Unfreeze backbone after warm-up; then use single LR for all params
        if args.pretrained and epoch == args.freeze_epochs + 1:
            for p in backbone_params:
                p.requires_grad = True
            for pg in optimizer.param_groups:
                pg["lr"] = args.lr
            print(f">> Unfroze backbone and set all LRs to {args.lr}")

        print(f"Epoch {epoch}/{args.epochs}")
        tr_loss, tr_acc, tr_f1 = train_epoch(
            model, train_loader, criterion, optimizer, device,
            grad_clip=args.grad_clip, mixup_alpha=args.mixup_alpha, ema=ema
        )
        va_loss, va_acc, va_f1, va_preds, va_gts = eval_epoch(model, val_loader, criterion, device)

        # step with epoch fraction to align restart phase
        scheduler.step(epoch - 1 + 1e-8)

        # Log current LR (from the first param group)
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"  LR={current_lr:.6e}")
        _append_history_row(args.out_dir, {
            "epoch": epoch,
            "lr": f"{current_lr:.8f}",
            "train_loss": f"{tr_loss:.6f}",
            "train_acc": f"{tr_acc:.6f}",
            "train_f1": f"{tr_f1:.6f}",
            "val_loss": f"{va_loss:.6f}",
            "val_acc": f"{va_acc:.6f}",
            "val_f1": f"{va_f1:.6f}",
        })

        # Save best by validation macro-F1; if EMA used, save shadow weights
        if va_f1 > best_f1:
            best_f1 = va_f1
            no_improve = 0
            ckpt = os.path.join(args.out_dir, "best.pt")

            if ema is not None:
                bak = deepcopy(model.state_dict())
                ema.load_shadow(model)        # swap in EMA weights
                torch.save(model.state_dict(), ckpt)
                model.load_state_dict(bak)    # restore live weights
            else:
                torch.save(model.state_dict(), ckpt)

            print("  saved:", ckpt)

            # Persist best report + confusion matrix arrays for later plotting
            report_txt = classification_report(va_gts, va_preds, digits=4, zero_division=0)
            with open(os.path.join(args.out_dir, "report_best.txt"), "w") as f:
                f.write(report_txt)
            cm = confusion_matrix(va_gts, va_preds)
            np.save(os.path.join(args.out_dir, "cm_best.npy"), cm)
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print("Early stopping.")
                break

    print(f"Best val macro-F1: {best_f1:.4f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", type=str, required=True)
    ap.add_argument("--val_csv", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="runs/densenet121_pretrained")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--img_size", type=int, default=224)

    # LRs
    ap.add_argument("--lr", type=float, default=3e-4)              # after unfreeze (single LR for all)
    ap.add_argument("--head_lr", type=float, default=1e-3)         # warmup head LR
    ap.add_argument("--backbone_lr", type=float, default=1e-4)     # warmup backbone LR
    ap.add_argument("--weight_decay", type=float, default=1e-4)

    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--grad_clip", type=float, default=1.0, help="Max grad-norm (0 to disable)")

    # pretrained controls
    ap.add_argument("--pretrained", action="store_true")
    ap.add_argument("--freeze_epochs", type=int, default=3)

    # imbalance-aware loss controls
    ap.add_argument("--focal", action="store_true", help="Use focal loss instead of CE")
    ap.add_argument("--weighted_loss", action="store_true", help="Apply class weights in the loss")
    ap.add_argument("--label_smoothing", type=float, default=0.0, help="Label smoothing for CE/Focal")

    # new options
    ap.add_argument("--ema", action="store_true", help="Use EMA of model weights")
    ap.add_argument("--mixup_alpha", type=float, default=0.2, help="MixUp alpha; 0 disables MixUp")

    args = ap.parse_args()
    main(args)