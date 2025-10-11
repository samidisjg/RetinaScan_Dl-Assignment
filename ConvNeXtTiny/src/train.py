import os, torch, timm
from pathlib import Path
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch import nn
from tqdm import tqdm
from .dataset import CSVDataset
from sklearn.metrics import classification_report, balanced_accuracy_score
import pandas as pd
import numpy as np

DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ROOT / "splits"

print(SPLITS)

# -------------------- Imbalance: Weighted Sampler --------------------
def _build_weighted_sampler(train_csv, classes):
    df = pd.read_csv(train_csv).reset_index(drop=True)
    class_to_idx = {c: i for i, c in enumerate(sorted(classes))}
    labels_idx = df["label"].map(class_to_idx).to_numpy()

    num_classes = len(classes)
    counts = np.bincount(labels_idx, minlength=num_classes)
    counts = np.maximum(counts, 1)
    class_weights = 1.0 / torch.tensor(counts, dtype=torch.float)

    sample_weights = class_weights[torch.from_numpy(labels_idx)]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    print("Class counts:", {i: int(c) for i, c in enumerate(counts)})
    print("Class weights (inv freq):", class_weights.tolist())
    return sampler

# -------------------- Freeze / Unfreeze helpers --------------------
def _head_param_names(model):
    names = [n for (n, _) in model.named_parameters() if n.startswith("head")]
    if not names and hasattr(model, "classifier"):
        names = [n for (n, _) in model.named_parameters() if n.startswith("classifier")]
    return set(names)

def freeze_backbone_train_head(model):
    head_names = _head_param_names(model)
    for n, p in model.named_parameters():
        p.requires_grad = (n in head_names)

def unfreeze_stages_3_4_and_head(model):
    # ConvNeXt in timm: model.stages[0..3], model.head
    for p in model.parameters():
        p.requires_grad = False
    # unfreeze deeper stages + head
    for p in model.stages[2].parameters():  # stage 3
        p.requires_grad = True
    for p in model.stages[3].parameters():  # stage 4
        p.requires_grad = True
    for p in model.head.parameters():
        p.requires_grad = True

def unfreeze_all(model):
    for p in model.parameters():
        p.requires_grad = True

def build_optimizer(model, lr_backbone, lr_head, weight_decay, train_head_only=False):
    head_names = _head_param_names(model)
    if train_head_only:
        params = [p for n, p in model.named_parameters() if p.requires_grad and (n in head_names)]
        return torch.optim.AdamW(params, lr=lr_head, weight_decay=weight_decay)

    backbone_params = [p for n, p in model.named_parameters() if p.requires_grad and (n not in head_names)]
    head_params     = [p for n, p in model.named_parameters() if p.requires_grad and (n in head_names)]
    groups = []
    if backbone_params:
        groups.append({"params": backbone_params, "lr": lr_backbone})
    if head_params:
        groups.append({"params": head_params, "lr": lr_head})
    return torch.optim.AdamW(groups, weight_decay=weight_decay)

def current_lrs(optimizer):
    return [g["lr"] for g in optimizer.param_groups]

# -------------------- Train / Eval --------------------
def run(train_csv=SPLITS / "train.csv",
        val_csv=SPLITS / "val.csv",
        out="checkpoints/convnext_tiny_second.pt",
        size=224, batch_size=32, epochs=37,
        # Phase controls
        freeze_epochs=2,                 # Phase-1 (head only)
        partial_unfreeze=True,           # unfreeze stages 3-4 + head first
        full_unfreeze_at=None,           # e.g., 8 => after 8 epochs of phase-2, unfreeze all; None to skip
        # LRs & WD
        lr_head_phase1=1e-3,
        lr_backbone_phase2=5e-5,
        lr_head_phase2=1e-4,
        weight_decay_phase1=1e-4,
        weight_decay_phase2=1e-5,
        # Warmup
        warmup_epochs_phase2=2,
        # Other
        grad_clip=1.0,
        label_smoothing=0.05,
        use_weighted_sampler=True):

    # Classes
    classes = sorted(pd.read_csv(train_csv)["label"].unique().tolist())

    # Datasets / Loaders
    train_ds = CSVDataset(train_csv, classes=classes, train=True,  size=size)
    val_ds   = CSVDataset(val_csv,   classes=classes, train=False, size=size)

    if use_weighted_sampler:
        sampler = _build_weighted_sampler(train_csv, classes)
        train_dl = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                              num_workers=0, pin_memory=True)
    else:
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)

    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                          num_workers=0, pin_memory=True)

    # Model
    model = timm.create_model("convnext_tiny.in12k_ft_in1k",
                              pretrained=True, num_classes=len(classes))
    model.to(DEVICE)

    # Loss
    loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    best_acc = 0.0
    os.makedirs(os.path.dirname(out), exist_ok=True)

    # -------------------- Phase-1: train head only --------------------
    phase1_epochs = max(0, min(freeze_epochs, epochs))
    if phase1_epochs > 0:
        freeze_backbone_train_head(model)
        opt = build_optimizer(model,
                              lr_backbone=0.0,
                              lr_head=lr_head_phase1,
                              weight_decay=weight_decay_phase1,
                              train_head_only=True)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=phase1_epochs)

        for ep in range(1, phase1_epochs + 1):
            # Train
            model.train()
            total, correct, running_loss = 0, 0, 0.0
            for xb, yb in tqdm(train_dl, desc=f"Train (Phase-1) {ep}/{phase1_epochs}"):
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                opt.zero_grad()
                logits = model(xb)
                loss = loss_fn(logits, yb)
                loss.backward()
                if grad_clip:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                opt.step()

                running_loss += loss.item() * xb.size(0)
                pred = logits.argmax(1)
                total += yb.size(0)
                correct += (pred == yb).sum().item()

            # Validate
            model.eval()
            vtotal, vcorrect, vloss = 0, 0, 0.0
            all_pred, all_true = [], []
            with torch.no_grad():
                for xb, yb in tqdm(val_dl, desc="Val"):
                    xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                    logits = model(xb)
                    loss = loss_fn(logits, yb)
                    vloss += loss.item() * xb.size(0)
                    pred = logits.argmax(1)
                    vtotal += yb.size(0)
                    vcorrect += (pred == yb).sum().item()
                    all_pred.extend(pred.cpu().tolist())
                    all_true.extend(yb.cpu().tolist())

            train_acc = correct / max(1, total)
            train_loss = running_loss / max(1, total)
            val_acc = vcorrect / max(1, vtotal)
            val_loss = vloss / max(1, vtotal)
            bal_acc = balanced_accuracy_score(all_true, all_pred) if all_true else 0.0

            print(f"[Phase-1] Epoch {ep}: train_loss={train_loss:.4f} acc={train_acc:.3f} | "
                  f"val_loss={val_loss:.4f} acc={val_acc:.3f} bal_acc={bal_acc:.3f} | lrs={current_lrs(opt)}")

            if val_acc > best_acc:
                best_acc = val_acc
                torch.save({"model": model.state_dict(), "classes": classes}, out)
                print("✓ Saved:", out)

            sched.step()

    # -------------------- Phase-2: fine-tune  --------------------
    phase2_epochs = epochs - phase1_epochs
    if phase2_epochs > 0:
        if partial_unfreeze:
            unfreeze_stages_3_4_and_head(model)
        else:
            unfreeze_all(model)

        opt = build_optimizer(model,
                              lr_backbone=lr_backbone_phase2,
                              lr_head=lr_head_phase2,
                              weight_decay=weight_decay_phase2,
                              train_head_only=False)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=phase2_epochs)

        # start at 0.1x lr then linearly ramp to base over warmup_epochs
        warmup_epochs = min(warmup_epochs_phase2, phase2_epochs)
        bb_base, hd_base = lr_backbone_phase2, lr_head_phase2
        for g in opt.param_groups:
            if "params" in g and len(g["params"]) > 0:
                if abs(g["lr"] - bb_base) < 1e-12:
                    g["lr"] = bb_base * 0.1
                elif abs(g["lr"] - hd_base) < 1e-12:
                    g["lr"] = hd_base * 0.1

        for ep in range(1, phase2_epochs + 1):
            # optional switch to full unfreeze after some epochs
            if full_unfreeze_at is not None and ep == full_unfreeze_at:
                unfreeze_all(model)
                # rebuild optimizer to include all params
                opt = build_optimizer(model,
                                      lr_backbone=bb_base,
                                      lr_head=hd_base,
                                      weight_decay=weight_decay_phase2,
                                      train_head_only=False)

            # Warmup step
            if ep <= warmup_epochs:
                scale = 0.1 + 0.9 * (ep / max(1, warmup_epochs))
                for g in opt.param_groups:
                    if abs(g["lr"] - bb_base * 0.1) < 1e-12 or abs(g["lr"] - bb_base) < 1e-12:
                        g["lr"] = bb_base * scale
                    elif abs(g["lr"] - hd_base * 0.1) < 1e-12 or abs(g["lr"] - hd_base) < 1e-12:
                        g["lr"] = hd_base * scale

            # Train
            model.train()
            total, correct, running_loss = 0, 0, 0.0
            for xb, yb in tqdm(train_dl, desc=f"Train (Phase-2) {ep}/{phase2_epochs}"):
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                opt.zero_grad()
                logits = model(xb)
                loss = loss_fn(logits, yb)
                loss.backward()
                if grad_clip:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                opt.step()

                running_loss += loss.item() * xb.size(0)
                pred = logits.argmax(1)
                total += yb.size(0)
                correct += (pred == yb).sum().item()

            # Validate
            model.eval()
            vtotal, vcorrect, vloss = 0, 0, 0.0
            all_pred, all_true = [], []
            with torch.no_grad():
                for xb, yb in tqdm(val_dl, desc="Val"):
                    xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                    logits = model(xb)
                    loss = loss_fn(logits, yb)
                    vloss += loss.item() * xb.size(0)
                    pred = logits.argmax(1)
                    vtotal += yb.size(0)
                    vcorrect += (pred == yb).sum().item()
                    all_pred.extend(pred.cpu().tolist())
                    all_true.extend(yb.cpu().tolist())

            train_acc = correct / max(1, total)
            train_loss = running_loss / max(1, total)
            val_acc = vcorrect / max(1, vtotal)
            val_loss = vloss / max(1, vtotal)
            bal_acc = balanced_accuracy_score(all_true, all_pred) if all_true else 0.0

            print(f"[Phase-2] Epoch {ep}: train_loss={train_loss:.4f} acc={train_acc:.3f} | "
                  f"val_loss={val_loss:.4f} acc={val_acc:.3f} bal_acc={bal_acc:.3f} | lrs={current_lrs(opt)}")

            if val_acc > best_acc:
                best_acc = val_acc
                torch.save({"model": model.state_dict(), "classes": classes}, out)
                print("✓ Saved:", out)

            sched.step()

    # --------- Final per-class report on validation set ---------
    label_ids = list(range(len(classes)))
    names = [str(c) for c in classes]
    print(classification_report(
        all_true, all_pred,
        labels=label_ids,
        target_names=names,
        zero_division=0,
        digits=4
    ))

if __name__ == "__main__":
    run()