import os, torch, timm, numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch import nn
from tqdm import tqdm
from .dataset import CSVDataset
from sklearn.metrics import classification_report, balanced_accuracy_score, f1_score
import pandas as pd

DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ROOT / "splits"

print(SPLITS)

# -------------------- Losses & helpers --------------------
def class_balanced_weights(counts, beta=0.999):
    c = torch.tensor(counts, dtype=torch.float)
    w = (1 - beta) / (1 - torch.pow(beta, c))
    w = w / w.mean()
    return w.float()

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction
    def forward(self, logits, target):
        logp = nn.functional.log_softmax(logits, dim=1)
        p = torch.exp(logp)
        pt = p.gather(1, target.unsqueeze(1)).squeeze(1)
        nll = nn.functional.nll_loss(logp, target, weight=self.weight, reduction='none')
        loss = ((1 - pt) ** self.gamma) * nll
        return loss.mean() if self.reduction == 'mean' else loss.sum()

def mixup_batch(x, y, alpha=0.4):
    if alpha <= 0:
        return x, y, None
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    mixed = lam * x + (1 - lam) * x[idx]
    return mixed, (y, y[idx], lam), idx

def mixup_criterion(criterion, pred, y_tuple):
    y_a, y_b, lam = y_tuple
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# -------------------- Imbalance: Weighted Sampler --------------------
def _build_weighted_sampler(train_csv, classes, class2_boost=1.4):
    df = pd.read_csv(train_csv).reset_index(drop=True)
    class_to_idx = {c: i for i, c in enumerate(sorted(classes))}
    labels_idx = df["label"].map(class_to_idx).to_numpy()

    num_classes = len(classes)
    counts = np.bincount(labels_idx, minlength=num_classes)
    counts = np.maximum(counts, 1)

    # inverse-sqrt weights
    class_weights = 1.0 / torch.sqrt(torch.tensor(counts, dtype=torch.float))
    # gentle push for class 2
    if num_classes > 2:
        class_weights[2] *= class2_boost
    class_weights = class_weights / class_weights.mean()

    sample_weights = class_weights[torch.from_numpy(labels_idx)]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    print("Class counts:", {i: int(c) for i, c in enumerate(counts)})
    print("Class weights (inv sqrt, c2 boosted):", class_weights.tolist())
    return sampler, counts

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
    for p in model.parameters(): p.requires_grad = False
    for p in model.stages[2].parameters(): p.requires_grad = True  # stage 3
    for p in model.stages[3].parameters(): p.requires_grad = True  # stage 4
    for p in model.head.parameters():      p.requires_grad = True

def unfreeze_all(model):
    for p in model.parameters(): p.requires_grad = True

def build_optimizer(model, lr_backbone, lr_head, weight_decay, train_head_only=False):
    head_names = _head_param_names(model)
    if train_head_only:
        params = [p for n, p in model.named_parameters() if p.requires_grad and (n in head_names)]
        return torch.optim.AdamW(params, lr=lr_head, weight_decay=weight_decay)
    backbone_params = [p for n, p in model.named_parameters() if p.requires_grad and (n not in head_names)]
    head_params     = [p for n, p in model.named_parameters() if p.requires_grad and (n in head_names)]
    groups = []
    if backbone_params: groups.append({"params": backbone_params, "lr": lr_backbone})
    if head_params:     groups.append({"params": head_params,     "lr": lr_head})
    return torch.optim.AdamW(groups, weight_decay=weight_decay)

def current_lrs(optimizer):
    return [round(g["lr"], 8) for g in optimizer.param_groups]

# -------------------- Train / Eval --------------------
def run(train_csv=SPLITS / "train.csv",
        val_csv=SPLITS / "val.csv",
        out="checkpoints/convnext_tiny_third.pt",
        size=320, batch_size=24, epochs=24,
        # Phase controls
        freeze_epochs=3,
        partial_unfreeze=True,
        full_unfreeze_at=None,
        # LRs & WD
        lr_head_phase1=1e-3,
        lr_backbone_phase2=3e-5,
        lr_head_phase2=8e-5,
        weight_decay_phase1=1e-4,
        weight_decay_phase2=1e-5,
        # Warmup
        warmup_epochs_phase2=3,
        # Other
        grad_clip=1.0,
        mixup_alpha=0.35,
        # Sampler
        use_weighted_sampler=True,
        class2_sampler_boost=1.4,
        # Loss tweaks
        focal_gamma=2.0,
        cb_beta=0.999,
        class2_loss_boost=1.3,
        # Logit adjustment
        tau_logit_adjust=0.5):

    # Classes
    classes = sorted(pd.read_csv(train_csv)["label"].unique().tolist())

    # Datasets / Loaders
    train_ds = CSVDataset(train_csv, classes=classes, train=True,  size=size)
    val_ds   = CSVDataset(val_csv,   classes=classes, train=False, size=size)

    if use_weighted_sampler:
        sampler, counts = _build_weighted_sampler(train_csv, classes, class2_boost=class2_sampler_boost)
        train_dl = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                              num_workers=0, pin_memory=True, drop_last=True)
    else:
        # still need counts for priors/loss
        df_tmp = pd.read_csv(train_csv)
        counts = df_tmp["label"].value_counts().sort_index().reindex(classes).fillna(0).astype(int).to_numpy()
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=True, drop_last=True)

    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                          num_workers=0, pin_memory=True)

    # Priors for logit adjustment
    priors = torch.tensor(counts / np.maximum(counts.sum(), 1), dtype=torch.float, device=DEVICE)

    def adjust_logits(logits):
        # prior correction
        return logits + tau_logit_adjust * torch.log(priors + 1e-12)

    # Model
    model = timm.create_model("convnext_tiny.in12k_ft_in1k",
                              pretrained=True, num_classes=len(classes))
    model.to(DEVICE)

    # Class-balanced focal loss
    cbw = class_balanced_weights(counts, beta=cb_beta).to(DEVICE)
    if len(cbw) > 2:
        cbw[2] *= class2_loss_boost  # extra nudge for class 2
    loss_fn = FocalLoss(gamma=focal_gamma, weight=cbw)

    best_score = -1.0  # macro-F1 based saving
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
                x_in, mix = xb, None
                if mixup_alpha and mixup_alpha > 0:
                    x_in, mix, _ = mixup_batch(xb, yb, alpha=mixup_alpha)

                logits = model(x_in)
                logits = adjust_logits(logits)

                if mix is None:
                    loss = loss_fn(logits, yb)
                    pred = logits.argmax(1)
                    total += yb.size(0)
                    correct += (pred == yb).sum().item()
                else:
                    loss = mixup_criterion(loss_fn, logits, mix)
                    # for acc, use argmax vs original y
                    pred = logits.argmax(1)
                    total += yb.size(0)
                    correct += (pred == yb).sum().item()

                loss.backward()
                if grad_clip: nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                opt.step()
                running_loss += loss.item() * xb.size(0)

            # Validate
            model.eval()
            vtotal, vcorrect, vloss = 0, 0, 0.0
            all_pred, all_true = [], []
            with torch.no_grad():
                for xb, yb in tqdm(val_dl, desc="Val"):
                    xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                    logits = model(xb)
                    logits = adjust_logits(logits)
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
            macro_f1 = f1_score(all_true, all_pred, average='macro') if all_true else 0.0

            print(f"[Phase-1] Epoch {ep}: train_loss={train_loss:.4f} acc={train_acc:.3f} | "
                  f"val_loss={val_loss:.4f} acc={val_acc:.3f} bal_acc={bal_acc:.3f} macroF1={macro_f1:.3f} | lrs={current_lrs(opt)}")

            if macro_f1 > best_score:
                best_score = macro_f1
                torch.save({"model": model.state_dict(), "classes": classes}, out)
                print(f"✓ Saved (macroF1={macro_f1:.3f}):", out)

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
                x_in, mix = xb, None
                if mixup_alpha and mixup_alpha > 0:
                    x_in, mix, _ = mixup_batch(xb, yb, alpha=mixup_alpha)

                logits = model(x_in)
                logits = adjust_logits(logits)

                if mix is None:
                    loss = loss_fn(logits, yb)
                    pred = logits.argmax(1)
                    total += yb.size(0)
                    correct += (pred == yb).sum().item()
                else:
                    loss = mixup_criterion(loss_fn, logits, mix)
                    pred = logits.argmax(1)
                    total += yb.size(0)
                    correct += (pred == yb).sum().item()

                loss.backward()
                if grad_clip: nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                opt.step()
                running_loss += loss.item() * xb.size(0)

            # Validate
            model.eval()
            vtotal, vcorrect, vloss = 0, 0, 0.0
            all_pred, all_true = [], []
            with torch.no_grad():
                for xb, yb in tqdm(val_dl, desc="Val"):
                    xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                    logits = model(xb)
                    logits = adjust_logits(logits)
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
            macro_f1 = f1_score(all_true, all_pred, average='macro') if all_true else 0.0

            print(f"[Phase-2] Epoch {ep}: train_loss={train_loss:.4f} acc={train_acc:.3f} | "
                  f"val_loss={val_loss:.4f} acc={val_acc:.3f} bal_acc={bal_acc:.3f} macroF1={macro_f1:.3f} | lrs={current_lrs(opt)}")

            if macro_f1 > best_score:
                best_score = macro_f1
                torch.save({"model": model.state_dict(), "classes": classes}, out)
                print(f"✓ Saved (macroF1={macro_f1:.3f}):", out)

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