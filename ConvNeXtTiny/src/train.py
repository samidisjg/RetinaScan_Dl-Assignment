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

def _build_weighted_sampler(train_csv, classes):
    """
    Create a WeightedRandomSampler so minority classes are sampled more often.
    Assumes CSVDataset reads rows in the same order as the CSV.
    """
    df = pd.read_csv(train_csv).reset_index(drop=True)

    # Map labels in CSV to class indices [0..C-1] using 'classes' order
    class_to_idx = {c: i for i, c in enumerate(sorted(classes))}
    labels_idx = df["label"].map(class_to_idx).to_numpy()

    # Count per class and compute inverse-frequency weights
    num_classes = len(classes)
    counts = np.bincount(labels_idx, minlength=num_classes)
    counts = np.maximum(counts, 1)  # avoid div-by-zero just in case
    class_weights = 1.0 / torch.tensor(counts, dtype=torch.float)

    # Per-sample weight = weight of its class
    sample_weights = class_weights[torch.from_numpy(labels_idx)]

    # Replacement=True => true oversampling (minority samples can repeat)
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    # Helpful prints
    print("Class counts:", {i: int(c) for i, c in enumerate(counts)})
    print("Class weights (inv freq):", class_weights.tolist())

    return sampler

def run(train_csv=SPLITS / "train.csv",
        val_csv=SPLITS / "val.csv",
        out="checkpoints/convnext_tiny.pt",
        size=224, batch_size=32, epochs=10, lr=3e-4,
        use_weighted_sampler=True):
    # Discover classes from train csv (sorted for stable id order)
    classes = sorted(pd.read_csv(train_csv)["label"].unique().tolist())

    # Datasets
    train_ds = CSVDataset(train_csv, classes=classes, train=True,  size=size)
    val_ds   = CSVDataset(val_csv,   classes=classes, train=False, size=size)

    # DataLoaders
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

    # Opt/Sched/Loss (unchanged)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss()

    best_acc = 0.0
    os.makedirs(os.path.dirname(out), exist_ok=True)

    for ep in range(1, epochs + 1):
        # --------------------- Train ---------------------
        model.train()
        total, correct, running_loss = 0, 0, 0.0
        for xb, yb in tqdm(train_dl, desc=f"Train {ep}/{epochs}"):
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()

            running_loss += loss.item() * xb.size(0)
            pred = logits.argmax(1)
            total += yb.size(0)
            correct += (pred == yb).sum().item()

        train_acc = correct / total if total else 0.0
        train_loss = running_loss / total if total else 0.0

        # --------------------- Validate ---------------------
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

        val_acc = vcorrect / vtotal if vtotal else 0.0
        val_loss = vloss / vtotal if vtotal else 0.0
        bal_acc = balanced_accuracy_score(all_true, all_pred) if all_true else 0.0

        print(f"Epoch {ep}: train_loss={train_loss:.4f} acc={train_acc:.3f} | "
              f"val_loss={val_loss:.4f} acc={val_acc:.3f} bal_acc={bal_acc:.3f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({"model": model.state_dict(), "classes": classes}, out)
            print("✓ Saved:", out)

        sched.step()

    # --------- Final per-class report on validation set ---------
    label_ids = list(range(len(classes)))  # [0, 1, 2, ...]
    names = [str(c) for c in classes]     # ["0","1",...]
    print(classification_report(
        all_true, all_pred,
        labels=label_ids,
        target_names=names,
        zero_division=0,
        digits=4
    ))

if __name__ == "__main__":
    run()