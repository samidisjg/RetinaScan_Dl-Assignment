import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

CSV_PATH = "checkpoints/history/history_convnext_tiny_new.csv"
OUT_DIR = "training_plots"
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(CSV_PATH).reset_index(drop=True)

# Running epoch index for x-axis
df["epoch_idx"] = np.arange(1, len(df) + 1)

def save_line(x, y, xlabel, ylabel, title, fname):
    plt.figure()
    plt.plot(x, y)
    plt.xlabel(xlabel); plt.ylabel(ylabel); plt.title(title)
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, fname), dpi=150)
    plt.close()

# Core curves
save_line(df["epoch_idx"], df["train_loss"], "Epoch", "Train Loss", "Train Loss vs Epoch", "train_loss.png")
save_line(df["epoch_idx"], df["val_loss"],   "Epoch", "Val Loss",   "Val Loss vs Epoch",   "val_loss.png")
save_line(df["epoch_idx"], df["train_acc"],  "Epoch", "Train Accuracy", "Train Accuracy vs Epoch", "train_acc.png")
save_line(df["epoch_idx"], df["val_acc"],    "Epoch", "Val Accuracy",   "Val Accuracy vs Epoch",   "val_acc.png")
save_line(df["epoch_idx"], df["balanced_acc"], "Epoch", "Balanced Accuracy", "Balanced Accuracy vs Epoch", "balanced_acc.png")
save_line(df["epoch_idx"], df["macro_f1"],     "Epoch", "Macro F1",        "Macro F1 vs Epoch",        "macro_f1.png")
save_line(df["epoch_idx"], df["lr_0"], "Epoch", "Learning Rate (group 0)", "LR Group 0 vs Epoch", "lr_group0.png")
save_line(df["epoch_idx"], df.get("lr_1", df["lr_0"]), "Epoch", "Learning Rate (group 1)", "LR Group 1 vs Epoch", "lr_group1.png")
save_line(df["epoch_idx"], df["epoch_seconds"], "Epoch", "Seconds", "Epoch Time vs Epoch", "epoch_time.png")

# Per-class recall / F1 curves (if present)
recall_cols = [c for c in df.columns if c.startswith("recall_")]
f1_cols     = [c for c in df.columns if c.startswith("f1_")]

if recall_cols:
    for c in recall_cols:
        save_line(df["epoch_idx"], df[c], "Epoch", "Recall", f"{c} vs Epoch", f"{c}.png")

if f1_cols:
    for c in f1_cols:
        save_line(df["epoch_idx"], df[c], "Epoch", "F1-score", f"{c} vs Epoch", f"{c}.png")

print(f"Saved plots to: {OUT_DIR}/")



