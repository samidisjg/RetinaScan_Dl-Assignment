import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

CSV_PATH = "checkpoints/history/history_convnext_tiny_new.csv"
OUT_DIR = "training_plots"
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(CSV_PATH).reset_index(drop=True)
df["epoch_idx"] = np.arange(1, len(df) + 1)

def save_plot(fig, name):
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, name), dpi=150)
    plt.close(fig)

# ------------------------
# Combined plots (train + val)
# ------------------------

# 1️⃣ Loss: Train vs Validation
fig, ax = plt.subplots()
ax.plot(df["epoch_idx"], df["train_loss"], label="Train Loss")
ax.plot(df["epoch_idx"], df["val_loss"], label="Validation Loss")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.set_title("Train vs Validation Loss")
ax.grid(True, linestyle="--", linewidth=0.5)
ax.legend()
save_plot(fig, "loss_combined.png")

# 2️⃣ Accuracy: Train vs Validation
fig, ax = plt.subplots()
ax.plot(df["epoch_idx"], df["train_acc"], label="Train Accuracy")
ax.plot(df["epoch_idx"], df["val_acc"], label="Validation Accuracy")
ax.set_xlabel("Epoch")
ax.set_ylabel("Accuracy")
ax.set_title("Train vs Validation Accuracy")
ax.grid(True, linestyle="--", linewidth=0.5)
ax.legend()
save_plot(fig, "accuracy_combined.png")

# 3️⃣ Balanced Accuracy & Macro F1 together
fig, ax = plt.subplots()
ax.plot(df["epoch_idx"], df["balanced_acc"], label="Balanced Accuracy")
ax.plot(df["epoch_idx"], df["macro_f1"], label="Macro F1-score")
ax.set_xlabel("Epoch")
ax.set_ylabel("Score")
ax.set_title("Balanced Accuracy and Macro F1 vs Epoch")
ax.grid(True, linestyle="--", linewidth=0.5)
ax.legend()
save_plot(fig, "balanced_acc_macro_f1.png")

# 4️⃣ Learning rates for both groups
fig, ax = plt.subplots()
ax.plot(df["epoch_idx"], df["lr_0"], label="LR Group 0")
if "lr_1" in df.columns:
    ax.plot(df["epoch_idx"], df["lr_1"], label="LR Group 1")
ax.set_xlabel("Epoch")
ax.set_ylabel("Learning Rate")
ax.set_title("Learning Rates vs Epoch")
ax.grid(True, linestyle="--", linewidth=0.5)
ax.legend()
save_plot(fig, "learning_rates_combined.png")

# 5️⃣ Per-class Recall in one plot
recall_cols = [c for c in df.columns if c.startswith("recall_")]
if recall_cols:
    fig, ax = plt.subplots()
    for c in recall_cols:
        ax.plot(df["epoch_idx"], df[c], label=c)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Recall")
    ax.set_title("Per-class Recall vs Epoch")
    ax.grid(True, linestyle="--", linewidth=0.5)
    ax.legend()
    save_plot(fig, "recall_all_classes.png")

# 6️⃣ Per-class F1 in one plot
f1_cols = [c for c in df.columns if c.startswith("f1_")]
if f1_cols:
    fig, ax = plt.subplots()
    for c in f1_cols:
        ax.plot(df["epoch_idx"], df[c], label=c)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("F1-score")
    ax.set_title("Per-class F1 vs Epoch")
    ax.grid(True, linestyle="--", linewidth=0.5)
    ax.legend()
    save_plot(fig, "f1_all_classes.png")

# 7️⃣ Summary grid (optional)
fig, axs = plt.subplots(2, 2, figsize=(10, 8))
axs = axs.ravel()
axs[0].plot(df["epoch_idx"], df["train_loss"], label="Train Loss")
axs[0].plot(df["epoch_idx"], df["val_loss"], label="Val Loss")
axs[0].set_title("Loss"); axs[0].legend(); axs[0].grid(True, linestyle="--")

axs[1].plot(df["epoch_idx"], df["train_acc"], label="Train Acc")
axs[1].plot(df["epoch_idx"], df["val_acc"], label="Val Acc")
axs[1].set_title("Accuracy"); axs[1].legend(); axs[1].grid(True, linestyle="--")

axs[2].plot(df["epoch_idx"], df["balanced_acc"], label="Balanced Acc")
axs[2].plot(df["epoch_idx"], df["macro_f1"], label="Macro F1")
axs[2].set_title("Balanced Acc & F1"); axs[2].legend(); axs[2].grid(True, linestyle="--")

axs[3].plot(df["epoch_idx"], df["lr_0"], label="LR Group 0")
if "lr_1" in df.columns:
    axs[3].plot(df["epoch_idx"], df["lr_1"], label="LR Group 1")
axs[3].set_title("Learning Rates"); axs[3].legend(); axs[3].grid(True, linestyle="--")

for ax in axs:
    ax.set_xlabel("Epoch")

fig.suptitle("Training Overview", fontsize=14)
save_plot(fig, "overview_grid.png")

print(f"✅ Combined graphs saved to: {OUT_DIR}/")
