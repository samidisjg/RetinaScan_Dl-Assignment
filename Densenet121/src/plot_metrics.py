import argparse, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_curve(df, x, ys, labels, title, ylabel, out_path, best_idx=None):
    plt.figure(figsize=(7,5))
    for y, label in zip(ys, labels):
        plt.plot(df[x], df[y], label=label)
    if best_idx is not None:
        bepoch = int(df.loc[best_idx, x])
        yval   = float(df.loc[best_idx, ys[1]])  # assume ys[1] is the validation metric
        plt.scatter([bepoch], [yval], s=60, marker="o")
        plt.annotate(f"best @ {bepoch}", (bepoch, yval), xytext=(5,5),
                     textcoords="offset points", fontsize=9)
    plt.title(title)
    plt.xlabel("epoch")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def plot_confusion(cm, out_path, class_names=None):
    plt.figure(figsize=(6,5))
    im = plt.imshow(cm, aspect="auto")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    if class_names is None:
        class_names = [str(i) for i in range(cm.shape[0])]
    plt.xticks(ticks=np.arange(len(class_names)), labels=class_names, rotation=45, ha="right")
    plt.yticks(ticks=np.arange(len(class_names)), labels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    # annotate cells
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=9, color="white" if cm[i,j] > cm.max()*0.6 else "black")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def main(args):
    run_dir = args.run_dir
    hist_path = os.path.join(run_dir, "history.csv")
    if not os.path.exists(hist_path):
        raise FileNotFoundError(f"history.csv not found in {run_dir}")

    df = pd.read_csv(hist_path)
    # ensure numeric types
    for col in ["epoch","train_loss","val_loss","train_acc","val_acc","train_f1","val_f1"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # best val-f1 index (for marker)
    best_idx = None
    if "val_f1" in df.columns:
        best_idx = int(df["val_f1"].idxmax())

    # Loss curve
    plot_curve(
        df, x="epoch",
        ys=["train_loss", "val_loss"],
        labels=["train loss", "val loss"],
        title="Loss vs Epoch",
        ylabel="loss",
        out_path=os.path.join(run_dir, "loss_curve.png"),
        best_idx=None
    )

    # Accuracy curve
    if "train_acc" in df.columns and "val_acc" in df.columns:
        plot_curve(
            df, x="epoch",
            ys=["train_acc", "val_acc"],
            labels=["train acc", "val acc"],
            title="Accuracy vs Epoch",
            ylabel="accuracy",
            out_path=os.path.join(run_dir, "acc_curve.png"),
            best_idx=None
        )

    # F1 curve (mark best val-F1)
    if "train_f1" in df.columns and "val_f1" in df.columns:
        plot_curve(
            df, x="epoch",
            ys=["train_f1", "val_f1"],
            labels=["train macro-F1", "val macro-F1"],
            title="Macro-F1 vs Epoch",
            ylabel="macro-F1",
            out_path=os.path.join(run_dir, "f1_curve.png"),
            best_idx=best_idx
        )

    # Confusion matrix (optional, from evaluate.py)
    cm_path = os.path.join(run_dir, "cm_eval.npy")
    if os.path.exists(cm_path):
        cm = np.load(cm_path)
        # Try to infer class names if available
        class_names = None
        # If your class_to_idx.json exists in run_dir, you can load and invert it to get names.
        # For now we default to numeric labels:
        plot_confusion(cm, os.path.join(run_dir, "confusion_matrix.png"), class_names)

    print("Saved plots to:", run_dir)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True, help="Folder that contains history.csv (e.g., runs/d121_448_focal)")
    args = ap.parse_args()
    main(args)