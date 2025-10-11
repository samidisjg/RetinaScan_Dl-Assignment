import argparse, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_curve(df, x, ys, labels, title, ylabel, out_path, best_idx=None):
    plt.figure(figsize=(7,5))
    for y, label in zip(ys, labels):
        plt.plot(df[x], df[y], label=label, linewidth=2)
    if best_idx is not None:
        bepoch = int(df.loc[best_idx, x])
        yval   = float(df.loc[best_idx, ys[1]])  # assume val metric second
        plt.scatter([bepoch], [yval], s=60, marker="o", color="red")
        plt.annotate(f"best @ {bepoch}", (bepoch, yval), xytext=(5,5),
                     textcoords="offset points", fontsize=9, color="red")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_confusion(cm, out_path, class_names=None, normalize=False):
    plt.figure(figsize=(6,5))
    if normalize:
        cm = cm.astype(np.float32)
        cm = cm / cm.sum(axis=1, keepdims=True)
        fmt = ".2f"
        cmap = "YlGnBu"
        title = "Normalized Confusion Matrix"
    else:
        fmt = "d"
        cmap = "coolwarm"
        title = "Confusion Matrix"

    sns.heatmap(cm, annot=True, fmt=fmt, cmap=cmap,
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main(args):
    run_dir = args.run_dir
    hist_path = os.path.join(run_dir, "history.csv")
    if not os.path.exists(hist_path):
        raise FileNotFoundError(f"❌ history.csv not found in {run_dir}")

    df = pd.read_csv(hist_path)
    for col in ["epoch","train_loss","val_loss","train_acc","val_acc","train_f1","val_f1"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    best_idx = None
    if "val_f1" in df.columns:
        best_idx = int(df["val_f1"].idxmax())

    # ---- Curves ----
    plot_curve(df, "epoch", ["train_loss","val_loss"],
               ["Train Loss","Val Loss"], "Loss vs Epoch",
               "Loss", os.path.join(run_dir,"loss_curve.png"))
    if "train_acc" in df.columns and "val_acc" in df.columns:
        plot_curve(df, "epoch", ["train_acc","val_acc"],
                   ["Train Accuracy","Val Accuracy"], "Accuracy vs Epoch",
                   "Accuracy", os.path.join(run_dir,"acc_curve.png"))
    if "train_f1" in df.columns and "val_f1" in df.columns:
        plot_curve(df, "epoch", ["train_f1","val_f1"],
                   ["Train F1","Val F1"], "Macro-F1 vs Epoch",
                   "Macro-F1", os.path.join(run_dir,"f1_curve.png"), best_idx)

    # ---- Confusion Matrices ----
    cm_path = os.path.join(run_dir, "cm_eval.npy")
    if os.path.exists(cm_path):
        cm = np.load(cm_path)
        class_names = ["No DR","Mild","Moderate","Severe","Proliferative"]

        raw_path = os.path.join(run_dir, "cm_heatmap_raw.png")
        norm_path = os.path.join(run_dir, "cm_heatmap_normalized.png")

        plot_confusion(cm, raw_path, class_names, normalize=False)
        plot_confusion(cm, norm_path, class_names, normalize=True)
        print(f"✅ Saved confusion matrix heatmaps →\n  - {raw_path}\n  - {norm_path}")
    else:
        print("⚠️ No cm_eval.npy found. Skipping confusion matrix plotting.")

    print(f"✅ All plots saved in: {run_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True,
                    help="Folder that contains history.csv & cm_eval.npy (e.g., runs/d121_448_focal)")
    args = ap.parse_args()
    main(args)