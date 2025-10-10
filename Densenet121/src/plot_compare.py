import argparse, os
import pandas as pd
import matplotlib.pyplot as plt

def to_num(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def plot_together(runs, metric_col, out_path, ylabel, title):
    plt.figure(figsize=(7,5))
    for label, df in runs:
        if metric_col not in df.columns or "epoch" not in df.columns:
            continue
        plt.plot(df["epoch"], df[metric_col], label=label, linewidth=2)
    plt.grid(True, alpha=0.3)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def bar_best(runs, metric_col, out_path, title):
    labels, bests = [], []
    for label, df in runs:
        if metric_col in df.columns:
            bests.append(df[metric_col].max())
            labels.append(label)
    plt.figure(figsize=(7,5))
    bars = plt.bar(labels, bests, color=["#4e79a7", "#f28e2b", "#e15759"])
    for bar, v in zip(bars, bests):
        plt.text(bar.get_x() + bar.get_width()/2, v, f"{v:.3f}", ha="center", va="bottom")
    plt.ylabel(metric_col)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def main(args):
    labels = args.labels if args.labels else [os.path.basename(d.rstrip("/")) for d in args.run_dirs]
    runs = []
    for label, d in zip(labels, args.run_dirs):
        csv_path = os.path.join(d, "history.csv")
        if not os.path.exists(csv_path):
            print(f"[skip] No history.csv in {d}")
            continue
        df = pd.read_csv(csv_path)
        df = to_num(df, ["epoch","train_loss","val_loss","train_acc","val_acc","train_f1","val_f1"])
        runs.append((label, df))

    if not runs:
        raise SystemExit("No valid runs found.")

    os.makedirs(args.out_dir, exist_ok=True)

    # Plot validation accuracy
    plot_together(runs, "val_acc", os.path.join(args.out_dir, "compare_val_acc.png"),
                  "Validation Accuracy", "Validation Accuracy vs Epoch")

    # Plot validation macro-F1
    plot_together(runs, "val_f1", os.path.join(args.out_dir, "compare_val_f1.png"),
                  "Validation Macro-F1", "Validation Macro-F1 vs Epoch")

    # Bar chart of best macro-F1
    bar_best(runs, "val_f1", os.path.join(args.out_dir, "best_val_f1_bar.png"),
             "Best Validation Macro-F1 per Model")

    print(f"✅ Plots saved in: {args.out_dir}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dirs", nargs="+", required=True,
                    help="List of run directories (each with history.csv)")
    ap.add_argument("--labels", nargs="*", help="Optional labels for the plots")
    ap.add_argument("--out_dir", type=str, default="runs/compare_plots")
    args = ap.parse_args()
    main(args)