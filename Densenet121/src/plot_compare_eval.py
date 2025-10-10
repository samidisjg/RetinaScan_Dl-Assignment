import argparse, os, re
import pandas as pd
import matplotlib.pyplot as plt

def read_eval_report(path):
    acc = f1 = None
    if not os.path.exists(path):
        return acc, f1
    with open(path, "r") as f:
        txt = f.read()
    m_acc = re.search(r"Accuracy:\s*([0-9.]+)", txt)
    m_f1  = re.search(r"Macro-F1:\s*([0-9.]+)", txt)
    if m_acc: acc = float(m_acc.group(1))
    if m_f1:  f1  = float(m_f1.group(1))
    return acc, f1

def bar_plot(labels, values, ylabel, title, out_path):
    plt.figure(figsize=(7,5))
    bars = plt.bar(labels, values)
    for b, v in zip(bars, values):
        plt.text(b.get_x()+b.get_width()/2, v, f"{v:.3f}", ha="center", va="bottom")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def main(args):
    labels = args.labels if args.labels else [os.path.basename(d.rstrip("/")) for d in args.run_dirs]
    accs, f1s = [], []
    for d in args.run_dirs:
        report = os.path.join(d, "report_eval.txt")
        acc, f1 = read_eval_report(report)
        if acc is None or f1 is None:
            raise SystemExit(f"Missing/invalid report_eval.txt in {d}")
        accs.append(acc)
        f1s.append(f1)

    os.makedirs(args.out_dir, exist_ok=True)
    bar_plot(labels, accs, "Accuracy", "Final Accuracy (eval)", os.path.join(args.out_dir, "compare_eval_accuracy.png"))
    bar_plot(labels, f1s,  "Macro-F1", "Final Macro-F1 (eval)", os.path.join(args.out_dir, "compare_eval_macro_f1.png"))

    # Also export a small CSV summary for your report
    df = pd.DataFrame({"run": labels, "accuracy": accs, "macro_f1": f1s})
    df.to_csv(os.path.join(args.out_dir, "compare_eval_summary.csv"), index=False)
    print(f"Saved plots & summary to: {args.out_dir}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dirs", nargs="+", required=True,
                    help="Run directories that contain report_eval.txt")
    ap.add_argument("--labels", nargs="*", help="Optional labels for plots")
    ap.add_argument("--out_dir", type=str, default="runs/compare_eval")
    args = ap.parse_args()
    main(args)