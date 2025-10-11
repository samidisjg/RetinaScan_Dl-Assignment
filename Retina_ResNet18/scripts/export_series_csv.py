# scripts/export_series_csv.py
import os, json, csv, numpy as np

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--history", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    with open(args.history) as f:
        hist = json.load(f)

    train, valid = hist.get("train", []), hist.get("valid", [])
    n = max(len(train), len(valid))
    rows = []
    for i in range(n):
        t = train[i] if i < len(train) else {}
        v = valid[i] if i < len(valid) else {}
        rows.append({
            "epoch": i+1,
            "train_loss": t.get("loss", np.nan),
            "train_grade_loss": t.get("loss_grade", np.nan),
            "train_grade_acc": t.get("grade_acc", np.nan),
            "train_edema_auc": t.get("edema_auc", np.nan),
            "val_loss": v.get("val_loss", np.nan),
            "val_grade_loss": v.get("loss_grade", np.nan),
            "val_grade_acc": v.get("grade_acc", np.nan),
            "val_edema_auc": v.get("edema_auc", np.nan),
        })

    out = args.out or os.path.join(os.path.dirname(args.history), "training_series.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print("wrote:", out)
