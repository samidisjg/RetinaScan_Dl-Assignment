# scripts/visualize_like_paper.py
import os, json, math
import numpy as np
import matplotlib.pyplot as plt

def _safe_list(seq, key):
    return [e.get(key, np.nan) if isinstance(e, dict) else np.nan for e in seq]

def _maybe(path):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="path to outputs/<run-id> folder")
    ap.add_argument("--smooth", type=int, default=0, help="moving-average window (epochs), 0=off")
    args = ap.parse_args()

    run = os.path.abspath(args.run)
    hist_path = os.path.join(run, "history.json")
    if not os.path.exists(hist_path):
        raise FileNotFoundError(f"history.json not found: {hist_path}")

    with open(hist_path, "r") as f:
        hist = json.load(f)

    train_hist = hist.get("train", [])
    valid_hist = hist.get("valid", [])

    # per-epoch series (train may be missing if you didn't patch the trainer)
    tr_loss = _safe_list(train_hist, "loss")
    tr_grade_loss = _safe_list(train_hist, "loss_grade")
    tr_acc = _safe_list(train_hist, "grade_acc")
    tr_auc = _safe_list(train_hist, "edema_auc")

    va_loss = _safe_list(valid_hist, "val_loss")
    va_grade_loss = _safe_list(valid_hist, "loss_grade")  # some versions save this
    # if not present, approximate via val_loss - (edema loss) … but we likely don't have it. leave NaN if missing.
    va_acc = _safe_list(valid_hist, "grade_acc")
    va_auc = _safe_list(valid_hist, "edema_auc")

    # optional smoothing
    def smooth(y, k):
        if k <= 1: return y
        y = np.array(y, dtype=float)
        mask = np.isfinite(y)
        if not mask.any(): return y.tolist()
        out = y.copy()
        valid_idx = np.where(mask)[0]
        vals = y[mask]
        sm = np.convolve(vals, np.ones(k)/k, mode="valid")
        # place smoothed section centered over valid span
        start = valid_idx[0] + k//2
        out[:] = np.nan
        out[start:start+len(sm)] = sm
        return out.tolist()

    k = int(args.smooth)
    tr_loss, va_loss = smooth(tr_loss, k), smooth(va_loss, k)
    tr_acc,  va_acc  = smooth(tr_acc,  k), smooth(va_acc,  k)
    tr_auc,  va_auc  = smooth(tr_auc,  k), smooth(va_auc,  k)
    tr_grade_loss, va_grade_loss = smooth(tr_grade_loss, k), smooth(va_grade_loss, k)

    # bring in final split metrics if available
    m_train = _maybe(os.path.join(run, "report_train", "metrics.json"))
    m_test  = _maybe(os.path.join(run, "report_test",  "metrics.json"))

    # === 2x2 PANEL ============================================================
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5))
    E = range(len(va_acc))  # x-axis = epochs on validation series length

    # (1) grade accuracy
    ax = axes[0,0]
    if any(np.isfinite(tr_acc)): ax.plot(E, tr_acc, label="train")
    ax.plot(E, va_acc, label="val")
    if m_train: ax.plot([len(va_acc)-1], [m_train.get("grade_acc", np.nan)], "o", label="train final")
    if m_test:  ax.plot([len(va_acc)-1], [m_test.get("grade_acc",  np.nan)], "s", label="test final")
    ax.set_title("Grade accuracy"); ax.set_xlabel("epoch"); ax.set_ylabel("accuracy"); ax.legend()

    # (2) edema AUC
    ax = axes[0,1]
    if any(np.isfinite(tr_auc)): ax.plot(E, tr_auc, label="train")
    ax.plot(E, va_auc, label="val")
    if m_train: ax.plot([len(va_acc)-1], [m_train.get("edema_auc", np.nan)], "o", label="train final")
    if m_test:  ax.plot([len(va_acc)-1], [m_test.get("edema_auc",  np.nan)], "s", label="test final")
    ax.set_title("Edema AUC"); ax.set_xlabel("epoch"); ax.set_ylabel("AUC"); ax.legend()

    # (3) total loss
    ax = axes[1,0]
    if any(np.isfinite(tr_loss)): ax.plot(E, tr_loss, label="train")
    ax.plot(E, va_loss, label="val")
    ax.set_title("Total loss"); ax.set_xlabel("epoch"); ax.set_ylabel("loss"); ax.legend()

    # (4) grade loss
    ax = axes[1,1]
    if any(np.isfinite(tr_grade_loss)): ax.plot(E, tr_grade_loss, label="train")
    if any(np.isfinite(va_grade_loss)): ax.plot(E, va_grade_loss, label="val")
    ax.set_title("Grade loss"); ax.set_xlabel("epoch"); ax.set_ylabel("loss"); ax.legend()

    plt.tight_layout()
    out_panel = os.path.join(run, "realistic_training_curves.png")
    plt.savefig(out_panel, dpi=200)
    print("saved:", out_panel)

    # === BAR CHART: final train/val/test =====================================
    # derive final val metrics from history.json
    v_last = valid_hist[-1] if valid_hist else {}
    v_acc_last = v_last.get("grade_acc", np.nan)
    v_auc_last = v_last.get("edema_auc", np.nan)

    # train/test may come only from reports
    t_acc = m_train.get("grade_acc", np.nan) if m_train else np.nan
    t_auc = m_train.get("edema_auc",  np.nan) if m_train else np.nan
    te_acc = m_test.get("grade_acc",  np.nan) if m_test else np.nan
    te_auc = m_test.get("edema_auc",   np.nan) if m_test else np.nan

    labels = ["train","val","test"]
    accs = [t_acc, v_acc_last, te_acc]
    aucs = [t_auc, v_auc_last, te_auc]

    fig2, ax2 = plt.subplots(1,2, figsize=(9.5,4.2))
    xpos = np.arange(len(labels))
    ax2[0].bar(xpos, accs); ax2[0].set_xticks(xpos, labels); ax2[0].set_ylim(0,1)
    ax2[0].set_title("Grade accuracy (final)")
    ax2[0].set_ylabel("accuracy")

    ax2[1].bar(xpos, aucs); ax2[1].set_xticks(xpos, labels); ax2[1].set_ylim(0,1)
    ax2[1].set_title("Edema AUC (final)")

    plt.tight_layout()
    out_bars = os.path.join(run, "final_split_compare.png")
    plt.savefig(out_bars, dpi=200)
    print("saved:", out_bars)
