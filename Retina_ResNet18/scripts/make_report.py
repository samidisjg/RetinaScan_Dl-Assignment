# scripts/make_report.py

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report

# Local helpers ---------------------------------------------------------------

def ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p

def to_jsonable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float16, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int8, np.int16, np.int32, np.int64)):
        return int(obj)
    return obj

# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="configs/resnet18.yaml")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="test", choices=["train", "valid", "test"])
    args = ap.parse_args()

    # --- Load config
    from src.config import load_config
    from src.datasets.retina import RetinaDataset
    from src.models.resnet18_multitask import ResNet18MultiTask

    cfg = load_config(args.cfg)

    if args.split == "train":
        csv_rel, img_rel = cfg.train_csv, cfg.train_img_dir
    elif args.split == "valid":
        csv_rel, img_rel = cfg.valid_csv, cfg.valid_img_dir
    else:
        csv_rel, img_rel = cfg.test_csv, cfg.test_img_dir

    csv_path = os.path.join(cfg.data_dir, csv_rel)
    img_dir = os.path.join(cfg.data_dir, img_rel)

    df = pd.read_csv(csv_path)

    # --- Dataset/Dataloader
    ds = RetinaDataset(
        csv_path=csv_path,
        img_dir=img_dir,
        col_image=cfg.columns.image,
        col_grade=cfg.columns.grade,
        col_edema=cfg.columns.edema,
        img_size=int(cfg.img_size),
        center_crop=getattr(cfg, "center_crop", None),
        is_train=False,
    )
    dl = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=False, num_workers=int(cfg.num_workers))

    # --- Model
    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = ResNet18MultiTask(num_classes_grade=int(cfg.num_classes_grade))
    model.load_state_dict(state["model_state"])
    model.eval()

    # --- Inference
    g_true, g_pred, e_true, e_prob = [], [], [], []
    with torch.no_grad():
        for imgs, yg, ye in dl:
            lg, le = model(imgs)
            g_true.append(yg.numpy())
            g_pred.append(lg.argmax(1).numpy())
            e_true.append(ye.numpy())
            e_prob.append(torch.sigmoid(le).numpy())

    g_true = np.concatenate(g_true)
    g_pred = np.concatenate(g_pred)
    e_true = np.concatenate(e_true)
    e_prob = np.concatenate(e_prob)

    # --- Metrics
    from src.engine.metrics import compute_metrics
    metrics = compute_metrics(g_true, g_pred, e_true, e_prob)

    out_dir = ensure_dir(os.path.join(os.path.dirname(args.ckpt), f"report_{args.split}"))

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump({k: to_jsonable(v) for k, v in metrics.items()}, f, indent=2)

    # --- Confusion matrices
    sns.set_context("talk")
    labels = [str(i) for i in range(int(cfg.num_classes_grade))]
    cm = confusion_matrix(g_true, g_pred, labels=list(range(int(cfg.num_classes_grade))))
    cm_norm = cm.astype(float) / np.clip(cm.sum(axis=1, keepdims=True), a_min=1, a_max=None)

    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix (Counts)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "confusion_matrix_counts.png"), dpi=250)

    plt.figure(figsize=(7, 6))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Greens", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix (Normalized)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "confusion_matrix_normalized.png"), dpi=250)

    # --- ROC for edema
    fpr, tpr, _ = roc_curve(e_true, e_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], "--")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("Edema ROC")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "roc_edema.png"), dpi=250)

    # --- TXT classification report (with zero_division=0 to avoid warnings)
    report_txt = classification_report(g_true, g_pred, digits=4, zero_division=0)
    with open(os.path.join(out_dir, "classification_report.txt"), "w") as f:
        f.write(report_txt)

    # --- Markdown quick report
    pos = int((df[cfg.columns.edema] == 1).sum())
    neg = int((df[cfg.columns.edema] == 0).sum())
    grade_counts = df[cfg.columns.grade].value_counts().sort_index().to_dict()

    md = []
    md.append(f"# ResNet18 Retina — {args.split.capitalize()} Report")
    md.append("")
    md.append("## Dataset")
    md.append(f"Rows: **{len(df)}**  |  Grade col: `{cfg.columns.grade}`  |  Edema col: `{cfg.columns.edema}`")
    md.append(f"Grade counts: `{grade_counts}`")
    md.append(f"Edema pos/neg: **{pos}/{neg}**")
    md.append("")
    md.append("## Metrics")
    md.append(f"- Grade Accuracy: **{metrics['grade_acc']:.4f}**")
    md.append(f"- Grade Macro-F1: **{metrics['grade_f1_macro']:.4f}**")
    md.append(f"- Edema ROC-AUC: **{metrics['edema_auc']:.4f}**")
    md.append("")
    md.append("## Figures")
    md.append("![](confusion_matrix_counts.png)")
    md.append("![](confusion_matrix_normalized.png)")
    md.append("![](roc_edema.png)")
    with open(os.path.join(out_dir, "report.md"), "w") as f:
        f.write("\n".join(md))

    print("Saved detailed report to:", out_dir)
