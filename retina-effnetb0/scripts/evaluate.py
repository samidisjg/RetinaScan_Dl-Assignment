import os, sys, argparse, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import tensorflow as tf
import tf_keras as keras

# Make src importable when called from scripts/
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.config import load_config
from src.data.tf_dataset import DataBuilder
from src.train.tf_trainer import compile_model

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def evaluate_split(cfg, split="valid"):
    out_dir = cfg.out["run_dir"]
    ensure_dir(out_dir)
    plots_dir = os.path.join(out_dir, "plots"); ensure_dir(plots_dir)
    reports_dir = os.path.join(out_dir, "reports"); ensure_dir(reports_dir)

    # Paths
    split_csv = cfg.paths[f"{split}_csv"]
    split_dir = cfg.paths[f"{split}_dir"]

    # Dataset (no aug, no shuffle)
    db = DataBuilder(tuple(cfg.model["input_shape"]), augment=False)
    ds = db.dataframe_to_ds(split_csv, split_dir, shuffle=False, batch=cfg.train["batch_size"])

    # Load model & compile for eval
    model_path = cfg.out["ckpt"]
    model = keras.models.load_model(model_path, compile=False)
    model = compile_model(model, lr=1e-5, w_grade=cfg.model["loss_weights"]["grade"], w_edema=cfg.model["loss_weights"]["edema"])

    # In-order labels (to attach filenames to preds)
    df = pd.read_csv(split_csv)
    n = len(df)

    # Predict
    preds = model.predict(ds, verbose=1)
    if isinstance(preds, list) or isinstance(preds, tuple):
        # Safety, but we expect dict outputs
        raise RuntimeError("Model outputs not dict-like. Expecting {'grade', 'edema'}")
    p_grade = preds["grade"]        # [N, 5]
    p_edema = preds["edema"].ravel()# [N]

    # True labels
    yg_true = df["Retinopathy grade"].to_numpy().astype(int)
    ye_true = df["Risk of macular edema"].to_numpy().astype(int)

    # Argmax for grade; threshold 0.5 for edema
    yg_pred = np.argmax(p_grade, axis=1)
    ye_pred = (p_edema >= 0.5).astype(int)

    # Reports
    rep_grade = classification_report(yg_true, yg_pred, digits=4)
    cm_grade = confusion_matrix(yg_true, yg_pred)

    report_txt = os.path.join(reports_dir, f"{split}_report.txt")
    with open(report_txt, "w", encoding="utf-8") as f:
        f.write("=== Grade (5-class) ===\n")
        f.write(rep_grade + "\n")
        f.write("Confusion Matrix (rows=true, cols=pred):\n")
        f.write(str(cm_grade) + "\n\n")

        # ROC AUC for edema
        try:
            auc = roc_auc_score(ye_true, p_edema)
            fpr, tpr, _ = roc_curve(ye_true, p_edema)
            f.write(f"=== Edema (binary) ===\nAUC: {auc:.4f}\n")
            # Plot ROC
            plt.figure()
            plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
            plt.plot([0,1], [0,1], linestyle="--")
            plt.xlabel("FPR")
            plt.ylabel("TPR")
            plt.title(f"ROC - Edema ({split})")
            plt.legend(loc="lower right")
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f"roc_edema_{split}.png"), dpi=150)
            plt.close()
        except Exception as e:
            f.write(f"=== Edema (binary) ===\nAUC: N/A ({e})\n")

    # Confusion matrix plot
    plt.figure(figsize=(5,4))
    plt.imshow(cm_grade, interpolation="nearest")
    plt.title(f"Confusion Matrix - Grade ({split})")
    plt.colorbar()
    ticks = np.arange(5)
    plt.xticks(ticks, ticks); plt.yticks(ticks, ticks)
    plt.xlabel("Pred"); plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f"cm_grade_{split}.png"), dpi=150)
    plt.close()

    # Save predictions CSV (top-1 + probs)
    top1_prob = p_grade.max(axis=1)
    pred_df = pd.DataFrame({
        "Image name": df["Image name"],
        "grade_true": yg_true,
        "grade_pred": yg_pred,
        "grade_pred_conf": top1_prob,
        "edema_true": ye_true,
        "edema_pred": ye_pred,
        "edema_proba": p_edema,
    })
    pred_csv = os.path.join(reports_dir, f"predictions_{split}.csv")
    pred_df.to_csv(pred_csv, index=False)

    print(f"\nSaved:\n  - {report_txt}\n  - {plots_dir}\\cm_grade_{split}.png\n  - {plots_dir}\\roc_edema_{split}.png\n  - {pred_csv}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/efficientnet_b0.yaml")
    p.add_argument("--split", default="valid", choices=["train","valid","test"])
    args = p.parse_args()
    cfg = load_config(args.config)
    evaluate_split(cfg, args.split)
