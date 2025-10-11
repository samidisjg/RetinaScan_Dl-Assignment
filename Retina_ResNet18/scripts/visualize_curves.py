# scripts/visualize_curves.py
import os
import json
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--history", required=True, help="path to outputs/<run-id>/history.json")
    ap.add_argument("--outdir", default=None, help="folder to save plots (defaults to history.json's folder)")
    args = ap.parse_args()

    with open(args.history, "r") as f:
        hist = json.load(f)

    outdir = args.outdir or os.path.dirname(args.history)
    os.makedirs(outdir, exist_ok=True)

    # Safely extract metrics from history
    train_loss = [e.get("loss", np.nan) for e in hist.get("train", [])]
    val_loss = [e.get("val_loss", np.nan) for e in hist.get("valid", [])]
    f1 = [e.get("grade_f1_macro", np.nan) for e in hist.get("valid", [])]
    acc = [e.get("grade_acc", np.nan) for e in hist.get("valid", [])]
    auc = [e.get("edema_auc", np.nan) for e in hist.get("valid", [])]

    # Loss (train vs val)
    plt.figure()
    plt.plot(train_loss, label="train")
    plt.plot(val_loss, label="val")
    plt.title("Loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "loss_curve.png"), dpi=200)

    # F1 (macro)
    plt.figure()
    plt.plot(f1)
    plt.title("Grade F1 (macro)")
    plt.xlabel("epoch")
    plt.ylabel("F1")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "f1_curve.png"), dpi=200)

    # Accuracy
    plt.figure()
    plt.plot(acc)
    plt.title("Grade Accuracy")
    plt.xlabel("epoch")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "acc_curve.png"), dpi=200)

    # AUC
    plt.figure()
    plt.plot(auc)
    plt.title("Edema ROC-AUC")
    plt.xlabel("epoch")
    plt.ylabel("AUC")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "auc_curve.png"), dpi=200)

    print(f"Saved curves to {outdir}")
