import torch, timm, matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader
from .dataset import CSVDataset
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import pandas as pd
import numpy as np

DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ROOT / "splits"
CHECKPOINTS = ROOT / "checkpoints"

def _normalize_classes(classes_obj):
    """
    Return classes as a list[str] in index order: 0..C-1.
    Supports list/tuple, list of ints, or dicts keyed by '0','1',...
    """
    if isinstance(classes_obj, dict):
        n = len(classes_obj)
        if all(str(i) in classes_obj for i in range(n)):
            ordered = [classes_obj[str(i)] for i in range(n)]
        else:
            ordered = [classes_obj[k] for k in sorted(classes_obj.keys(), key=lambda x: int(x) if str(x).isdigit() else x)]
        return [str(c) for c in ordered]
    else:
        return [str(c) for c in classes_obj]

def _tta_logits(model, xb, use_tta=True):
    if not use_tta:
        return model(xb)
    outs = []
    outs.append(model(xb))                    # identity
    outs.append(model(torch.flip(xb, dims=[3])))  # hflip
    return torch.stack(outs, dim=0).mean(0)

def evaluate(ckpt=CHECKPOINTS/"convnext_tiny_second.pt",
             test_csv=SPLITS/"test.csv",
             size=224, batch_size=32,
             use_tta=True,
             out_dir="eval_plots"):
    # Be explicit about weights_only to silence the FutureWarning
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)

    classes = _normalize_classes(ck["classes"])
    num_classes = len(classes)
    labels = list(range(num_classes))

    ds = CSVDataset(test_csv, classes=classes, train=False, size=size)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2)

    model = timm.create_model("convnext_tiny.in12k_ft_in1k", pretrained=False, num_classes=num_classes)
    model.load_state_dict(ck["model"], strict=True)
    model.to(DEVICE)
    model.eval()

    out_path = Path(out_dir); out_path.mkdir(parents=True, exist_ok=True)

    y_true, y_pred = [], []
    with torch.no_grad():
        for xb, yb in dl:
            xb = xb.to(DEVICE)
            logits = _tta_logits(model, xb, use_tta=use_tta)
            y_pred.extend(logits.argmax(1).cpu().tolist())
            y_true.extend(yb.tolist())

    # Text report + confusion matrix df
    report = classification_report(
        y_true, y_pred,
        labels=labels,
        target_names=classes,
        zero_division=0,
        digits=4
    )
    print(report)

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm,
                         index=[f"true_{c}" for c in classes],
                         columns=[f"pred_{c}" for c in classes])
    print("Confusion Matrix:\n", cm_df)

    # ---------- FIGURES ----------
    # Confusion matrix (normalized)
    cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    plt.figure()
    plt.imshow(cm_norm, interpolation='nearest')
    plt.title("Confusion Matrix (normalized)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(ticks=range(len(classes)), labels=classes, rotation=45, ha="right")
    plt.yticks(ticks=range(len(classes)), labels=classes)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, f"{cm_norm[i,j]:.2f}", ha="center", va="center")
    plt.tight_layout()
    plt.savefig(out_path / "confusion_matrix.png", dpi=150)
    plt.close()

    # Per-class PRF bars
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    x = np.arange(len(classes))

    def _bar(y, title, fname, ylabel):
        plt.figure()
        plt.bar(x, y)
        plt.xticks(x, classes, rotation=45, ha="right")
        plt.ylabel(ylabel); plt.title(title)
        plt.tight_layout()
        plt.savefig(out_path / fname, dpi=150)
        plt.close()

    _bar(prec, "Per-class Precision", "per_class_precision.png", "Precision")
    _bar(rec,  "Per-class Recall",    "per_class_recall.png",    "Recall")
    _bar(f1,   "Per-class F1",        "per_class_f1.png",        "F1-score")

    print(f"Saved figures to: {out_path}/ "
          f"(confusion_matrix.png, per_class_precision.png, per_class_recall.png, per_class_f1.png)")

if __name__ == "__main__":
    evaluate()
