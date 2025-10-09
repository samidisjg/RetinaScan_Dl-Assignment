import argparse, os
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

from dataset import RetinaDataset
from transforms import build_transforms
from model_densenet121 import build_densenet121


def main(args):
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    # Dataset / model
    ds = RetinaDataset(args.csv, tfm=build_transforms(args.img_size, is_train=False))
    num_classes = len(ds.classes)
    model = build_densenet121(num_classes, pretrained=False)
    state = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.eval()

    preds, gts = [], []
    with torch.no_grad():
        for i in range(len(ds)):
            x, y = ds[i]
            logits = model(x.unsqueeze(0))
            preds.append(int(logits.argmax(1).item()))
            gts.append(int(y))

    # Metrics
    acc = accuracy_score(gts, preds)
    f1  = f1_score(gts, preds, average="macro")
    print(classification_report(gts, preds, digits=4))
    print("Accuracy:", f"{acc:.4f}")
    print("Macro-F1:", f"{f1:.4f}")
    cm = confusion_matrix(gts, preds)
    print("Confusion matrix:\n", cm)

    # Save alongside checkpoint dir
    out_dir = os.path.dirname(args.checkpoint)
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "cm_eval.npy"), cm)
    with open(os.path.join(out_dir, "report_eval.txt"), "w") as f:
        f.write(classification_report(gts, preds, digits=4))
        f.write("\n")
        f.write(f"Accuracy: {acc:.6f}\nMacro-F1: {f1:.6f}\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, default="runs/densenet121_pretrained_focal/best.pt")
    ap.add_argument("--csv", type=str, default="data/valid.csv")  # change to test.csv if you have it
    ap.add_argument("--img_size", type=int, default=224)
    args = ap.parse_args()
    main(args)