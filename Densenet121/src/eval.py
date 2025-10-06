import argparse, os
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

from dataset import RetinaDataset
from transforms import build_transforms
from model_densenet121 import build_densenet121
from utils import load_json, get_device

@torch.no_grad()
def main(args):
    device = get_device()
    class_to_idx = load_json(args.class_map)
    idx_to_class = {v:k for k,v in class_to_idx.items()}
    num_classes = len(idx_to_class)

    ds = RetinaDataset(args.test_csv, tfm=build_transforms(args.img_size, is_train=False))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = build_densenet121(num_classes, pretrained=False).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    y_true, y_pred = [], []
    for x,y in loader:
        x = x.to(device)
        out = model(x)
        y_true.extend(y.numpy().tolist())
        y_pred.extend(out.argmax(1).cpu().numpy().tolist())

    print(classification_report(y_true, y_pred, target_names=[idx_to_class[i] for i in range(num_classes)], digits=4))

    # Confusion matrix plot
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    fig = plt.figure(figsize=(8,8))
    plt.imshow(cm, interpolation='nearest')
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, [idx_to_class[i] for i in range(num_classes)], rotation=45, ha='right')
    plt.yticks(tick_marks, [idx_to_class[i] for i in range(num_classes)])
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "confusion_matrix.png")
    plt.savefig(out_path, bbox_inches='tight', dpi=150)
    print("Saved:", out_path)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_csv", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--class_map", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="report/figs")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--num_workers", type=int, default=2)
    args = ap.parse_args()
    main(args)
