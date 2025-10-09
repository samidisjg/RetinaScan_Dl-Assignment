# src/evaluate.py
import argparse, os, csv
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

from dataset import RetinaDataset
from transforms import build_transforms
from model_densenet121 import build_densenet121


def pick_device(arg):
    if arg and arg.lower() != "auto":
        return torch.device(arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    # Apple Silicon
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def forward_logits(model, x, tta=False):
    """
    x: (B, C, H, W) Tensor on the correct device.
    If tta=True, do simple horizontal-flip TTA with logits averaging.
    """
    logits = model(x)
    if tta:
        logits_flip = model(torch.flip(x, dims=[-1]))
        logits = (logits + logits_flip) / 2.0
    return logits


def main(args):
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    device = pick_device(args.device)
    print("Eval device:", device)

    # Dataset / DataLoader
    tfm = build_transforms(args.img_size, is_train=False)
    ds  = RetinaDataset(args.csv, tfm=tfm)
    num_classes = len(ds.classes)

    # Model
    model = build_densenet121(num_classes, pretrained=False).to(device)
    state = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.eval()

    # Batched inference
    all_preds, all_gts = [], []
    all_probs = [] if args.save_probs or args.save_preds else None

    # Iterate in mini-batches to speed up GPU/MPS and avoid OOM
    batch_size = args.batch_size
    n = len(ds)
    for start in range(0, n, batch_size):
        batch = [ds[i] for i in range(start, min(start + batch_size, n))]
        xs = torch.stack([x for (x, _) in batch], dim=0).to(device)
        ys = torch.tensor([int(y) for (_, y) in batch], device=device)

        if args.half and device.type == "cuda":
            xs = xs.half()
            model.half()

        logits = forward_logits(model, xs, tta=args.tta)
        probs  = F.softmax(logits.float(), dim=1).cpu().numpy()
        preds  = logits.argmax(1).cpu().numpy()

        all_preds.extend(preds.tolist())
        all_gts.extend(ys.cpu().numpy().tolist())
        if all_probs is not None:
            all_probs.append(probs)

    if all_probs is not None:
        all_probs = np.concatenate(all_probs, axis=0)

    # Metrics
    acc = accuracy_score(all_gts, all_preds)
    f1  = f1_score(all_gts, all_preds, average="macro")
    print(classification_report(all_gts, all_preds, digits=4))
    print("Accuracy:", f"{acc:.4f}")
    print("Macro-F1:", f"{f1:.4f}")
    cm = confusion_matrix(all_gts, all_preds)
    print("Confusion matrix:\n", cm)

    # Save alongside checkpoint dir
    out_dir = os.path.dirname(args.checkpoint)
    os.makedirs(out_dir, exist_ok=True)

    # Save numpy CM + text report
    np.save(os.path.join(out_dir, "cm_eval.npy"), cm)
    with open(os.path.join(out_dir, "report_eval.txt"), "w") as f:
        f.write(classification_report(all_gts, all_preds, digits=4))
        f.write("\n")
        f.write(f"Accuracy: {acc:.6f}\nMacro-F1: {f1:.6f}\n")

    # Optionally save per-image predictions/probabilities
    if args.save_preds:
        pred_path = os.path.join(out_dir, "preds_eval.csv")
        with open(pred_path, "w", newline="") as f:
            writer = csv.writer(f)
            header = ["index", "true_label", "pred_label"]
            if args.save_probs:
                header += [f"prob_class_{c}" for c in range(num_classes)]
            writer.writerow(header)
            for i, (gt, pr) in enumerate(zip(all_gts, all_preds)):
                row = [i, gt, int(pr)]
                if args.save_probs:
                    row += list(all_probs[i])
                writer.writerow(row)
        print("Saved:", pred_path)

    print("Saved:", os.path.join(out_dir, "cm_eval.npy"))
    print("Saved:", os.path.join(out_dir, "report_eval.txt"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, default="runs/densenet121_pretrained_focal/best.pt")
    ap.add_argument("--csv", type=str, default="data/valid.csv")  # change to test.csv if you have it
    ap.add_argument("--img_size", type=int, default=224)

    # speed/accuracy toggles
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--tta", action="store_true", help="Enable simple horizontal-flip TTA")
    ap.add_argument("--half", action="store_true", help="Use FP16 on CUDA (ignored on CPU/MPS)")
    ap.add_argument("--device", type=str, default="auto", help="'auto', 'cuda', 'mps', or 'cpu'")

    # saving
    ap.add_argument("--save_preds", action="store_true", help="Save per-sample predictions to CSV")
    ap.add_argument("--save_probs", action="store_true", help="Also save class probabilities")

    args = ap.parse_args()
    main(args)