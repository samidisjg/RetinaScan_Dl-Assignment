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
    """
    Resolve evaluation device:
      - if user passed --device (not "auto"), honor it;
      - else prefer CUDA, then Apple MPS, else CPU.
    """
    if arg and arg.lower() != "auto":
        return torch.device(arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")  # Apple Silicon
    return torch.device("cpu")


@torch.no_grad()
def forward_logits(model, x, tta=False):
    """
    Forward pass helper.
    If tta=True, apply a horizontal-flip Test-Time Augmentation (TTA)
    and average logits from original and flipped views.
    """
    logits = model(x)
    if tta:
        logits_flip = model(torch.flip(x, dims=[-1]))  # flip width dimension
        logits = (logits + logits_flip) / 2.0
    return logits


def main(args):
    # ---------- Load checkpoint ----------
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    device = pick_device(args.device)
    print("Eval device:", device)

    # ---------- Dataset / transforms ----------
    tfm = build_transforms(args.img_size, is_train=False)
    ds  = RetinaDataset(args.csv, tfm=tfm)
    num_classes = len(ds.classes)

    # ---------- Build model skeleton and load weights ----------
    model = build_densenet121(num_classes, pretrained=False).to(device)
    state = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.eval()

    # ---------- Batched inference (memory-safe & fast) ----------
    all_preds, all_gts = [], []
    all_probs = [] if args.save_probs or args.save_preds else None

    batch_size = args.batch_size
    n = len(ds)
    for start in range(0, n, batch_size):
        # assemble a mini-batch from the dataset
        batch = [ds[i] for i in range(start, min(start + batch_size, n))]
        xs = torch.stack([x for (x, _) in batch], dim=0).to(device)
        ys = torch.tensor([int(y) for (_, y) in batch], device=device)

        # optional half precision (only meaningful on CUDA)
        if args.half and device.type == "cuda":
            xs = xs.half()
            model.half()

        # forward + optional flip TTA
        logits = forward_logits(model, xs, tta=args.tta)
        # use float softmax even if half-precision path was used
        probs  = F.softmax(logits.float(), dim=1).cpu().numpy()
        preds  = logits.argmax(1).cpu().numpy()

        # collect batch outputs
        all_preds.extend(preds.tolist())
        all_gts.extend(ys.cpu().numpy().tolist())
        if all_probs is not None:
            all_probs.append(probs)

    # concat per-batch probability arrays if saving
    if all_probs is not None:
        all_probs = np.concatenate(all_probs, axis=0)

    # ---------- Metrics ----------
    acc = accuracy_score(all_gts, all_preds)
    f1  = f1_score(all_gts, all_preds, average="macro")
    print(classification_report(all_gts, all_preds, digits=4))
    print("Accuracy:", f"{acc:.4f}")
    print("Macro-F1:", f"{f1:.4f}")
    cm = confusion_matrix(all_gts, all_preds)
    print("Confusion matrix:\n", cm)

    # ---------- Save outputs next to the checkpoint ----------
    out_dir = os.path.dirname(args.checkpoint)
    os.makedirs(out_dir, exist_ok=True)

    # Save confusion matrix (npy) + textual report
    np.save(os.path.join(out_dir, "cm_eval.npy"), cm)
    with open(os.path.join(out_dir, "report_eval.txt"), "w") as f:
        f.write(classification_report(all_gts, all_preds, digits=4))
        f.write("\n")
        f.write(f"Accuracy: {acc:.6f}\nMacro-F1: {f1:.6f}\n")

    # Optionally save per-image predictions and probabilities
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
    ap.add_argument("--csv", type=str, default="data/valid.csv")  # or test.csv if you have it
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