import argparse, os, json
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

from dataset import RetinaDataset
from transforms import build_transforms
from model_densenet121 import build_densenet121
from utils import seed_everything, get_device, save_json

def build_loaders(train_csv, val_csv, batch_size, img_size, num_workers=2):
    train_ds = RetinaDataset(train_csv, tfm=build_transforms(img_size, is_train=True))
    val_ds = RetinaDataset(val_csv, tfm=build_transforms(img_size, is_train=False))

    # class weights for imbalance
    class_counts = train_ds.df['label'].value_counts().reindex(train_ds.classes).fillna(0).to_numpy()
    weights = 1.0 / np.maximum(class_counts, 1)
    sample_weights = train_ds.df['label'].map({c:w for c,w in zip(train_ds.classes, weights)}).to_numpy()
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, train_ds.classes

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    all_preds, all_gts = [], []
    running_loss = 0.0
    for x,y in tqdm(loader, desc="Train", leave=False):
        x,y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()*x.size(0)
        all_preds.extend(out.argmax(1).detach().cpu().tolist())
        all_gts.extend(y.detach().cpu().tolist())
    acc = accuracy_score(all_gts, all_preds)
    f1 = f1_score(all_gts, all_preds, average='macro')
    return running_loss/len(loader.dataset), acc, f1

@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    all_preds, all_gts = [], []
    running_loss = 0.0
    for x,y in tqdm(loader, desc="Val  ", leave=False):
        x,y = x.to(device), y.to(device)
        out = model(x)
        loss = criterion(out, y)
        running_loss += loss.item()*x.size(0)
        all_preds.extend(out.argmax(1).cpu().tolist())
        all_gts.extend(y.cpu().tolist())
    acc = accuracy_score(all_gts, all_preds)
    f1 = f1_score(all_gts, all_preds, average='macro')
    return running_loss/len(loader.dataset), acc, f1

def main(args):
    seed_everything(args.seed)
    device = get_device()
    print("Device:", device)

    train_loader, val_loader, classes = build_loaders(args.train_csv, args.val_csv, args.batch_size, args.img_size, args.num_workers)
    num_classes = len(classes)

    model = build_densenet121(num_classes, pretrained=True).to(device)

    os.makedirs(args.out_dir, exist_ok=True)
    class_to_idx = {str(c): int(i) for i, c in enumerate(classes)}  # cast keys to str (or int(c))
    save_json(class_to_idx, os.path.join(args.out_dir, "class_to_idx.json"))

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_f1 = -1.0
    patience = args.patience
    no_improve = 0

    for epoch in range(1, args.epochs+1):
        print(f"Epoch {epoch}/{args.epochs}")
        tr_loss, tr_acc, tr_f1 = train_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc, va_f1 = eval_epoch(model, val_loader, criterion, device)
        scheduler.step()

        print(f"  train: loss={tr_loss:.4f} acc={tr_acc:.4f} f1={tr_f1:.4f}")
        print(f"  valid: loss={va_loss:.4f} acc={va_acc:.4f} f1={va_f1:.4f}")

        if va_f1 > best_f1:
            best_f1 = va_f1
            no_improve = 0
            ckpt_path = os.path.join(args.out_dir, "best.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"  saved: {ckpt_path}")
        else:
            no_improve += 1
            if no_improve >= patience:
                print("Early stopping.")
                break

    print(f"Best val macro-F1: {best_f1:.4f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", type=str, required=True)
    ap.add_argument("--val_csv", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="runs/densenet121")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--patience", type=int, default=7)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    main(args)
