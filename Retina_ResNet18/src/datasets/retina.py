# src/datasets/retina.py
from __future__ import annotations
import os
import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.v2 as T  # if this errors, see note below


class RetinaDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        img_dir: str,
        col_image: str,
        col_grade: str,
        col_edema: str,
        img_size: int = 384,
        center_crop: int | None = None,
        is_train: bool = False,
    ):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.col_image = col_image
        self.col_grade = col_grade
        self.col_edema = col_edema
        self.is_train = is_train
        self.center_crop = center_crop
        self.img_size = img_size

        train_tfms = [
            T.ToImage(),
            T.Resize((img_size, img_size), antialias=True),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.1),
            T.RandomRotation(10),
            T.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        ]
        eval_tfms = [
            T.ToImage(),
            T.Resize((img_size, img_size), antialias=True),
        ]
        common = [
            T.CenterCrop(center_crop) if center_crop else T.Identity(),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
        self.tfms = T.Compose((train_tfms if is_train else eval_tfms) + common)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_name = str(row[self.col_image])
        img_path = os.path.join(self.img_dir, img_name)
        if not os.path.exists(img_path):
            alt = os.path.join(self.img_dir, "images", img_name)
            if os.path.exists(alt):
                img_path = alt

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Let torchvision v2 convert HWC->CHW and normalize
        img = self.tfms(img)

        grade = int(row[self.col_grade])
        edema = int(row[self.col_edema])
        return img, torch.tensor(grade, dtype=torch.long), torch.tensor(edema, dtype=torch.float32)


def make_loaders(cfg):
    train_ds = RetinaDataset(
        csv_path=os.path.join(cfg.data_dir, cfg.train_csv),
        img_dir=os.path.join(cfg.data_dir, cfg.train_img_dir),
        col_image=cfg.columns.image,
        col_grade=cfg.columns.grade,
        col_edema=cfg.columns.edema,
        img_size=int(cfg.img_size),
        center_crop=getattr(cfg, "center_crop", None),
        is_train=True,
    )
    valid_ds = RetinaDataset(
        csv_path=os.path.join(cfg.data_dir, cfg.valid_csv),
        img_dir=os.path.join(cfg.data_dir, cfg.valid_img_dir),
        col_image=cfg.columns.image,
        col_grade=cfg.columns.grade,
        col_edema=cfg.columns.edema,
        img_size=int(cfg.img_size),
        center_crop=getattr(cfg, "center_crop", None),
        is_train=False,
    )

    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg.batch_size),
        shuffle=True,
        num_workers=int(cfg.num_workers),
        pin_memory=pin,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=int(cfg.batch_size),
        shuffle=False,
        num_workers=int(cfg.num_workers),
        pin_memory=pin,
    )
    return train_loader, valid_loader
