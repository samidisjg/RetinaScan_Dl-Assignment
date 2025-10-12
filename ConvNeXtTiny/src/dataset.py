# src/dataset.py
import cv2, pandas as pd
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

def _rrc(sz):
    try:
        return A.RandomResizedCrop(size=(sz, sz), scale=(0.8, 1.0))
    except TypeError:
        return A.RandomResizedCrop(height=sz, width=sz, scale=(0.8, 1.0))

def _cc(sz):
    try:
        return A.CenterCrop(height=sz, width=sz)
    except TypeError:
        return A.CenterCrop(sz, sz)

def build_transforms(train=True, size=224):
    if train:
        return A.Compose([
            A.LongestMaxSize(max_size=size),
            A.PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_REFLECT_101),
            _rrc(size),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.4),
            A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.LongestMaxSize(max_size=size),
            A.PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_REFLECT_101),
            _cc(size),
            A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
            ToTensorV2(),
        ])

class CSVDataset(Dataset):
    def __init__(self, csv_path, classes=None, train=True, size=224):
        df = pd.read_csv(csv_path)

        # force labels to STRING everywhere
        df["label"] = df["label"].astype(str)

        self.paths = df["filepath"].tolist()
        self.labels_str = df["label"].tolist()

        # classes must also be STRING and in a fixed order
        if classes is None:
            self.classes = sorted(list(set(self.labels_str)))
        else:
            self.classes = [str(c) for c in classes]

        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.labels = [self.class_to_idx[s] for s in self.labels_str]

        self.tfm = build_transforms(train=train, size=size)

    def __len__(self): return len(self.paths)

    def __getitem__(self, i):
        p = self.paths[i]
        im = cv2.imread(p, cv2.IMREAD_COLOR)
        if im is None:
            raise FileNotFoundError(p)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        aug = self.tfm(image=im)
        x = aug["image"]
        y = self.labels[i]
        return x, y
