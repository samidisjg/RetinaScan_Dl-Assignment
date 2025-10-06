import cv2, pandas as pd, torch
from torch.utils.data import Dataset

class RetinaDataset(Dataset):
    def __init__(self, csv_path, tfm=None):
        self.df = pd.read_csv(csv_path)
        self.tfm = tfm
        self.classes = sorted(self.df['label'].unique())
        self.class_to_idx = {c:i for i,c in enumerate(self.classes)}

    def __len__(self): return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = cv2.imread(row.filepath)
        if img is None:
            raise RuntimeError(f'Failed to read image: {row.filepath}')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.tfm:
            img = self.tfm(image=img)['image']
        y = self.class_to_idx[row.label]
        return img, torch.tensor(y).long()
