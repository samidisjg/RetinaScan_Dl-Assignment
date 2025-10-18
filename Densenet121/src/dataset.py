import cv2, pandas as pd, torch
import numpy as np
from torch.utils.data import Dataset


class RetinaDataset(Dataset):
    def __init__(self, csv_path, tfm=None):
        # Load the CSV file that contains image file paths and labels
        self.df = pd.read_csv(csv_path)

        # Store the image transformation pipeline (augmentations, resizing, normalization, etc.)
        self.tfm = tfm

        # Get the unique label classes and sort them
        self.classes = sorted(self.df['label'].unique())

        # Create a mapping from class label
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}


    # Return total number of samples in the dataset
    def __len__(self):
        return len(self.df)


    def __getitem__(self, i):
        # Select image record from the dataframe using the index
        row = self.df.iloc[i]

        # Read the image using OpenCV
        img = cv2.imread(row.filepath)

        # Convert from BGR (OpenCV default) to RGB (PyTorch expects RGB)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Apply image transformations if provided (augmentations, normalization, etc.)
        if self.tfm:
            img = self.tfm(image=img)["image"]

        # Convert NumPy array to PyTorch tensor if still a NumPy array
        # and rearrange dimensions from (H, W, C) → (C, H, W), normalize to [0,1]
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0

        # Convert class label to numeric tensor using mapping
        y = self.class_to_idx[row.label]

        # Return the image tensor and its label as a PyTorch tensor
        return img, torch.tensor(y).long()