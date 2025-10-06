from albumentations import (
    Compose, Resize, RandomResizedCrop, HorizontalFlip, ShiftScaleRotate,
    CLAHE, RandomBrightnessContrast, CoarseDropout, Normalize
)
from albumentations.pytorch import ToTensorV2

def build_transforms(img_size=224, is_train=True):
    if is_train:
        return Compose([
            RandomResizedCrop(img_size, img_size, scale=(0.8, 1.0)),
            HorizontalFlip(p=0.5),
            ShiftScaleRotate(shift_limit=0.02, scale_limit=0.1, rotate_limit=15, p=0.5),
            CLAHE(clip_limit=2.0, p=0.2),
            RandomBrightnessContrast(p=0.2),
            CoarseDropout(max_holes=8, max_height=img_size//16, max_width=img_size//16, p=0.2),
            Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
            ToTensorV2()
        ])
    else:
        return Compose([
            Resize(img_size, img_size),
            Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
            ToTensorV2()
        ])
