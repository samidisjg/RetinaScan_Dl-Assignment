import albumentations as A
from albumentations.pytorch import ToTensorV2

def _rrc(img_size, **kwargs):
    """
    Create RandomResizedCrop compatibly across Albumentations versions:
    - Newer: RandomResizedCrop(size=(H, W), ...)
    - Older: RandomResizedCrop(height=H, width=W, ...)
    """
    try:
        # New API (>=1.4)
        return A.RandomResizedCrop(size=(img_size, img_size), **kwargs)
    except TypeError:
        # Old API
        return A.RandomResizedCrop(height=img_size, width=img_size, **kwargs)

def build_transforms(img_size: int, is_train: bool):
    if is_train:
        return A.Compose([
            _rrc(img_size, scale=(0.8, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.7),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            # Center-crop to desired size for eval
            A.LongestMaxSize(max_size=img_size),
            A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=0),
            A.CenterCrop(height=img_size, width=img_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])