import albumentations as A
from albumentations.pytorch import ToTensorV2

def _rrc(img_size, **kwargs):
    try:
        return A.RandomResizedCrop(size=(img_size, img_size), **kwargs)  # Albumentations >=1.4
    except TypeError:
        return A.RandomResizedCrop(height=img_size, width=img_size, **kwargs)

def build_transforms(img_size: int, is_train: bool):
    if is_train:
        return A.Compose([
            # keep lesions visible while adding variety
            _rrc(img_size, scale=(0.8, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.15, p=0.6),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.4),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.ShiftScaleRotate(shift_limit=0.03, scale_limit=0.05, rotate_limit=10, p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.LongestMaxSize(max_size=img_size),
            A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=0),
            A.CenterCrop(height=img_size, width=img_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])