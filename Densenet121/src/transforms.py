import albumentations as A
from albumentations.pytorch import ToTensorV2

def _rrc(img_size, **kwargs):
    try:
        return A.RandomResizedCrop(size=(img_size, img_size), **kwargs)
    except TypeError:
        return A.RandomResizedCrop(height=img_size, width=img_size, **kwargs)

def build_transforms(img_size: int, is_train: bool):
    #Defines separate transformation pipelines for training and validation/testing.
    if is_train:
        # Data Augmentation — increases diversity and reduces overfitting
        return A.Compose([
            _rrc(img_size, scale=(0.8, 1.0)),  # random crop and resize for variety
            A.HorizontalFlip(p=0.5),           # 50% chance to flip horizontally
            A.RandomBrightnessContrast(        # adjust brightness & contrast slightly
                brightness_limit=0.1, contrast_limit=0.15, p=0.6
            ),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.4),  # enhance local contrast
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),               # mimic focus variations
            A.ShiftScaleRotate(shift_limit=0.03, scale_limit=0.05, rotate_limit=10, p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406),                 # standard ImageNet normalization
                        std=(0.229, 0.224, 0.225)),
            ToTensorV2(),                                           # convert to PyTorch tensor
        ])
    else:
        # For valid/test — only resizing and normalization (no random changes)
        return A.Compose([
            A.LongestMaxSize(max_size=img_size),
            A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=0),
            A.CenterCrop(height=img_size, width=img_size),          # deterministic center crop
            A.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])