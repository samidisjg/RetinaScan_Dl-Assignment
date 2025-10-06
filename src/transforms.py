import albumentations as A

def build_transforms(img_size: int, is_train: bool):
    if is_train:
        return A.Compose([
            A.RandomResizedCrop(size=(img_size, img_size), scale=(0.8, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Normalize(),
        ])
    else:
        return A.Compose([
            # Resize still accepts height/width across versions; it's fine to keep:
            A.Resize(height=img_size, width=img_size),
            A.Normalize(),
        ])