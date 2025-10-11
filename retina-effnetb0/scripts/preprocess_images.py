import os, cv2, argparse, pandas as pd
from tqdm import tqdm

# CLAHE for local contrast enhancement (green-channel)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

def square_center_crop(img):
    h, w = img.shape[:2]
    s = min(h, w)
    y0 = (h - s) // 2
    x0 = (w - s) // 2
    return img[y0:y0 + s, x0:x0 + s]

def apply_clahe_green(img):
    # OpenCV uses BGR; convert to RGB to isolate green, then back
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    g = rgb[:, :, 1]
    g = clahe.apply(g)
    rgb[:, :, 1] = g
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

def unsharp(img, ksize=(0, 0), sigma=1.0, amount=0.4):
    blur = cv2.GaussianBlur(img, ksize, sigma)
    return cv2.addWeighted(img, 1 + amount, blur, -amount, 0)

def process_one(src_path, dst_path, size=380, use_clahe=True, use_unsharp=True):
    img = cv2.imread(src_path, cv2.IMREAD_COLOR)
    if img is None:
        return False
    img = square_center_crop(img)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    if use_clahe:
        img = apply_clahe_green(img)
    if use_unsharp:
        img = unsharp(img, sigma=1.0, amount=0.4)
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    cv2.imwrite(dst_path, img)
    return True

def mirror_split(split_dir, out_split_dir, ann_csv, size=380):
    df = pd.read_csv(ann_csv)
    in_img_dir = os.path.join(split_dir, "images")
    out_img_dir = os.path.join(out_split_dir, "images")
    os.makedirs(out_img_dir, exist_ok=True)
    # copy annotations
    df.to_csv(os.path.join(out_split_dir, "annotations.csv"), index=False)

    for name in tqdm(df["Image name"].values, desc=f"Preprocess {os.path.basename(split_dir)}"):
        src_path = os.path.join(in_img_dir, name)
        dst_path = os.path.join(out_img_dir, name)
        ok = process_one(src_path, dst_path, size=size)
        if not ok:
            print(f"[WARN] Could not read: {src_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--out_root", default="data_preprocessed")
    parser.add_argument("--size", type=int, default=380)
    args = parser.parse_args()

    for split in ["train", "valid", "test"]:
        mirror_split(
            os.path.join(args.data_root, split),
            os.path.join(args.out_root, split),
            os.path.join(args.data_root, split, "annotations.csv"),
            size=args.size,
        )
    print("Done.")
