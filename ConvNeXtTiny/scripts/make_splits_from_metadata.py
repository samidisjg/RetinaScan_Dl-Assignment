
import pandas as pd
from pathlib import Path
import csv, os
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "DiabeticRetinopathy"


SPLITS = {
    "train": DATA_ROOT / "train",
    "val":   DATA_ROOT / "valid",
    "test":  DATA_ROOT / "test",
}

def load_annotations(split_dir: Path) -> pd.DataFrame:
    ann = split_dir / "annotations.csv"
    df = pd.read_csv(ann)
    need = {"Image name","Retinopathy grade"}
    assert need.issubset(df.columns), f"{ann} must have {need}, got {df.columns.tolist()}"
    df = df[["Image name","Retinopathy grade"]].copy()
    # clean up
    df["image"] = df["Image name"].astype(str).str.strip()
    df["label"] = df["Retinopathy grade"].astype(str).str.strip()
    df = df.drop(columns=["Image name","Retinopathy grade"])
    return df

def index_images(img_root: Path):
    """
    Build a case-insensitive index of filenames -> absolute paths.
    """
    idx = {}
    for p in img_root.rglob("*"):
        if p.is_file():
            idx[p.name.lower()] = p.resolve()
    return idx

def write_split(split_name: str, out_path: Path):
    sd = SPLITS[split_name]
    img_root = sd / "images"
    df = load_annotations(sd)

    # index files case-insensitively
    file_idx = index_images(img_root)

    # map image names to absolute paths
    def build_path(name: str) -> str:
        key = name.lower()
        p = file_idx.get(key)
        return str(p) if p else ""

    df["filepath"] = df["image"].astype(str).map(build_path)
    df = df[["filepath","label"]]

    # drop missing & duplicates
    missing = df["filepath"].eq("").sum()
    if missing:
        print(f"[WARN] {split_name}: {missing} missing (filename not found under {img_root})")
    df = df[df["filepath"] != ""].drop_duplicates()

    # report counts
    counts = Counter(df["label"].tolist())
    print(f"[{split_name}] samples={len(df)} | class counts={dict(counts)}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filepath","label"])
        for fp, lb in df.itertuples(index=False):
            w.writerow([fp, lb])

def main():
    out_dir = ROOT / "splits"
    write_split("train", out_dir / "train.csv")
    write_split("val",   out_dir / "val.csv")
    write_split("test",  out_dir / "test.csv")
    print("✓ Wrote splits/train.csv, splits/val.csv, splits/test.csv")

if __name__ == "__main__":
    main()
