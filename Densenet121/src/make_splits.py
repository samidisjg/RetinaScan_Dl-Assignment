import argparse, os, pandas as pd, glob
from sklearn.model_selection import train_test_split

def infer_label_from_path(path):
    return os.path.basename(os.path.dirname(path))

def main(args):
    # 1) Collect all image file paths recursively using the glob pattern
    files = glob.glob(os.path.join(args.data_root, args.glob_pattern), recursive=True)

    # 2) Build a list of (filepath, label) pairs by inferring label from folder name
    rows = []
    for fp in files:
        if os.path.isfile(fp):
            rows.append((fp, infer_label_from_path(fp)))

    # 3) Create a DataFrame and shuffle deterministically
    df = pd.DataFrame(rows, columns=['filepath','label'])
    df = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    # 4) Stratified split: 70% train, 15% val, 15% test
    #    First split off 30% temp (val+test), keeping class distribution with stratify
    train_df, tmp = train_test_split(
        df, test_size=0.30, stratify=df['label'], random_state=args.seed
    )
    #    Then split the temp set evenly into val and test (each 15%)
    val_df, test_df = train_test_split(
        tmp, test_size=0.50, stratify=tmp['label'], random_state=args.seed
    )

    # 5) Ensure output folder exists and write CSVs
    os.makedirs(os.path.dirname(args.out_prefix), exist_ok=True)
    train_df.to_csv(args.out_prefix + "train.csv", index=False)
    val_df.to_csv(args.out_prefix + "val.csv", index=False)
    test_df.to_csv(args.out_prefix + "test.csv", index=False)

    print("Wrote:", args.out_prefix + "train.csv", args.out_prefix + "val.csv", args.out_prefix + "test.csv")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True, help="Root folder with class subfolders")
    ap.add_argument("--glob_pattern", type=str, default="**/*.jpg", help="File pattern (use ** for recursive)")
    ap.add_argument("--out_prefix", type=str, default="data/", help="Where to save train/val/test CSVs")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = ap.parse_args()
    main(args)