import argparse, os, pandas as pd, glob
from sklearn.model_selection import train_test_split

def infer_label_from_path(path):
    # Assumes data_root/class_name/xxx.jpg
    return os.path.basename(os.path.dirname(path))

def main(args):
    files = glob.glob(os.path.join(args.data_root, args.glob_pattern), recursive=True)
    rows = []
    for fp in files:
        if os.path.isfile(fp):
            rows.append((fp, infer_label_from_path(fp)))
    df = pd.DataFrame(rows, columns=['filepath','label'])
    df = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    train_df, tmp = train_test_split(df, test_size=0.30, stratify=df['label'], random_state=args.seed)
    val_df, test_df = train_test_split(tmp, test_size=0.50, stratify=tmp['label'], random_state=args.seed)

    os.makedirs(os.path.dirname(args.out_prefix), exist_ok=True)
    train_df.to_csv(args.out_prefix + "train.csv", index=False)
    val_df.to_csv(args.out_prefix + "val.csv", index=False)
    test_df.to_csv(args.out_prefix + "test.csv", index=False)
    print("Wrote:", args.out_prefix + "train.csv", args.out_prefix + "val.csv", args.out_prefix + "test.csv")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True, help="Root folder with class subfolders")
    ap.add_argument("--glob_pattern", type=str, default="**/*.jpg")
    ap.add_argument("--out_prefix", type=str, default="data/")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    main(args)
