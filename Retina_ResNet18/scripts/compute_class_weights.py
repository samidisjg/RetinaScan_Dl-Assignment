# scripts/compute_class_weights.py
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import numpy as np
import pandas as pd
from src.config import load_config


def _detect_columns(df):
    """Heuristically detect image / grade / edema columns."""
    orig_cols = list(df.columns)
    norm_cols = [c.strip().lower() for c in orig_cols]
    back = dict(zip(norm_cols, orig_cols))

    # image-like columns
    img_keys = [
        "image",
        "file",
        "filename",
        "img",
        "image_name",
        "path",
        "imageid",
        "image_id",
    ]
    img_cands = [c for c in norm_cols if any(k in c for k in img_keys)]
    image_col = back[img_cands[0]] if img_cands else None

    # grade-like columns (0..4)
    grade_col = None
    grade_keys = ["grade", "retinopathy", "dr", "severity"]
    for c in norm_cols:
        if any(k in c for k in grade_keys):
            ser = pd.to_numeric(df[back[c]], errors="coerce")
            uniq = sorted([int(x) for x in pd.Series(ser).dropna().unique()])
            if set(uniq).issubset({0, 1, 2, 3, 4}) and len(set(uniq)) >= 3:
                grade_col = back[c]
                break

    # edema-like columns (0/1)
    edema_col = None
    edema_keys = ["edema", "macular", "risk", "dme"]
    for c in norm_cols:
        if any(k in c for k in edema_keys):
            ser = pd.to_numeric(df[back[c]], errors="coerce")
            uniq = sorted([int(x) for x in pd.Series(ser).dropna().unique()])
            if set(uniq).issubset({0, 1}) and len(set(uniq)) >= 1:
                edema_col = back[c]
                break

    return image_col, grade_col, edema_col


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="configs/resnet18.yaml")
    ap.add_argument("--image-col", default=None, help="override image column name")
    ap.add_argument("--grade-col", default=None, help="override grade column name")
    ap.add_argument("--edema-col", default=None, help="override edema column name")
    args = ap.parse_args()

    cfg = load_config(args.cfg)
    csv_path = os.path.join(cfg.data_dir, cfg.train_csv)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Train CSV not found at: {csv_path}")

    df = pd.read_csv(csv_path)

    # Use YAML first (if present), else auto-detect, then CLI overrides (highest priority)
    yml_img = getattr(getattr(cfg, "columns", None), "image", None)
    yml_grade = getattr(getattr(cfg, "columns", None), "grade", None)
    yml_edema = getattr(getattr(cfg, "columns", None), "edema", None)

    det_img, det_grade, det_edema = _detect_columns(df)

    image_col = args.image_col or yml_img or det_img
    grade_col = args.grade_col or yml_grade or det_grade
    edema_col = args.edema_col or yml_edema or det_edema

    # Validate
    missing = [
        name for name in [image_col, grade_col, edema_col] if name not in df.columns
    ]
    if missing:
        cols = ", ".join(df.columns.astype(str).tolist())
        raise ValueError(
            "Could not find required columns.\n"
            f"  image: {image_col}\n  grade: {grade_col}\n  edema: {edema_col}\n\n"
            f"CSV columns are:\n  {cols}\n\n"
            "Pass explicit names with --image-col/--grade-col/--edema-col or update configs/resnet18.yaml:columns."
        )

    # Compute weights
    gser = pd.to_numeric(df[grade_col], errors="coerce")
    counts = (
        gser.value_counts()
        .reindex(range(getattr(cfg, "num_classes_grade", 5)), fill_value=0)
        .astype(int)
    )
    N = counts.sum()
    C = getattr(cfg, "num_classes_grade", 5)

    weights = []
    for c in range(C):
        n_i = max(1, int(counts.get(c, 0)))
        weights.append(N / (C * n_i))
    weights = (np.array(weights) / np.mean(weights)).round(4).tolist()

    eser = pd.to_numeric(df[edema_col], errors="coerce").fillna(0).astype(int)
    P = int((eser == 1).sum())
    Nneg = int((eser == 0).sum())
    pos_weight = round((Nneg / max(1, P)), 4)

    out = {
        "detected_columns": {
            "image": image_col,
            "grade": grade_col,
            "edema": edema_col,
        },
        "grade_class_weights": weights,
        "edema_pos_weight": pos_weight,
        "grade_counts": {int(k): int(v) for k, v in counts.to_dict().items()},
        "edema_pos": P,
        "edema_neg": Nneg,
    }

    out_dir = os.path.join(cfg.output_dir, "weights")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "weights.json"), "w") as f:
        json.dump(out, f, indent=2)

    print("\nDetected columns:")
    print(f"  image: {image_col}")
    print(f"  grade: {grade_col}")
    print(f"  edema: {edema_col}\n")

    print("Suggested YAML snippet:\n")
    print("columns:")
    print(f'  image: "{image_col}"')
    print(f'  grade: "{grade_col}"')
    print(f'  edema: "{edema_col}"\n')
    print("loss:")
    print(f"  grade_class_weights: {weights}")
    print(f"  edema_pos_weight: {pos_weight}\n")
    print("Details saved to:", os.path.join(out_dir, "weights.json"))
