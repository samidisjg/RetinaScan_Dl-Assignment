import os
import sys
import argparse

# --- Make project root importable when running from scripts/ ---
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.config import load_config
from src.utils.seed import set_seed
from src.data.tf_dataset import DataBuilder
from src.models.tf_efficientnet_b0 import build_effnet_b0
from src.train.tf_trainer import enable_perf, two_stage_finetune
import tensorflow as tf


def main(cfg_path: str):
    cfg = load_config(cfg_path)
    set_seed(cfg.train["seed"])
    enable_perf()

    # Datasets
    db_train = DataBuilder(tuple(cfg.model["input_shape"]), augment=cfg.train["augment"])
    db_val   = DataBuilder(tuple(cfg.model["input_shape"]), augment=False)

    train_ds = db_train.dataframe_to_ds(
        cfg.paths["train_csv"], cfg.paths["train_dir"], shuffle=True,  batch=cfg.train["batch_size"]
    )
    val_ds   = db_val.dataframe_to_ds(
        cfg.paths["valid_csv"], cfg.paths["valid_dir"], shuffle=False, batch=cfg.train["batch_size"]
    )

    # Model
    model, base = build_effnet_b0(tuple(cfg.model["input_shape"]), dropout=cfg.model["dropout"])

    # Train
    os.makedirs(cfg.out["run_dir"], exist_ok=True)
    two_stage_finetune(
    model, base, train_ds, val_ds,
    cfg.out['ckpt'], cfg.out['run_dir'],   # <-- pass run_dir here
    epochs1=cfg.train['epochs_stage1'],
    epochs2=cfg.train['epochs_stage2'],
    unfreeze_ratio=cfg.train['unfreeze_ratio'],
    lr1=cfg.train['base_lr_stage1'],
    lr2=cfg.train['base_lr_stage2'],
)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/efficientnet_b0.yaml")
    args = parser.parse_args()
    main(args.config)
