# scripts/train.py
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
from src.config import load_config, set_seed, ensure_dir
from src.datasets.retina import make_loaders
from src.models.resnet18_multitask import ResNet18MultiTask
from src.engine.trainer import Trainer
from src.utils.common import timestamp_dir, save_json

if __name__ == "__main__":
    cfg = load_config("configs/resnet18.yaml")
    set_seed(cfg.seed)

    out_dir = timestamp_dir(cfg.output_dir, cfg.project_name)
    ensure_dir(out_dir)

    train_loader, valid_loader = make_loaders(cfg)
    model = ResNet18MultiTask(num_classes_grade=cfg.num_classes_grade)
    trainer = Trainer(cfg, model, train_loader, valid_loader, out_dir)

    history = trainer.fit()

    # save curves-ready JSON & a summary (NumPy-safe)
    save_json(history, os.path.join(out_dir, "history.json"))

    last = history.get("valid", [{}])[-1] if history.get("valid") else {}
    save_json(last, os.path.join(out_dir, "summary.json"))

    print(f"Training finished. Outputs saved to: {out_dir}")

