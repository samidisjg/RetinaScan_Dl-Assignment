import yaml
from dataclasses import dataclass
from typing import Any, Dict

@dataclass
class Cfg:
    project: str
    paths: Dict[str, str]
    out: Dict[str, str]
    model: Dict[str, Any]
    train: Dict[str, Any]

def load_config(path: str) -> Cfg:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if raw is None:
        raw = {}  # file was empty or only comments

    cfg = {
        "project": raw.get("project", "retina-effnetb0"),
        "paths": dict(raw.get("paths", {})),
        "out": dict(raw.get("out", {})),
        "model": dict(raw.get("model", {})),
        "train": dict(raw.get("train", {})),
    }

    # sensible defaults & path glue
    cfg["paths"].setdefault("data_root", "data_preprocessed")
    cfg["paths"].setdefault("train_dir", f"{cfg['paths']['data_root']}/train")
    cfg["paths"].setdefault("valid_dir", f"{cfg['paths']['data_root']}/valid")
    cfg["paths"].setdefault("test_dir",  f"{cfg['paths']['data_root']}/test")
    cfg["paths"].setdefault("train_csv", f"{cfg['paths']['train_dir']}/annotations.csv")
    cfg["paths"].setdefault("valid_csv", f"{cfg['paths']['valid_dir']}/annotations.csv")
    cfg["paths"].setdefault("test_csv",  f"{cfg['paths']['test_dir']}/annotations.csv")

    cfg["out"].setdefault("run_dir", "runs/effnet_b0")
    cfg["out"].setdefault("ckpt", f"{cfg['out']['run_dir']}/best.keras")

    cfg["model"].setdefault("input_shape", [380, 380, 3])
    cfg["model"].setdefault("dropout", 0.3)
    cfg["model"].setdefault("loss_weights", {"grade": 0.7, "edema": 0.3})

    cfg["train"].setdefault("batch_size", 16)
    cfg["train"].setdefault("epochs_stage1", 5)
    cfg["train"].setdefault("epochs_stage2", 20)
    cfg["train"].setdefault("base_lr_stage1", 1e-3)
    cfg["train"].setdefault("base_lr_stage2", 3e-5)
    cfg["train"].setdefault("unfreeze_ratio", 0.34)
    cfg["train"].setdefault("augment", True)
    cfg["train"].setdefault("seed", 42)

    return Cfg(**cfg)
