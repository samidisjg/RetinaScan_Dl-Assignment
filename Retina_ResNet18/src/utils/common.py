# src/utils/common.py
from __future__ import annotations
import os
import json
from datetime import datetime
import torch
import numpy as np

class AvgMeter:
    def __init__(self):
        self.sum = 0.0
        self.n = 0

    def update(self, val, k: int = 1):
        self.sum += float(val) * k
        self.n += k

    @property
    def avg(self) -> float:
        return self.sum / max(1, self.n)


def timestamp_dir(root: str, project: str) -> str:
    t = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = os.path.join(root, f"{project}-{t}")
    os.makedirs(path, exist_ok=True)
    return path


def save_checkpoint(state: dict, is_best: bool, folder: str, max_keep: int = 3):
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"ckpt-epoch{state['epoch']:03d}.pt")
    torch.save(state, path)
    if is_best:
        torch.save(state, os.path.join(folder, "best.pt"))
    # housekeeping: keep only latest N ckpts
    ckpts = sorted([f for f in os.listdir(folder) if f.startswith("ckpt-")])
    for f in ckpts[:-max_keep]:
        try:
            os.remove(os.path.join(folder, f))
        except OSError:
            pass


def save_json(d: dict, path: str):
    def _default(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        # numpy scalars
        if isinstance(o, (np.float32, np.float64, np.float16)):
            return float(o)
        if isinstance(o, (np.int32, np.int64, np.int16, np.int8)):
            return int(o)
        return str(o)  # fallback for anything odd
    with open(path, "w") as f:
        json.dump(d, f, indent=2, default=_default)
