# src/config.py
from __future__ import annotations
import yaml
import os
import random
import numpy as np
import torch
from dataclasses import dataclass


@dataclass
class Cfg:
    d: dict

    def __getattr__(self, k):
        v = self.d.get(k)
        if isinstance(v, dict):
            return Cfg(v)
        return v


def load_config(path: str) -> Cfg:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return Cfg(cfg)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
