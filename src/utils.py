import os, random, json
import numpy as np
import torch

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def save_json(obj, path):
    # Convert NumPy types (like np.int64) to native Python types
    def default(o):
        if hasattr(o, "item"):  # NumPy scalar
            return o.item()
        raise TypeError(f"Type {type(o)} not serializable")

    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=default)

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)
