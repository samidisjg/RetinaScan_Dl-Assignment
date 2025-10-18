import os, random, json
import numpy as np
import torch

def seed_everything(seed: int = 42):
    """
    Ensures reproducibility across runs.
    Sets fixed random seeds for Python, NumPy, and PyTorch
    so model training results are consistent.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # ensures same conv outputs
    torch.backends.cudnn.benchmark = False     # disables auto-optimization

def get_device():
    """
    Automatically selects the best available device:
    - Apple MPS (Metal GPU on Mac)
    - CUDA GPU (if available)
    - CPU fallback
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def save_json(obj, path):
    """
    Saves Python objects (like dicts) into JSON format.
    Handles NumPy data types safely.
    """
    def default(o):
        if hasattr(o, "item"):  # converts NumPy scalar (e.g., np.int64 → int)
            return o.item()
        raise TypeError(f"Type {type(o)} not serializable")

    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=default)

def load_json(path):
    """Loads a JSON file and returns its contents as a Python object."""
    with open(path, "r") as f:
        return json.load(f)