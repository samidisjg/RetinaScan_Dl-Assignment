import sys, os, json, torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils.common import save_json

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="path to best.pt or any ckpt-epochxxx.pt")
    ap.add_argument("--out", default=None, help="history.json output path")
    args = ap.parse_args()

    state = torch.load(args.ckpt, map_location="cpu")
    hist = state.get("history", None)
    if hist is None:
        raise RuntimeError("No 'history' found in checkpoint.")
    out = args.out or os.path.join(os.path.dirname(args.ckpt), "history.json")
    save_json(hist, out)
    print("Saved:", out)
