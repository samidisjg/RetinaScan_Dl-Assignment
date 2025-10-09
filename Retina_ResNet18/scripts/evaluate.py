import os, json
import pandas as pd
from src.config import load_config
from src.datasets.retina import RetinaDataset
from src.models.resnet18_multitask import ResNet18MultiTask
from src.engine.metrics import compute_metrics
import numpy as np


if __name__ == "__main__":
import argparse
ap = argparse.ArgumentParser()
ap.add_argument('--cfg', default='configs/resnet18.yaml')
ap.add_argument('--ckpt', required=True)
ap.add_argument('--split', default='test', choices=['train','valid','test'])
ap.add_argument('--out', default=None)
args = ap.parse_args()


cfg = load_config(args.cfg)
if args.split=='train':
csv, imgdir = cfg.train_csv, cfg.train_img_dir
elif args.split=='valid':
csv, imgdir = cfg.valid_csv, cfg.valid_img_dir
else:
csv, imgdir = cfg.test_csv, cfg.test_img_dir


ds = RetinaDataset(
csv_path=os.path.join(cfg.data_dir, csv),
img_dir=os.path.join(cfg.data_dir, imgdir),
col_image=cfg.columns.image, col_grade=cfg.columns.grade, col_edema=cfg.columns.edema,
img_size=cfg.img_size, center_crop=cfg.center_crop, is_train=False)


model = ResNet18MultiTask(num_classes_grade=cfg.num_classes_grade)
state = torch.load(args.ckpt, map_location='cpu')
model.load_state_dict(state['model_state'])
model.eval()


from torch.utils.data import DataLoader
dl = DataLoader(ds, batch_size=64, shuffle=False, num_workers=cfg.num_workers)


g_true, g_pred, e_true, e_prob = [], [], [], []
with torch.no_grad():
for imgs, yg, ye in dl:
lg, le = model(imgs)
g_true.append(yg.numpy())
g_pred.append(lg.argmax(1).numpy())
e_true.append(ye.numpy())
e_prob.append(torch.sigmoid(le).numpy())
import numpy as np
g_true = np.concatenate(g_true); g_pred = np.concatenate(g_pred)
e_true = np.concatenate(e_true); e_prob = np.concatenate(e_prob)
metrics = compute_metrics(g_true, g_pred, e_true, e_prob)


out = args.out or os.path.join(os.path.dirname(args.ckpt), f"eval_{args.split}.json")
with open(out, 'w') as f: json.dump({k:(v.tolist() if hasattr(v, 'tolist') else v) for k,v in metrics.items()}, f, indent=2)
print(json.dumps(metrics, indent=2, default=lambda x: x.tolist()))