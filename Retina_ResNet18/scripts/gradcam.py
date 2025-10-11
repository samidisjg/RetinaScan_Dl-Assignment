# scripts/gradcam.py
import sys, os, torch, cv2, numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.config import load_config
from src.datasets.retina import RetinaDataset
from src.models.resnet18_multitask import ResNet18MultiTask

def apply_colormap(img, mask):
    mask = (255 * (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)).astype(np.uint8)
    heat = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    out = (0.5 * img + 0.5 * heat).astype(np.uint8)
    return out

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="configs/resnet18.yaml")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="valid", choices=["train","valid","test"])
    ap.add_argument("--num", type=int, default=12)
    args = ap.parse_args()

    cfg = load_config(args.cfg)
    if args.split == "train":
        csv_rel, img_rel = cfg.train_csv, cfg.train_img_dir
    elif args.split == "valid":
        csv_rel, img_rel = cfg.valid_csv, cfg.valid_img_dir
    else:
        csv_rel, img_rel = cfg.test_csv, cfg.test_img_dir

    ds = RetinaDataset(
        csv_path=os.path.join(cfg.data_dir, csv_rel),
        img_dir=os.path.join(cfg.data_dir, img_rel),
        col_image=cfg.columns.image,
        col_grade=cfg.columns.grade,
        col_edema=cfg.columns.edema,
        img_size=int(cfg.img_size),
        center_crop=getattr(cfg, "center_crop", None),
        is_train=False,
    )

    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = ResNet18MultiTask(num_classes_grade=int(cfg.num_classes_grade))
    model.load_state_dict(state["model_state"])
    model.eval()

    # target conv layer
    target = model.features[-1]  # layer4.1.conv2 output after pool block
    feats, grads = [], []
    def f_hook(_, __, output): feats.append(output.detach())
    def b_hook(_, grad_in, grad_out): grads.append(grad_out[0].detach())
    h1 = target.register_forward_hook(f_hook)
    h2 = target.register_full_backward_hook(b_hook)

    out_dir = os.path.join(os.path.dirname(args.ckpt), f"gradcam_{args.split}")
    os.makedirs(out_dir, exist_ok=True)

    for i in range(min(args.num, len(ds))):
        img_t, yg, ye = ds[i]  # CHW
        img = (img_t.permute(1,2,0).numpy() * 255.0).clip(0,255).astype(np.uint8)  # de-normalized *roughly*
        x = img_t.unsqueeze(0)
        feats.clear(); grads.clear()

        # backprop from predicted grade class
        lg, le = model(x)
        cls = lg.argmax(1).item()
        score = lg[0, cls]
        model.zero_grad(set_to_none=True)
        score.backward()

        A = feats[-1][0]           # [C, H, W]
        G = grads[-1][0]           # [C, H, W]
        weights = G.mean(dim=(1,2), keepdim=True)  # [C,1,1]
        cam = torch.relu((weights * A).sum(0)).cpu().numpy()  # [H, W]
        cam = cv2.resize(cam, (img.shape[1], img.shape[0]))

        over = apply_colormap(img, cam)
        cv2.imwrite(os.path.join(out_dir, f"idx{i:03d}_pred{cls}_gradcam.png"),
                    cv2.cvtColor(over, cv2.COLOR_RGB2BGR))
    h1.remove(); h2.remove()
    print("Saved Grad-CAMs to:", out_dir)
