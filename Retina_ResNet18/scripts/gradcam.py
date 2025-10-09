import os, random
ap = argparse.ArgumentParser()
ap.add_argument('--cfg', default='configs/resnet18.yaml')
ap.add_argument('--ckpt', required=True)
ap.add_argument('--num', type=int, default=12)
args = ap.parse_args()


cfg = load_config(args.cfg)
ds = RetinaDataset(
csv_path=os.path.join(cfg.data_dir, cfg.valid_csv),
img_dir=os.path.join(cfg.data_dir, cfg.valid_img_dir),
col_image=cfg.columns.image, col_grade=cfg.columns.grade, col_edema=cfg.columns.edema,
img_size=cfg.img_size, center_crop=cfg.center_crop, is_train=False)


model = ResNet18MultiTask(num_classes_grade=cfg.num_classes_grade)
state = torch.load(args.ckpt, map_location='cpu')
model.load_state_dict(state['model_state'])


gc = GradCAM(model, target_layer=cfg.gradcam.target_layer)
out_dir = os.path.join(os.path.dirname(args.ckpt), 'gradcam')
os.makedirs(out_dir, exist_ok=True)


idxs = random.sample(range(len(ds)), k=min(args.num, len(ds)))
for i in idxs:
img, yg, ye = ds[i]
cam = gc(img, target='grade')
# to numpy image (H,W,3)
img_np = (img.permute(1,2,0).numpy() * [0.229,0.224,0.225] + [0.485,0.456,0.406])
img_np = np.clip(img_np, 0, 1)
cam_resized = cv2.resize(cam, (img_np.shape[1], img_np.shape[0]))
heatmap = cv2.applyColorMap((cam_resized*255).astype(np.uint8), cv2.COLORMAP_JET)
heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)/255.0
overlay = (0.4*heatmap + 0.6*img_np)
out = (overlay*255).astype(np.uint8)
cv2.imwrite(os.path.join(out_dir, f"cam_{i}.png"), cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
print(f"Saved Grad-CAM images to {out_dir}")