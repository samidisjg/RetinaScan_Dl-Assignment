import os, sys, argparse
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
import tf_keras as keras
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.config import load_config

def find_last_conv_layer(model):
    # Prefer last 4D feature map layer (Conv2D / DepthwiseConv2D)
    for layer in reversed(model.layers):
        try:
            shp = getattr(layer, "output_shape", None)
            if shp is not None and len(shp) == 4:
                if isinstance(layer, (keras.layers.Conv2D, keras.layers.DepthwiseConv2D)):
                    return layer.name
        except Exception:
            pass
    # Fallback to known name in EfficientNetB0
    try:
        model.get_layer("top_conv")
        return "top_conv"
    except:
        pass
    raise ValueError("Could not find a 2D conv layer for Grad-CAM.")

def load_rgb(path, target_hw=None):
    img = cv2.imread(path, cv2.IMREAD_COLOR)  # BGR
    if img is None:
        raise FileNotFoundError(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if target_hw is not None:
        img = cv2.resize(img, (target_hw[1], target_hw[0]), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    return img

def make_gradcam(model, img_tensor, head="grade", class_index=None, last_conv_layer_name=None):
    # Build a model that maps input -> (conv_outputs, grade_out, edema_out)
    last_conv = model.get_layer(last_conv_layer_name)
    grad_model = keras.models.Model(
        [model.inputs],
        [last_conv.output, model.get_layer("grade").output, model.get_layer("edema").output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, grade_pred, edema_pred = grad_model(img_tensor, training=False)
        if head == "grade":
            if class_index is None:
                class_index = tf.argmax(grade_pred[0])
            target = grade_pred[:, class_index]
        else:
            # Binary sigmoid: maximize the logit ~ y; use output directly
            target = edema_pred[:, 0]

        grads = tape.gradient(target, conv_outputs)

    conv_outputs = conv_outputs[0]          # [Hc, Wc, C]
    grads = grads[0]                        # [Hc, Wc, C]
    weights = tf.reduce_mean(grads, axis=(0,1))  # [C]
    cam = tf.reduce_sum(weights * conv_outputs, axis=-1)  # [Hc, Wc]

    cam = tf.nn.relu(cam)
    cam = cam / (tf.reduce_max(cam) + 1e-8)
    cam = cam.numpy()
    return cam  # 0..1

def overlay(image_rgb, heatmap, alpha=0.35):
    heatmap = cv2.resize(heatmap, (image_rgb.shape[1], image_rgb.shape[0]))
    heatmap = (255 * heatmap).astype(np.uint8)
    heatmap_c = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)      # BGR
    overlay_bgr = cv2.addWeighted(cv2.cvtColor((image_rgb*255).astype(np.uint8), cv2.COLOR_RGB2BGR),
                                  1.0, heatmap_c, alpha, 0)
    return cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)

def main(cfg_path, split, num, out_dir=None):
    cfg = load_config(cfg_path)

    csv_path = cfg.paths[f"{split}_csv"]
    img_dir  = os.path.join(cfg.paths[f"{split}_dir"], "images")

    out_root = out_dir or os.path.join(cfg.out["run_dir"], "gradcam", split)
    os.makedirs(out_root, exist_ok=True)

    model = keras.models.load_model(cfg.out["ckpt"], compile=False)

    # Find last conv layer inside the full model
    last_conv_name = find_last_conv_layer(model)
    print(f"[Grad-CAM] Using conv layer: {last_conv_name}")

    df = pd.read_csv(csv_path)
    if num > 0:
        df = df.sample(n=min(num, len(df)), random_state=42)

    for _, row in df.iterrows():
        fname = row["Image name"]
        path = os.path.join(img_dir, fname)
        img = load_rgb(path, target_hw=tuple(cfg.model["input_shape"][:2]))
        x = np.expand_dims(img, axis=0)  # [1,H,W,3]

        # Forward pass for class choices
        outputs = model.predict(x, verbose=0)
        p_grade = outputs["grade"][0]
        p_edema = outputs["edema"][0,0]
        k = int(np.argmax(p_grade))

        # Grade Grad-CAM (for predicted class)
        cam_g = make_gradcam(model, x, head="grade", class_index=k, last_conv_layer_name=last_conv_name)
        ov_g  = overlay(img, cam_g)
        out_g = os.path.join(out_root, f"{os.path.splitext(fname)[0]}_grade{str(k)}.png")
        plt.imsave(out_g, ov_g)

        # Edema Grad-CAM (for positive direction)
        cam_e = make_gradcam(model, x, head="edema", class_index=None, last_conv_layer_name=last_conv_name)
        ov_e  = overlay(img, cam_e)
        out_e = os.path.join(out_root, f"{os.path.splitext(fname)[0]}_edema.png")
        plt.imsave(out_e, ov_e)

    print(f"[Grad-CAM] Saved overlays to: {out_root}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/efficientnet_b0.yaml")
    ap.add_argument("--split", default="valid", choices=["train","valid","test"])
    ap.add_argument("--num", type=int, default=24, help="number of images to sample (0=all)")
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()
    main(args.config, args.split, args.num, args.out_dir)
