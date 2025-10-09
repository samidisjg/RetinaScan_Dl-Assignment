# scripts/evaluate_and_viz.py
import os, sys, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- make project importable ----------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import tensorflow as tf
import tf_keras as keras
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

from src.config import load_config
from src.data.tf_dataset import DataBuilder


# ---------- output/label adapters ----------
def _outputs_to_grade_edema(model, out):
    """
    Return (grade_probs, edema_probs) regardless of whether `out` is a dict or list/tuple.
    grade_probs shape: [B, 5], edema_probs shape: [B, 1]
    """
    if isinstance(out, dict):
        d = out
    else:
        # Map list/tuple to names using model.outputs
        names = []
        for t in model.outputs:
            n = getattr(t, "name", "")
            n = n.split(":")[0].split("/")[0] if n else f"out{len(names)}"
            names.append(n)
        d = {names[i]: out[i] for i in range(len(out))}

    if "grade" in d and "edema" in d:
        return d["grade"], d["edema"]

    # Fallback by shape
    grade_arr, edema_arr = None, None
    for v in d.values():
        arr = np.asarray(v)
        if arr.ndim >= 2 and arr.shape[-1] == 5:
            grade_arr = arr
        elif arr.ndim >= 1 and arr.shape[-1] == 1:
            edema_arr = arr
    if grade_arr is None or edema_arr is None:
        raise RuntimeError("Could not identify outputs for 'grade' (5-way) and 'edema' (1-way).")
    return grade_arr, edema_arr


def _targets_to_grade_edema(batch_y):
    """Return (grade_labels, edema_labels) whether labels are a dict or a (grade, edema) tuple/list."""
    if isinstance(batch_y, dict) and "grade" in batch_y and "edema" in batch_y:
        return batch_y["grade"], batch_y["edema"]

    if isinstance(batch_y, (tuple, list)) and len(batch_y) == 2:
        return batch_y[0], batch_y[1]

    # Best-effort: detect by dtype/shape
    try:
        t_list = list(batch_y) if isinstance(batch_y, (tuple, list)) else [batch_y]
        g, e = None, None
        for item in t_list:
            t = tf.convert_to_tensor(item)
            # grade is integer labels; edema is float/bool or numeric
            if t.dtype.is_integer:
                g = t
            else:
                e = t
        if g is not None and e is not None:
            return g, e
    except Exception:
        pass

    raise RuntimeError("Could not parse batch_y; expected dict with keys 'grade'/'edema' or tuple/list (grade, edema).")


# ---------- small helpers ----------
def _ensure_dir(d):
    os.makedirs(d, exist_ok=True)
    return d

def _plot_confusion_matrix(cm, classes, save_path, normalize=True, title="Confusion matrix"):
    plt.figure(figsize=(6, 5), dpi=150)
    if normalize:
        cm = cm.astype("float") / (cm.sum(axis=1, keepdims=True) + 1e-9)
    im = plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    fmt = ".2f" if normalize else "d"
    thresh = cm.max() * 0.6
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

def _plot_roc(y_true, y_score, save_path, title="ROC (edema)"):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(5, 5), dpi=150)
    plt.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], lw=1, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

def _plot_training_curves(history_csv, save_path):
    if not os.path.exists(history_csv):
        return
    hist = pd.read_csv(history_csv)
    epochs = hist["epoch"].values if "epoch" in hist.columns else np.arange(len(hist))

    fig = plt.figure(figsize=(8, 6), dpi=150)
    ax1 = fig.add_subplot(2, 2, 1); ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3); ax4 = fig.add_subplot(2, 2, 4)

    if "grade_accuracy" in hist and "val_grade_accuracy" in hist:
        ax1.plot(epochs, hist["grade_accuracy"], label="train")
        ax1.plot(epochs, hist["val_grade_accuracy"], label="val")
        ax1.set_title("Grade accuracy"); ax1.legend()

    if "edema_auc" in hist and "val_edema_auc" in hist:
        ax2.plot(epochs, hist["edema_auc"], label="train")
        ax2.plot(epochs, hist["val_edema_auc"], label="val")
        ax2.set_title("Edema AUC"); ax2.legend()

    if "loss" in hist and "val_loss" in hist:
        ax3.plot(epochs, hist["loss"], label="train")
        ax3.plot(epochs, hist["val_loss"], label="val")
        ax3.set_title("Total loss"); ax3.legend()

    if "grade_loss" in hist and "val_grade_loss" in hist:
        ax4.plot(epochs, hist["grade_loss"], label="train")
        ax4.plot(epochs, hist["val_grade_loss"], label="val")
        ax4.set_title("Grade loss"); ax4.legend()

    fig.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


# ---------- Grad-CAM (no graph disconnect) ----------
def _get_layer_by_name(model, name):
    try:
        return model.get_layer(name)
    except Exception:
        return None

def gradcam(model, images, class_indices=None, last_conv_name="top_conv", head="grade"):
    """
    images: float32 tensors in model input scale (your pipeline feeds 0..255)
    returns: heatmaps in [0,1], shape (N, H, W)
    """
    # Grab key layers from your trained model
    resize = model.get_layer("resize_to_224")
    base = model.get_layer("efficientnetb0")
    grade_head = model.get_layer("grade")
    edema_head = model.get_layer("edema")

    if resize is None or base is None or grade_head is None or edema_head is None:
        raise RuntimeError("Expected layers 'resize_to_224', 'efficientnetb0', 'grade', 'edema' not found.")

    # Locate the last conv layer inside EfficientNet
    try:
        conv_layer = base.get_layer(last_conv_name)
    except Exception as e:
        raise RuntimeError(f"Conv layer '{last_conv_name}' not found inside EfficientNet.") from e

    # --- Build a single-pass graph that yields BOTH conv maps and head preds ---
    # Inner feature extractor: EfficientNet input -> (top_conv, pooled_features)
    inner = keras.Model(
        inputs=base.input,
        outputs=[conv_layer.output, base.output],
        name="inner_effnet_cam"
    )

    # Joint model: original input -> resize -> inner -> heads
    cam_inp = keras.Input(shape=model.input_shape[1:], name="cam_input")
    x224 = resize(cam_inp)                                   # (B,224,224,3)
    conv_maps_t, feats_t = inner(x224, training=False)       # (B,h,w,C), (B,1280)
    grade_preds_t = grade_head(feats_t)                      # (B,5)
    edema_preds_t = edema_head(feats_t)                      # (B,1)
    sub_out_t = grade_preds_t if head == "grade" else edema_preds_t

    joint = keras.Model(inputs=cam_inp, outputs=[conv_maps_t, sub_out_t], name="cam_joint")

    # --- Gradients through the single forward pass ---
    with tf.GradientTape() as tape:
        conv_maps, preds = joint(images, training=False)     # 1 pass for both tensors
        if head == "grade":
            if class_indices is None:
                class_indices = tf.argmax(preds, axis=-1)
            idx = tf.stack([tf.range(tf.shape(preds)[0]), tf.cast(class_indices, tf.int32)], axis=1)
            target = tf.gather_nd(preds, idx)                # (B,)
        else:
            target = tf.squeeze(preds, axis=-1)              # (B,)

    grads = tape.gradient(target, conv_maps)                 # (B,h,w,C)
    if grads is None:
        raise RuntimeError(
            "Gradients are None. Make sure images are float32 in 0..255 and "
            f"'{last_conv_name}' exists inside 'efficientnetb0'."
        )

    # Global-average weights and CAM
    weights = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)  # (B,1,1,C)
    cam = tf.reduce_sum(weights * conv_maps, axis=-1)            # (B,h,w)

    # ReLU + normalize to [0,1], then resize to input size
    cam = tf.nn.relu(cam)
    cam_min = tf.reduce_min(cam, axis=(1, 2), keepdims=True)
    cam_max = tf.reduce_max(cam, axis=(1, 2), keepdims=True)
    cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

    in_h, in_w = images.shape[1], images.shape[2]
    cam = tf.image.resize(cam[..., tf.newaxis], (in_h, in_w))
    return tf.squeeze(cam, axis=-1).numpy()



def save_gradcam_overlays(model, df, root_dir, out_dir, n_samples=8, head="grade"):
    _ensure_dir(out_dir)
    paths = [os.path.join(root_dir, "images", name) for name in df["Image name"].head(n_samples).tolist()]
    xs = []
    for p in paths:
        img = tf.io.read_file(p)
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
        img = tf.image.convert_image_dtype(img, tf.float32)  # 0..1
        img = img * 255.0  # keep 0..255 because EfficientNet has internal rescale
        xs.append(tf.image.resize(img, (model.input_shape[1], model.input_shape[2]), antialias=True))
    x = tf.stack(xs, axis=0)

    class_idx = None
    if head == "grade":
        out = model.predict(x, verbose=0)
        preds, _ = _outputs_to_grade_edema(model, out)  # grade probs [N,5]
        class_idx = np.argmax(preds, axis=-1)

    cams = gradcam(model, x, class_indices=class_idx, head=head)  # [N, H, W]

    for i, (img, cam, name) in enumerate(zip(x.numpy() / 255.0, cams, df["Image name"].head(n_samples))):
        plt.figure(figsize=(4, 4), dpi=150)
        plt.imshow(img)
        plt.imshow(cam, cmap="jet", alpha=0.35)
        plt.axis("off")
        plt.title(f"{head.upper()} Grad-CAM: {name}")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{i:02d}_{head}_gradcam_{os.path.splitext(name)[0]}.png"),
                    bbox_inches="tight")
        plt.close()


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/efficientnet_b0.yaml")
    ap.add_argument("--split", choices=["valid", "test"], default="valid")
    ap.add_argument("--samples_for_gradcam", type=int, default=8)
    args = ap.parse_args()

    cfg = load_config(args.config)

    # where to save
    run_dir = cfg.out["run_dir"]
    plots_dir = _ensure_dir(os.path.join(run_dir, "plots"))
    reports_dir = _ensure_dir(os.path.join(run_dir, "reports"))

    # data
    split = args.split
    csv_path = cfg.paths[f"{split}_csv"]
    root_dir = cfg.paths[f"{split}_dir"]
    df = pd.read_csv(csv_path)

    # dataset (no shuffle)
    db = DataBuilder(tuple(cfg.model["input_shape"]), augment=False)
    ds = db.dataframe_to_ds(csv_path, root_dir, shuffle=False, batch=cfg.train["batch_size"])

    # load best model
    model_path = cfg.out["ckpt"]
    model = keras.models.load_model(model_path)

    # collect preds
    y_true_g, y_pred_g = [], []
    y_true_e, y_pred_e, y_score_e = [], [], []

    for batch_x, batch_y in ds:
        out = model.predict(batch_x, verbose=0)
        g_prob, e_prob = _outputs_to_grade_edema(model, out)
        e_prob = e_prob.reshape(-1)   # [B]

        g_hat = np.argmax(g_prob, axis=1)
        e_hat = (e_prob >= 0.5).astype(np.int32)

        g_true_batch, e_true_batch = _targets_to_grade_edema(batch_y)

        y_true_g.extend(np.array(g_true_batch).astype(int).tolist())
        y_pred_g.extend(g_hat.tolist())

        y_true_e.extend(np.array(e_true_batch).astype(int).tolist())
        y_pred_e.extend(e_hat.tolist())
        y_score_e.extend(e_prob.tolist())

    y_true_g = np.array(y_true_g); y_pred_g = np.array(y_pred_g)
    y_true_e = np.array(y_true_e); y_pred_e = np.array(y_pred_e)
    y_score_e = np.array(y_score_e)

    # ---- reports ----
    # grade
    cls_names = [str(i) for i in range(5)]
    rep_grade = classification_report(y_true_g, y_pred_g, target_names=cls_names, digits=4)
    with open(os.path.join(reports_dir, f"classification_report_grade_{split}.txt"), "w") as f:
        f.write(rep_grade + "\n")

    cm = confusion_matrix(y_true_g, y_pred_g, labels=list(range(5)))
    _plot_confusion_matrix(cm, classes=cls_names,
                           save_path=os.path.join(plots_dir, f"cm_grade_{split}.png"),
                           normalize=True, title=f"Grade confusion matrix ({split})")

    # edema
    rep_edema = classification_report(y_true_e, y_pred_e, target_names=["no_edema", "edema"], digits=4)
    with open(os.path.join(reports_dir, f"classification_report_edema_{split}.txt"), "w") as f:
        f.write(rep_edema + "\n")

    _plot_roc(y_true_e, y_score_e, save_path=os.path.join(plots_dir, f"roc_edema_{split}.png"),
              title=f"ROC (edema) on {split}")

    # training curves (if CSVLogger was enabled)
    _plot_training_curves(os.path.join(run_dir, "history.csv"),
                          save_path=os.path.join(plots_dir, "training_curves.png"))

    # ---- Grad-CAM ----
    save_gradcam_overlays(model, df, root_dir, os.path.join(plots_dir, f"gradcam_{split}_grade"),
                          n_samples=args.samples_for_gradcam, head="grade")
    save_gradcam_overlays(model, df, root_dir, os.path.join(plots_dir, f"gradcam_{split}_edema"),
                          n_samples=args.samples_for_gradcam, head="edema")

    print("Saved:")
    print(f"  Confusion matrix: {os.path.join(plots_dir, f'cm_grade_{split}.png')}")
    print(f"  Grade report:     {os.path.join(reports_dir, f'classification_report_grade_{split}.txt')}")
    print(f"  Edema report:     {os.path.join(reports_dir, f'classification_report_edema_{split}.txt')}")
    print(f"  Edema ROC:        {os.path.join(plots_dir, f'roc_edema_{split}.png')}")
    print(f"  Train curves:     {os.path.join(plots_dir, 'training_curves.png')}")
    print(f"  Grad-CAM samples: {os.path.join(plots_dir, f'gradcam_{split}_grade')}")
    print("Done.")


if __name__ == "__main__":
    main()
