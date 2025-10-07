# 🧠 Retina Disease Detection — DenseNet-121

## 📌 Overview

### 🧠 About DenseNet-121
DenseNet-121 (Dense Convolutional Network) is a deep CNN architecture introduced by **Huang et al. (2017)**.  
Unlike traditional CNNs where each layer connects only to the next one, DenseNet connects **each layer to every other layer** in a feed-forward manner.  
This *dense connectivity* encourages **feature reuse**, reduces the number of parameters, and helps gradients flow efficiently during training.  
DenseNet-121 consists of **121 layers** organized into dense blocks and transition layers, making it both **efficient and highly accurate** for image classification tasks.

---

### 💡 Model Summary
This module implements **DenseNet-121** for **retinal disease classification** using the *Diabetic Retinopathy* dataset.  
It is trained **from scratch** (no pretrained ImageNet weights) as part of the **RetinaScan Deep Learning Assignment**.

Each image is labeled with:
- **Retinopathy Grade** — the disease severity level  
- **Risk of Macular Edema** — optional secondary label (for future multi-task extension)

The model outputs the **disease class (0–4)** for each input image.

---

## 🧱 Project Structure
```
Densenet121/
├── data/
│   ├── train.csv              # Training data mapping
│   ├── valid.csv                # Validation data mapping
│   ├── test.csv               # Test data mapping
│   └── Diabetic Retinopathy/  # Image dataset (ignored by Git)
│
├── src/
│   ├── dataset.py             # Custom PyTorch dataset
│   ├── transforms.py          # Image augmentations
│   ├── model_densenet121.py   # DenseNet-121 architecture
│   ├── train.py               # Training loop
│   ├── eval.py                # Evaluation script
│   ├── utils.py               # Helper functions (JSON, metrics)
│   └── make_splits.py         # CSV generator
│
├── runs/                      # Saved checkpoints & logs
├── Densenet121_README.md      # You are here
└── requirements.txt           # Python dependencies
```

---

## ⚙️ Environment Setup
```bash
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

If you are on macOS:
```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1   # Enables Apple Metal backend
```

---

## 📂 Dataset Setup
Place your downloaded dataset under:

```
Densenet121/data/Diabetic Retinopathy/
  ├── train/images/
  ├── valid/images/
  ├── test/images/
  ├── train/annotations.csv
  ├── valid/annotations.csv
  └── test/annotations.csv
```

Then generate CSVs for training:
```bash
python src/make_splits.py
```

---

## 🚀 Training (from scratch)
```bash
python src/train.py   --train_csv data/train.csv   --val_csv data/val.csv   --out_dir runs/densenet121_scratch   --epochs 20   --batch_size 16   --img_size 224   --lr 3e-4   --patience 5
```

- Uses **DenseNet-121** with `pretrained=False`.
- Uses **CrossEntropyLoss** for multi-class classification.
- Augmentations: `RandomResizedCrop`, `HorizontalFlip`, `Brightness/Contrast`, `Normalize`.
- Device: **MPS** (Apple GPU) automatically detected.

---

## 🧩 Evaluation
After training:
```bash
python src/eval.py   --test_csv data/test.csv   --ckpt runs/densenet121_scratch/best.pt   --class_map runs/densenet121_scratch/class_to_idx.json   --out_dir report/figs
```

Outputs:
- `classification_report` with accuracy, precision, recall, F1.
- `confusion_matrix.png` saved in `report/figs/`.

---

## 📊 Sample Results (Baseline – 3 Epochs)
| Metric | Training | Validation |
|---------|-----------|------------|
| Loss | 0.76 | 0.91 |
| Accuracy | 70.5% | 62.2% |
| Macro-F1 | 0.70 | 0.56 |

*(Trained from scratch, 3 epochs on Apple M-series GPU)*

---

## 🔧 Notes
- Increase epochs (15–30) for better performance.
- Use smaller LR (1e-4) if loss oscillates.
- Add stronger augmentations (CLAHE, rotation) for robustness.
- `runs/` and `data/Diabetic Retinopathy/` are ignored in `.gitignore` to avoid large pushes.

---

## 🧾 Citation
Dataset: [Kaggle — Retinal Disease Detection](https://www.kaggle.com/datasets/mohamedabdalkader/retinal-disease-detection)

Model: **DenseNet-121** (Huang et al., CVPR 2017)

---

## 👤 Author
**Pradicksha Pradeepraj**  
SLIIT — Year 4 / Semester 1  
Module: SE4050 — Deep Learning  
Algorithm Used: **DenseNet-121**  
