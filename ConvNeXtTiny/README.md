# Retinal Disease Detection with ConvNeXt‑Tiny (PyTorch + timm)

> A compact, reproducible pipeline for retinal disease classification using **ConvNeXt‑Tiny**, trained and evaluated on the Kaggle *Retinal Disease Detection* dataset with strong baselines for imbalanced data.

---

## 📌 Overview

This repository implements a modern transfer‑learning workflow for multi‑class retinal disease classification. It uses **ConvNeXt‑Tiny** as the backbone, two‑phase fine‑tuning (head‑only → partial/full unfreeze), class‑imbalance handling, and thorough tracking of metrics and plots.

**Final Validation (from `src.eval`)**

- **Accuracy:** `0.6183`  
- **Macro Precision / Recall / F1:** `0.5110 / 0.5441 / 0.5212`

Per‑class metrics:

| class | precision | recall | f1-score | support |
|---:|---:|---:|---:|---:|
| 0 | 0.8355 | 0.7135 | 0.7697 | 178 |
| 1 | 0.2889 | 0.2955 | 0.2921 | 44 |
| 2 | 0.5000 | 0.6447 | 0.5632 | 76 |
| 3 | 0.4545 | 0.4000 | 0.4255 | 25 |
| 4 | 0.4762 | 0.6667 | 0.5556 | 15 |

Confusion matrix:

|       | pred_0 | pred_1 | pred_2 | pred_3 | pred_4 |
|:-----:|------:|------:|------:|------:|------:|
| true_0 | 127 | 25 | 25 | 0 | 1 |
| true_1 | 16 | 13 | 14 | 0 | 1 |
| true_2 | 9 | 7 | 49 | 9 | 2 |
| true_3 | 0 | 0 | 8 | 10 | 7 |
| true_4 | 0 | 0 | 2 | 3 | 10 |

---

## 🧠 About ConvNeXt‑Tiny

ConvNeXt is a convolutional architecture that revisits CNN design with insights from Vision Transformers (ViTs). The **Tiny** variant is lightweight and fast while retaining strong accuracy, making it suitable for laptops and mid‑range GPUs. We use the `timm` implementation and ImageNet‑pretrained weights.

- Depthwise separable convs + LayerNorm
- Large kernel sizes and modern training tricks
- Drop‑in replacement for many ResNet‑style backbones

---

## 📚 Dataset

- **Source:** Kaggle — *Retinal Disease Detection*  
- **Task:** 5‑class classification
- **Input:** Fundus images (resized to 448×448 for this project)
- **Splits:** Provided as CSVs (`splits/train.csv`, `splits/val.csv`, `splits/test.csv`) with columns: `filepath,label`

> Place image folders under `data/…` and make sure CSV paths in `splits/*.csv` point correctly.

---

## 🗂 Project Structure

```
.
├── checkpoints/
│   ├── convnext_tiny.pt
│   ├── convnext_tiny_first.pt
│   ├── convnext_tiny_second.pt
│   ├── convnext_tiny_third.pt
│   ├── convnext_tiny_fourth.pt
│   └── history/
│       └── history_convnext_tiny_new.csv
├── configs/
│   └── train_convnext_tiny.yaml
├── data/
├── eval_plots/
│   ├── confusion_matrix.png
│   ├── per_class_precision.png
│   ├── per_class_recall.png
│   └── per_class_f1.png
├── splits/
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
├── src/
│   ├── dataset.py
│   ├── eval.py
│   ├── models.py
│   ├── plot_training_curves.py
│   ├── train.py
│   └── utils.py
├── training_plots/
│   └── training_overview.png   # (from plot_training_curves.py)
├── README.md
└── requirements.txt
```

---

## ⚙️ Environment Setup

> Tested on macOS (Apple Silicon / Intel) and Linux. Python 3.11 recommended.

```bash
# (Optional) Conda env
conda create -n retina-convnext python=3.11 -y
conda activate retina-convnext

# Install deps
pip install -r requirements.txt

# Verify torch device
python -c "import torch;print('MPS:',torch.backends.mps.is_available(),'CUDA:',torch.cuda.is_available())"
```

---

## 🚀 Training & Evaluation

**Train (reads YAML config):**
```bash
python -m src.train --config configs/train_convnext_tiny.yaml
```

- Checkpoints will be written under `checkpoints/…`
- Training history CSV under `checkpoints/history/*.csv`

**Plot training curves:**
```bash
python -m src.plot_training_curves --history checkpoints/history/history_convnext_tiny_new.csv   --out training_plots/training_overview.png
```

**Evaluate the best checkpoint:**
```bash
python -m src.eval
# Saves figures to: eval_plots/
# (confusion_matrix.png, per_class_precision.png, per_class_recall.png, per_class_f1.png)
```

---

## 🧩 Key Techniques Used

- **Transfer Learning with ConvNeXt‑Tiny** (ImageNet pretrained)
- **Two‑Phase Fine‑Tuning**  
  1) Freeze backbone, train head only → 2) Unfreeze high‑level stages (3–4) then optionally full model
- **Cosine Annealing LR + Warmup**
- **Differential LRs** (backbone vs head)
- **Label Smoothing** for stability
- **Imbalance Handling** via `WeightedRandomSampler`
- **Gradient Clipping** for robust training
- **Comprehensive Metrics**: accuracy, balanced accuracy, macro‑F1, per‑class precision/recall/F1

---

## 📈 Final Experimental Results

**Summary (Validation):**
- Accuracy: **0.6183**
- Balanced Accuracy: **≈0.547**
- Macro F1: **≈0.521**

**Observations:**
- Majority class (0) shows strong precision/recall; tail classes improve with sampler + partial unfreeze.
- Class **2** benefits most from fine‑tuning (recall **0.6447**).  
- Class **1** remains challenging; targeted augmentation or class‑aware loss (e.g., focal loss) could help.

---

## 🖼️ Visualizations

Training overview (loss/acc, balanced acc & macro‑F1, LRs):

![Training Overview](training_plots/training_overview.png)

Confusion matrix & per‑class metrics:

- `eval_plots/confusion_matrix.png`  
- `eval_plots/per_class_precision.png`  
- `eval_plots/per_class_recall.png`  
- `eval_plots/per_class_f1.png`

> These figures are autogenerated by `src.eval` and `src.plot_training_curves`.

---

## 🔍 Key Insights & Next Steps

- **Partial unfreeze** stabilizes early fine‑tuning; full unfreeze can be introduced after several epochs.  
- **Weighted sampling** improved tail‑class recall without harming head‑class precision too much.  
- **Further gains** likely from:
  - Class‑balanced / focal loss
  - Stronger augmentations (MixUp/CutMix, color jitter tuned for fundus images)
  - Higher input resolution (e.g., 512–640) if memory allows
  - TTA (test‑time augmentation)
  - Mild EMA (exponential moving average) of weights

---

## 📎 References

1. **ConvNeXt:** A ConvNet for the 2020s — *Liu et al., 2022*  
2. **timm:** PyTorch Image Models — *Ross Wightman*  
3. **Kaggle Dataset:** Retinal Disease Detection  
4. **Cosine Annealing with Warm Restarts:** *Loshchilov & Hutter, 2016*  
5. **Label Smoothing:** *Szegedy et al., 2015*

> Note: See `requirements.txt` for exact library versions used in this repo.

---


## 🔁 Reproducibility

- Deterministic seeds are set in `src/train.py`.
- YAML config captures all hyper‑parameters for clean experiment tracking.
- History CSV enables post‑hoc analysis and plotting for reports.


---

## 🏗️ Model Architecture

Below is a high-level CNN architecture diagram used to illustrate the **feature extraction → classification** pipeline (convolutions, pooling, flatten + fully connected, softmax). This aligns with our transfer-learning setup where ConvNeXt-Tiny acts as the feature extractor and a classifier head performs the final prediction.

![CNN Architecture](/Users/thewandamnidu/Desktop/SLIIT/Y4S1/DL/Assignment-Projects/Retinal-Disease-Detection-ConvNeXt-Tiny-DL/assets/architecture.png)

> Note: This diagram is a generic CNN schematic for explanatory purposes.



## 👤 Author

- **Thewan Damindu** *(BSc (Hons) in Information Technology Specialized in Software Engineering— SLIIT  SLIIT – Y4S1)*  
  - DL Assignment – Retinal Disease Detection (ConvNeXt‑Tiny)
  - Email / Contact: _it22602978@my.sllit.lk_

---