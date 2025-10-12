# 🩺 Retinal Disease Detection — Multi‑Model Deep Learning

A supervised learning project using **four CNN architectures** — ResNet‑18, EfficientNet‑B0, ConvNeXt‑Tiny, and DenseNet‑121 — for automated detection of **Diabetic Retinopathy (DR)** and **Macular Edema (ME)** from retinal fundus images.👁️🧑‍⚕️

---

## 🚀 Overview
This project explores the use of deep learning for retinal image analysis to assist ophthalmologists.  
Each model was trained on the **Kaggle Retinal Disease Detection dataset**, targeting:  
- **DR Grade (0–4)** → Multi‑class classification  
- **Macular Edema Risk (0–1)** → Binary classification  

---

## ⚙️ Tech Stack
- **Frameworks:** PyTorch, Torchvision, Timm  
- **Techniques:** Transfer Learning, Focal Loss, Weighted Sampling, CLAHE  
- **Visuals:** Grad‑CAM, ROC Curves, Confusion Matrices  

---

## 📊 Dataset
- **Source:** [Kaggle — Retinal Disease Detection](https://www.kaggle.com/datasets/mohamedabdalkader/retinal-disease-detection)  
- **Splits:** Train 1,577 • Validation 339 • Test 338  
- **Preprocessing:** Resize (224–448), CLAHE, ImageNet normalization, random flips & rotations  

---

## 🔍 Key Insights
- **DenseNet‑121** → highest grading accuracy (~0.68).  
- **ResNet‑18** → best edema detection (AUC ≈ 0.99).  
- **EfficientNet‑B0** & **ConvNeXt‑Tiny** → best efficiency vs accuracy trade‑off.  
- Grad‑CAMs confirmed medical interpretability (focus on macula and lesion zones).

---

## 👩‍⚕️ Clinical Relevance
Provides an AI‑assisted screening tool to support ophthalmologists in early DR and ME detection.  
---

---

## Note 📝
-  All the Relevant Model Readme.md are added inside each model folder.
---

## 👥 Team — Group 11(Deep Learning Module SE4050)
- IT22607232 — **Gamage S S J**  
- IT22577160 — **Nimesh R H R**  
- IT22602978 — **Damnidu T W T**  
- IT22603418 — **P Pradicksha**  

---



## 📜 License
Academic / Research use only.  
© 2025 — SLIIT SE4050 | Retinal Disease Detection Project.
