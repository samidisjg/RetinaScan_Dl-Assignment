# ResNet18 Retina — Test Report

## Dataset
Rows: **338**  |  Grade col: `Retinopathy grade`  |  Edema col: `Risk of macular edema`
Grade counts: `{0: 178, 1: 44, 2: 76, 3: 25, 4: 15}`
Edema pos/neg: **66/272**

## Metrics
- Grade Accuracy: **0.6686**
- Grade Macro-F1: **0.5829**
- Edema ROC-AUC: **0.9875**

## Figures
![](confusion_matrix_counts.png)
![](confusion_matrix_normalized.png)
![](roc_edema.png)