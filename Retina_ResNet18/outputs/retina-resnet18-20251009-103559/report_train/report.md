# ResNet18 Retina — Train Report

## Dataset
Rows: **1577**  |  Grade col: `Retinopathy grade`  |  Edema col: `Risk of macular edema`
Grade counts: `{0: 829, 1: 206, 2: 358, 3: 116, 4: 68}`
Edema pos/neg: **309/1268**

## Metrics
- Grade Accuracy: **0.7679**
- Grade Macro-F1: **0.7750**
- Edema ROC-AUC: **0.9997**

## Figures
![](confusion_matrix_counts.png)
![](confusion_matrix_normalized.png)
![](roc_edema.png)