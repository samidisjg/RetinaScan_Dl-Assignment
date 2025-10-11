# src/engine/metrics.py
from __future__ import annotations
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)


def multitask_loss(
    logits_grade,
    logit_edema,
    y_grade,
    y_edema,
    label_smoothing: float = 0.0,
    grade_weights: torch.Tensor | None = None,
    edema_pos_weight: torch.Tensor | None = None,
):
    """
    Grade head: multi-class CE (optionally weighted/smoothed)
    Edema head: BCEWithLogits (optionally pos_weight)
    Returns: total_loss, loss_grade_scalar, loss_edema_scalar
    """
    if label_smoothing > 0:
        num_classes = logits_grade.size(1)
        y_onehot = torch.nn.functional.one_hot(y_grade, num_classes).float()
        y_onehot = y_onehot * (1 - label_smoothing) + label_smoothing / num_classes
        if grade_weights is not None:
            # per-sample weights based on true class
            w = grade_weights.gather(0, y_grade)
            loss_grade = (
                -(y_onehot * F.log_softmax(logits_grade, dim=1)).sum(dim=1) * w
            ).mean()
        else:
            loss_grade = (
                -(y_onehot * F.log_softmax(logits_grade, dim=1)).sum(dim=1)
            ).mean()
    else:
        loss_grade = F.cross_entropy(logits_grade, y_grade, weight=grade_weights)

    if edema_pos_weight is not None:
        loss_edema = F.binary_cross_entropy_with_logits(
            logit_edema, y_edema, pos_weight=edema_pos_weight
        )
    else:
        loss_edema = F.binary_cross_entropy_with_logits(logit_edema, y_edema)

    return loss_grade + loss_edema, loss_grade.item(), loss_edema.item()


def compute_metrics(y_true_grade, y_pred_grade, y_true_edema, y_prob_edema):
    acc = accuracy_score(y_true_grade, y_pred_grade)
    f1_macro = f1_score(y_true_grade, y_pred_grade, average="macro")
    cm = confusion_matrix(y_true_grade, y_pred_grade)
    report = classification_report(y_true_grade, y_pred_grade, digits=4)
    try:
        auc = roc_auc_score(y_true_edema, y_prob_edema)
    except ValueError:
        auc = float("nan")
    return {
        "grade_acc": acc,
        "grade_f1_macro": f1_macro,
        "grade_confusion_matrix": cm,
        "grade_classification_report": report,
        "edema_auc": auc,
    }
