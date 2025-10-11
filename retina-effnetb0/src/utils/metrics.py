import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score


def grade_report(y_true, y_pred):
return classification_report(y_true, y_pred, digits=4, output_dict=False)


def grade_cm(y_true, y_pred):
return confusion_matrix(y_true, y_pred)


def edema_report(y_true, y_pred):
return classification_report(y_true, y_pred, digits=4, output_dict=False)


def edema_auc(y_true, y_proba):
try:
return roc_auc_score(y_true, y_proba)
except Exception:
return None