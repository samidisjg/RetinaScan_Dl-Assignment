import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

@tf.function
def _predict_step(model, x):
    return model(x, training=False)

def evaluate_model(model, test_ds):
    y_true_g, y_pred_g = [], []
    y_true_e, y_pred_e, y_proba_e = [], [], []

    for x, y in test_ds:
        p = model.predict(x, verbose=0)
        g = np.argmax(p["grade"], axis=1)
        e = (p["edema"] > 0.5).astype(int).ravel()

        y_pred_g.extend(g)
        y_pred_e.extend(e)
        y_proba_e.extend(p["edema"].ravel())

        y_true_g.extend(y["grade"].numpy())
        y_true_e.extend(y["edema"].numpy())

    rep_g = classification_report(y_true_g, y_pred_g, digits=4)
    cm_g = confusion_matrix(y_true_g, y_pred_g)

    rep_e = classification_report(y_true_e, y_pred_e, digits=4)
    try:
        auc_e = roc_auc_score(y_true_e, y_proba_e)
    except Exception:
        auc_e = None

    return rep_g, cm_g, rep_e, auc_e
