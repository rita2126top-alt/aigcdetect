from __future__ import annotations
import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score, f1_score


def binary_metrics(y_true, fake_prob, threshold=0.5):
    y = np.asarray(y_true, dtype=np.int64)
    s = np.asarray(fake_prob, dtype=np.float64)
    pred = (s >= threshold).astype(np.int64)
    real_mask = y == 0
    fake_mask = y == 1
    return {
        "acc": float(accuracy_score(y, pred)),
        "ap": float(average_precision_score(y, s)) if len(np.unique(y)) > 1 else float("nan"),
        "auc": float(roc_auc_score(y, s)) if len(np.unique(y)) > 1 else float("nan"),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "real_acc": float((pred[real_mask] == 0).mean()) if real_mask.any() else float("nan"),
        "fake_acc": float((pred[fake_mask] == 1).mean()) if fake_mask.any() else float("nan"),
        "n": int(len(y)),
    }
