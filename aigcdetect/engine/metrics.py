from __future__ import annotations
import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score, f1_score


def binary_metrics(y_true, fake_prob, threshold=0.5):
    y, s = np.asarray(y_true), np.asarray(fake_prob, dtype=np.float64)
    if y.ndim != 1 or s.shape != y.shape or len(y) == 0:
        raise ValueError("Expected matching nonempty 1-D labels and probabilities")
    if not np.isin(y, [0, 1]).all() or not np.isfinite(s).all() or ((s < 0) | (s > 1)).any():
        raise ValueError("Labels must be binary and probabilities finite in [0,1]")
    pred = (s >= threshold).astype(np.int64)
    real, fake = y == 0, y == 1
    r = float((pred[real] == 0).mean()) if real.any() else None
    f = float((pred[fake] == 1).mean()) if fake.any() else None
    both = real.any() and fake.any()
    return {"acc": float(accuracy_score(y, pred)),
            "ap": float(average_precision_score(y, s)) if both else None,
            "auc": float(roc_auc_score(y, s)) if both else None,
            "f1": float(f1_score(y, pred, zero_division=0)),
            "real_acc": r, "fake_acc": f,
            "balanced_acc": (r + f) / 2 if both else None,
            "n": len(y), "n_real": int(real.sum()), "n_fake": int(fake.sum()),
            "threshold": float(threshold)}
