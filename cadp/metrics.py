"""Binary metrics on P(fake), not logits. Undefined metrics are JSON null."""
from __future__ import annotations
import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                              f1_score, roc_auc_score, roc_curve)


def binary_metrics(labels, probabilities, threshold=0.5, bins=15, bootstrap=0, seed=42):
    y = np.asarray(labels, dtype=np.int64)
    p = np.asarray(probabilities, dtype=np.float64)
    if y.ndim != 1 or p.shape != y.shape or not len(y):
        raise ValueError('Nonempty matching one-dimensional labels and probabilities required')
    if not np.isin(y, [0, 1]).all() or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError('Invalid binary labels or probabilities')
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0,1]).ravel().tolist()
    real_acc = tn/(tn+fp) if tn+fp else None
    fake_acc = tp/(tp+fn) if tp+fn else None
    both = len(np.unique(y)) == 2
    confidence = np.maximum(p, 1-p)
    # Calibration is for the natural argmax classifier; operating threshold metrics are separate.
    correct = ((p >= 0.5).astype(int) == y).astype(float)
    indexes = np.minimum((confidence*bins).astype(int), bins-1)
    ece = sum(np.mean(indexes == i) * abs(correct[indexes == i].mean()-confidence[indexes == i].mean())
              for i in range(bins) if np.any(indexes == i))
    q = np.clip(p, 1e-12, 1-1e-12)
    result = {'n': len(y), 'n_real': int((y==0).sum()), 'n_fake': int((y==1).sum()),
              'threshold': threshold, 'accuracy': float(accuracy_score(y, pred)),
              'balanced_accuracy': (real_acc+fake_acc)/2 if both else None,
              'real_accuracy': real_acc, 'fake_accuracy': fake_acc,
              'f1': float(f1_score(y, pred, zero_division=0)),
              'auroc': float(roc_auc_score(y, p)) if both else None,
              'ap': float(average_precision_score(y, p)) if both else None,
              'fpr': fp/(tn+fp) if tn+fp else None, 'tpr': fake_acc,
              'tpr_at_fpr_1pct': None, 'tpr_at_fpr_5pct': None,
              'ece': float(ece), 'brier': float(((p-y)**2).mean()),
              'nll': float(-(y*np.log(q)+(1-y)*np.log1p(-q)).mean()),
              'confusion': {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp}}
    if both:
        fpr, tpr, _ = roc_curve(y, p)
        # Empirical operating point, no optimistic interpolation beyond the FPR budget.
        result['tpr_at_fpr_1pct'] = float(tpr[fpr <= 0.01].max())
        result['tpr_at_fpr_5pct'] = float(tpr[fpr <= 0.05].max())
    if bootstrap:
        rng, aucs = np.random.default_rng(seed), []
        for _ in range(bootstrap):
            ids = rng.integers(0, len(y), len(y))
            if len(np.unique(y[ids])) == 2:
                aucs.append(roc_auc_score(y[ids], p[ids]))
        result['auroc_bootstrap_95ci'] = np.quantile(aucs, [.025,.975]).tolist() if aucs else None
        result['bootstrap_note'] = 'Image-level bootstrap; not a substitute for group-level uncertainty or multiple training seeds.'
    return result


def choose_threshold(labels, probabilities):
    """Validation-only Youden J calibration; never call on held-out test labels."""
    y, p = np.asarray(labels), np.asarray(probabilities)
    if set(np.unique(y)) != {0, 1}:
        raise ValueError('Threshold calibration requires both validation classes')
    fpr, tpr, thresholds = roc_curve(y, p)
    valid = np.isfinite(thresholds) & (thresholds >= 0) & (thresholds <= 1)
    ids = np.flatnonzero(valid)
    return float(thresholds[ids[np.argmax((tpr-fpr)[ids])]])
