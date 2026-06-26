"""Phase 1 — classification metrics + calibration (METHODS_SECTION §3.3.1, §3.3.2).

Computes the standard metric suite (accuracy + bootstrap CI, P/R/F1 macro & weighted,
specificity, Cohen's kappa, MCC, ROC-AUC, PR-AUC) and calibration (ECE, Brier, NLL,
temperature scaling) for one model's predictions on a test set.

All functions take (y_true_int, y_prob) where y_prob is (N, C) softmax output.
"""
from __future__ import annotations

import numpy as np

CLASSES = ["Covid", "Normal", "Pneumonia", "TB"]


def bootstrap_accuracy_ci(y_true, y_pred, n_boot=1000, seed=42):
    """Accuracy with 95% bootstrap CI (§3.3.1)."""
    rng = np.random.default_rng(seed)
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    acc = float((y_true == y_pred).mean())
    n = len(y_true)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        boots.append((y_true[idx] == y_pred[idx]).mean())
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return acc, float(lo), float(hi)


def standard_metrics(y_true, y_prob):
    """Full standard metric suite for one model. Returns a flat dict."""
    from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                                 cohen_kappa_score, matthews_corrcoef,
                                 roc_auc_score, average_precision_score,
                                 confusion_matrix)

    y_true = np.asarray(y_true)
    y_pred = y_prob.argmax(1)
    C = y_prob.shape[1]
    onehot = np.eye(C)[y_true]

    acc, lo, hi = bootstrap_accuracy_ci(y_true, y_pred)
    pw, rw, fw, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    pm, rm, fm, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=range(C))
    # per-class specificity from the confusion matrix
    spec = []
    for c in range(C):
        tn = cm.sum() - cm[c, :].sum() - cm[:, c].sum() + cm[c, c]
        fp = cm[:, c].sum() - cm[c, c]
        spec.append(tn / (tn + fp) if (tn + fp) else 0.0)

    out = {
        "accuracy": acc, "acc_ci_lo": lo, "acc_ci_hi": hi,
        "precision_w": pw, "recall_w": rw, "f1_w": fw,
        "precision_m": pm, "recall_m": rm, "f1_m": fm,
        "specificity_m": float(np.mean(spec)),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "confusion_matrix": cm.tolist(),
    }
    # AUCs guarded against single-class test subsets
    try:
        out["roc_auc_macro"] = float(roc_auc_score(onehot, y_prob, average="macro", multi_class="ovr"))
        out["pr_auc_macro"] = float(average_precision_score(onehot, y_prob, average="macro"))
    except ValueError:
        out["roc_auc_macro"] = float("nan")
        out["pr_auc_macro"] = float("nan")
    return out


# ---------------------------------------------------------------- calibration

def expected_calibration_error(y_true, y_prob, n_bins=15):
    """ECE over confidence bins (§3.3.2)."""
    y_true = np.asarray(y_true)
    conf = y_prob.max(1)
    pred = y_prob.argmax(1)
    correct = (pred == y_true).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        m = (conf > bins[i]) & (conf <= bins[i + 1])
        if m.sum():
            ece += abs(correct[m].mean() - conf[m].mean()) * m.sum() / len(y_true)
    return float(ece)


def brier_score(y_true, y_prob):
    C = y_prob.shape[1]
    onehot = np.eye(C)[np.asarray(y_true)]
    return float(((y_prob - onehot) ** 2).sum(1).mean())


def nll(y_true, y_prob, eps=1e-12):
    y_true = np.asarray(y_true)
    p = np.clip(y_prob[np.arange(len(y_true)), y_true], eps, 1.0)
    return float(-np.log(p).mean())


def fit_temperature(y_true, logits):
    """Fit scalar temperature T on validation logits by minimizing NLL (§3.3.2, Guo 2017)."""
    from scipy.optimize import minimize_scalar

    y_true = np.asarray(y_true)

    def _nll_T(T):
        z = logits / T
        z = z - z.max(1, keepdims=True)
        p = np.exp(z) / np.exp(z).sum(1, keepdims=True)
        return nll(y_true, p)

    res = minimize_scalar(_nll_T, bounds=(0.1, 10.0), method="bounded")
    return float(res.x)


def calibration_metrics(y_true, y_prob, n_bins=15):
    return {
        "ece": expected_calibration_error(y_true, y_prob, n_bins),
        "brier": brier_score(y_true, y_prob),
        "nll": nll(y_true, y_prob),
    }
