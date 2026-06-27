"""Certificate-gated abstention — C5 (PAPER_OUTLINE.md §5.7, P7).

Selective prediction: rank test predictions by confidence (max softmax prob), abstain on
the least-confident fraction, and trace the accuracy/coverage curve. The SRC ties in as a
per-model gate — a model that FAILS its SRC validity gate, or whose SRC is high, should
require more abstention to reach a target accuracy.

Returns coverage/accuracy points + the area under the accuracy-coverage curve (AURC-style).
"""
from __future__ import annotations

import numpy as np


def accuracy_coverage_curve(y_true, y_prob, n_points=20):
    """Accuracy at each coverage level when abstaining on the least-confident predictions.

    y_prob: (N, C) softmax. Returns dict(coverage:[...], accuracy:[...], aurc: float).
    """
    y_true = np.asarray(y_true)
    conf = y_prob.max(axis=1)
    pred = y_prob.argmax(axis=1)
    correct = (pred == y_true).astype("float32")

    order = np.argsort(conf)[::-1]              # most confident first
    correct_sorted = correct[order]

    coverages = np.linspace(1.0 / n_points, 1.0, n_points)
    cov_out, acc_out = [], []
    N = len(y_true)
    for cov in coverages:
        k = max(1, int(round(cov * N)))
        acc_out.append(float(correct_sorted[:k].mean()))
        cov_out.append(float(k / N))
    # AURC: lower = better (risk under coverage); here use 1 - mean(accuracy) as a simple proxy
    aurc = float(1.0 - np.trapz(acc_out, cov_out) / (cov_out[-1] - cov_out[0] + 1e-8))
    return {"coverage": cov_out, "accuracy": acc_out, "aurc": aurc}


def coverage_at_target_accuracy(curve, target_acc=0.95):
    """Max coverage achievable while keeping accuracy >= target (or 0 if never reached)."""
    cov, acc = curve["coverage"], curve["accuracy"]
    feasible = [c for c, a in zip(cov, acc) if a >= target_acc]
    return float(max(feasible)) if feasible else 0.0
