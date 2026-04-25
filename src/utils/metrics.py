from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import f1_score


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def binary_metrics(y_true: np.ndarray, probs: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """Compute publication-friendly multilabel UC commitment metrics.

    y_true and probs can be shaped [N, G, T] or [N, G*T].
    """
    y_true = np.asarray(y_true).astype(int)
    probs = np.asarray(probs)
    y_pred = (probs >= threshold).astype(int)

    yt_flat = y_true.reshape(y_true.shape[0], -1)
    yp_flat = y_pred.reshape(y_pred.shape[0], -1)

    bitwise_acc = float((yt_flat == yp_flat).mean())
    exact_match = float(np.all(yt_flat == yp_flat, axis=1).mean())
    f1_micro = float(f1_score(yt_flat.ravel(), yp_flat.ravel(), average="micro", zero_division=0))
    f1_macro = float(f1_score(yt_flat.ravel(), yp_flat.ravel(), average="macro", zero_division=0))

    return {
        "bitwise_accuracy": bitwise_acc,
        "exact_schedule_match_accuracy": exact_match,
        "f1_micro": f1_micro,
        "f1_macro": f1_macro,
    }


def per_generator_accuracy(y_true: np.ndarray, probs: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    y_pred = (probs >= threshold).astype(int)
    return (y_true == y_pred).mean(axis=(0, 2))


def per_time_accuracy(y_true: np.ndarray, probs: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    y_pred = (probs >= threshold).astype(int)
    return (y_true == y_pred).mean(axis=(0, 1))


def regression_safe_div(num: float, den: float) -> float:
    if den is None or abs(float(den)) < 1e-12:
        return float("nan")
    return float(num) / float(den)
