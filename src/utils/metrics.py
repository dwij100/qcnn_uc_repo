from __future__ import annotations

from typing import Dict

import numpy as np
# from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def binary_metrics(y_true: np.ndarray, probs: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """
    Compute UC binary commitment prediction metrics.

    Parameters
    ----------
    y_true:
        Ground-truth commitment labels.
        Shape: [samples, num_generators, time_horizon]

    y_prob:
        Predicted probabilities from the model.
        Shape: [samples, num_generators, time_horizon]

    threshold:
        Probability threshold used to convert probabilities to binary predictions.

    Returns
    -------
    dict
        Dictionary containing accuracy, precision, recall, and F1 metrics.
    """

    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    if y_true.shape != y_prob.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape}, y_prob {y_prob.shape}")

    y_pred = (y_prob >= threshold).astype(int)

    y_true_flat = y_true.reshape(-1)
    y_pred_flat = y_pred.reshape(-1)

    return {
        "threshold": threshold,

        # Overall bitwise accuracy
        "bitwise_accuracy": accuracy_score(y_true_flat, y_pred_flat),

        # Metrics for predicting generator ON
        "precision_on": precision_score(
            y_true_flat, y_pred_flat, pos_label=1, zero_division=0
        ),
        "recall_on": recall_score(
            y_true_flat, y_pred_flat, pos_label=1, zero_division=0
        ),
        "f1_on": f1_score(
            y_true_flat, y_pred_flat, pos_label=1, zero_division=0
        ),

        # Metrics for predicting generator OFF
        "precision_off": precision_score(
            y_true_flat, y_pred_flat, pos_label=0, zero_division=0
        ),
        "recall_off": recall_score(
            y_true_flat, y_pred_flat, pos_label=0, zero_division=0
        ),
        "f1_off": f1_score(
            y_true_flat, y_pred_flat, pos_label=0, zero_division=0
        ),

        # Balanced single-number versions
        "macro_precision": precision_score(
            y_true_flat, y_pred_flat, average="macro", zero_division=0
        ),
        "macro_recall": recall_score(
            y_true_flat, y_pred_flat, average="macro", zero_division=0
        ),
        "macro_f1": f1_score(
            y_true_flat, y_pred_flat, average="macro", zero_division=0
        ),
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
