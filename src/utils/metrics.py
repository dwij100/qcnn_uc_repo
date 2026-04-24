"""Metric helpers for later ML stages."""

from __future__ import annotations

import numpy as np


def binary_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Overall binary accuracy for commitment matrices."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: {y_true.shape} vs {y_pred.shape}")
    return float((y_true == y_pred).mean())


def exact_schedule_match(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of scenarios where the full GxT binary schedule matches exactly."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: {y_true.shape} vs {y_pred.shape}")
    if y_true.ndim < 3:
        raise ValueError("Expected arrays with shape [n_scenarios, n_generators, time_horizon]")
    return float(np.all(y_true == y_pred, axis=(1, 2)).mean())
