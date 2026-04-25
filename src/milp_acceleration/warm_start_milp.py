from __future__ import annotations

import numpy as np


def probabilities_to_commitment(probs: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    return (np.asarray(probs) >= threshold).astype(int)


def confidence_matrix(probs: np.ndarray) -> np.ndarray:
    return np.abs(np.asarray(probs, dtype=float) - 0.5) * 2.0


def full_warm_start_from_probs(probs: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    return probabilities_to_commitment(probs, threshold)
