from __future__ import annotations

import numpy as np

from src.milp_acceleration.warm_start_milp import confidence_matrix, probabilities_to_commitment


def make_partial_fix_matrix(probs: np.ndarray, threshold: float = 0.5, confidence_threshold: float = 0.85) -> np.ndarray:
    """Return [G,T] matrix with 0/1 where confident, nan where free."""
    binaries = probabilities_to_commitment(probs, threshold)
    conf = confidence_matrix(probs)
    fixed = np.full_like(probs, fill_value=np.nan, dtype=float)
    fixed[conf >= confidence_threshold] = binaries[conf >= confidence_threshold]
    return fixed


def fixed_binary_stats(fixed_matrix: np.ndarray) -> tuple[int, float]:
    total = fixed_matrix.size
    fixed_count = int(np.sum(~np.isnan(fixed_matrix)))
    return fixed_count, fixed_count / max(1, total)
