"""Stage 2 placeholder: ML/QCNN dataset preparation.

Next implementation will:
- load Stage 1 CSV/NPY outputs,
- filter feasible MILP solutions,
- split by scenario_id to avoid leakage,
- normalize features using train-only statistics,
- reshape labels to [n_scenarios, n_generators, time_horizon].
"""

from __future__ import annotations


def prepare_dataset(*args, **kwargs):
    raise NotImplementedError("Stage 2 will be implemented after Stage 1 dataset generation is validated.")
