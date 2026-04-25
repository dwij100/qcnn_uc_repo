from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

from src.utils.config_loader import ensure_dir, project_path
from src.utils.logger import get_logger
from src.utils.seed import set_seed


_LABEL_RE = re.compile(r"GOn_g(\d+)_t(\d+)")


def _sorted_label_columns(columns: List[str]) -> Tuple[List[str], int, int]:
    parsed = []
    for c in columns:
        m = _LABEL_RE.fullmatch(c)
        if m:
            parsed.append((int(m.group(1)), int(m.group(2)), c))
    if not parsed:
        raise ValueError("No commitment label columns found. Expected columns like GOn_g0_t0.")
    parsed.sort()
    n_gen = max(p[0] for p in parsed) + 1
    T = max(p[1] for p in parsed) + 1
    return [p[2] for p in parsed], n_gen, T


def load_and_align_dataset(cfg: Dict[str, Any], case_name: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data_dir = project_path(cfg, "data", "processed", case_name)
    features_path = data_dir / "features.csv"
    labels_path = data_dir / "labels_commitment.csv"
    summary_path = data_dir / "milp_summary.csv"

    if not features_path.exists() or not labels_path.exists():
        raise FileNotFoundError(f"Missing features/labels in {data_dir}. Run dataset generation first.")

    features = pd.read_csv(features_path)
    labels = pd.read_csv(labels_path)
    summary = pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame()

    if cfg["dataset"].get("only_optimal", True) and not summary.empty:
        valid_ids = summary.loc[summary["feasible"].astype(bool), "scenario_id"]
        features = features[features["scenario_id"].isin(valid_ids)]
        labels = labels[labels["scenario_id"].isin(valid_ids)]

    common = sorted(set(features["scenario_id"]).intersection(set(labels["scenario_id"])))
    features = features[features["scenario_id"].isin(common)].sort_values("scenario_id").reset_index(drop=True)
    labels = labels[labels["scenario_id"].isin(common)].sort_values("scenario_id").reset_index(drop=True)

    if len(features) == 0:
        raise ValueError("No feasible aligned scenarios found.")
    return features, labels, summary


def make_feature_matrix(features: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    drop_cols = {"scenario_id", "case_name"}
    numeric = features.drop(columns=[c for c in drop_cols if c in features.columns], errors="ignore")
    numeric = numeric.select_dtypes(include=[np.number])
    feature_cols = list(numeric.columns)
    return numeric.to_numpy(dtype=np.float32), feature_cols


def make_label_tensor(labels: pd.DataFrame) -> tuple[np.ndarray, list[str], int, int]:
    label_cols, n_gen, T = _sorted_label_columns(list(labels.columns))
    y_flat = labels[label_cols].to_numpy(dtype=np.float32)
    y = y_flat.reshape(len(labels), n_gen, T)
    return y, label_cols, n_gen, T


def prepare_dataset(cfg: Dict[str, Any], case_name: str | None = None) -> Path:
    seed = int(cfg["project"]["seed"])
    set_seed(seed)
    logger = get_logger("preprocessing", level=cfg["project"].get("log_level", "INFO"))

    case_name = case_name or cfg["case"]["name"]
    features, labels, _ = load_and_align_dataset(cfg, case_name)

    X, feature_cols = make_feature_matrix(features)
    y, label_cols, n_gen, T = make_label_tensor(labels)
    scenario_ids = features["scenario_id"].to_numpy(dtype=int)

    groups = scenario_ids
    train_ratio = float(cfg["dataset"]["train_ratio"])
    val_ratio = float(cfg["dataset"]["val_ratio"])
    test_ratio = float(cfg["dataset"]["test_ratio"])

    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.")

    gss1 = GroupShuffleSplit(n_splits=1, train_size=train_ratio, random_state=seed)
    train_idx, temp_idx = next(gss1.split(X, y, groups=groups))

    relative_val = val_ratio / (val_ratio + test_ratio)
    gss2 = GroupShuffleSplit(n_splits=1, train_size=relative_val, random_state=seed + 1)
    val_rel, test_rel = next(gss2.split(X[temp_idx], y[temp_idx], groups=groups[temp_idx]))
    val_idx = temp_idx[val_rel]
    test_idx = temp_idx[test_rel]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[train_idx]).astype(np.float32)
    X_val = scaler.transform(X[val_idx]).astype(np.float32)
    X_test = scaler.transform(X[test_idx]).astype(np.float32)

    out_dir = project_path(cfg, "data", "processed", case_name)
    ensure_dir(out_dir)

    np.savez_compressed(out_dir / "train.npz", X=X_train, y=y[train_idx], scenario_id=scenario_ids[train_idx])
    np.savez_compressed(out_dir / "val.npz", X=X_val, y=y[val_idx], scenario_id=scenario_ids[val_idx])
    np.savez_compressed(out_dir / "test.npz", X=X_test, y=y[test_idx], scenario_id=scenario_ids[test_idx])

    joblib.dump(scaler, out_dir / "scaler.joblib")

    metadata = {
        "case_name": case_name,
        "n_samples": int(len(X)),
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_test": int(len(test_idx)),
        "feature_dim": int(X.shape[1]),
        "num_generators": int(n_gen),
        "time_horizon": int(T),
        "label_shape": [int(n_gen), int(T)],
        "feature_columns": feature_cols,
        "label_columns": label_cols,
    }
    (out_dir / "preprocessing_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    logger.info("Prepared dataset for %s: %s", case_name, metadata)
    return out_dir
