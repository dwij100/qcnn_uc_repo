from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.models.model_utils import build_model
from src.training.train_model import UCNPZDataset, resolve_device
from src.utils.config_loader import project_path
from src.utils.metrics import binary_metrics, per_generator_accuracy, per_time_accuracy


def evaluate_saved_model(cfg: Dict[str, Any], model_name: str, case_name: str | None = None) -> Path:
    case_name = case_name or cfg["case"]["name"]
    data_dir = project_path(cfg, "data", "processed", case_name)
    out_dir = project_path(cfg, "data", "results", case_name, model_name)
    metadata = json.loads((data_dir / "preprocessing_metadata.json").read_text(encoding="utf-8"))

    device = resolve_device(cfg)
    model = build_model(
        model_name,
        cfg,
        feature_dim=int(metadata["feature_dim"]),
        num_generators=int(metadata["num_generators"]),
        time_horizon=int(metadata["time_horizon"]),
    ).to(device)

    ckpt = torch.load(out_dir / "best_model.pt", map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    ds = UCNPZDataset(data_dir / "test.npz")
    loader = DataLoader(ds, batch_size=int(cfg["training"]["batch_size"]), shuffle=False)

    all_probs, all_y = [], []
    with torch.no_grad():
        for X, y, _ in loader:
            probs = torch.sigmoid(model(X.to(device))).cpu().numpy()
            all_probs.append(probs)
            all_y.append(y.numpy())

    probs = np.concatenate(all_probs, axis=0)
    y_true = np.concatenate(all_y, axis=0)
    threshold = float(cfg["training"].get("threshold", 0.5))
    metrics = binary_metrics(y_true, probs, threshold)
    pd.DataFrame([metrics]).to_csv(out_dir / "evaluation_metrics.csv", index=False)

    G, T = y_true.shape[1], y_true.shape[2]
    pd.DataFrame({"generator": np.arange(G), "accuracy": per_generator_accuracy(y_true, probs, threshold)}).to_csv(
        out_dir / "per_generator_accuracy_eval.csv", index=False
    )
    pd.DataFrame({"time": np.arange(T), "accuracy": per_time_accuracy(y_true, probs, threshold)}).to_csv(
        out_dir / "per_time_accuracy_eval.csv", index=False
    )
    return out_dir
