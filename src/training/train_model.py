from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

from src.models.model_utils import build_model, count_parameters, save_checkpoint
from src.utils.config_loader import ensure_dir, project_path
from src.utils.logger import get_logger
from src.utils.metrics import binary_metrics, per_generator_accuracy, per_time_accuracy
from src.utils.seed import set_seed


class UCNPZDataset(Dataset):
    def __init__(self, npz_path: str | Path):
        data = np.load(npz_path)
        self.X = data["X"].astype(np.float32)
        self.y = data["y"].astype(np.float32)
        self.scenario_id = data["scenario_id"].astype(int)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return (
            torch.from_numpy(self.X[idx]),
            torch.from_numpy(self.y[idx]),
            int(self.scenario_id[idx]),
        )


def resolve_device(cfg: Dict[str, Any]) -> torch.device:
    setting = cfg["project"].get("device", "auto")
    if setting == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(setting)


def _limit_dataset(ds: Dataset, max_samples: Optional[int]) -> Dataset:
    if max_samples is None or max_samples <= 0 or max_samples >= len(ds):
        return ds
    return Subset(ds, list(range(max_samples)))


def _run_epoch(model, loader, criterion, optimizer, device, train: bool, use_amp: bool = False):
    model.train(train)
    total_loss = 0.0
    all_probs, all_y = [], []

    scaler = torch.amp.GradScaler('cuda', enabled=use_amp and train)# torch.cuda.amp.GradScaler(enabled=use_amp and train)

    for X, y, _ in loader:
        X = X.to(device)
        y = y.to(device)

        with torch.set_grad_enabled(train):
            with torch.amp.autocast('cuda',enabled=use_amp):#torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(X)
                loss = criterion(logits, y)

            if train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        total_loss += float(loss.detach().cpu()) * X.shape[0]
        all_probs.append(torch.sigmoid(logits.detach()).cpu().numpy())
        all_y.append(y.detach().cpu().numpy())

    probs = np.concatenate(all_probs, axis=0)
    y_true = np.concatenate(all_y, axis=0)
    return total_loss / max(1, len(loader.dataset)), probs, y_true


def train_model(cfg: Dict[str, Any], model_name: str, case_name: str | None = None) -> Path:
    seed = int(cfg["project"]["seed"])
    set_seed(seed)
    logger = get_logger(f"train_{model_name}", level=cfg["project"].get("log_level", "INFO"))

    case_name = case_name or cfg["case"]["name"]
    data_dir = project_path(cfg, "data", "processed", case_name)
    metadata = json.loads((data_dir / "preprocessing_metadata.json").read_text(encoding="utf-8"))

    train_ds = UCNPZDataset(data_dir / "train.npz")
    val_ds = UCNPZDataset(data_dir / "val.npz")
    test_ds = UCNPZDataset(data_dir / "test.npz")

    # Quantum models can be slow. Config allows training on smaller subsets for quick tests.
    mcfg = cfg["models"].get(model_name, {})
    train_ds = _limit_dataset(train_ds, mcfg.get("max_train_samples"))
    val_ds = _limit_dataset(val_ds, mcfg.get("max_val_samples"))

    train_cfg = cfg["training"]
    batch_size = int(train_cfg["batch_size"])
    device = resolve_device(cfg)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=int(train_cfg.get("num_workers", 0)))
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=int(train_cfg.get("num_workers", 0)))
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=int(train_cfg.get("num_workers", 0)))

    model = build_model(
        model_name=model_name,
        cfg=cfg,
        feature_dim=int(metadata["feature_dim"]),
        num_generators=int(metadata["num_generators"]),
        time_horizon=int(metadata["time_horizon"]),
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )

    out_dir = project_path(cfg, "data", "results", case_name, model_name)
    ensure_dir(out_dir)

    logger.info("Training %s on %s | params=%s | device=%s", model_name, case_name, count_parameters(model), device)

    best_val_loss = float("inf")
    best_epoch = -1
    patience = int(train_cfg.get("patience", 8))
    history = []
    start_train = time.perf_counter()

    for epoch in range(1, int(train_cfg["epochs"]) + 1):
        train_loss, train_probs, train_y = _run_epoch(
            model, train_loader, criterion, optimizer, device, train=True, use_amp=bool(train_cfg.get("use_amp", False))
        )
        val_loss, val_probs, val_y = _run_epoch(model, val_loader, criterion, optimizer, device, train=False)

        threshold = float(train_cfg.get("threshold", 0.5))
        train_metrics = binary_metrics(train_y, train_probs, threshold)
        val_metrics = binary_metrics(val_y, val_probs, threshold)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_bitwise_accuracy": train_metrics["bitwise_accuracy"],
            "val_bitwise_accuracy": val_metrics["bitwise_accuracy"],
            "train_exact_match": train_metrics["exact_schedule_match_accuracy"],
            "val_exact_match": val_metrics["exact_schedule_match_accuracy"],
            "val_f1_micro": val_metrics["f1_micro"],
        }
        history.append(row)
        pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)

        logger.info(
            "epoch=%03d train_loss=%.5f val_loss=%.5f val_acc=%.4f val_exact=%.4f",
            epoch,
            train_loss,
            val_loss,
            val_metrics["bitwise_accuracy"],
            val_metrics["exact_schedule_match_accuracy"],
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            save_checkpoint(
                out_dir / "best_model.pt",
                model,
                {
                    "model_name": model_name,
                    "case_name": case_name,
                    "feature_dim": metadata["feature_dim"],
                    "num_generators": metadata["num_generators"],
                    "time_horizon": metadata["time_horizon"],
                    "best_epoch": best_epoch,
                    "best_val_loss": best_val_loss,
                    "num_parameters": count_parameters(model),
                },
            )

        if epoch - best_epoch >= patience:
            logger.info("Early stopping at epoch %s", epoch)
            break

    training_time = time.perf_counter() - start_train

    # Load best checkpoint and evaluate on test.
    ckpt = torch.load(out_dir / "best_model.pt", map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    all_probs, all_y, all_ids = [], [], []
    pred_start = time.perf_counter()
    with torch.no_grad():
        for X, y, ids in test_loader:
            X = X.to(device)
            probs = torch.sigmoid(model(X)).cpu().numpy()
            all_probs.append(probs)
            all_y.append(y.numpy())
            all_ids.extend([int(i) for i in ids])
    prediction_time = time.perf_counter() - pred_start

    probs = np.concatenate(all_probs, axis=0)
    y_true = np.concatenate(all_y, axis=0)
    threshold = float(train_cfg.get("threshold", 0.5))

    metrics = binary_metrics(y_true, probs, threshold)
    metrics.update(
        {
            "model_name": model_name,
            "case_name": case_name,
            "training_time": training_time,
            "prediction_time_total": prediction_time,
            "prediction_time_per_sample": prediction_time / max(1, len(y_true)),
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "num_parameters": count_parameters(model),
        }
    )

    pd.DataFrame([metrics]).to_csv(out_dir / "test_metrics.csv", index=False)

    # Save probability predictions in wide format for feasibility and MILP acceleration.
    G = int(metadata["num_generators"])
    T = int(metadata["time_horizon"])
    pred_rows = []
    true_rows = []
    for i, sid in enumerate(all_ids):
        prow = {"scenario_id": sid}
        trow = {"scenario_id": sid}
        for g in range(G):
            for t in range(T):
                prow[f"prob_g{g}_t{t}"] = float(probs[i, g, t])
                trow[f"true_g{g}_t{t}"] = int(y_true[i, g, t])
        pred_rows.append(prow)
        true_rows.append(trow)

    pd.DataFrame(pred_rows).to_csv(out_dir / "predictions_test.csv", index=False)
    pd.DataFrame(true_rows).to_csv(out_dir / "targets_test.csv", index=False)

    per_g = per_generator_accuracy(y_true, probs, threshold)
    per_t = per_time_accuracy(y_true, probs, threshold)
    pd.DataFrame({"generator": np.arange(G), "accuracy": per_g}).to_csv(out_dir / "per_generator_accuracy.csv", index=False)
    pd.DataFrame({"time": np.arange(T), "accuracy": per_t}).to_csv(out_dir / "per_time_accuracy.csv", index=False)

    logger.info("Saved training outputs to %s", out_dir)
    return out_dir
