from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable

import matplotlib.pyplot as plt
import pandas as pd

from src.utils.config_loader import ensure_dir, project_path


def plot_training_curves(cfg: Dict[str, Any], case_name: str, model_names: Iterable[str]) -> Path:
    out_dir = project_path(cfg, "data", "results", case_name, "plots")
    ensure_dir(out_dir)

    for model_name in model_names:
        hist_path = project_path(cfg, "data", "results", case_name, model_name, "history.csv")
        if not hist_path.exists():
            continue
        hist = pd.read_csv(hist_path)

        plt.figure()
        plt.plot(hist["epoch"], hist["train_loss"], label="train_loss")
        plt.plot(hist["epoch"], hist["val_loss"], label="val_loss")
        plt.xlabel("Epoch")
        plt.ylabel("BCE loss")
        plt.title(f"Loss curve: {case_name} / {model_name}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"{model_name}_loss_curve.png", dpi=200)
        plt.close()

        plt.figure()
        plt.plot(hist["epoch"], hist["train_bitwise_accuracy"], label="train_acc")
        plt.plot(hist["epoch"], hist["val_bitwise_accuracy"], label="val_acc")
        plt.xlabel("Epoch")
        plt.ylabel("Bitwise accuracy")
        plt.title(f"Accuracy curve: {case_name} / {model_name}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"{model_name}_accuracy_curve.png", dpi=200)
        plt.close()

    return out_dir


def plot_model_comparison(cfg: Dict[str, Any], case_name: str, model_names: Iterable[str]) -> Path:
    out_dir = project_path(cfg, "data", "results", case_name, "plots")
    ensure_dir(out_dir)

    rows = []
    for model_name in model_names:
        path = project_path(cfg, "data", "results", case_name, model_name, "test_metrics.csv")
        if path.exists():
            rows.append(pd.read_csv(path).iloc[0].to_dict())
    if not rows:
        return out_dir

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "model_comparison_metrics.csv", index=False)

    for metric in ["bitwise_accuracy", "exact_schedule_match_accuracy", "f1_micro", "prediction_time_per_sample"]:
        if metric not in df.columns:
            continue
        plt.figure()
        plt.bar(df["model_name"], df[metric])
        plt.xticks(rotation=25, ha="right")
        plt.ylabel(metric)
        plt.title(f"{metric}: {case_name}")
        plt.tight_layout()
        plt.savefig(out_dir / f"comparison_{metric}.png", dpi=200)
        plt.close()

    return out_dir
