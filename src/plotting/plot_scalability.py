from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import pandas as pd

from src.utils.config_loader import ensure_dir, project_path


def plot_scalability_results(cfg: Dict[str, Any]) -> Path:
    out_dir = project_path(cfg, "data", "results", "scalability")
    ensure_dir(out_dir)
    path = out_dir / "scalability_results.csv"
    if not path.exists():
        return out_dir

    df = pd.read_csv(path)
    for metric in ["mean_full_milp_time", "mean_assisted_milp_time", "mean_speedup", "feasibility_rate"]:
        if metric not in df.columns:
            continue
        plt.figure()
        plt.plot(df["case_name"], df[metric], marker="o")
        plt.xlabel("Case")
        plt.ylabel(metric)
        plt.title(f"Scalability trend: {metric}")
        plt.tight_layout()
        plt.savefig(out_dir / f"{metric}.png", dpi=200)
        plt.close()
    return out_dir
