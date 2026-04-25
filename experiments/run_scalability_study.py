from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_generation.generate_dataset import generate_uc_dataset
from src.milp_acceleration.compare_speedup import run_milp_acceleration
from src.plotting.plot_scalability import plot_scalability_results
from src.preprocessing.prepare_dataset import prepare_dataset
from src.training.train_model import train_model
from src.utils.config_loader import ensure_dir, load_config, project_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="cnn", help="Default cnn for scalable assisted-MILP comparison.")
    parser.add_argument("--skip-training", action="store_true")
    args = parser.parse_args()

    cfg = load_config()
    rows = []

    for case_name in cfg["scalability"]["cases"]:
        n = int(cfg["scalability"]["n_scenarios"].get(case_name, cfg["dataset"]["n_scenarios"]))
        cfg["dataset"]["n_scenarios"] = n
        cfg["training"]["epochs"] = int(cfg["scalability"].get("train_epochs_small", cfg["training"]["epochs"]))
        cfg["milp_acceleration"]["max_scenarios"] = int(cfg["scalability"].get("max_milp_scenarios", 10))

        generate_uc_dataset(cfg, case_name=case_name, n_scenarios=n)
        prepare_dataset(cfg, case_name=case_name)

        if not args.skip_training:
            train_model(cfg, args.model, case_name=case_name)

        acc_dir = run_milp_acceleration(cfg, args.model, case_name=case_name)
        summary_path = acc_dir / "acceleration_summary.csv"
        if summary_path.exists():
            s = pd.read_csv(summary_path)
            full = s[s["mode"] == "full_milp"]
            assisted = s[s["mode"] == "partial_fix_confident"]
            rows.append(
                {
                    "case_name": case_name,
                    "n_scenarios": n,
                    "mean_full_milp_time": float(full["mean_solve_time"].iloc[0]) if len(full) else None,
                    "mean_assisted_milp_time": float(assisted["mean_solve_time"].iloc[0]) if len(assisted) else None,
                    "mean_speedup": float(assisted["mean_speedup"].iloc[0]) if len(assisted) else None,
                    "feasibility_rate": float(assisted["feasible_rate"].iloc[0]) if len(assisted) else None,
                    "mean_cost_deviation": float(assisted["mean_cost_deviation"].iloc[0]) if len(assisted) else None,
                }
            )

    out_dir = project_path(cfg, "data", "results", "scalability")
    ensure_dir(out_dir)
    pd.DataFrame(rows).to_csv(out_dir / "scalability_results.csv", index=False)
    plot_scalability_results(cfg)


if __name__ == "__main__":
    main()
