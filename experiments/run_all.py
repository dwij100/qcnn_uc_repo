from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_generation.generate_dataset import generate_uc_dataset
from src.feasibility.check_uc_feasibility import run_feasibility_check
from src.milp_acceleration.compare_speedup import run_milp_acceleration
from src.plotting.plot_results import plot_model_comparison, plot_training_curves
from src.preprocessing.prepare_dataset import prepare_dataset
from src.training.train_model import train_model
from src.utils.config_loader import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default=None)
    parser.add_argument("--models", nargs="+", default=["cnn", "henderson_quanv", "pqc_qcnn"])
    parser.add_argument("--n-scenarios", type=int, default=None)
    parser.add_argument("--skip-quantum", action="store_true", help="Run only CNN for quick smoke tests.")
    args = parser.parse_args()

    cfg = load_config()
    case_name = args.case or cfg["case"]["name"]

    generate_uc_dataset(cfg, case_name=case_name, n_scenarios=args.n_scenarios)
    prepare_dataset(cfg, case_name=case_name)

    models = ["cnn"] if args.skip_quantum else args.models
    for model_name in models:
        train_model(cfg, model_name, case_name=case_name)
        run_feasibility_check(cfg, model_name, case_name=case_name)
        run_milp_acceleration(cfg, model_name, case_name=case_name)

    plot_training_curves(cfg, case_name, models)
    plot_model_comparison(cfg, case_name, models)


if __name__ == "__main__":
    main()
