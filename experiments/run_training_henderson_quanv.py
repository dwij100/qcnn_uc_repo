from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.training.train_model import train_model
from src.plotting.plot_results import plot_model_comparison, plot_training_curves
from src.utils.config_loader import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default=None)
    parser.add_argument("--trainable", action="store_true", help="Use trainable quantum filters instead of fixed random filters.")
    args = parser.parse_args()
    cfg = load_config()
    case_name = args.case or cfg["case"]["name"]
    model_name = "henderson_quanv_trainable" if args.trainable else "henderson_quanv"
    train_model(cfg, model_name, case_name=case_name)
    plot_training_curves(cfg, case_name, [model_name])
    plot_model_comparison(cfg, case_name, [model_name])


if __name__ == "__main__":
    main()
