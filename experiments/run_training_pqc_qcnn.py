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
    args = parser.parse_args()
    cfg = load_config()
    case_name = args.case or cfg["case"]["name"]
    train_model(cfg, "pqc_qcnn", case_name=case_name)
    plot_training_curves(cfg, case_name, ["pqc_qcnn"])
    plot_model_comparison(cfg, case_name, ["pqc_qcnn"])


if __name__ == "__main__":
    main()
