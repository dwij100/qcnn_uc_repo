"""CLI wrapper for Stage 1: MILP UC dataset generation."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.data_generation.generate_dataset import generate_dataset
from src.utils.config_loader import load_config, project_root_from_config
from src.utils.logging_utils import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate UC dataset using classical MILP solves.")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config YAML")
    args = parser.parse_args()

    cfg = load_config(args.config)
    project_root = project_root_from_config(args.config)
    setup_logging(project_root / cfg["paths"]["logs_dir"], cfg["project"].get("log_level", "INFO"))
    output_dir = generate_dataset(cfg, project_root=project_root)
    print(f"Dataset generated: {output_dir}")


if __name__ == "__main__":
    main()
