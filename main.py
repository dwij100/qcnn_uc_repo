"""Main entry point for the QCNN-UC research pipeline.

Current working command:
    python main.py generate-data --config config/config.yaml

Later stages will be added as separate commands while preserving the same
configuration-driven interface.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.data_generation.generate_dataset import generate_dataset
from src.utils.config_loader import load_config, project_root_from_config
from src.utils.logging_utils import setup_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="QCNN-UC research pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_generate = subparsers.add_parser("generate-data", help="Solve UC MILPs and generate labelled dataset")
    p_generate.add_argument("--config", default="config/config.yaml", help="Path to YAML config")

    # Commands reserved for the next implementation stages.
    for name, help_text in [
        ("preprocess", "Prepare train/validation/test datasets"),
        ("train", "Train CNN/QCNN models"),
        ("evaluate", "Evaluate trained models"),
        ("check-feasibility", "Check predicted UC feasibility"),
        ("accelerate-milp", "Run warm-start/partial-fixing MILP experiments"),
        ("scalability", "Run scalability study"),
    ]:
        p = subparsers.add_parser(name, help=help_text)
        p.add_argument("--config", default="config/config.yaml", help="Path to YAML config")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    cfg = load_config(args.config)
    project_root = project_root_from_config(args.config)
    setup_logging(project_root / cfg["paths"]["logs_dir"], cfg["project"].get("log_level", "INFO"))

    if args.command == "generate-data":
        output_dir = generate_dataset(cfg, project_root=project_root)
        print(f"Dataset generated: {output_dir}")
        return

    raise NotImplementedError(
        f"Command '{args.command}' is scaffolded but not implemented yet. "
        "Stage 1 currently implements generate-data."
    )


if __name__ == "__main__":
    main()
