from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.milp_acceleration.compare_speedup import run_milp_acceleration
from src.utils.config_loader import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default=None)
    parser.add_argument("--model", default="cnn")
    args = parser.parse_args()

    cfg = load_config()
    run_milp_acceleration(cfg, args.model, case_name=args.case)


if __name__ == "__main__":
    main()
