from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_generation.generate_dataset import generate_uc_dataset
from src.utils.config_loader import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default=None, help="case10 | case24 | case118_reduced")
    parser.add_argument("--n-scenarios", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config()
    generate_uc_dataset(cfg, case_name=args.case, n_scenarios=args.n_scenarios)


if __name__ == "__main__":
    main()
