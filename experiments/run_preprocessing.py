from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.preprocessing.prepare_dataset import prepare_dataset
from src.utils.config_loader import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default=None)
    args = parser.parse_args()
    cfg = load_config()
    prepare_dataset(cfg, case_name=args.case)


if __name__ == "__main__":
    main()
