from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.feasibility.check_uc_feasibility import run_feasibility_check
from src.utils.config_loader import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default=None)
    parser.add_argument("--model", default="cnn", help="cnn | henderson_quanv | henderson_quanv_trainable | pqc_qcnn")
    args = parser.parse_args()

    cfg = load_config()
    run_feasibility_check(cfg, args.model, case_name=args.case)


if __name__ == "__main__":
    main()
