from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.data_generation.case_factory import create_uc_case
from src.data_generation.generate_scenarios import extract_net_demand, generate_scenarios
from src.data_generation.uc_milp_model import commitment_to_row, dispatch_to_row, solve_uc_milp
from src.utils.config_loader import ensure_dir, project_path
from src.utils.logger import get_logger
from src.utils.seed import set_seed


def generate_uc_dataset(cfg: Dict[str, Any], case_name: str | None = None, n_scenarios: int | None = None) -> Path:
    """Generate MILP-labelled UC data and save CSV files.

    Outputs:
    - features.csv
    - labels_commitment.csv
    - dispatch.csv
    - milp_summary.csv
    """
    seed = int(cfg["project"]["seed"])
    set_seed(seed)
    logger = get_logger("dataset_generation", level=cfg["project"].get("log_level", "INFO"))

    case_name = case_name or cfg["case"]["name"]
    T = int(cfg["case"]["time_horizon"])
    reserve_margin = float(cfg["case"]["reserve_margin"])
    n_scenarios = int(n_scenarios or cfg["dataset"]["n_scenarios"])

    case = create_uc_case(case_name=case_name, time_horizon=T, reserve_margin=reserve_margin, seed=seed)

    out_dir = project_path(cfg, "data", "processed", case.name)
    ensure_dir(out_dir)

    case.generators.to_csv(out_dir / "generators.csv", index=False)
    if case.buses is not None:
        case.buses.to_csv(out_dir / "buses_reduced.csv", index=False)
    if case.lines is not None:
        case.lines.to_csv(out_dir / "lines_reduced.csv", index=False)

    features_df = generate_scenarios(
        case=case,
        n_scenarios=n_scenarios,
        seed=seed + 7,
        demand_noise_std=float(cfg["dataset"]["demand_noise_std"]),
        demand_scale_low=float(cfg["dataset"]["demand_scale_low"]),
        demand_scale_high=float(cfg["dataset"]["demand_scale_high"]),
        include_renewables=bool(cfg["case"]["include_renewables"]),
        renewable_penetration=float(cfg["case"]["renewable_penetration"]),
    )

    label_rows = []
    dispatch_rows = []
    summary_rows = []

    solver_cfg = cfg["solver"]

    logger.info("Generating %s scenarios for %s with %s generators and %s hours", n_scenarios, case.name, case.num_generators, T)

    for _, row in tqdm(features_df.iterrows(), total=len(features_df), desc=f"Solve {case.name}"):
        sid = int(row["scenario_id"])
        demand = extract_net_demand(row, T)

        sol = solve_uc_milp(
            case=case,
            demand=demand,
            solver_name=str(solver_cfg["name"]),
            tee=bool(solver_cfg.get("tee", False)),
            time_limit=float(solver_cfg.get("time_limit", 120)),
            mip_gap=float(solver_cfg.get("mip_gap", 0.001)),
            threads=int(solver_cfg.get("threads", 0)),
        )

        summary_rows.append(
            {
                "scenario_id": sid,
                "case_name": case.name,
                "num_generators": case.num_generators,
                "time_horizon": T,
                "objective": sol["objective"],
                "solve_time": sol["solve_time"],
                "gap": sol["gap"],
                "solver_status": sol["status"],
                "termination_condition": sol["termination_condition"],
                "feasible": bool(sol["feasible"]),
                "error": sol.get("error", ""),
            }
        )

        if sol["feasible"] and sol["commitment"] is not None:
            label_rows.append(commitment_to_row(sid, sol["commitment"]))
            dispatch_rows.append(dispatch_to_row(sid, sol["dispatch"]))

        save_every = int(cfg["dataset"].get("save_every", 25))
        if save_every > 0 and (sid + 1) % save_every == 0:
            _write_dataset(out_dir, features_df, label_rows, dispatch_rows, summary_rows)

    _write_dataset(out_dir, features_df, label_rows, dispatch_rows, summary_rows)
    logger.info("Saved dataset to %s", out_dir)
    return out_dir


def _write_dataset(out_dir: Path, features_df: pd.DataFrame, label_rows: list, dispatch_rows: list, summary_rows: list) -> None:
    features_df.to_csv(out_dir / "features.csv", index=False)
    pd.DataFrame(label_rows).to_csv(out_dir / "labels_commitment.csv", index=False)
    pd.DataFrame(dispatch_rows).to_csv(out_dir / "dispatch.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(out_dir / "milp_summary.csv", index=False)
