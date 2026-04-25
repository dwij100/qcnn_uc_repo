from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.data_generation.case_factory import create_uc_case
from src.data_generation.generate_scenarios import extract_net_demand
from src.data_generation.uc_milp_model import solve_uc_milp
from src.feasibility.check_uc_feasibility import check_schedule_feasibility, load_prediction_tensor
from src.milp_acceleration.partial_fixing_milp import fixed_binary_stats, make_partial_fix_matrix
from src.milp_acceleration.warm_start_milp import probabilities_to_commitment
from src.utils.config_loader import ensure_dir, project_path
from src.utils.logger import get_logger
from src.utils.metrics import regression_safe_div


def _cost_deviation(candidate: float, baseline: float) -> float:
    if baseline is None or not np.isfinite(baseline) or abs(baseline) < 1e-9:
        return float("nan")
    return float((candidate - baseline) / baseline)


def run_milp_acceleration(cfg: Dict[str, Any], model_name: str, case_name: str | None = None) -> Path:
    logger = get_logger("milp_acceleration", level=cfg["project"].get("log_level", "INFO"))
    case_name = case_name or cfg["case"]["name"]
    T = int(cfg["case"]["time_horizon"])
    case = create_uc_case(case_name, time_horizon=T, reserve_margin=float(cfg["case"]["reserve_margin"]), seed=int(cfg["project"]["seed"]))

    pred_path = project_path(cfg, "data", "results", case.name, model_name, "predictions_test.csv")
    ids, probs, _ = load_prediction_tensor(pred_path)

    max_scenarios = int(cfg["milp_acceleration"].get("max_scenarios", len(ids)))
    ids = ids[:max_scenarios]
    probs = probs[:max_scenarios]

    features = pd.read_csv(project_path(cfg, "data", "processed", case.name, "features.csv")).set_index("scenario_id")
    solver_cfg = cfg["solver"]
    threshold = float(cfg["training"].get("threshold", 0.5))
    conf_threshold = float(cfg["milp_acceleration"].get("confidence_threshold", 0.85))

    rows = []

    for i, sid in enumerate(tqdm(ids, desc=f"MILP acceleration {case.name}/{model_name}")):
        demand = extract_net_demand(features.loc[int(sid)], T)
        pred_u = probabilities_to_commitment(probs[i], threshold)
        pred_feas = check_schedule_feasibility(case, demand, pred_u)["fully_feasible"]

        full_sol = solve_uc_milp(
            case,
            demand,
            solver_name=solver_cfg["name"],
            tee=bool(solver_cfg.get("tee", False)),
            time_limit=float(solver_cfg.get("time_limit", 120)),
            mip_gap=float(solver_cfg.get("mip_gap", 0.001)),
            threads=int(solver_cfg.get("threads", 0)),
        )
        baseline_time = float(full_sol["solve_time"])
        baseline_obj = float(full_sol["objective"]) if full_sol["feasible"] else np.nan

        def record(mode: str, sol: Dict[str, Any], fixed_count: int = 0, fixed_pct: float = 0.0):
            rows.append(
                {
                    "scenario_id": int(sid),
                    "case_name": case.name,
                    "model_name": model_name,
                    "mode": mode,
                    "solve_time": sol["solve_time"],
                    "objective": sol["objective"],
                    "gap": sol["gap"],
                    "solver_status": sol["status"],
                    "termination_condition": sol["termination_condition"],
                    "feasible": bool(sol["feasible"]),
                    "predicted_schedule_feasible": bool(pred_feas),
                    "num_binaries_fixed": int(fixed_count),
                    "pct_binaries_fixed": float(fixed_pct),
                    "baseline_full_milp_time": baseline_time,
                    "baseline_full_milp_objective": baseline_obj,
                    "speedup_vs_full_milp": regression_safe_div(baseline_time, float(sol["solve_time"])),
                    "cost_deviation_vs_full_milp": _cost_deviation(float(sol["objective"]), baseline_obj) if sol["feasible"] else np.nan,
                }
            )

        record("full_milp", full_sol, fixed_count=0, fixed_pct=0.0)

        warm_sol = solve_uc_milp(
            case,
            demand,
            solver_name=solver_cfg["name"],
            tee=bool(solver_cfg.get("tee", False)),
            time_limit=float(solver_cfg.get("time_limit", 120)),
            mip_gap=float(solver_cfg.get("mip_gap", 0.001)),
            threads=int(solver_cfg.get("threads", 0)),
            warm_start_commitment=pred_u,
        )
        record("warm_start", warm_sol, fixed_count=0, fixed_pct=0.0)

        if pred_feas or not bool(cfg["milp_acceleration"].get("full_fix_only_if_feasible", True)):
            fixed_full = pred_u.astype(float)
            fix_sol = solve_uc_milp(
                case,
                demand,
                solver_name=solver_cfg["name"],
                tee=bool(solver_cfg.get("tee", False)),
                time_limit=float(solver_cfg.get("time_limit", 120)),
                mip_gap=float(solver_cfg.get("mip_gap", 0.001)),
                threads=int(solver_cfg.get("threads", 0)),
                fixed_commitment=fixed_full,
            )
            record("full_fix_feasible", fix_sol, fixed_count=pred_u.size, fixed_pct=1.0)
        else:
            rows.append(
                {
                    "scenario_id": int(sid),
                    "case_name": case.name,
                    "model_name": model_name,
                    "mode": "full_fix_feasible",
                    "solve_time": np.nan,
                    "objective": np.nan,
                    "gap": np.nan,
                    "solver_status": "skipped_predicted_schedule_infeasible",
                    "termination_condition": "skipped",
                    "feasible": False,
                    "predicted_schedule_feasible": bool(pred_feas),
                    "num_binaries_fixed": pred_u.size,
                    "pct_binaries_fixed": 1.0,
                    "baseline_full_milp_time": baseline_time,
                    "baseline_full_milp_objective": baseline_obj,
                    "speedup_vs_full_milp": np.nan,
                    "cost_deviation_vs_full_milp": np.nan,
                }
            )

        fixed_partial = make_partial_fix_matrix(probs[i], threshold=threshold, confidence_threshold=conf_threshold)
        fixed_count, fixed_pct = fixed_binary_stats(fixed_partial)
        partial_sol = solve_uc_milp(
            case,
            demand,
            solver_name=solver_cfg["name"],
            tee=bool(solver_cfg.get("tee", False)),
            time_limit=float(solver_cfg.get("time_limit", 120)),
            mip_gap=float(solver_cfg.get("mip_gap", 0.001)),
            threads=int(solver_cfg.get("threads", 0)),
            fixed_commitment=fixed_partial,
            warm_start_commitment=pred_u,
        )
        record("partial_fix_confident", partial_sol, fixed_count=fixed_count, fixed_pct=fixed_pct)

    out_dir = project_path(cfg, "data", "results", case.name, model_name, "milp_acceleration")
    ensure_dir(out_dir)
    result_df = pd.DataFrame(rows)
    result_df.to_csv(out_dir / "acceleration_results.csv", index=False)

    summary = (
        result_df.groupby("mode")
        .agg(
            n=("scenario_id", "count"),
            feasible_rate=("feasible", "mean"),
            mean_solve_time=("solve_time", "mean"),
            median_solve_time=("solve_time", "median"),
            mean_speedup=("speedup_vs_full_milp", "mean"),
            mean_cost_deviation=("cost_deviation_vs_full_milp", "mean"),
            mean_pct_binaries_fixed=("pct_binaries_fixed", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(out_dir / "acceleration_summary.csv", index=False)
    logger.info("Saved MILP acceleration results to %s", out_dir)
    return out_dir
