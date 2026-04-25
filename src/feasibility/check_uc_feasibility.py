from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from src.data_generation.case_factory import create_uc_case
from src.data_generation.generate_scenarios import extract_net_demand
from src.utils.config_loader import ensure_dir, project_path
from src.utils.logger import get_logger


_PROB_RE = re.compile(r"prob_g(\d+)_t(\d+)")


def load_prediction_tensor(pred_path: str | Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    df = pd.read_csv(pred_path)
    parsed = []
    for c in df.columns:
        m = _PROB_RE.fullmatch(c)
        if m:
            parsed.append((int(m.group(1)), int(m.group(2)), c))
    if not parsed:
        raise ValueError(f"No probability columns found in {pred_path}")
    parsed.sort()
    G = max(p[0] for p in parsed) + 1
    T = max(p[1] for p in parsed) + 1
    cols = [p[2] for p in parsed]
    probs = df[cols].to_numpy(dtype=float).reshape(len(df), G, T)
    ids = df["scenario_id"].to_numpy(dtype=int)
    return ids, probs, cols


def greedy_dispatch_for_schedule(case, demand: np.ndarray, u: np.ndarray) -> tuple[bool, np.ndarray, list[str]]:
    """Create a simple economic dispatch for a fixed commitment schedule.

    This is a feasibility approximation for screening predictions, not a
    replacement for MILP feasibility repair.
    """
    gens = case.generators.reset_index(drop=True)
    G, T = u.shape
    p = np.zeros((G, T), dtype=float)
    violations = []

    merit = np.argsort(gens["marginal_cost"].to_numpy())
    p_min = gens["p_min"].to_numpy()
    p_max = gens["p_max"].to_numpy()

    for t in range(T):
        on = u[:, t].astype(bool)
        min_total = float(np.sum(p_min[on]))
        max_total = float(np.sum(p_max[on]))
        if max_total + 1e-6 < demand[t]:
            violations.append("demand_capacity")
            continue
        if max_total + 1e-6 < demand[t] * (1.0 + case.reserve_margin):
            violations.append("reserve")

        # If committed minimum exceeds demand, this is infeasible in the exact UC.
        if min_total - 1e-6 > demand[t]:
            violations.append("minimum_generation_exceeds_demand")
            # Keep p at minimum so ramp checks still have useful values.
            p[on, t] = p_min[on]
            continue

        p[on, t] = p_min[on]
        remaining = demand[t] - min_total
        for g in merit:
            if not on[g] or remaining <= 1e-9:
                continue
            add = min(remaining, p_max[g] - p_min[g])
            p[g, t] += add
            remaining -= add

        if remaining > 1e-5:
            violations.append("dispatch_shortfall")

    return len(violations) == 0, p, violations


def check_schedule_feasibility(case, demand: np.ndarray, u: np.ndarray) -> Dict[str, Any]:
    gens = case.generators.reset_index(drop=True)
    G, T = u.shape
    violations: list[str] = []

    p_min = gens["p_min"].to_numpy()
    p_max = gens["p_max"].to_numpy()
    ramp_up = gens["ramp_up"].to_numpy()
    ramp_down = gens["ramp_down"].to_numpy()
    min_up = gens["min_up_time"].to_numpy(dtype=int)
    min_down = gens["min_down_time"].to_numpy(dtype=int)
    init_status = gens["initial_status"].to_numpy(dtype=int)
    init_dispatch = gens["initial_dispatch"].to_numpy(dtype=float)

    committed_cap = (u * p_max[:, None]).sum(axis=0)
    committed_min = (u * p_min[:, None]).sum(axis=0)

    demand_coverage_ok = bool(np.all(committed_cap + 1e-6 >= demand))
    reserve_ok = bool(np.all(committed_cap + 1e-6 >= demand * (1.0 + case.reserve_margin)))
    min_gen_ok = bool(np.all(committed_min <= demand + 1e-6))

    if not demand_coverage_ok:
        violations.append("demand_capacity")
    if not reserve_ok:
        violations.append("reserve")
    if not min_gen_ok:
        violations.append("minimum_generation_exceeds_demand")

    dispatch_ok, p, dispatch_viol = greedy_dispatch_for_schedule(case, demand, u)
    violations.extend(dispatch_viol)

    ramp_ok = True
    for g in range(G):
        for t in range(T):
            prev_p = init_dispatch[g] if t == 0 else p[g, t - 1]
            if p[g, t] - prev_p > ramp_up[g] + p_max[g] * max(0, u[g, t] - (init_status[g] if t == 0 else u[g, t - 1])) + 1e-5:
                ramp_ok = False
            if prev_p - p[g, t] > ramp_down[g] + p_max[g] * max(0, (init_status[g] if t == 0 else u[g, t - 1]) - u[g, t]) + 1e-5:
                ramp_ok = False
    if not ramp_ok:
        violations.append("ramp")

    min_up_ok = True
    min_down_ok = True
    for g in range(G):
        prev = init_status[g]
        for t in range(T):
            startup = int(u[g, t] == 1 and prev == 0)
            shutdown = int(u[g, t] == 0 and prev == 1)
            if startup:
                end = min(T, t + min_up[g])
                if np.any(u[g, t:end] == 0):
                    min_up_ok = False
            if shutdown:
                end = min(T, t + min_down[g])
                if np.any(u[g, t:end] == 1):
                    min_down_ok = False
            prev = u[g, t]

    if not min_up_ok:
        violations.append("min_up")
    if not min_down_ok:
        violations.append("min_down")

    fully_feasible = demand_coverage_ok and reserve_ok and min_gen_ok and ramp_ok and min_up_ok and min_down_ok and dispatch_ok
    capacity_ratio = float(np.min(committed_cap / np.maximum(demand * (1.0 + case.reserve_margin), 1e-9)))

    return {
        "fully_feasible": bool(fully_feasible),
        "partially_feasible": bool(capacity_ratio >= 0.95),
        "demand_coverage_ok": demand_coverage_ok,
        "reserve_ok": reserve_ok,
        "min_generation_ok": min_gen_ok,
        "ramp_ok": bool(ramp_ok),
        "min_up_ok": bool(min_up_ok),
        "min_down_ok": bool(min_down_ok),
        "capacity_ratio_min": capacity_ratio,
        "violations": sorted(set(violations)),
    }


def run_feasibility_check(cfg: Dict[str, Any], model_name: str, case_name: str | None = None) -> Path:
    logger = get_logger("feasibility", level=cfg["project"].get("log_level", "INFO"))
    case_name = case_name or cfg["case"]["name"]
    T = int(cfg["case"]["time_horizon"])
    case = create_uc_case(case_name, time_horizon=T, reserve_margin=float(cfg["case"]["reserve_margin"]), seed=int(cfg["project"]["seed"]))

    pred_path = project_path(cfg, "data", "results", case.name, model_name, "predictions_test.csv")
    if not pred_path.exists():
        raise FileNotFoundError(f"Missing predictions: {pred_path}. Train model first.")

    ids, probs, _ = load_prediction_tensor(pred_path)
    threshold = float(cfg["feasibility"].get("threshold", cfg["training"].get("threshold", 0.5)))
    schedules = (probs >= threshold).astype(int)

    features = pd.read_csv(project_path(cfg, "data", "processed", case.name, "features.csv")).set_index("scenario_id")

    rows = []
    violation_rows = []
    for i, sid in enumerate(ids):
        demand = extract_net_demand(features.loc[int(sid)], T)
        result = check_schedule_feasibility(case, demand, schedules[i])
        row = {"scenario_id": int(sid), "model_name": model_name, **{k: v for k, v in result.items() if k != "violations"}}
        row["violation_types"] = ";".join(result["violations"])
        rows.append(row)
        for v in result["violations"]:
            violation_rows.append({"scenario_id": int(sid), "model_name": model_name, "violation_type": v})

    out_dir = project_path(cfg, "data", "results", case.name, model_name, "feasibility")
    ensure_dir(out_dir)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "feasibility_by_scenario.csv", index=False)
    df[df["fully_feasible"]].to_csv(out_dir / "feasible_predictions.csv", index=False)
    df[~df["fully_feasible"]].to_csv(out_dir / "infeasible_predictions.csv", index=False)

    if violation_rows:
        vdf = pd.DataFrame(violation_rows)
        breakdown = vdf.groupby("violation_type").size().reset_index(name="count").sort_values("count", ascending=False)
    else:
        breakdown = pd.DataFrame(columns=["violation_type", "count"])
    breakdown.to_csv(out_dir / "violation_breakdown.csv", index=False)

    summary = {
        "case_name": case.name,
        "model_name": model_name,
        "n_scenarios": int(len(df)),
        "feasibility_rate": float(df["fully_feasible"].mean()) if len(df) else 0.0,
        "partial_feasibility_rate": float(df["partially_feasible"].mean()) if len(df) else 0.0,
        "threshold": threshold,
    }
    pd.DataFrame([summary]).to_csv(out_dir / "feasibility_summary.csv", index=False)
    logger.info("Feasibility summary: %s", summary)
    return out_dir
