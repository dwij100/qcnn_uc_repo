from __future__ import annotations

import time
from typing import Any, Dict, Optional

import numpy as np
import pyomo.environ as pyo
from pyomo.opt import SolverStatus, TerminationCondition

from src.data_generation.case_factory import UCCase


def _solver_gap(results: Any) -> float:
    """Best-effort MIP gap extraction across solver plugins."""
    try:
        ub = float(results.problem.upper_bound)
        lb = float(results.problem.lower_bound)
        if abs(ub) > 1e-9 and np.isfinite(ub) and np.isfinite(lb):
            return abs(ub - lb) / (abs(ub) + 1e-9)
    except Exception:
        pass
    try:
        return float(results.solver.mip_gap)
    except Exception:
        return float("nan")


def build_uc_model(
    case: UCCase,
    demand: np.ndarray,
    warm_start_commitment: Optional[np.ndarray] = None,
    fixed_commitment: Optional[np.ndarray] = None,
) -> pyo.ConcreteModel:
    """Build a single-bus UC MILP.

    fixed_commitment may be a [G, T] float array where nan means free, and 0/1
    means fixed. This supports confidence-based partial fixing.
    """
    gens = case.generators.reset_index(drop=True)
    G = range(case.num_generators)
    T = range(case.time_horizon)

    m = pyo.ConcreteModel(name=f"UC_{case.name}")
    m.G = pyo.Set(initialize=list(G))
    m.T = pyo.Set(initialize=list(T))

    p_min = gens["p_min"].to_dict()
    p_max = gens["p_max"].to_dict()
    ramp_up = gens["ramp_up"].to_dict()
    ramp_down = gens["ramp_down"].to_dict()
    startup_cost = gens["startup_cost"].to_dict()
    shutdown_cost = gens["shutdown_cost"].to_dict()
    no_load_cost = gens["no_load_cost"].to_dict()
    marginal_cost = gens["marginal_cost"].to_dict()
    min_up = gens["min_up_time"].astype(int).to_dict()
    min_down = gens["min_down_time"].astype(int).to_dict()
    initial_status = gens["initial_status"].astype(int).to_dict()
    initial_dispatch = gens["initial_dispatch"].to_dict()

    m.u = pyo.Var(m.G, m.T, within=pyo.Binary)
    m.p = pyo.Var(m.G, m.T, within=pyo.NonNegativeReals)
    m.startup = pyo.Var(m.G, m.T, within=pyo.Binary)
    m.shutdown = pyo.Var(m.G, m.T, within=pyo.Binary)

    def objective_rule(mm: pyo.ConcreteModel) -> pyo.Expression:
        return sum(
            marginal_cost[g] * mm.p[g, t]
            + no_load_cost[g] * mm.u[g, t]
            + startup_cost[g] * mm.startup[g, t]
            + shutdown_cost[g] * mm.shutdown[g, t]
            for g in mm.G
            for t in mm.T
        )

    m.obj = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

    def demand_balance_rule(mm, t):
        return sum(mm.p[g, t] for g in mm.G) == float(demand[t])

    m.demand_balance = pyo.Constraint(m.T, rule=demand_balance_rule)

    def reserve_rule(mm, t):
        return sum(p_max[g] * mm.u[g, t] for g in mm.G) >= float(demand[t]) * (1.0 + case.reserve_margin)

    m.reserve = pyo.Constraint(m.T, rule=reserve_rule)

    def pmin_rule(mm, g, t):
        return mm.p[g, t] >= p_min[g] * mm.u[g, t]

    def pmax_rule(mm, g, t):
        return mm.p[g, t] <= p_max[g] * mm.u[g, t]

    m.p_min = pyo.Constraint(m.G, m.T, rule=pmin_rule)
    m.p_max = pyo.Constraint(m.G, m.T, rule=pmax_rule)

    def transition_rule(mm, g, t):
        prev_u = initial_status[g] if t == 0 else mm.u[g, t - 1]
        return mm.u[g, t] - prev_u == mm.startup[g, t] - mm.shutdown[g, t]

    m.transition = pyo.Constraint(m.G, m.T, rule=transition_rule)

    def no_simultaneous_rule(mm, g, t):
        return mm.startup[g, t] + mm.shutdown[g, t] <= 1

    m.no_simultaneous = pyo.Constraint(m.G, m.T, rule=no_simultaneous_rule)

    def ramp_up_rule(mm, g, t):
        prev_p = initial_dispatch[g] if t == 0 else mm.p[g, t - 1]
        return mm.p[g, t] - prev_p <= ramp_up[g] + p_max[g] * mm.startup[g, t]

    def ramp_down_rule(mm, g, t):
        prev_p = initial_dispatch[g] if t == 0 else mm.p[g, t - 1]
        return prev_p - mm.p[g, t] <= ramp_down[g] + p_max[g] * mm.shutdown[g, t]

    m.ramp_up = pyo.Constraint(m.G, m.T, rule=ramp_up_rule)
    m.ramp_down = pyo.Constraint(m.G, m.T, rule=ramp_down_rule)

    # Minimum up/down constraints. Near the end of the horizon we enforce over
    # the available remaining hours, which is standard for finite-horizon data generation.
    def min_up_rule(mm, g, t):
        window = range(t, min(case.time_horizon, t + int(min_up[g])))
        return sum(mm.u[g, tau] for tau in window) >= len(list(window)) * mm.startup[g, t]

    def min_down_rule(mm, g, t):
        window = range(t, min(case.time_horizon, t + int(min_down[g])))
        return sum(1 - mm.u[g, tau] for tau in window) >= len(list(window)) * mm.shutdown[g, t]

    m.min_up = pyo.Constraint(m.G, m.T, rule=min_up_rule)
    m.min_down = pyo.Constraint(m.G, m.T, rule=min_down_rule)

    if warm_start_commitment is not None:
        arr = np.asarray(warm_start_commitment)
        if arr.shape != (case.num_generators, case.time_horizon):
            raise ValueError(f"warm_start_commitment shape {arr.shape} must be {(case.num_generators, case.time_horizon)}")
        for g in G:
            for t in T:
                m.u[g, t].value = int(round(float(arr[g, t])))

    if fixed_commitment is not None:
        arr = np.asarray(fixed_commitment, dtype=float)
        if arr.shape != (case.num_generators, case.time_horizon):
            raise ValueError(f"fixed_commitment shape {arr.shape} must be {(case.num_generators, case.time_horizon)}")
        for g in G:
            for t in T:
                if not np.isnan(arr[g, t]):
                    m.u[g, t].fix(int(round(float(arr[g, t]))))

    return m


def solve_uc_milp(
    case: UCCase,
    demand: np.ndarray,
    solver_name: str = "gurobi",
    tee: bool = False,
    time_limit: Optional[float] = None,
    mip_gap: Optional[float] = None,
    threads: int = 0,
    warm_start_commitment: Optional[np.ndarray] = None,
    fixed_commitment: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Solve UC MILP and return a serializable solution dictionary."""
    model = build_uc_model(
        case=case,
        demand=demand,
        warm_start_commitment=warm_start_commitment,
        fixed_commitment=fixed_commitment,
    )

    solver = pyo.SolverFactory(solver_name)
    if not solver.available(exception_flag=False):
        return {
            "status": "solver_unavailable",
            "termination_condition": "solver_unavailable",
            "feasible": False,
            "objective": np.nan,
            "gap": np.nan,
            "solve_time": 0.0,
            "commitment": None,
            "dispatch": None,
            "error": f"Solver {solver_name} is not available. Install/configure Gurobi or change solver.name.",
        }

    if time_limit is not None:
        if solver_name.lower() == "gurobi":
            solver.options["TimeLimit"] = float(time_limit)
        else:
            solver.options["tmlim"] = float(time_limit)
    if mip_gap is not None and solver_name.lower() == "gurobi":
        solver.options["MIPGap"] = float(mip_gap)
    if threads and solver_name.lower() == "gurobi":
        solver.options["Threads"] = int(threads)

    start = time.perf_counter()
    try:
        results = solver.solve(model, tee=tee, load_solutions=True)
        elapsed = time.perf_counter() - start
    except Exception as exc:
        return {
            "status": "solve_error",
            "termination_condition": "exception",
            "feasible": False,
            "objective": np.nan,
            "gap": np.nan,
            "solve_time": time.perf_counter() - start,
            "commitment": None,
            "dispatch": None,
            "error": repr(exc),
        }

    status = str(results.solver.status)
    term = str(results.solver.termination_condition)
    feasible_terms = {
        str(TerminationCondition.optimal),
        str(TerminationCondition.feasible),
        str(TerminationCondition.maxTimeLimit),
        "optimal",
        "feasible",
    }
    feasible = (results.solver.status in {SolverStatus.ok, SolverStatus.warning}) and (term in feasible_terms)

    commitment = None
    dispatch = None
    objective = np.nan

    if feasible:
        try:
            objective = float(pyo.value(model.obj))
            commitment = np.zeros((case.num_generators, case.time_horizon), dtype=int)
            dispatch = np.zeros((case.num_generators, case.time_horizon), dtype=float)
            for g in range(case.num_generators):
                for t in range(case.time_horizon):
                    commitment[g, t] = int(round(pyo.value(model.u[g, t])))
                    dispatch[g, t] = float(pyo.value(model.p[g, t]))
        except Exception:
            feasible = False

    return {
        "status": status,
        "termination_condition": term,
        "feasible": bool(feasible),
        "objective": objective,
        "gap": _solver_gap(results),
        "solve_time": elapsed,
        "commitment": commitment,
        "dispatch": dispatch,
        "error": "",
    }


def commitment_to_row(scenario_id: int, commitment: np.ndarray) -> Dict[str, int]:
    G, T = commitment.shape
    row: Dict[str, int] = {"scenario_id": int(scenario_id)}
    for g in range(G):
        for t in range(T):
            row[f"GOn_g{g}_t{t}"] = int(commitment[g, t])
    return row


def dispatch_to_row(scenario_id: int, dispatch: np.ndarray) -> Dict[str, float]:
    G, T = dispatch.shape
    row: Dict[str, float] = {"scenario_id": int(scenario_id)}
    for g in range(G):
        for t in range(T):
            row[f"P_g{g}_t{t}"] = float(dispatch[g, t])
    return row
