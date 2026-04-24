"""Pyomo MILP formulation for Unit Commitment dataset generation."""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pyomo.environ as pyo

from src.data_generation.generate_scenarios import ScenarioData
from src.data_generation.load_ieee_case import SystemData

LOGGER = logging.getLogger(__name__)


@dataclass
class UCSolveResult:
    scenario_id: int
    feasible: bool
    solver_status: str
    termination_condition: str
    objective_cost: Optional[float]
    solve_time_sec: float
    mip_gap: Optional[float]
    commitment: Optional[np.ndarray]  # [n_generators, time_horizon]
    dispatch: Optional[np.ndarray]    # [n_generators, time_horizon]
    message: str = ""


def build_uc_model(system: SystemData, scenario: ScenarioData, cfg: dict) -> pyo.ConcreteModel:
    """Build the UC MILP for one demand scenario.

    The formulation is intentionally standard and upgradeable:
    - binary commitment u[g,t]
    - startup/shutdown binaries v[g,t], w[g,t]
    - dispatch p[g,t]
    - generation limits, demand balance, ramping, min up/down, reserve
    - optional DC network constraints with bus angle variables
    """
    uc_cfg = cfg["uc"]
    H = int(uc_cfg["time_horizon"])
    enable_network = bool(uc_cfg.get("enable_network_constraints", False))
    enable_reserve = bool(uc_cfg.get("enable_reserve", True))
    enable_ramping = bool(uc_cfg.get("enable_ramping", True))
    enable_min_up_down = bool(uc_cfg.get("enable_min_up_down", True))
    reserve_margin = float(uc_cfg.get("reserve_margin", 0.0))

    if scenario.demand_mw.shape != (system.n_buses, H):
        raise ValueError(
            f"Demand shape mismatch. Expected {(system.n_buses, H)}, got {scenario.demand_mw.shape}"
        )

    m = pyo.ConcreteModel(name=f"UC_{system.name}_scenario_{scenario.scenario_id}")
    m.G = pyo.RangeSet(0, system.n_generators - 1)
    m.T = pyo.RangeSet(0, H - 1)
    m.B = pyo.Set(initialize=system.bus_ids, ordered=True)
    m.L = pyo.RangeSet(0, max(0, len(system.branches) - 1)) if system.branches else pyo.Set(initialize=[])

    gen_by_id = {g.gen_id: g for g in system.generators}
    bus_index_by_id = {bus.bus_id: idx for idx, bus in enumerate(system.buses)}

    p_min = {g.gen_id: g.p_min_mw for g in system.generators}
    p_max = {g.gen_id: g.p_max_mw for g in system.generators}
    ramp_up = {g.gen_id: g.ramp_up_mw for g in system.generators}
    ramp_down = {g.gen_id: g.ramp_down_mw for g in system.generators}
    min_up = {g.gen_id: g.min_up_time for g in system.generators}
    min_down = {g.gen_id: g.min_down_time for g in system.generators}
    no_load = {g.gen_id: g.no_load_cost for g in system.generators}
    startup = {g.gen_id: g.startup_cost for g in system.generators}
    shutdown = {g.gen_id: g.shutdown_cost for g in system.generators}
    linear = {g.gen_id: g.linear_cost for g in system.generators}
    init_status = {g.gen_id: int(g.initial_status) for g in system.generators}
    init_power = {g.gen_id: float(g.initial_power_mw) for g in system.generators}
    gen_bus = {g.gen_id: g.bus for g in system.generators}

    demand_total = {t: float(scenario.total_demand_mw[t]) for t in range(H)}
    demand_bus = {
        (bus_id, t): float(scenario.demand_mw[bus_index_by_id[bus_id], t])
        for bus_id in system.bus_ids
        for t in range(H)
    }

    # Decision variables.
    m.u = pyo.Var(m.G, m.T, within=pyo.Binary)  # on/off commitment
    m.v = pyo.Var(m.G, m.T, within=pyo.Binary)  # startup
    m.w = pyo.Var(m.G, m.T, within=pyo.Binary)  # shutdown
    m.p = pyo.Var(m.G, m.T, within=pyo.NonNegativeReals)  # dispatch MW

    if enable_network:
        if not system.branches:
            raise ValueError("Network constraints enabled, but system has no branches.")
        m.theta = pyo.Var(m.B, m.T, within=pyo.Reals, bounds=(-math.pi, math.pi))

    # Objective: linear production + no-load + transition costs.
    def objective_rule(model: pyo.ConcreteModel) -> pyo.Expression:
        return sum(
            linear[g] * model.p[g, t]
            + no_load[g] * model.u[g, t]
            + startup[g] * model.v[g, t]
            + shutdown[g] * model.w[g, t]
            for g in model.G
            for t in model.T
        )

    m.obj = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

    # Commitment transition: u_t - u_{t-1} = startup - shutdown.
    def transition_rule(model: pyo.ConcreteModel, g: int, t: int) -> pyo.Constraint:
        prev_u = init_status[g] if t == 0 else model.u[g, t - 1]
        return model.u[g, t] - prev_u == model.v[g, t] - model.w[g, t]

    m.transition = pyo.Constraint(m.G, m.T, rule=transition_rule)

    # Avoid simultaneous startup and shutdown.
    def no_simultaneous_transition_rule(model: pyo.ConcreteModel, g: int, t: int) -> pyo.Constraint:
        return model.v[g, t] + model.w[g, t] <= 1

    m.no_simultaneous_transition = pyo.Constraint(m.G, m.T, rule=no_simultaneous_transition_rule)

    # Generation limits linked to commitment.
    def pmax_rule(model: pyo.ConcreteModel, g: int, t: int) -> pyo.Constraint:
        return model.p[g, t] <= p_max[g] * model.u[g, t]

    def pmin_rule(model: pyo.ConcreteModel, g: int, t: int) -> pyo.Constraint:
        return model.p[g, t] >= p_min[g] * model.u[g, t]

    m.pmax_limit = pyo.Constraint(m.G, m.T, rule=pmax_rule)
    m.pmin_limit = pyo.Constraint(m.G, m.T, rule=pmin_rule)

    # Demand balance: copper-plate or DC network form.
    if not enable_network:

        def system_balance_rule(model: pyo.ConcreteModel, t: int) -> pyo.Constraint:
            return sum(model.p[g, t] for g in model.G) == demand_total[t]

        m.system_balance = pyo.Constraint(m.T, rule=system_balance_rule)
    else:
        # DC flow: F_l,t = baseMVA / x_l * (theta_from - theta_to)
        branch_from = {l.branch_id: l.from_bus for l in system.branches}
        branch_to = {l.branch_id: l.to_bus for l in system.branches}
        branch_x = {l.branch_id: max(float(l.x_pu), 1e-6) for l in system.branches}
        branch_rate = {l.branch_id: float(l.rate_mw) for l in system.branches}
        gen_at_bus: Dict[int, list[int]] = system.generators_by_bus()
        outgoing = {b: [l.branch_id for l in system.branches if l.from_bus == b] for b in system.bus_ids}
        incoming = {b: [l.branch_id for l in system.branches if l.to_bus == b] for b in system.bus_ids}

        def flow_expr(model: pyo.ConcreteModel, l: int, t: int) -> pyo.Expression:
            return system.base_mva / branch_x[l] * (
                model.theta[branch_from[l], t] - model.theta[branch_to[l], t]
            )

        def nodal_balance_rule(model: pyo.ConcreteModel, b: int, t: int) -> pyo.Constraint:
            generation = sum(model.p[g, t] for g in gen_at_bus.get(b, []))
            net_outflow = sum(flow_expr(model, l, t) for l in outgoing[b]) - sum(
                flow_expr(model, l, t) for l in incoming[b]
            )
            return generation - demand_bus[b, t] == net_outflow

        m.nodal_balance = pyo.Constraint(m.B, m.T, rule=nodal_balance_rule)

        ref_bus = system.bus_ids[0]

        def ref_angle_rule(model: pyo.ConcreteModel, t: int) -> pyo.Constraint:
            return model.theta[ref_bus, t] == 0.0

        m.reference_angle = pyo.Constraint(m.T, rule=ref_angle_rule)

        def branch_upper_rule(model: pyo.ConcreteModel, l: int, t: int) -> pyo.Constraint:
            return flow_expr(model, l, t) <= branch_rate[l]

        def branch_lower_rule(model: pyo.ConcreteModel, l: int, t: int) -> pyo.Constraint:
            return flow_expr(model, l, t) >= -branch_rate[l]

        m.branch_upper = pyo.Constraint(m.L, m.T, rule=branch_upper_rule)
        m.branch_lower = pyo.Constraint(m.L, m.T, rule=branch_lower_rule)

    if enable_reserve:

        def reserve_rule(model: pyo.ConcreteModel, t: int) -> pyo.Constraint:
            return sum(p_max[g] * model.u[g, t] for g in model.G) >= (1.0 + reserve_margin) * demand_total[t]

        m.reserve = pyo.Constraint(m.T, rule=reserve_rule)

    if enable_ramping:

        def ramp_up_rule(model: pyo.ConcreteModel, g: int, t: int) -> pyo.Constraint:
            prev_p = init_power[g] if t == 0 else model.p[g, t - 1]
            # Startup term relaxes ramping when a unit is switched on.
            return model.p[g, t] - prev_p <= ramp_up[g] + p_max[g] * model.v[g, t]

        def ramp_down_rule(model: pyo.ConcreteModel, g: int, t: int) -> pyo.Constraint:
            prev_p = init_power[g] if t == 0 else model.p[g, t - 1]
            # Shutdown term relaxes ramping when a unit is switched off.
            return prev_p - model.p[g, t] <= ramp_down[g] + p_max[g] * model.w[g, t]

        m.ramp_up = pyo.Constraint(m.G, m.T, rule=ramp_up_rule)
        m.ramp_down = pyo.Constraint(m.G, m.T, rule=ramp_down_rule)

    if enable_min_up_down:

        def min_up_rule(model: pyo.ConcreteModel, g: int, t: int) -> pyo.Constraint:
            mu = int(min_up[g])
            if mu <= 1 or t + mu - 1 >= H:
                return pyo.Constraint.Skip
            return sum(model.u[g, tau] for tau in range(t, t + mu)) >= mu * model.v[g, t]

        def min_down_rule(model: pyo.ConcreteModel, g: int, t: int) -> pyo.Constraint:
            md = int(min_down[g])
            if md <= 1 or t + md - 1 >= H:
                return pyo.Constraint.Skip
            return sum(1 - model.u[g, tau] for tau in range(t, t + md)) >= md * model.w[g, t]

        m.min_up = pyo.Constraint(m.G, m.T, rule=min_up_rule)
        m.min_down = pyo.Constraint(m.G, m.T, rule=min_down_rule)

    return m


def _configure_solver(solver: pyo.SolverFactory, solver_name: str, solver_cfg: dict) -> None:
    """Apply common solver options when supported."""
    time_limit = solver_cfg.get("time_limit_sec")
    mip_gap = solver_cfg.get("mip_gap")
    threads = solver_cfg.get("threads")

    name = solver_name.lower()
    try:
        if "gurobi" in name:
            if time_limit is not None:
                solver.options["TimeLimit"] = float(time_limit)
            if mip_gap is not None:
                solver.options["MIPGap"] = float(mip_gap)
            if threads is not None and int(threads) > 0:
                solver.options["Threads"] = int(threads)
        elif "highs" in name:
            if time_limit is not None:
                solver.options["time_limit"] = float(time_limit)
            if mip_gap is not None:
                solver.options["mip_rel_gap"] = float(mip_gap)
        elif "cbc" in name:
            if time_limit is not None:
                solver.options["seconds"] = float(time_limit)
            if mip_gap is not None:
                solver.options["ratio"] = float(mip_gap)
        elif "glpk" in name:
            if time_limit is not None:
                solver.options["tmlim"] = int(time_limit)
            if mip_gap is not None:
                solver.options["mipgap"] = float(mip_gap)
    except Exception as exc:
        LOGGER.warning("Could not apply some solver options: %s", exc)


def _extract_mip_gap(results: object) -> Optional[float]:
    """Best-effort MIP gap extraction from Pyomo results."""
    try:
        lb = getattr(results.problem, "lower_bound", None)
        ub = getattr(results.problem, "upper_bound", None)
        if lb is None or ub is None:
            return None
        lb = float(lb)
        ub = float(ub)
        if not (math.isfinite(lb) and math.isfinite(ub)):
            return None
        denom = max(abs(ub), 1e-9)
        return abs(ub - lb) / denom
    except Exception:
        return None


def solve_uc_instance(system: SystemData, scenario: ScenarioData, cfg: dict) -> Tuple[pyo.ConcreteModel, UCSolveResult]:
    """Build and solve one UC instance."""
    solver_cfg = cfg["solver"]
    solver_name = str(solver_cfg.get("name", "gurobi"))
    tee = bool(solver_cfg.get("tee", False))

    model = build_uc_model(system, scenario, cfg)

    solver = pyo.SolverFactory(solver_name)
    if solver is None or not solver.available(False):
        msg = (
            f"Solver '{solver_name}' is not available. Install/configure it, or set solver.name "
            "to appsi_highs/cbc/glpk in config.yaml."
        )
        LOGGER.error(msg)
        return model, UCSolveResult(
            scenario_id=scenario.scenario_id,
            feasible=False,
            solver_status="unavailable",
            termination_condition="solver_unavailable",
            objective_cost=None,
            solve_time_sec=0.0,
            mip_gap=None,
            commitment=None,
            dispatch=None,
            message=msg,
        )

    _configure_solver(solver, solver_name, solver_cfg)

    start = time.perf_counter()
    try:
        results = solver.solve(model, tee=tee)
    except Exception as exc:
        elapsed = time.perf_counter() - start
        msg = f"Solver failed on scenario {scenario.scenario_id}: {exc}"
        LOGGER.exception(msg)
        return model, UCSolveResult(
            scenario_id=scenario.scenario_id,
            feasible=False,
            solver_status="error",
            termination_condition="solver_error",
            objective_cost=None,
            solve_time_sec=elapsed,
            mip_gap=None,
            commitment=None,
            dispatch=None,
            message=msg,
        )

    elapsed = time.perf_counter() - start
    status = str(results.solver.status)
    term = str(results.solver.termination_condition)
    feasible_terms = {"optimal", "feasible", "maxTimeLimit"}
    feasible = any(k.lower() in term.lower() for k in feasible_terms)

    if not feasible:
        return model, UCSolveResult(
            scenario_id=scenario.scenario_id,
            feasible=False,
            solver_status=status,
            termination_condition=term,
            objective_cost=None,
            solve_time_sec=elapsed,
            mip_gap=_extract_mip_gap(results),
            commitment=None,
            dispatch=None,
            message="No feasible UC solution returned.",
        )

    H = int(cfg["uc"]["time_horizon"])
    n_g = system.n_generators
    commitment = np.zeros((n_g, H), dtype=int)
    dispatch = np.zeros((n_g, H), dtype=float)

    try:
        for g in range(n_g):
            for t in range(H):
                commitment[g, t] = int(round(float(pyo.value(model.u[g, t]))))
                dispatch[g, t] = float(pyo.value(model.p[g, t]))
        objective = float(pyo.value(model.obj))
    except Exception as exc:
        LOGGER.exception("Failed to extract solution values for scenario %s", scenario.scenario_id)
        return model, UCSolveResult(
            scenario_id=scenario.scenario_id,
            feasible=False,
            solver_status=status,
            termination_condition=term,
            objective_cost=None,
            solve_time_sec=elapsed,
            mip_gap=_extract_mip_gap(results),
            commitment=None,
            dispatch=None,
            message=f"Solution extraction failed: {exc}",
        )

    return model, UCSolveResult(
        scenario_id=scenario.scenario_id,
        feasible=True,
        solver_status=status,
        termination_condition=term,
        objective_cost=objective,
        solve_time_sec=elapsed,
        mip_gap=_extract_mip_gap(results),
        commitment=commitment,
        dispatch=dispatch,
        message="ok",
    )
