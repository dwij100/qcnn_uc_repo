from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd


@dataclass
class UCCase:
    name: str
    num_generators: int
    time_horizon: int
    reserve_margin: float
    generators: pd.DataFrame
    base_demand: np.ndarray
    buses: pd.DataFrame | None = None
    lines: pd.DataFrame | None = None
    network_constraints: bool = False

    @property
    def total_capacity(self) -> float:
        return float(self.generators["p_max"].sum())


def _daily_load_shape(time_horizon: int) -> np.ndarray:
    """Typical normalized 24-hour demand shape with morning/evening peaks."""
    x = np.arange(time_horizon)
    morning = 0.18 * np.exp(-0.5 * ((x - 8) / 3.0) ** 2)
    evening = 0.28 * np.exp(-0.5 * ((x - 19) / 4.0) ** 2)
    base = 0.62 + morning + evening + 0.03 * np.sin(2 * np.pi * x / time_horizon)
    return base / base.max()


def _make_generators(num_generators: int, seed: int) -> pd.DataFrame:
    """Create synthetic but UC-realistic thermal units.

    The generated fleet mixes baseload, mid-merit, and peaking units. Costs are
    correlated inversely with capacity, which is a common stylized UC benchmark
    assumption.
    """
    rng = np.random.default_rng(seed)
    rows = []

    for g in range(num_generators):
        frac = g / max(1, num_generators - 1)

        if frac < 0.25:  # baseload
            p_max = rng.uniform(120, 260)
            p_min_ratio = rng.uniform(0.35, 0.55)
            marginal = rng.uniform(14, 26)
            no_load = rng.uniform(500, 1100)
            startup = rng.uniform(1800, 4500)
            min_up = rng.integers(4, 8)
            min_down = rng.integers(3, 6)
        elif frac < 0.70:  # mid-merit
            p_max = rng.uniform(60, 160)
            p_min_ratio = rng.uniform(0.25, 0.45)
            marginal = rng.uniform(25, 48)
            no_load = rng.uniform(180, 650)
            startup = rng.uniform(700, 2200)
            min_up = rng.integers(2, 5)
            min_down = rng.integers(2, 5)
        else:  # peaking
            p_max = rng.uniform(25, 90)
            p_min_ratio = rng.uniform(0.10, 0.30)
            marginal = rng.uniform(45, 85)
            no_load = rng.uniform(50, 250)
            startup = rng.uniform(120, 900)
            min_up = rng.integers(1, 3)
            min_down = rng.integers(1, 3)

        p_min = p_min_ratio * p_max
        rows.append(
            {
                "gen_id": f"G{g}",
                "bus": int(rng.integers(0, max(1, min(118, num_generators * 2)))),
                "p_min": round(float(p_min), 3),
                "p_max": round(float(p_max), 3),
                "ramp_up": round(float(rng.uniform(0.35, 0.75) * p_max), 3),
                "ramp_down": round(float(rng.uniform(0.35, 0.75) * p_max), 3),
                "startup_cost": round(float(startup), 3),
                "shutdown_cost": round(float(0.15 * startup), 3),
                "no_load_cost": round(float(no_load), 3),
                "marginal_cost": round(float(marginal), 3),
                "min_up_time": int(min_up),
                "min_down_time": int(min_down),
                "initial_status": int(rng.choice([0, 1], p=[0.35, 0.65])),
                "initial_dispatch": 0.0,
            }
        )

    df = pd.DataFrame(rows)
    df["initial_dispatch"] = np.where(
        df["initial_status"].to_numpy() == 1,
        0.5 * (df["p_min"].to_numpy() + df["p_max"].to_numpy()),
        0.0,
    )
    return df


def _make_reduced_network(num_buses: int, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create a lightweight IEEE-118-inspired bus/line skeleton.

    This is intentionally not a full AC/DC OPF network. It preserves a larger
    system structure for later PTDF/network extension without making Stage 1
    computationally heavy.
    """
    rng = np.random.default_rng(seed)
    buses = pd.DataFrame({"bus": np.arange(num_buses), "load_share": rng.dirichlet(np.ones(num_buses))})
    edges = []
    for i in range(num_buses - 1):
        edges.append({"from_bus": i, "to_bus": i + 1, "x": rng.uniform(0.01, 0.10), "rate": rng.uniform(100, 350)})
    extra = max(num_buses // 3, 1)
    for _ in range(extra):
        a, b = rng.choice(num_buses, size=2, replace=False)
        edges.append({"from_bus": int(a), "to_bus": int(b), "x": rng.uniform(0.01, 0.12), "rate": rng.uniform(80, 300)})
    return buses, pd.DataFrame(edges)


def create_uc_case(case_name: str, time_horizon: int = 24, reserve_margin: float = 0.10, seed: int = 42) -> UCCase:
    case_name = case_name.lower()
    if case_name == "case10":
        n_gen, demand_peak, buses_n = 10, 720.0, 10
    elif case_name == "case24":
        n_gen, demand_peak, buses_n = 24, 1850.0, 24
    elif case_name in {"case118", "case118_reduced", "ieee118_reduced"}:
        # IEEE 118-bus systems often have ~50+ generators in benchmark data.
        # This reduced model keeps 118 buses and a large thermal fleet, while
        # Stage 1 UC remains a tractable single-bus MILP unless network is enabled.
        n_gen, demand_peak, buses_n = 54, 4200.0, 118
        case_name = "case118_reduced"
    else:
        raise ValueError(f"Unknown case_name={case_name}. Use case10, case24, or case118_reduced.")

    gens = _make_generators(n_gen, seed=seed)
    load_shape = _daily_load_shape(time_horizon)
    base_demand = demand_peak * load_shape

    # Make sure demand does not exceed capacity after reserve.
    cap = gens["p_max"].sum()
    max_allowed = cap / (1.0 + reserve_margin) * 0.88
    if base_demand.max() > max_allowed:
        base_demand *= max_allowed / base_demand.max()

    buses, lines = _make_reduced_network(buses_n, seed + 1000)
    return UCCase(
        name=case_name,
        num_generators=n_gen,
        time_horizon=time_horizon,
        reserve_margin=reserve_margin,
        generators=gens,
        base_demand=base_demand,
        buses=buses,
        lines=lines,
        network_constraints=False,
    )


def case_to_metadata(case: UCCase) -> Dict[str, float | int | str]:
    return {
        "case_name": case.name,
        "num_generators": case.num_generators,
        "time_horizon": case.time_horizon,
        "reserve_margin": case.reserve_margin,
        "total_capacity": case.total_capacity,
        "peak_base_demand": float(np.max(case.base_demand)),
    }
