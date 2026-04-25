from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from src.data_generation.case_factory import UCCase


def solar_shape(time_horizon: int) -> np.ndarray:
    h = np.arange(time_horizon)
    curve = np.maximum(0.0, np.sin(np.pi * (h - 6) / 12))
    return curve / max(curve.max(), 1e-9)


def wind_shape(time_horizon: int, rng: np.random.Generator) -> np.ndarray:
    base = 0.45 + 0.12 * np.sin(2 * np.pi * (np.arange(time_horizon) + 4) / time_horizon)
    noise = rng.normal(0, 0.05, size=time_horizon)
    return np.clip(base + noise, 0.05, 0.95)


def generate_scenarios(
    case: UCCase,
    n_scenarios: int,
    seed: int,
    demand_noise_std: float = 0.08,
    demand_scale_low: float = 0.90,
    demand_scale_high: float = 1.12,
    include_renewables: bool = True,
    renewable_penetration: float = 0.15,
) -> pd.DataFrame:
    """Generate scenario features.

    Features contain gross demand, renewable production, and net demand. The UC
    solver uses net demand. Both gross and renewable values are saved to allow
    ML models to learn uncertainty patterns.
    """
    rng = np.random.default_rng(seed)
    rows: List[Dict[str, float | int | str]] = []
    T = case.time_horizon

    for sid in range(n_scenarios):
        scale = rng.uniform(demand_scale_low, demand_scale_high)
        correlated_noise = rng.normal(0.0, demand_noise_std, size=T)
        # Smooth the random profile to avoid unrealistic hour-to-hour jumps.
        correlated_noise = 0.60 * correlated_noise + 0.25 * np.roll(correlated_noise, 1) + 0.15 * np.roll(correlated_noise, -1)
        gross_demand = case.base_demand * scale * np.clip(1.0 + correlated_noise, 0.75, 1.30)

        if include_renewables and renewable_penetration > 0:
            solar = solar_shape(T) * renewable_penetration * gross_demand.max() * rng.uniform(0.65, 1.15)
            wind = wind_shape(T, rng) * 0.35 * renewable_penetration * gross_demand.max() * rng.uniform(0.70, 1.20)
            renewable = solar + wind
        else:
            solar = np.zeros(T)
            wind = np.zeros(T)
            renewable = np.zeros(T)

        net_demand = np.maximum(gross_demand - renewable, 0.10 * gross_demand)

        row: Dict[str, float | int | str] = {
            "scenario_id": sid,
            "case_name": case.name,
            "num_generators": case.num_generators,
            "time_horizon": T,
            "reserve_margin": case.reserve_margin,
            "total_capacity": case.total_capacity,
        }
        for t in range(T):
            row[f"demand_t{t}"] = float(gross_demand[t])
            row[f"renewable_t{t}"] = float(renewable[t])
            row[f"net_demand_t{t}"] = float(net_demand[t])
            row[f"solar_t{t}"] = float(solar[t])
            row[f"wind_t{t}"] = float(wind[t])
        rows.append(row)

    return pd.DataFrame(rows)


def extract_net_demand(feature_row: pd.Series, time_horizon: int) -> np.ndarray:
    return np.array([feature_row[f"net_demand_t{t}"] for t in range(time_horizon)], dtype=float)
