"""Scenario generation for UC dataset creation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from src.data_generation.load_ieee_case import SystemData


@dataclass(frozen=True)
class ScenarioData:
    scenario_id: int
    demand_mw: np.ndarray  # shape [n_buses, time_horizon]
    total_demand_mw: np.ndarray  # shape [time_horizon]
    seed: int


def _daily_profile(time_horizon: int, profile_24h: List[float]) -> np.ndarray:
    profile = np.asarray(profile_24h, dtype=float)
    if len(profile) != 24:
        raise ValueError("daily_profile must contain 24 hourly values")
    if time_horizon == 24:
        return profile
    if time_horizon < 24:
        return profile[:time_horizon]
    # Repeat if a longer horizon is requested.
    reps = int(np.ceil(time_horizon / 24))
    return np.tile(profile, reps)[:time_horizon]


def generate_uc_scenarios(system: SystemData, cfg: dict) -> List[ScenarioData]:
    """Generate demand scenarios around the system base-load profile."""
    uc_cfg = cfg["uc"]
    seed = int(cfg["project"]["seed"])
    rng = np.random.default_rng(seed)

    n_scenarios = int(uc_cfg["n_scenarios"])
    time_horizon = int(uc_cfg["time_horizon"])
    scale_low = float(uc_cfg["demand_scale_low"])
    scale_high = float(uc_cfg["demand_scale_high"])
    noise_std = float(uc_cfg["bus_load_noise_std"])
    profile = _daily_profile(time_horizon, uc_cfg["daily_profile"])

    base_load = np.asarray([b.base_load_mw for b in system.buses], dtype=float)
    if np.allclose(base_load.sum(), 0.0):
        # Some power-flow cases may have missing static loads; create a small
        # artificial load so the UC problem is not empty.
        base_load = np.ones(system.n_buses) * (0.5 * sum(g.p_max_mw for g in system.generators) / system.n_buses)

    scenarios: List[ScenarioData] = []
    for s in range(n_scenarios):
        scenario_seed = int(rng.integers(0, 2**31 - 1))
        srng = np.random.default_rng(scenario_seed)

        global_scale = srng.uniform(scale_low, scale_high)
        demand = np.zeros((system.n_buses, time_horizon), dtype=float)

        for t in range(time_horizon):
            bus_noise = srng.normal(loc=1.0, scale=noise_std, size=system.n_buses)
            bus_noise = np.clip(bus_noise, 0.70, 1.30)
            demand[:, t] = base_load * profile[t] * global_scale * bus_noise

        scenarios.append(
            ScenarioData(
                scenario_id=s,
                demand_mw=demand,
                total_demand_mw=demand.sum(axis=0),
                seed=scenario_seed,
            )
        )

    return scenarios
