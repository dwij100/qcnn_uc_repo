"""Load or construct UC test systems.

Version 1 provides guaranteed-runnable synthetic UC systems and an optional
pandapower/MATPOWER-style network loader. Standard power-flow cases usually do
not include UC-specific data such as startup cost, min up/down time, or ramp
limits. For those fields, this module assigns reproducible synthetic parameters
and marks the resulting system as a UC-augmented power-flow case.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np


@dataclass(frozen=True)
class BusData:
    bus_id: int
    base_load_mw: float


@dataclass(frozen=True)
class BranchData:
    branch_id: int
    from_bus: int
    to_bus: int
    x_pu: float
    rate_mw: float


@dataclass(frozen=True)
class GeneratorData:
    gen_id: int
    name: str
    bus: int
    p_min_mw: float
    p_max_mw: float
    ramp_up_mw: float
    ramp_down_mw: float
    min_up_time: int
    min_down_time: int
    no_load_cost: float
    startup_cost: float
    shutdown_cost: float
    linear_cost: float
    initial_status: int = 0
    initial_power_mw: float = 0.0


@dataclass(frozen=True)
class SystemData:
    name: str
    base_mva: float
    buses: List[BusData]
    generators: List[GeneratorData]
    branches: List[BranchData]
    notes: str = ""

    @property
    def n_buses(self) -> int:
        return len(self.buses)

    @property
    def n_generators(self) -> int:
        return len(self.generators)

    @property
    def bus_ids(self) -> List[int]:
        return [b.bus_id for b in self.buses]

    @property
    def gen_ids(self) -> List[int]:
        return [g.gen_id for g in self.generators]

    def generators_by_bus(self) -> Dict[int, List[int]]:
        mapping: Dict[int, List[int]] = {b.bus_id: [] for b in self.buses}
        for g in self.generators:
            mapping.setdefault(g.bus, []).append(g.gen_id)
        return mapping


def load_system(
    case_name: str,
    case_source: str = "synthetic",
    n_generators: Optional[int] = None,
    n_buses: Optional[int] = None,
    base_mva: float = 100.0,
    seed: int = 42,
) -> SystemData:
    """Load a system by name.

    Parameters
    ----------
    case_name:
        synthetic_10, synthetic_24, ieee118_synthetic_uc, or a pandapower case
        such as case118 when case_source='pandapower'.
    case_source:
        synthetic or pandapower.
    """
    source = case_source.lower()
    if source == "synthetic":
        if case_name.startswith("synthetic_"):
            parsed_n = int(case_name.split("_")[-1])
            n_generators = n_generators or parsed_n
            n_buses = n_buses or max(3, min(parsed_n // 2, 12))
        elif case_name == "ieee118_synthetic_uc":
            n_generators = n_generators or 24
            n_buses = n_buses or 118
        else:
            n_generators = n_generators or 10
            n_buses = n_buses or 5
        return build_synthetic_uc_system(
            name=case_name,
            n_generators=n_generators,
            n_buses=n_buses,
            base_mva=base_mva,
            seed=seed,
        )

    if source == "pandapower":
        return load_pandapower_system(case_name=case_name, base_mva=base_mva, seed=seed)

    raise ValueError(f"Unsupported case_source={case_source!r}. Use synthetic or pandapower.")


def build_synthetic_uc_system(
    name: str,
    n_generators: int,
    n_buses: int,
    base_mva: float,
    seed: int,
) -> SystemData:
    """Build a reproducible synthetic UC system.

    This gives a controlled first benchmark where the UC constraints are real,
    but network data are not claimed to represent an actual grid. It is useful
    for debugging the full QCNN/MILP pipeline before scaling to IEEE cases.
    """
    if n_generators < 2:
        raise ValueError("n_generators must be >= 2")
    if n_buses < 1:
        raise ValueError("n_buses must be >= 1")

    rng = np.random.default_rng(seed)

    # Generator sizes are deliberately heterogeneous to avoid trivial schedules.
    p_max = np.sort(rng.uniform(60, 220, size=n_generators))[::-1]
    p_min = rng.uniform(0.15, 0.30, size=n_generators) * p_max
    linear_cost = np.sort(rng.uniform(12, 55, size=n_generators))
    no_load_cost = rng.uniform(5, 25, size=n_generators)
    startup_cost = rng.uniform(50, 400, size=n_generators)
    shutdown_cost = 0.25 * startup_cost
    ramp_frac = rng.uniform(0.35, 0.75, size=n_generators)
    ramp = ramp_frac * p_max
    min_up = rng.integers(1, 4, size=n_generators)
    min_down = rng.integers(1, 4, size=n_generators)

    buses: List[BusData] = []
    total_capacity = float(p_max.sum())
    base_total_load = 0.52 * total_capacity
    weights = rng.dirichlet(np.ones(n_buses))
    for b in range(n_buses):
        buses.append(BusData(bus_id=b, base_load_mw=float(base_total_load * weights[b])))

    generators: List[GeneratorData] = []
    for g in range(n_generators):
        bus = g % n_buses
        initial_status = 1 if g < max(1, n_generators // 3) else 0
        initial_power = float(p_min[g]) if initial_status else 0.0
        generators.append(
            GeneratorData(
                gen_id=g,
                name=f"G{g}",
                bus=bus,
                p_min_mw=float(p_min[g]),
                p_max_mw=float(p_max[g]),
                ramp_up_mw=float(ramp[g]),
                ramp_down_mw=float(ramp[g]),
                min_up_time=int(min_up[g]),
                min_down_time=int(min_down[g]),
                no_load_cost=float(no_load_cost[g]),
                startup_cost=float(startup_cost[g]),
                shutdown_cost=float(shutdown_cost[g]),
                linear_cost=float(linear_cost[g]),
                initial_status=initial_status,
                initial_power_mw=initial_power,
            )
        )

    branches: List[BranchData] = []
    if n_buses >= 2:
        # A ring with generous rates. Network constraints can be enabled later.
        for i in range(n_buses):
            branches.append(
                BranchData(
                    branch_id=i,
                    from_bus=i,
                    to_bus=(i + 1) % n_buses,
                    x_pu=float(rng.uniform(0.05, 0.18)),
                    rate_mw=float(0.80 * total_capacity),
                )
            )

    return SystemData(
        name=name,
        base_mva=base_mva,
        buses=buses,
        generators=generators,
        branches=branches,
        notes="Synthetic UC case with reproducible generator and network parameters.",
    )


def load_pandapower_system(case_name: str, base_mva: float, seed: int) -> SystemData:
    """Load a pandapower network and augment it with UC parameters.

    Requires pandapower. Example config values:
        case_source: pandapower
        case_name: case118

    Important: IEEE/MATPOWER power-flow cases do not usually contain full UC
    operating data. Startup/shutdown/ramp/min-up/min-down values are therefore
    generated reproducibly here and should be replaced with validated data before
    making final publication claims.
    """
    try:
        import pandapower.networks as pn
    except ImportError as exc:
        raise ImportError(
            "pandapower is not installed. Install it or use case_source='synthetic'."
        ) from exc

    if not hasattr(pn, case_name):
        raise ValueError(f"pandapower.networks has no case named {case_name!r}")

    net = getattr(pn, case_name)()
    rng = np.random.default_rng(seed)

    # Bus loads: aggregate static loads per bus. Missing load means zero.
    n_buses = len(net.bus)
    load_by_bus = {int(i): 0.0 for i in net.bus.index}
    if hasattr(net, "load") and len(net.load) > 0:
        for _, row in net.load.iterrows():
            load_by_bus[int(row.bus)] += float(row.p_mw)
    buses = [BusData(bus_id=int(i), base_load_mw=float(load_by_bus[int(i)])) for i in net.bus.index]

    gen_rows = []
    if hasattr(net, "gen") and len(net.gen) > 0:
        for _, row in net.gen.iterrows():
            gen_rows.append((int(row.bus), float(row.get("min_p_mw", 0.0) or 0.0), float(row.get("max_p_mw", row.p_mw) or row.p_mw)))
    if hasattr(net, "ext_grid") and len(net.ext_grid) > 0:
        for _, row in net.ext_grid.iterrows():
            # External grid is approximated as a large generator.
            gen_rows.append((int(row.bus), 0.0, max(200.0, 0.25 * max(1.0, sum(load_by_bus.values())))))

    if not gen_rows:
        raise ValueError(f"No generators/ext_grid found in pandapower case {case_name!r}")

    generators: List[GeneratorData] = []
    for g, (bus, pmin, pmax) in enumerate(gen_rows):
        pmax = max(pmax, pmin + 10.0)
        generators.append(
            GeneratorData(
                gen_id=g,
                name=f"G{g}",
                bus=bus,
                p_min_mw=float(max(0.0, pmin)),
                p_max_mw=float(pmax),
                ramp_up_mw=float(rng.uniform(0.30, 0.70) * pmax),
                ramp_down_mw=float(rng.uniform(0.30, 0.70) * pmax),
                min_up_time=int(rng.integers(1, 4)),
                min_down_time=int(rng.integers(1, 4)),
                no_load_cost=float(rng.uniform(5, 30)),
                startup_cost=float(rng.uniform(80, 500)),
                shutdown_cost=float(rng.uniform(20, 150)),
                linear_cost=float(rng.uniform(10, 60)),
                initial_status=0,
                initial_power_mw=0.0,
            )
        )

    branches: List[BranchData] = []
    branch_id = 0
    if hasattr(net, "line") and len(net.line) > 0:
        for _, row in net.line.iterrows():
            length = float(row.length_km) if "length_km" in row else 1.0
            x_ohm = float(row.x_ohm_per_km) * max(length, 1e-6) if "x_ohm_per_km" in row else 0.1
            vn_kv = float(net.bus.loc[int(row.from_bus), "vn_kv"])
            z_base = (vn_kv**2) / max(base_mva, 1e-9)
            x_pu = max(x_ohm / max(z_base, 1e-9), 1e-4)
            rate = float(row.max_i_ka * vn_kv * np.sqrt(3.0)) if "max_i_ka" in row else 1e4
            branches.append(BranchData(branch_id, int(row.from_bus), int(row.to_bus), x_pu, max(rate, 1e3)))
            branch_id += 1

    return SystemData(
        name=case_name,
        base_mva=base_mva,
        buses=buses,
        generators=generators,
        branches=branches,
        notes="pandapower network augmented with synthetic UC-specific parameters.",
    )
