"""End-to-end UC MILP dataset generation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.data_generation.generate_scenarios import ScenarioData, generate_uc_scenarios
from src.data_generation.load_ieee_case import SystemData, load_system
from src.data_generation.uc_milp_model import UCSolveResult, solve_uc_instance
from src.utils.seed import set_global_seed

LOGGER = logging.getLogger(__name__)


def _resolve_path(project_root: Path, path_like: str) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else project_root / path


def _write_system_tables(system: SystemData, output_dir: Path) -> None:
    """Save bus, generator, and branch parameters for reproducibility."""
    pd.DataFrame([b.__dict__ for b in system.buses]).to_csv(output_dir / "system_buses.csv", index=False)
    pd.DataFrame([g.__dict__ for g in system.generators]).to_csv(output_dir / "system_generators.csv", index=False)
    pd.DataFrame([br.__dict__ for br in system.branches]).to_csv(output_dir / "system_branches.csv", index=False)

    with (output_dir / "system_notes.txt").open("w", encoding="utf-8") as f:
        f.write(f"System name: {system.name}\n")
        f.write(f"Base MVA: {system.base_mva}\n")
        f.write(f"Buses: {system.n_buses}\n")
        f.write(f"Generators: {system.n_generators}\n")
        f.write(f"Branches: {len(system.branches)}\n")
        f.write(f"Notes: {system.notes}\n")


def _scenario_feature_row(system: SystemData, scenario: ScenarioData, cfg: dict) -> Dict[str, float | int | str]:
    H = int(cfg["uc"]["time_horizon"])
    row: Dict[str, float | int | str] = {
        "scenario_id": scenario.scenario_id,
        "system_name": system.name,
        "n_generators": system.n_generators,
        "n_buses": system.n_buses,
        "time_horizon": H,
        "scenario_seed": scenario.seed,
    }
    for t in range(H):
        row[f"total_demand_t{t}"] = float(scenario.total_demand_mw[t])

    for b_idx, bus in enumerate(system.buses):
        for t in range(H):
            row[f"busload_b{bus.bus_id}_t{t}"] = float(scenario.demand_mw[b_idx, t])
    return row


def _commitment_row(result: UCSolveResult, system: SystemData, cfg: dict) -> Dict[str, float | int]:
    H = int(cfg["uc"]["time_horizon"])
    row: Dict[str, float | int] = {"scenario_id": result.scenario_id}
    for g in range(system.n_generators):
        for t in range(H):
            key = f"u_g{g}_t{t}"
            row[key] = int(result.commitment[g, t]) if result.commitment is not None else np.nan
    return row


def _dispatch_row(result: UCSolveResult, system: SystemData, cfg: dict) -> Dict[str, float | int]:
    H = int(cfg["uc"]["time_horizon"])
    row: Dict[str, float | int] = {"scenario_id": result.scenario_id}
    for g in range(system.n_generators):
        for t in range(H):
            key = f"p_g{g}_t{t}"
            row[key] = float(result.dispatch[g, t]) if result.dispatch is not None else np.nan
    return row


def _metadata_row(result: UCSolveResult, system: SystemData, scenario: ScenarioData, cfg: dict) -> Dict[str, float | int | str | bool | None]:
    return {
        "scenario_id": result.scenario_id,
        "system_name": system.name,
        "n_generators": system.n_generators,
        "n_buses": system.n_buses,
        "time_horizon": int(cfg["uc"]["time_horizon"]),
        "scenario_seed": scenario.seed,
        "feasible": bool(result.feasible),
        "solver_status": result.solver_status,
        "termination_condition": result.termination_condition,
        "objective_cost": result.objective_cost,
        "solve_time_sec": result.solve_time_sec,
        "mip_gap": result.mip_gap,
        "message": result.message,
    }


def generate_dataset(cfg: dict, project_root: str | Path) -> Path:
    """Generate a UC dataset by repeatedly solving the classical MILP.

    Returns
    -------
    Path
        Output directory containing CSV/NPY files.
    """
    project_root = Path(project_root).resolve()
    seed = int(cfg["project"]["seed"])
    set_global_seed(seed)

    dataset_name = str(cfg["outputs"]["dataset_name"])
    results_root = _resolve_path(project_root, cfg["paths"]["results_dir"])
    output_dir = results_root / dataset_name

    overwrite = bool(cfg["outputs"].get("overwrite", False))
    if output_dir.exists() and not overwrite:
        raise FileExistsError(
            f"Output directory already exists: {output_dir}. Set outputs.overwrite=true to overwrite."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Loading system: %s", cfg["uc"]["case_name"])
    system = load_system(
        case_name=str(cfg["uc"]["case_name"]),
        case_source=str(cfg["uc"].get("case_source", "synthetic")),
        n_generators=int(cfg["uc"].get("n_generators", 10)),
        n_buses=int(cfg["uc"].get("n_buses", 5)),
        base_mva=float(cfg["uc"].get("base_mva", 100.0)),
        seed=seed,
    )
    _write_system_tables(system, output_dir)

    scenarios = generate_uc_scenarios(system, cfg)
    LOGGER.info("Generated %d demand scenarios", len(scenarios))

    feature_rows: List[dict] = []
    commitment_rows: List[dict] = []
    dispatch_rows: List[dict] = []
    metadata_rows: List[dict] = []

    save_lp = bool(cfg["outputs"].get("save_model_lp", False))
    lp_dir = output_dir / "lp_models"
    if save_lp:
        lp_dir.mkdir(exist_ok=True)

    for scenario in scenarios:
        LOGGER.info("Solving scenario %s/%s", scenario.scenario_id + 1, len(scenarios))
        feature_rows.append(_scenario_feature_row(system, scenario, cfg))
        model, result = solve_uc_instance(system, scenario, cfg)

        if save_lp:
            try:
                model.write(str(lp_dir / f"scenario_{scenario.scenario_id}.lp"), io_options={"symbolic_solver_labels": True})
            except Exception as exc:
                LOGGER.warning("Could not write LP for scenario %s: %s", scenario.scenario_id, exc)

        commitment_rows.append(_commitment_row(result, system, cfg))
        dispatch_rows.append(_dispatch_row(result, system, cfg))
        metadata_rows.append(_metadata_row(result, system, scenario, cfg))

        LOGGER.info(
            "Scenario %s: feasible=%s, status=%s, term=%s, time=%.3fs, obj=%s",
            scenario.scenario_id,
            result.feasible,
            result.solver_status,
            result.termination_condition,
            result.solve_time_sec,
            result.objective_cost,
        )

    features_df = pd.DataFrame(feature_rows)
    commitment_df = pd.DataFrame(commitment_rows)
    dispatch_df = pd.DataFrame(dispatch_rows)
    metadata_df = pd.DataFrame(metadata_rows)

    features_df.to_csv(output_dir / "features.csv", index=False)
    commitment_df.to_csv(output_dir / "labels_commitment.csv", index=False)
    dispatch_df.to_csv(output_dir / "labels_dispatch.csv", index=False)
    metadata_df.to_csv(output_dir / "metadata.csv", index=False)

    # Save compact arrays for ML preprocessing. Infeasible rows contain NaN labels.
    feature_cols = [c for c in features_df.columns if c.startswith("total_demand_") or c.startswith("busload_")]
    commitment_cols = [c for c in commitment_df.columns if c.startswith("u_g")]
    dispatch_cols = [c for c in dispatch_df.columns if c.startswith("p_g")]
    np.save(output_dir / "features.npy", features_df[feature_cols].to_numpy(dtype=float))
    np.save(output_dir / "labels_commitment.npy", commitment_df[commitment_cols].to_numpy(dtype=float))
    np.save(output_dir / "labels_dispatch.npy", dispatch_df[dispatch_cols].to_numpy(dtype=float))

    feasible_rate = float(metadata_df["feasible"].mean()) if len(metadata_df) else 0.0
    LOGGER.info("Dataset saved to %s", output_dir)
    LOGGER.info("Feasible scenario rate: %.2f%%", 100.0 * feasible_rate)

    return output_dir
