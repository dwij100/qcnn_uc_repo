from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch

from src.models.classical_cnn import ClassicalCNN
from src.models.henderson_quanv import HendersonQuanvNet
from src.models.trainable_pqc_qcnn import TrainablePQCQCNN


def build_model(model_name: str, cfg: Dict[str, Any], feature_dim: int, num_generators: int, time_horizon: int) -> torch.nn.Module:
    name = model_name.lower()
    if name == "cnn":
        mcfg = cfg["models"]["cnn"]
        return ClassicalCNN(
            feature_dim=feature_dim,
            num_generators=num_generators,
            time_horizon=time_horizon,
            conv_channels=mcfg.get("conv_channels", [32, 64, 64]),
            hidden_dim=int(mcfg.get("hidden_dim", 256)),
            dropout=float(mcfg.get("dropout", 0.2)),
        )

    if name in {"henderson_quanv", "henderson_quanv_trainable"}:
        mcfg = cfg["models"][name]
        return HendersonQuanvNet(
            feature_dim=feature_dim,
            num_generators=num_generators,
            time_horizon=time_horizon,
            n_qubits=int(mcfg.get("n_qubits", 4)),
            n_filters=int(mcfg.get("n_filters", 4)),
            quantum_layers=int(mcfg.get("quantum_layers", 2)),
            patch_size=int(mcfg.get("patch_size", 4)),
            stride=int(mcfg.get("stride", 2)),
            trainable_filters=bool(mcfg.get("trainable_filters", name.endswith("trainable"))),
            backend=str(mcfg.get("backend", "default.qubit")),
            hidden_dim=int(mcfg.get("hidden_dim", 128)),
            dropout=float(mcfg.get("dropout", 0.15)),
        )

    if name == "pqc_qcnn":
        mcfg = cfg["models"]["pqc_qcnn"]
        return TrainablePQCQCNN(
            feature_dim=feature_dim,
            num_generators=num_generators,
            time_horizon=time_horizon,
            n_qubits=int(mcfg.get("n_qubits", 6)),
            quantum_layers=int(mcfg.get("quantum_layers", 3)),
            data_reuploading=bool(mcfg.get("data_reuploading", True)),
            backend=str(mcfg.get("backend", "default.qubit")),
            classical_hidden_dim=int(mcfg.get("classical_hidden_dim", 128)),
            dropout=float(mcfg.get("dropout", 0.15)),
        )

    raise ValueError(f"Unknown model_name={model_name}")


def count_parameters(model: torch.nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def save_checkpoint(path: str | Path, model: torch.nn.Module, metadata: Dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "metadata": metadata}, path)


def load_checkpoint(path: str | Path, model: torch.nn.Module, map_location: str | torch.device = "cpu") -> Dict[str, Any]:
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["state_dict"])
    return ckpt.get("metadata", {})
