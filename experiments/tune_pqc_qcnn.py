from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any
import re
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

try:
    import optuna
except ImportError as exc:
    raise ImportError(
        "Optuna is required for hyperparameter tuning.\n"
        "Install it with:\n"
        "    pip install optuna"
    ) from exc


from src.models.trainable_pqc_qcnn import TrainablePQCQCNN


# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------
def decode_conv_channels(key: str) -> tuple[int, ...]:
    mapping = {
        "16": (16,),
        "32": (32,),
        "16_32": (16, 32),
        "32_64": (32, 64),
        "32_64_64": (32, 64, 64),
    }

    if key not in mapping:
        raise ValueError(f"Unknown conv_channels_key: {key}")

    return mapping[key]
def _sort_gon_columns(label_cols: list[str]) -> list[str]:
    """Sort labels as GOn_g0_t0 ... GOn_g0_t23, GOn_g1_t0 ..."""
    pattern = re.compile(r"^GOn_g(\d+)_t(\d+)$")

    parsed = []
    for col in label_cols:
        match = pattern.match(col)
        if match is None:
            return label_cols
        g = int(match.group(1))
        t = int(match.group(2))
        parsed.append((g, t, col))

    return [col for _, _, col in sorted(parsed)]


def load_uc_dataset(
    features_path: str,
    labels_path: str,
    num_generators: int,
    time_horizon: int,
    test_size: float,
    val_size: float,
    seed: int,
    max_samples: int | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    StandardScaler,
]:
    features_df = pd.read_csv(features_path)
    labels_df = pd.read_csv(labels_path)

    expected_output_dim = num_generators * time_horizon

    # ------------------------------------------------------------
    # Align rows safely using scenario_id
    # ------------------------------------------------------------
    if "scenario_id" in features_df.columns and "scenario_id" in labels_df.columns:
        if not np.array_equal(features_df["scenario_id"].to_numpy(), labels_df["scenario_id"].to_numpy()):
            print("scenario_id order differs between features and labels. Merging by scenario_id.")

        merged = features_df.merge(
            labels_df,
            on="scenario_id",
            how="inner",
            validate="one_to_one",
        )

        if len(merged) != len(features_df) or len(merged) != len(labels_df):
            raise ValueError(
                f"Merge mismatch: features={len(features_df)}, "
                f"labels={len(labels_df)}, merged={len(merged)}"
            )

    else:
        if len(features_df) != len(labels_df):
            raise ValueError(
                f"features and labels have different row counts: "
                f"{len(features_df)} vs {len(labels_df)}"
            )

        merged = pd.concat(
            [
                features_df.reset_index(drop=True),
                labels_df.reset_index(drop=True),
            ],
            axis=1,
        )

    # ------------------------------------------------------------
    # Optional sample limit
    # ------------------------------------------------------------
    if max_samples is not None and max_samples > 0:
        merged = merged.iloc[:max_samples].copy()

    # ------------------------------------------------------------
    # Get label columns
    # ------------------------------------------------------------
    label_cols = [
        c for c in merged.columns
        if re.match(r"^GOn_g\d+_t\d+$", c)
    ]

    label_cols = _sort_gon_columns(label_cols)

    if len(label_cols) != expected_output_dim:
        raise ValueError(
            f"Found {len(label_cols)} commitment label columns, but expected "
            f"{expected_output_dim} = {num_generators} * {time_horizon}."
        )

    labels_clean = merged[label_cols]

    # ------------------------------------------------------------
    # Get feature columns
    # ------------------------------------------------------------
    metadata_cols = {
        "scenario_id",
        "case_name",
        "case",
        "dataset",
        "system",
        "split",
        "idx",
        "index",
        "sample_id",
        "num_generators",
        "time_horizon",
    }

    # Start only from original feature columns, not merged label columns.
    feature_cols = [
        c for c in features_df.columns
        if c in merged.columns and c not in metadata_cols
    ]

    features_clean = merged[feature_cols].copy()

    # Drop non-numeric columns, e.g. case_name = "case118_reduced"
    non_numeric_features = features_clean.select_dtypes(exclude=["number"]).columns.tolist()
    if non_numeric_features:
        print(f"Dropping non-numeric feature columns: {non_numeric_features}")
        features_clean = features_clean.drop(columns=non_numeric_features)

    # Drop constant feature columns.
    # In your uploaded file, columns like solar_t0...solar_t6 and metadata constants are constant.
    constant_features = [
        c for c in features_clean.columns
        if features_clean[c].nunique(dropna=False) <= 1
    ]

    if constant_features:
        print(f"Dropping constant feature columns: {constant_features}")
        features_clean = features_clean.drop(columns=constant_features)

    if features_clean.empty:
        raise ValueError("No usable numeric feature columns remain after cleaning.")

    # ------------------------------------------------------------
    # Convert to numpy
    # ------------------------------------------------------------
    X = features_clean.to_numpy(dtype=np.float32)
    y = labels_clean.to_numpy(dtype=np.float32)

    if not np.isfinite(X).all():
        raise ValueError("Features contain NaN or infinite values.")

    if not np.isfinite(y).all():
        raise ValueError("Labels contain NaN or infinite values.")

    unique_labels = np.unique(y)
    if not set(unique_labels.tolist()).issubset({0.0, 1.0}):
        raise ValueError(f"Labels must be binary 0/1. Found values: {unique_labels}")

    print(f"Loaded dataset:")
    print(f"  samples: {X.shape[0]}")
    print(f"  features used: {X.shape[1]}")
    print(f"  labels used: {y.shape[1]}")
    print(f"  output shape per sample: [{num_generators}, {time_horizon}]")

    # ------------------------------------------------------------
    # Train/val/test split
    # ------------------------------------------------------------
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        shuffle=True,
    )

    relative_val_size = val_size / (1.0 - test_size)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=relative_val_size,
        random_state=seed,
        shuffle=True,
    )

    # ------------------------------------------------------------
    # Standardize features
    # ------------------------------------------------------------
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    # Store feature names for later inference/debugging
    scaler.feature_columns_ = features_clean.columns.tolist()

    # ------------------------------------------------------------
    # Reshape labels to [samples, G, T]
    # ------------------------------------------------------------
    y_train = y_train.reshape(-1, num_generators, time_horizon)
    y_val = y_val.reshape(-1, num_generators, time_horizon)
    y_test = y_test.reshape(-1, num_generators, time_horizon)

    return (
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32),
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32),
        scaler,
    )


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    model.eval()

    total_loss = 0.0
    total_bits = 0
    correct_bits = 0

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        logits = model(xb)
        loss = criterion(logits, yb)

        probs = torch.sigmoid(logits)
        preds = probs >= 0.5

        correct_bits += (preds == yb.bool()).sum().item()
        total_bits += yb.numel()

        total_loss += loss.item() * xb.size(0)

    avg_loss = total_loss / len(loader.dataset)
    bit_accuracy = correct_bits / total_bits

    return {
        "loss": avg_loss,
        "bit_accuracy": bit_accuracy,
    }


# ---------------------------------------------------------------------
# Training one trial
# ---------------------------------------------------------------------

def train_one_trial(
    trial: optuna.Trial,
    data: dict[str, torch.Tensor],
    config: dict[str, Any],
) -> float:
    device = torch.device(config["device"])

    X_train = data["X_train"]
    y_train = data["y_train"]
    X_val = data["X_val"]
    y_val = data["y_val"]

    feature_dim = X_train.shape[1]

    # ------------------------------------------------------------
    # Hyperparameter search space
    # ------------------------------------------------------------

    n_qubits = trial.suggest_categorical("n_qubits", [4, 6, 8, 10])
    quantum_layers = trial.suggest_int("quantum_layers", 1, 4)
    data_reuploading = trial.suggest_categorical("data_reuploading", [True, False])

    classical_hidden_dim = trial.suggest_categorical(
        "classical_hidden_dim",
        [64, 128, 256, 512],
    )

    dropout = trial.suggest_float("dropout", 0.05, 0.40)

    lr = trial.suggest_float("lr", 1e-4, 3e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-7, 1e-3, log=True)

    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])

    use_cnn_head = trial.suggest_categorical("use_cnn_head", [True, False])

    # Optuna requires a fixed search space for each parameter name.
    # Therefore, we tune a string key instead of tuple values.
    if use_cnn_head:
        conv_channels_key = trial.suggest_categorical(
            "conv_channels_key",
            [
                "16",
                "32",
                "16_32",
                "32_64",
                "32_64_64",
            ],
        )

        # Avoid silly CNN heads when the quantum output sequence is too short.
        # This keeps the search meaningful without changing Optuna's search space.
        if n_qubits <= 4 and conv_channels_key in {"16_32", "32_64", "32_64_64"}:
            raise optuna.TrialPruned()

        if n_qubits <= 6 and conv_channels_key in {"32_64_64"}:
            raise optuna.TrialPruned()

        conv_channels = decode_conv_channels(conv_channels_key)

    else:
        conv_channels_key = "none"
        conv_channels = (32, 64)

    # ------------------------------------------------------------
    # DataLoaders
    # ------------------------------------------------------------

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )

    val_loader = DataLoader(
        TensorDataset(X_val, y_val),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    # ------------------------------------------------------------
    # Model
    # ------------------------------------------------------------

    model = TrainablePQCQCNN(
        feature_dim=feature_dim,
        num_generators=config["num_generators"],
        time_horizon=config["time_horizon"],
        n_qubits=n_qubits,
        quantum_layers=quantum_layers,
        data_reuploading=data_reuploading,
        backend=config["backend"],
        classical_hidden_dim=classical_hidden_dim,
        dropout=dropout,
        conv_channels=conv_channels,
        use_cnn_head=use_cnn_head,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
    )

    best_val_loss = float("inf")
    best_val_acc = 0.0
    patience_counter = 0

    # ------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------

    for epoch in range(config["epochs"]):
        model.train()

        total_train_loss = 0.0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad(set_to_none=True)

            logits = model(xb)
            loss = criterion(logits, yb)

            loss.backward()

            if config["grad_clip"] is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=config["grad_clip"],
                )

            optimizer.step()

            total_train_loss += loss.item() * xb.size(0)

        train_loss = total_train_loss / len(train_loader.dataset)

        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )

        val_loss = val_metrics["loss"]
        val_acc = val_metrics["bit_accuracy"]

        scheduler.step(val_loss)

        trial.report(val_loss, step=epoch)

        if trial.should_prune():
            raise optuna.TrialPruned()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1

        if config["verbose"]:
            print(
                f"Trial {trial.number:03d} | "
                f"Epoch {epoch + 1:03d}/{config['epochs']} | "
                f"train_loss={train_loss:.5f} | "
                f"val_loss={val_loss:.5f} | "
                f"val_acc={val_acc:.4f}"
            )

        if patience_counter >= config["early_stopping_patience"]:
            break

    # Store extra useful values.
    trial.set_user_attr("best_val_loss", best_val_loss)
    trial.set_user_attr("best_val_bit_accuracy", best_val_acc)

    # Objective:
    # minimize validation BCE loss.
    return best_val_loss


# ---------------------------------------------------------------------
# Train final best model
# ---------------------------------------------------------------------

def train_best_model(
    best_params: dict[str, Any],
    data: dict[str, torch.Tensor],
    config: dict[str, Any],
    output_dir: Path,
) -> None:
    device = torch.device(config["device"])

    X_train = torch.cat([data["X_train"], data["X_val"]], dim=0)
    y_train = torch.cat([data["y_train"], data["y_val"]], dim=0)

    X_test = data["X_test"]
    y_test = data["y_test"]

    feature_dim = X_train.shape[1]

    batch_size = int(best_params["batch_size"])

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        TensorDataset(X_test, y_test),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    model = TrainablePQCQCNN(
        feature_dim=feature_dim,
        num_generators=config["num_generators"],
        time_horizon=config["time_horizon"],
        n_qubits=int(best_params["n_qubits"]),
        quantum_layers=int(best_params["quantum_layers"]),
        data_reuploading=bool(best_params["data_reuploading"]),
        backend=config["backend"],
        classical_hidden_dim=int(best_params["classical_hidden_dim"]),
        dropout=float(best_params["dropout"]),
        conv_channels=decode_conv_channels(best_params.get("conv_channels_key", "32_64")),
        use_cnn_head=bool(best_params["use_cnn_head"]),
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(best_params["lr"]),
        weight_decay=float(best_params["weight_decay"]),
    )

    best_train_loss = float("inf")

    for epoch in range(config["final_epochs"]):
        model.train()
        total_loss = 0.0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad(set_to_none=True)

            logits = model(xb)
            loss = criterion(logits, yb)

            loss.backward()

            if config["grad_clip"] is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=config["grad_clip"],
                )

            optimizer.step()

            total_loss += loss.item() * xb.size(0)

        train_loss = total_loss / len(train_loader.dataset)

        if train_loss < best_train_loss:
            best_train_loss = train_loss

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "best_params": best_params,
                    "config": config,
                    "feature_dim": feature_dim,
                },
                output_dir / "best_pqc_qcnn_model.pt",
            )

        print(
            f"Final training epoch {epoch + 1:03d}/{config['final_epochs']} | "
            f"train_loss={train_loss:.5f}"
        )

    test_metrics = evaluate(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
    )

    with open(output_dir / "best_pqc_qcnn_test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)

    print("\nFinal test metrics:")
    print(json.dumps(test_metrics, indent=2))


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--features", type=str, default="data/processed/features.csv")
    parser.add_argument("--labels", type=str, default="data/processed/labels_commitment.csv")

    parser.add_argument("--num-generators", type=int, required=True)
    parser.add_argument("--time-horizon", type=int, default=24)

    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--final-epochs", type=int, default=30)

    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--val-size", type=float, default=0.15)

    parser.add_argument("--max-samples", type=int, default=0)

    parser.add_argument("--backend", type=str, default="default.qubit")
    parser.add_argument("--device", type=str, default="cpu")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--early-stopping-patience", type=int, default=5)

    parser.add_argument("--study-name", type=str, default="pqc_qcnn_tuning")
    parser.add_argument("--output-dir", type=str, default="results/hparam_pqc_qcnn")
    parser.add_argument("--storage", type=str, default=None)

    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--train-best", action="store_true")

    args = parser.parse_args()

    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    max_samples = args.max_samples if args.max_samples > 0 else None

    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        scaler,
    ) = load_uc_dataset(
        features_path=args.features,
        labels_path=args.labels,
        num_generators=args.num_generators,
        time_horizon=args.time_horizon,
        test_size=args.test_size,
        val_size=args.val_size,
        seed=args.seed,
        max_samples=max_samples,
    )

    data = {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
    }

    config = {
        "num_generators": args.num_generators,
        "time_horizon": args.time_horizon,
        "epochs": args.epochs,
        "final_epochs": args.final_epochs,
        "backend": args.backend,
        "device": args.device,
        "grad_clip": args.grad_clip,
        "early_stopping_patience": args.early_stopping_patience,
        "verbose": args.verbose,
    }

    with open(output_dir / "tuning_config.json", "w") as f:
        json.dump(config, f, indent=2)

    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=3,
        interval_steps=1,
    )

    sampler = optuna.samplers.TPESampler(seed=args.seed)

    study = optuna.create_study(
        study_name=args.study_name,
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        storage=args.storage,
        load_if_exists=True if args.storage else False,
    )

    study.optimize(
        lambda trial: train_one_trial(trial, data, config),
        n_trials=args.n_trials,
    )

    print("\nBest trial:")
    print(f"  Trial number: {study.best_trial.number}")
    print(f"  Best val loss: {study.best_value:.6f}")
    print("  Best params:")
    for key, value in study.best_params.items():
        print(f"    {key}: {value}")

    # Save best params.
    with open(output_dir / "best_pqc_qcnn_params.json", "w") as f:
        json.dump(study.best_params, f, indent=2)

    # Save trial table.
    trials_df = study.trials_dataframe()
    trials_df.to_csv(output_dir / "pqc_qcnn_trials.csv", index=False)

    if args.train_best:
        train_best_model(
            best_params=study.best_params,
            data=data,
            config=config,
            output_dir=output_dir,
        )


if __name__ == "__main__":
    main()