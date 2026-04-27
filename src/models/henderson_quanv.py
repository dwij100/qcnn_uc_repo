# from __future__ import annotations

# import math
# from typing import Iterable

# import torch
# from torch import nn


# class HendersonQuanvNet(nn.Module):
#     """Henderson-style quanvolutional model for vector UC features.

#     This adapts the image-patch idea in Henderson et al. (2019) to 1D UC
#     scenario features: local feature patches are passed through small quantum
#     circuits, and quantum expectation values feed a classical classifier head.

#     The implementation is intentionally simple and research-transparent. It is
#     slower than pure PyTorch because it evaluates quantum circuits patch by patch.
#     Use max_train_samples in config during development.
#     """

#     def __init__(
#         self,
#         feature_dim: int,
#         num_generators: int,
#         time_horizon: int,
#         n_qubits: int = 4,
#         n_filters: int = 4,
#         quantum_layers: int = 2,
#         patch_size: int = 4,
#         stride: int = 2,
#         trainable_filters: bool = False,
#         backend: str = "default.qubit",
#         hidden_dim: int = 128,
#         dropout: float = 0.15,
#     ) -> None:
#         super().__init__()
#         try:
#             import pennylane as qml  # noqa: F401
#         except Exception as exc:  # pragma: no cover
#             raise ImportError("PennyLane is required for HendersonQuanvNet. Install pennylane.") from exc

#         self.feature_dim = int(feature_dim)
#         self.num_generators = int(num_generators)
#         self.time_horizon = int(time_horizon)
#         self.n_qubits = int(n_qubits)
#         self.n_filters = int(n_filters)
#         self.quantum_layers = int(quantum_layers)
#         self.patch_size = int(patch_size)
#         self.stride = int(stride)
#         self.trainable_filters = bool(trainable_filters)
#         self.backend = backend

#         if self.patch_size > self.feature_dim:
#             self.patch_size = self.feature_dim

#         self.n_patches = 1 + max(0, math.floor((self.feature_dim - self.patch_size) / self.stride))

#         # Each filter is a separate random circuit. If trainable_filters=True,
#         # these angles receive gradients through PennyLane.
#         init = 0.01 * torch.randn(self.n_filters, self.quantum_layers, self.n_qubits, 3)
#         if self.trainable_filters:
#             self.q_weights = nn.Parameter(init)
#         else:
#             self.register_buffer("q_weights", init)

#         self._build_qnode()

#         q_feature_dim = self.n_patches * self.n_filters * self.n_qubits
#         self.head = nn.Sequential(
#             nn.Linear(q_feature_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim, self.num_generators * self.time_horizon),
#         )

#     def _build_qnode(self) -> None:
#         import pennylane as qml

#         self.dev = qml.device(self.backend, wires=self.n_qubits)

#         @qml.qnode(self.dev, interface="torch", diff_method="backprop")
#         def circuit(x_patch: torch.Tensor, weights: torch.Tensor):
#             # Pad or truncate the patch to n_qubits.
#             for i in range(self.n_qubits):
#                 val = x_patch[i] if i < x_patch.shape[0] else torch.tensor(0.0, device=x_patch.device)
#                 qml.RY(val, wires=i)

#             for layer in range(self.quantum_layers):
#                 for q in range(self.n_qubits):
#                     qml.Rot(weights[layer, q, 0], weights[layer, q, 1], weights[layer, q, 2], wires=q)
#                 for q in range(self.n_qubits - 1):
#                     qml.CNOT(wires=[q, q + 1])
#                 if self.n_qubits > 2:
#                     qml.CNOT(wires=[self.n_qubits - 1, 0])

#             return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

#         self._circuit = circuit

#     def _patches(self, x: torch.Tensor) -> Iterable[torch.Tensor]:
#         for start in range(0, self.feature_dim - self.patch_size + 1, self.stride):
#             yield x[start : start + self.patch_size]

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         batch_features = []
#         for sample in x:
#             sample_q = []
#             for patch in self._patches(sample):
#                 # If patch_size < n_qubits, pad at the tensor level.
#                 if patch.numel() < self.n_qubits:
#                     patch = torch.nn.functional.pad(patch, (0, self.n_qubits - patch.numel()))
#                 elif patch.numel() > self.n_qubits:
#                     patch = patch[: self.n_qubits]

#                 for f in range(self.n_filters):
#                     out = self._circuit(patch, self.q_weights[f])
#                     if isinstance(out, (list, tuple)):
#                         out = torch.stack(list(out))
#                     sample_q.append(out)
#             batch_features.append(torch.cat(sample_q))

#         z = torch.stack(batch_features, dim=0).to(x.device).float()
#         logits = self.head(z)
#         return logits.view(-1, self.num_generators, self.time_horizon)
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class GateOp:
    name: str
    wires: tuple[int, ...]
    params: tuple[float, ...] = ()


class HendersonQuanvNet(nn.Module):
    """
    Fast Henderson-style quanvolutional model for Unit Commitment.

    Drop-in compatibility with the earlier HendersonQuanvNet:
        - Same class name.
        - Same core constructor arguments.
        - Same forward(x) output shape:
              [batch_size, num_generators, time_horizon]
        - Still exposes q_weights so old state/inspection code does not crash.
        - Adds a fast lookup-table quanvolutional layer.

    Paper-inspired architecture:
        UC feature vector
            -> threshold/basis encoding into local patches
            -> random quantum filters
            -> scalar decoded quanvolutional feature maps
            -> Conv1D + Pool1D
            -> Conv1D + Pool1D
            -> FC1 + Dropout + FC2
            -> G x T logits

    Important:
        This is intentionally a FAST Henderson-style implementation.
        The quantum filters are static random filters, not variationally trained.
        This matches the original Henderson idea more closely than a trainable PQC.

    During training:
        No quantum circuit is executed.
        The quantum filter outputs are obtained from a precomputed lookup table.
    """

    def __init__(
        self,
        feature_dim: int,
        num_generators: int,
        time_horizon: int,
        n_qubits: Optional[int] = None,
        n_filters: int = 25,
        quantum_layers: int = 2,
        patch_size: int = 9,
        stride: int = 1,
        trainable_filters: bool = False,
        backend: str = "default.qubit",
        hidden_dim: int = 128,
        dropout: float = 0.15,
        *,
        threshold: float = 0.0,
        connection_probability: float = 0.15,
        seed: int = 123,
        normalize_decode: bool = True,
        lut_cache_path: Optional[str] = "cache/henderson_quanv_lut.pt",
        conv_channels: list[int] | tuple[int, ...] = (32, 64, 64),
        conv1_filters: int = 50,
        conv2_filters: int = 64,
        conv_kernel_size: int = 5,
        fc_hidden_dim: Optional[int] = None,
        max_lut_qubits: int = 12,
        use_conv_head: bool = False#True,
    ) -> None:
        super().__init__()

        try:
            import pennylane as qml  # noqa: F401
        except Exception as exc:
            raise ImportError(
                "PennyLane is required for HendersonQuanvNet. "
                "Install it with: pip install pennylane"
            ) from exc

        self.feature_dim = int(feature_dim)
        self.num_generators = int(num_generators)
        self.time_horizon = int(time_horizon)

        self.patch_size = int(min(patch_size, self.feature_dim))
        self.stride = int(stride)

        if self.patch_size <= 0:
            raise ValueError("patch_size must be positive.")

        if self.stride <= 0:
            raise ValueError("stride must be positive.")

        # Drop-in behavior:
        # Old model had n_qubits as an explicit argument.
        # For Henderson paper style, n_qubits should usually equal patch_size.
        if n_qubits is None:
            self.n_qubits = min(self.patch_size, max_lut_qubits)
        else:
            self.n_qubits = min(int(n_qubits), self.patch_size, max_lut_qubits)

        if self.n_qubits <= 0:
            raise ValueError("n_qubits must be positive.")

        self.n_filters = int(n_filters)
        self.quantum_layers = int(quantum_layers)
        self.trainable_filters = bool(trainable_filters)
        self.backend = str(backend)

        self.threshold = float(threshold)
        self.connection_probability = float(connection_probability)
        self.seed = int(seed)
        self.normalize_decode = bool(normalize_decode)
        self.max_lut_qubits = int(max_lut_qubits)
        self.use_conv_head = bool(use_conv_head)

        self.n_patches = 1 + max(
            0,
            math.floor((self.feature_dim - self.patch_size) / self.stride),
        )

        if self.n_patches <= 0:
            raise ValueError(
                f"Invalid patch setup: feature_dim={self.feature_dim}, "
                f"patch_size={self.patch_size}, stride={self.stride}"
            )

        # Kept for backward compatibility with old code.
        # In this fast Henderson implementation, these are not used for training.
        # The actual filters are random gate lists stored in self._filter_ops.
        q_init = 0.01 * torch.randn(
            self.n_filters,
            max(1, self.quantum_layers),
            self.n_qubits,
            3,
        )

        if self.trainable_filters:
            self.q_weights = nn.Parameter(q_init)
        else:
            self.register_buffer("q_weights", q_init)

        # Powers for binary patch -> integer lookup index.
        powers = 2 ** torch.arange(self.n_qubits, dtype=torch.long)
        self.register_buffer("pattern_powers", powers)

        self._filter_ops = self._make_all_filter_specs()

        lut = self._load_or_build_lut(lut_cache_path)
        self.register_buffer("quanv_lut", lut.float())

        # Debug QNode for printing/drawing one quantum filter.
        # Not used during training.
        self._build_debug_qnode(filter_idx=0)

        q_feature_dim = self.n_patches * self.n_filters

        if fc_hidden_dim is None:
            # hidden_dim is from the old implementation.
            # fc_hidden_dim is the paper-style FC1 size override.
            fc_hidden_dim = int(hidden_dim)

        if self.use_conv_head:
            # padding = conv_kernel_size // 2

            # self.classical_conv_stack = nn.Sequential(
            #     nn.Conv1d(
            #         in_channels=self.n_filters,
            #         out_channels=conv1_filters,
            #         kernel_size=conv_kernel_size,
            #         padding=padding,
            #     ),
            #     nn.ReLU(),
            #     nn.MaxPool1d(kernel_size=2, stride=2, ceil_mode=True),
            #     nn.Conv1d(
            #         in_channels=conv1_filters,
            #         out_channels=conv2_filters,
            #         kernel_size=conv_kernel_size,
            #         padding=padding,
            #     ),
            #     nn.ReLU(),
            #     nn.MaxPool1d(kernel_size=2, stride=2, ceil_mode=True),
            # )

            # with torch.no_grad():
            #     dummy = torch.zeros(1, self.n_filters, self.n_patches)
            #     flat_dim = self.classical_conv_stack(dummy).flatten(start_dim=1).shape[1]

            # self.head = nn.Sequential(
            #     nn.Linear(flat_dim, fc_hidden_dim),
            #     nn.ReLU(),
            #     nn.Dropout(dropout),
            #     nn.Linear(fc_hidden_dim, self.num_generators * self.time_horizon),
            # )
            # ---------------------------------------------------------------------
            # Classical CNN head after Henderson-style quanvolution
            # ---------------------------------------------------------------------
            # _quanvolution(x) returns:
            #     [batch_size, n_filters, n_patches]
            #
            # So the CNN sees n_filters channels, not 1 channel.

            layers = []
            in_ch = self.n_filters

            for ch in conv_channels:
                ch = int(ch)
                layers += [
                    nn.Conv1d(
                        in_channels=in_ch,
                        out_channels=ch,
                        kernel_size=3,
                        padding=1,
                    ),
                    nn.BatchNorm1d(ch),
                    nn.ReLU(),
                    nn.MaxPool1d(
                        kernel_size=2,
                        stride=2,
                        ceil_mode=True,
                    ),
                ]
                in_ch = ch

            self.classical_conv_stack = nn.Sequential(*layers)

            with torch.no_grad():
                dummy = torch.zeros(1, self.n_filters, self.n_patches)
                enc_dim = int(self.classical_conv_stack(dummy).numel())

            self.head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(enc_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, self.num_generators * self.time_horizon),
            )

        else:
            # Legacy-style simple MLP head.
            # This is useful if you want the old external behavior with less architecture change.
            self.classical_conv_stack = nn.Identity()
            self.head = nn.Sequential(
                nn.Linear(q_feature_dim, fc_hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(fc_hidden_dim, self.num_generators * self.time_horizon),
            )

    # ---------------------------------------------------------------------
    # Random Henderson-style quantum filter generation
    # ---------------------------------------------------------------------

    def _make_all_filter_specs(self) -> list[list[GateOp]]:
        rng = np.random.default_rng(self.seed)
        return [self._make_random_filter_specs(rng) for _ in range(self.n_filters)]

    def _make_random_filter_specs(self, rng: np.random.Generator) -> list[GateOp]:
        """
        Generate one random quanvolutional filter.

        Paper-inspired version:
            - each qubit is a graph node
            - random 2-qubit gates are added with a connection probability
            - random 1-qubit gates are added
            - the full gate list is shuffled
        """

        ops: list[GateOp] = []

        one_qubit_gates = [
            "RX",
            "RY",
            "RZ",
            "Rot",
            "PhaseShift",
            "T",
            "Hadamard",
        ]

        two_qubit_gates = [
            "CNOT",
            "SWAP",
            "CZ",
            "CRot",
        ]

        # Henderson used a random number in [0, 2n^2].
        # Here n_qubits already equals the flattened local patch size.
        max_one_qubit_gates = 2 * self.n_qubits
        n_one_qubit_gates = int(rng.integers(0, max_one_qubit_gates + 1))

        for _ in range(n_one_qubit_gates):
            gate = str(rng.choice(one_qubit_gates))
            q = int(rng.integers(0, self.n_qubits))

            if gate in {"RX", "RY", "RZ", "PhaseShift"}:
                params = (float(rng.uniform(0.0, 2.0 * np.pi)),)
            elif gate == "Rot":
                params = tuple(float(v) for v in rng.uniform(0.0, 2.0 * np.pi, size=3))
            else:
                params = ()

            ops.append(GateOp(name=gate, wires=(q,), params=params))

        for i in range(self.n_qubits):
            for j in range(i + 1, self.n_qubits):
                if rng.random() < self.connection_probability:
                    gate = str(rng.choice(two_qubit_gates))

                    if rng.random() < 0.5:
                        wires = (i, j)
                    else:
                        wires = (j, i)

                    if gate == "CRot":
                        params = tuple(float(v) for v in rng.uniform(0.0, 2.0 * np.pi, size=3))
                    else:
                        params = ()

                    ops.append(GateOp(name=gate, wires=wires, params=params))

        if len(ops) > 1:
            order = rng.permutation(len(ops))
            ops = [ops[int(i)] for i in order]

        return ops

    @staticmethod
    def _apply_gate(qml: Any, op: GateOp) -> None:
        name = op.name
        wires = list(op.wires)

        if name == "RX":
            qml.RX(op.params[0], wires=wires[0])
        elif name == "RY":
            qml.RY(op.params[0], wires=wires[0])
        elif name == "RZ":
            qml.RZ(op.params[0], wires=wires[0])
        elif name == "Rot":
            qml.Rot(op.params[0], op.params[1], op.params[2], wires=wires[0])
        elif name == "PhaseShift":
            qml.PhaseShift(op.params[0], wires=wires[0])
        elif name == "T":
            qml.T(wires=wires[0])
        elif name == "Hadamard":
            qml.Hadamard(wires=wires[0])
        elif name == "CNOT":
            qml.CNOT(wires=wires)
        elif name == "SWAP":
            qml.SWAP(wires=wires)
        elif name == "CZ":
            qml.CZ(wires=wires)
        elif name == "CRot":
            qml.CRot(op.params[0], op.params[1], op.params[2], wires=wires)
        else:
            raise ValueError(f"Unsupported quantum gate: {name}")

    # ---------------------------------------------------------------------
    # Lookup table construction
    # ---------------------------------------------------------------------

    def _cache_metadata(self) -> dict[str, Any]:
        return {
            "n_filters": self.n_filters,
            "patch_size": self.patch_size,
            "n_qubits": self.n_qubits,
            "threshold": self.threshold,
            "connection_probability": self.connection_probability,
            "seed": self.seed,
            "normalize_decode": self.normalize_decode,
            "backend": self.backend,
        }

    def _cache_matches(self, payload: dict[str, Any]) -> bool:
        meta = payload.get("metadata", {})
        expected = self._cache_metadata()

        for key, value in expected.items():
            if meta.get(key) != value:
                return False

        lut = payload.get("lut", None)
        if lut is None:
            return False

        expected_shape = (self.n_filters, 2 ** self.n_qubits)
        return tuple(lut.shape) == expected_shape

    def _load_or_build_lut(self, lut_cache_path: Optional[str]) -> torch.Tensor:
        if lut_cache_path is not None:
            cache_path = Path(lut_cache_path)

            if cache_path.exists():
                try:
                    payload = torch.load(cache_path, map_location="cpu")

                    if isinstance(payload, dict) and self._cache_matches(payload):
                        return payload["lut"].float()

                    if torch.is_tensor(payload):
                        expected_shape = (self.n_filters, 2 ** self.n_qubits)
                        if tuple(payload.shape) == expected_shape:
                            return payload.float()

                except Exception:
                    # If cache is corrupted or incompatible, rebuild automatically.
                    pass

        lut = self._build_quanvolution_lut()

        if lut_cache_path is not None:
            cache_path = Path(lut_cache_path)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "metadata": self._cache_metadata(),
                    "lut": lut.cpu(),
                },
                cache_path,
            )

        return lut.float()

    def _build_quanvolution_lut(self) -> torch.Tensor:
        """
        Precompute all possible quantum outputs.

        For n_qubits = 9:
            number of binary patterns = 2^9 = 512

        LUT shape:
            [n_filters, 2^n_qubits]
        """

        import pennylane as qml

        num_patterns = 2 ** self.n_qubits
        lut = np.zeros((self.n_filters, num_patterns), dtype=np.float32)

        for filter_idx, ops in enumerate(self._filter_ops):
            dev = qml.device(self.backend, wires=self.n_qubits)

            @qml.qnode(dev)
            def circuit(bits: list[int]):
                # Basis encoding:
                # 0 -> |0>
                # 1 -> |1> via X gate
                for q in range(self.n_qubits):
                    if int(bits[q]) == 1:
                        qml.PauliX(wires=q)

                for op in ops:
                    self._apply_gate(qml, op)

                return qml.probs(wires=list(range(self.n_qubits)))

            for pattern_idx in range(num_patterns):
                bits = [(pattern_idx >> q) & 1 for q in range(self.n_qubits)]

                probs = np.asarray(circuit(bits), dtype=np.float64)
                most_likely_state = int(np.argmax(probs))

                # Henderson-style scalar decoding:
                # pick most likely output bitstring, then count number of |1> states.
                decoded_value = most_likely_state.bit_count()

                if self.normalize_decode:
                    decoded_value = decoded_value / self.n_qubits

                lut[filter_idx, pattern_idx] = float(decoded_value)

        return torch.from_numpy(lut)

    # ---------------------------------------------------------------------
    # Fast quanvolution layer
    # ---------------------------------------------------------------------

    def _quanvolution(self, x: torch.Tensor) -> torch.Tensor:
        """
        Fast lookup-table quanvolution.

        Input:
            x: [batch_size, feature_dim]

        Output:
            quanv_maps: [batch_size, n_filters, n_patches]
        """

        if x.ndim != 2:
            raise ValueError(
                f"HendersonQuanvNet expects x with shape [batch_size, feature_dim], "
                f"but got {tuple(x.shape)}"
            )

        if x.shape[1] != self.feature_dim:
            raise ValueError(
                f"Expected feature_dim={self.feature_dim}, but got {x.shape[1]}"
            )

        batch_size = x.shape[0]

        # Threshold encoding:
        # x > threshold -> 1
        # x <= threshold -> 0
        bits = (x > self.threshold).long()
        # print(self.threshold)
        # [B, n_patches, patch_size]
        patches = bits.unfold(
            dimension=1,
            size=self.patch_size,
            step=self.stride,
        )

        # Use first n_qubits from each patch.
        # This keeps old compatibility where patch_size and n_qubits could differ.
        quantum_bits = patches[..., : self.n_qubits]

        # Convert binary pattern to integer index.
        # [B, n_patches]
        pattern_indices = torch.sum(
            quantum_bits * self.pattern_powers.view(1, 1, -1),
            dim=-1,
        )

        flat_indices = pattern_indices.reshape(-1)

        # self.quanv_lut: [n_filters, 2^n_qubits]
        # selected: [n_filters, B*n_patches]
        selected = self.quanv_lut[:, flat_indices]

        # [B, n_patches, n_filters]
        selected = selected.T.reshape(batch_size, self.n_patches, self.n_filters)

        # Conv1D expects [B, channels, length]
        quanv_maps = selected.permute(0, 2, 1).contiguous()

        return quanv_maps.float()

    def extract_quantum_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Public helper.

        Returns the quanvolutional maps before the classical CNN head:
            [batch_size, n_filters, n_patches]
        """

        return self._quanvolution(x)

    def _patches(self, x: torch.Tensor) -> Iterable[torch.Tensor]:
        """
        Backward-compatible helper from the old implementation.

        It yields 1D patches from a single feature vector.
        """

        for start in range(0, self.feature_dim - self.patch_size + 1, self.stride):
            yield x[start : start + self.patch_size]

    # ---------------------------------------------------------------------
    # Debug/printing helpers
    # ---------------------------------------------------------------------

    def _build_debug_qnode(self, filter_idx: int = 0) -> None:
        """
        Builds self._circuit for backward-compatible circuit drawing.

        This is not used during forward training.
        """

        import pennylane as qml

        filter_idx = int(filter_idx)
        filter_idx = max(0, min(filter_idx, self.n_filters - 1))

        ops = self._filter_ops[filter_idx]
        dev = qml.device(self.backend, wires=self.n_qubits)

        @qml.qnode(dev, interface="torch")
        def circuit(x_patch: torch.Tensor, weights: Optional[torch.Tensor] = None):
            # Threshold/basis encoding.
            # weights is accepted only for old compatibility.
            for q in range(self.n_qubits):
                val = 0.0
                if q < x_patch.numel():
                    val = float(x_patch[q].detach().cpu())

                if val > self.threshold:
                    qml.PauliX(wires=q)

            for op in ops:
                self._apply_gate(qml, op)

            return [qml.expval(qml.PauliZ(q)) for q in range(self.n_qubits)]

        self._circuit = circuit

    def quantum_circuit_text(
        self,
        filter_idx: int = 0,
        x_patch: Optional[torch.Tensor] = None,
        decimals: int = 3,
    ) -> str:
        """
        Returns a printable text drawing of one random quantum filter.
        """

        import pennylane as qml

        self._build_debug_qnode(filter_idx=filter_idx)

        if x_patch is None:
            x_patch = torch.ones(self.n_qubits)

        return qml.draw(self._circuit, decimals=decimals)(x_patch, None)

    def print_quantum_circuit(
        self,
        filter_idx: int = 0,
        x_patch: Optional[torch.Tensor] = None,
        decimals: int = 3,
    ) -> None:
        """
        Prints one quanvolutional filter circuit.
        """

        print(
            self.quantum_circuit_text(
                filter_idx=filter_idx,
                x_patch=x_patch,
                decimals=decimals,
            )
        )

    # ---------------------------------------------------------------------
    # Forward
    # ---------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input:
            x: [batch_size, feature_dim]

        Output:
            logits: [batch_size, num_generators, time_horizon]

        Use with:
            BCEWithLogitsLoss
        """

        quanv_maps = self._quanvolution(x)

        if self.use_conv_head:
            z = self.classical_conv_stack(quanv_maps)
            z = z.flatten(start_dim=1)
        else:
            z = quanv_maps.flatten(start_dim=1)

        logits = self.head(z)

        return logits.view(-1, self.num_generators, self.time_horizon)


# Optional alias.
# This lets old or new code import either name.
FastHendersonQuanvNet = HendersonQuanvNet