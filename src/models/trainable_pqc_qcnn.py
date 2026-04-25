from __future__ import annotations

import torch
from torch import nn


class TrainablePQCQCNN(nn.Module):
    """Trainable PQC/QCNN-style model for UC commitment prediction.

    Pipeline:
    1. Classical linear encoder maps the scenario feature vector to n_qubits angles.
    2. A trainable PQC uses angle encoding, entanglement, and optional data re-uploading.
    3. Quantum expectation values are passed to a classical head for G x T logits.
    """

    def __init__(
        self,
        feature_dim: int,
        num_generators: int,
        time_horizon: int,
        n_qubits: int = 6,
        quantum_layers: int = 3,
        data_reuploading: bool = True,
        backend: str = "default.qubit",
        classical_hidden_dim: int = 128,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        try:
            import pennylane as qml  # noqa: F401
        except Exception as exc:  # pragma: no cover
            raise ImportError("PennyLane is required for TrainablePQCQCNN. Install pennylane.") from exc

        self.feature_dim = int(feature_dim)
        self.num_generators = int(num_generators)
        self.time_horizon = int(time_horizon)
        self.n_qubits = int(n_qubits)
        self.quantum_layers = int(quantum_layers)
        self.data_reuploading = bool(data_reuploading)
        self.backend = backend

        self.input_encoder = nn.Sequential(
            nn.Linear(self.feature_dim, max(32, self.n_qubits * 4)),
            nn.Tanh(),
            nn.Linear(max(32, self.n_qubits * 4), self.n_qubits),
        )

        self.q_weights = nn.Parameter(0.01 * torch.randn(self.quantum_layers, self.n_qubits, 3))
        self._build_qnode()

        self.head = nn.Sequential(
            nn.Linear(self.n_qubits, classical_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(classical_hidden_dim, self.num_generators * self.time_horizon),
        )

    def _build_qnode(self) -> None:
        import pennylane as qml

        self.dev = qml.device(self.backend, wires=self.n_qubits)

        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def circuit(x_angles: torch.Tensor, weights: torch.Tensor):
            for layer in range(self.quantum_layers):
                if layer == 0 or self.data_reuploading:
                    for q in range(self.n_qubits):
                        qml.RY(x_angles[q], wires=q)

                for q in range(self.n_qubits):
                    qml.Rot(weights[layer, q, 0], weights[layer, q, 1], weights[layer, q, 2], wires=q)

                # QCNN/MERA-inspired local entanglement pattern.
                for q in range(0, self.n_qubits - 1, 2):
                    qml.CNOT(wires=[q, q + 1])
                for q in range(1, self.n_qubits - 1, 2):
                    qml.CNOT(wires=[q, q + 1])
                if self.n_qubits > 2:
                    qml.CNOT(wires=[self.n_qubits - 1, 0])

            return [qml.expval(qml.PauliZ(q)) for q in range(self.n_qubits)]

        self._circuit = circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        angles = self.input_encoder(x)
        q_out = []
        for sample_angles in angles:
            out = self._circuit(sample_angles, self.q_weights)
            if isinstance(out, (list, tuple)):
                out = torch.stack(list(out))
            q_out.append(out)
        z = torch.stack(q_out, dim=0).to(x.device).float()
        logits = self.head(z)
        return logits.view(-1, self.num_generators, self.time_horizon)
