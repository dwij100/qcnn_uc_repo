from __future__ import annotations

import torch
from torch import nn


class TrainablePQCQCNN(nn.Module):
    """Trainable PQC/QCNN-style model with selectable PennyLane conv circuits.

    Supported conv_ansatz:
        c1, c2, c5, c6, c8, c9

    Input:
        x: [batch_size, feature_dim]

    Output:
        logits: [batch_size, num_generators, time_horizon]
    """

    CONV_PARAM_COUNTS = {
        "c1": 2,
        "c2": 2,
        "c5": 6,
        "c6": 6,
        "c8": 10,
        "c9": 15,
    }

    def __init__(
        self,
        feature_dim: int,
        num_generators: int,
        time_horizon: int,
        n_qubits: int = 6,
        quantum_layers: int = 3,
        data_reuploading: bool = True,
        backend: str = "default.qubit",
        classical_hidden_dim: int = 15,
        dropout: float = 0.15,
        conv_channels: list[int] | tuple[int, ...] = 128,#(10, 20),
        use_cnn_head: bool = True,
        conv_ansatz: str = "c9",
        shared_conv_params: bool = False,
    ) -> None:
        super().__init__()

        try:
            import pennylane as qml  # noqa: F401
        except Exception as exc:
            raise ImportError(
                "PennyLane is required for TrainablePQCQCNN. Install pennylane."
            ) from exc

        self.feature_dim = int(feature_dim)
        self.num_generators = int(num_generators)
        self.time_horizon = int(time_horizon)
        self.n_qubits = int(n_qubits)
        self.quantum_layers = int(quantum_layers)
        self.data_reuploading = bool(data_reuploading)
        self.backend = backend
        self.use_cnn_head = bool(use_cnn_head)

        self.conv_ansatz = conv_ansatz.lower().strip()
        self.shared_conv_params = bool(shared_conv_params)

        if self.conv_ansatz not in self.CONV_PARAM_COUNTS:
            raise ValueError(
                f"Unknown conv_ansatz='{conv_ansatz}'. "
                f"Choose from {list(self.CONV_PARAM_COUNTS.keys())}."
            )

        self.params_per_conv = self.CONV_PARAM_COUNTS[self.conv_ansatz]

        # Same input encoder as before:
        # UC feature vector -> n_qubits quantum angles
        encoder_hidden_dim = max(32, self.n_qubits * 4)

        self.input_encoder = nn.Sequential(
            nn.Linear(self.feature_dim, encoder_hidden_dim),
            nn.Tanh(),
            nn.Linear(encoder_hidden_dim, self.n_qubits),
        )

        # Pair pattern replacing your previous CNOT-only entanglement pattern.
        # For n_qubits=6:
        #   even pairs: (0,1), (2,3), (4,5)
        #   odd pairs:  (1,2), (3,4)
        #   wrap:       (5,0)
        self.conv_pairs = self._make_conv_pairs()
        self.n_conv_blocks = len(self.conv_pairs)

        # Trainable quantum parameters.
        #
        # shared_conv_params=False:
        #   each two-qubit block has its own parameters.
        #   shape = [quantum_layers, n_conv_blocks, params_per_conv]
        #
        # shared_conv_params=True:
        #   same conv kernel reused across all two-qubit blocks.
        #   shape = [quantum_layers, params_per_conv]
        if self.shared_conv_params:
            self.q_weights = nn.Parameter(
                0.01 * torch.randn(self.quantum_layers, self.params_per_conv)
            )
        else:
            self.q_weights = nn.Parameter(
                0.01
                * torch.randn(
                    self.quantum_layers,
                    self.n_conv_blocks,
                    self.params_per_conv,
                )
            )

        self._build_qnode()

        # Same CNN/MLP head as your current model
        # if self.use_cnn_head:
        #     layers = []
        #     in_ch = 1

        #     for ch in conv_channels:
        #         ch = int(ch)
        #         layers += [
        #             nn.Conv1d(
        #                 in_channels=in_ch,
        #                 out_channels=ch,
        #                 kernel_size=3,
        #                 padding=1,
        #             ),
        #             nn.BatchNorm1d(ch),
        #             nn.ReLU(),
        #             nn.MaxPool1d(
        #                 kernel_size=2,
        #                 stride=2,
        #                 ceil_mode=True,
        #             ),
        #         ]
        #         in_ch = ch

        #     self.quantum_cnn_head = nn.Sequential(*layers)

        #     with torch.no_grad():
        #         dummy = torch.zeros(1, 1, self.n_qubits)
        #         enc_dim = int(self.quantum_cnn_head(dummy).numel())

        #     self.head = nn.Sequential(
        #         nn.Flatten(),
        #         nn.Linear(enc_dim, classical_hidden_dim),
        #         nn.ReLU(),
        #         nn.Dropout(dropout),
        #         nn.Linear(classical_hidden_dim, self.num_generators * self.time_horizon),
        #     )

        # else:
        #     self.quantum_cnn_head = nn.Identity()

        #     self.head = nn.Sequential(
        #         nn.Linear(self.n_qubits, classical_hidden_dim),
        #         nn.ReLU(),
        #         nn.Dropout(dropout),
        #         nn.Linear(classical_hidden_dim, self.num_generators * self.time_horizon),
        #     )


        # Same tiny CNN/MLP head as the minimal ClassicalCNN baseline.
        #
        # If use_cnn_head=True:
        #     quantum output -> Conv1D -> ReLU -> Global Average Pooling -> Linear
        #
        # If use_cnn_head=False:
        #     quantum output -> small MLP -> Linear
        #
        # q_out from quantum circuit has shape:
        #     [batch_size, n_qubits]
        #
        # CNN head sees it as:
        #     [batch_size, 1, n_qubits]
        if self.use_cnn_head:
            if isinstance(conv_channels, (list, tuple)):
                if len(conv_channels) == 0:
                    tiny_channels = 8
                else:
                    tiny_channels = int(conv_channels[0])
            else:
                tiny_channels = int(conv_channels)

            self.quantum_cnn_head = nn.Sequential(
                nn.Conv1d(
                    in_channels=1,
                    out_channels=tiny_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
            )

            self.head = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(float(dropout)),
                nn.Linear(
                    tiny_channels,
                    self.num_generators * self.time_horizon,
                ),
            )

        else:
            self.quantum_cnn_head = nn.Identity()

            self.head = nn.Sequential(
                nn.Linear(self.n_qubits, classical_hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(
                    classical_hidden_dim,
                    self.num_generators * self.time_horizon,
                ),
            )

    def _make_conv_pairs(self) -> list[tuple[int, int]]:
        pairs: list[tuple[int, int]] = []

        # Even pairs: (0,1), (2,3), ...
        for q in range(0, self.n_qubits - 1, 2):
            pairs.append((q, q + 1))

        # Odd pairs: (1,2), (3,4), ...
        for q in range(1, self.n_qubits - 1, 2):
            pairs.append((q, q + 1))

        # Wrap-around pair, similar to your previous last-qubit-to-first-qubit CNOT
        if self.n_qubits > 2:
            pairs.append((self.n_qubits - 1, 0))

        return pairs

    def _apply_conv_ansatz(self, qml, params: torch.Tensor, wire0: int, wire1: int) -> None:
        """Apply one selected two-qubit conv circuit in PennyLane.

        wire0 and wire1 correspond to Qiskit qubits 0 and 1 inside each local block.
        """

        if self.conv_ansatz == "c1":
            # Qiskit:
            # ry(p0, 0)
            # ry(p1, 1)
            # cx(0, 1)
            qml.RY(params[0], wires=wire0)
            qml.RY(params[1], wires=wire1)
            qml.CNOT(wires=[wire0, wire1])

        elif self.conv_ansatz == "c2":
            # Qiskit:
            # h(0), h(1)
            # cx(0, 1)
            # cx(1, 0)
            # rx(p0, 0)
            # rx(p1, 1)
            qml.Hadamard(wires=wire0)
            qml.Hadamard(wires=wire1)
            qml.CNOT(wires=[wire0, wire1])
            qml.CNOT(wires=[wire1, wire0])
            qml.RX(params[0], wires=wire0)
            qml.RX(params[1], wires=wire1)

        elif self.conv_ansatz == "c5":
            # Qiskit:
            # ry(p0, 0)
            # ry(p1, 1)
            # crx(p2, 1, 0)
            # ry(p4, 1)
            # ry(p3, 0)
            # crx(p5, 0, 1)
            qml.RY(params[0], wires=wire0)
            qml.RY(params[1], wires=wire1)
            qml.CRX(params[2], wires=[wire1, wire0])
            qml.RY(params[4], wires=wire1)
            qml.RY(params[3], wires=wire0)
            qml.CRX(params[5], wires=[wire0, wire1])

        elif self.conv_ansatz == "c6":
            # Qiskit:
            # ry(p0, 0)
            # ry(p1, 1)
            # cx(0, 1)
            # ry(p2, 0)
            # ry(p3, 1)
            # cx(0, 1)
            # ry(p4, 0)
            # ry(p5, 1)
            qml.RY(params[0], wires=wire0)
            qml.RY(params[1], wires=wire1)
            qml.CNOT(wires=[wire0, wire1])
            qml.RY(params[2], wires=wire0)
            qml.RY(params[3], wires=wire1)
            qml.CNOT(wires=[wire0, wire1])
            qml.RY(params[4], wires=wire0)
            qml.RY(params[5], wires=wire1)

        elif self.conv_ansatz == "c8":
            # Qiskit:
            # rx(p0, 0)
            # rx(p1, 1)
            # rz(p2, 0)
            # rz(p3, 1)
            # crx(p4, 1, 0)
            # crx(p5, 0, 1)
            # rx(p6, 0)
            # rx(p7, 1)
            # rz(p8, 0)
            # rz(p9, 1)
            qml.RX(params[0], wires=wire0)
            qml.RX(params[1], wires=wire1)
            qml.RZ(params[2], wires=wire0)
            qml.RZ(params[3], wires=wire1)
            qml.CRX(params[4], wires=[wire1, wire0])
            qml.CRX(params[5], wires=[wire0, wire1])
            qml.RX(params[6], wires=wire0)
            qml.RX(params[7], wires=wire1)
            qml.RZ(params[8], wires=wire0)
            qml.RZ(params[9], wires=wire1)

        elif self.conv_ansatz == "c9":
            # Qiskit:
            # u(p0, p1, p2, 0)
            # u(p3, p4, p5, 1)
            # cx(0, 1)
            # ry(p6, 0)
            # rz(p7, 1)
            # cx(1, 0)
            # ry(p8, 0)
            # cx(0, 1)
            # u(p9, p10, p11, 0)
            # u(p12, p13, p14, 1)
            qml.U3(params[0], params[1], params[2], wires=wire0)
            qml.U3(params[3], params[4], params[5], wires=wire1)
            qml.CNOT(wires=[wire0, wire1])
            qml.RY(params[6], wires=wire0)
            qml.RZ(params[7], wires=wire1)
            qml.CNOT(wires=[wire1, wire0])
            qml.RY(params[8], wires=wire0)
            qml.CNOT(wires=[wire0, wire1])
            qml.U3(params[9], params[10], params[11], wires=wire0)
            qml.U3(params[12], params[13], params[14], wires=wire1)

        else:
            raise ValueError(f"Unsupported conv_ansatz='{self.conv_ansatz}'.")

    def _build_qnode(self) -> None:
        import pennylane as qml

        self.dev = qml.device(self.backend, wires=self.n_qubits)

        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def circuit(x_angles: torch.Tensor, weights: torch.Tensor):
            for layer in range(self.quantum_layers):

                # Data encoding / data re-uploading
                if layer == 0 or self.data_reuploading:
                    for q in range(self.n_qubits):
                        qml.RY(x_angles[q], wires=q)

                # Apply selected two-qubit conv circuit across the qubit line
                for block_idx, pair in enumerate(self.conv_pairs):
                    wire0, wire1 = pair

                    if self.shared_conv_params:
                        block_params = weights[layer]
                    else:
                        block_params = weights[layer, block_idx]

                    self._apply_conv_ansatz(qml, block_params, wire0, wire1)

            return [qml.expval(qml.PauliZ(q)) for q in range(self.n_qubits)]

        self._circuit = circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Encode full UC feature vector into quantum angles
        angles = self.input_encoder(x)

        # 2. Run trainable PQC sample-by-sample
        q_out = []

        for sample_angles in angles:
            out = self._circuit(sample_angles, self.q_weights)

            if isinstance(out, (list, tuple)):
                out = torch.stack(list(out))

            q_out.append(out)

        # q_out: [batch_size, n_qubits]
        z = torch.stack(q_out, dim=0).to(x.device).float()

        # 3. CNN or MLP head
        if self.use_cnn_head:
            z = z.unsqueeze(1)  # [batch, 1, n_qubits]
            z = self.quantum_cnn_head(z)
            logits = self.head(z)
        else:
            logits = self.head(z)

        return logits.view(-1, self.num_generators, self.time_horizon)