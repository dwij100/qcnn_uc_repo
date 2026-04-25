import torch
import pennylane as qml
import matplotlib.pyplot as plt

from src.models.henderson_quanv import HendersonQuanvNet

model = HendersonQuanvNet(
    feature_dim=24,
    num_generators=10,
    time_horizon=24,
    n_qubits=4,
    n_filters=4,
    quantum_layers=2,
)

x_patch = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float32)

fig, ax = qml.draw_mpl(model._circuit, decimals=3)(
    x_patch,
    model.q_weights[0],
)

plt.savefig("henderson_quanv_circuit.png", dpi=300, bbox_inches="tight")
plt.show()