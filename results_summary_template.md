# QCNN-Assisted Unit Commitment Results Summary

## 1. Experimental setup

| Item | Description |
|---|---|
| UC cases | case10, case24, case118_reduced |
| Time horizon | 24 hours |
| Solver | Gurobi through Pyomo |
| ML framework | PyTorch |
| QML framework | PennyLane |
| Train/val/test split | Scenario-level split |
| Main threshold | 0.5 |
| Partial-fixing confidence threshold | 0.85 |

---

## 2. Dataset generation summary

| Case | Scenarios solved | Feasible solved | Mean MILP time | Mean objective | Mean gap |
|---|---:|---:|---:|---:|---:|
| case10 |  |  |  |  |  |
| case24 |  |  |  |  |  |
| case118_reduced |  |  |  |  |  |

---

## 3. Prediction performance

| Case | Model | Bitwise accuracy | Exact schedule match | F1 micro | Training time | Prediction time/sample |
|---|---|---:|---:|---:|---:|---:|
| case10 | CNN |  |  |  |  |  |
| case10 | Henderson Quanv |  |  |  |  |  |
| case10 | Trainable PQC/QCNN |  |  |  |  |  |

---

## 4. Feasibility of predicted schedules

| Case | Model | Fully feasible % | Partially feasible % | Most common violation |
|---|---|---:|---:|---|
| case10 | CNN |  |  |  |
| case10 | Henderson Quanv |  |  |  |
| case10 | Trainable PQC/QCNN |  |  |  |

---

## 5. MILP acceleration

| Case | Model | Mode | Mean solve time | Mean speedup | Mean objective deviation | Fixed binaries % |
|---|---|---|---:|---:|---:|---:|
| case10 | CNN | Full MILP |  | 1.00 | 0.00 | 0 |
| case10 | CNN | Warm start |  |  |  | 0 |
| case10 | CNN | Full fixing feasible |  |  |  | 100 |
| case10 | CNN | Partial fixing |  |  |  |  |

---

## 6. Scalability

| Case | Full MILP time | QCNN-assisted time | Speedup | Objective deviation | Feasibility rate |
|---|---:|---:|---:|---:|---:|
| case10 |  |  |  |  |  |
| case24 |  |  |  |  |  |
| case118_reduced |  |  |  |  |  |

---

## 7. Main findings

1.
2.
3.

---

## 8. Publication-ready interpretation

Write the key result here:

> The proposed QCNN-assisted MILP framework reduces UC solve time by ___% on average while preserving feasible dispatch and maintaining objective deviations below ___%.

---

## 9. Limitations

- Quantum simulations are computationally expensive on classical hardware.
- The default large case is reduced IEEE-118-inspired UC, not a full network-constrained UC.
- Feasibility screening is approximate; final feasibility is enforced by the MILP acceleration stage.

---

## 10. Next improvements

- Add PTDF-based DC network constraints.
- Add confidence-threshold sweeps.
- Add rolling-horizon UC.
- Compare with transformer or graph neural network baselines.
- Add real IEEE RTS-GMLC or MATPOWER case data.
