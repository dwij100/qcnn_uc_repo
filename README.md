# QCNN-UC Research Pipeline

This repository is a modular research pipeline for applying Quantum Convolutional Neural Networks (QCNNs) to the Unit Commitment (UC) problem.

The current version implements **Stage 1: classical MILP dataset generation**. Later stages are scaffolded and will be filled in sequentially:

1. Classical MILP UC dataset generation
2. ML/QCNN preprocessing
3. Classical CNN, Henderson-style quanvolution, and trainable PQC/QCNN models
4. Training and evaluation
5. UC feasibility checking
6. QCNN-assisted MILP warm-start and partial fixing
7. Scalability experiments

## Current working command

From the project root:

```bash
python main.py generate-data --config config/config.yaml
```

or:

```bash
python experiments/run_dataset_generation.py --config config/config.yaml
```

## Solver setup

The default solver is Gurobi:

```yaml
solver:
  name: gurobi
```

If Gurobi is not available, install HiGHS and set:

```yaml
solver:
  name: appsi_highs
```

For publication-scale experiments, Gurobi is recommended because UC is a mixed-integer optimization problem and repeated dataset generation can become expensive.

## What Stage 1 saves

Dataset outputs are saved under:

```text
data/results/<dataset_name>/
```

Files include:

- `features.csv`: scenario-level demand features
- `features.npy`: compact feature array for ML preprocessing
- `labels_commitment.csv`: optimal generator on/off schedules
- `labels_commitment.npy`: compact commitment label array
- `labels_dispatch.csv`: optimal generator dispatch schedules
- `labels_dispatch.npy`: compact dispatch label array
- `metadata.csv`: objective cost, solve time, feasibility status, termination condition, MIP gap
- `system_generators.csv`: generator parameters used by the UC MILP
- `system_buses.csv`: bus load parameters
- `system_branches.csv`: branch parameters
- `system_notes.txt`: reproducibility notes

## UC formulation currently implemented

The first working MILP formulation includes:

- commitment binaries `u[g,t]`
- startup binaries `v[g,t]`
- shutdown binaries `w[g,t]`
- dispatch variables `p[g,t]`
- generation upper/lower bounds
- demand balance
- startup/shutdown logic
- ramp-up and ramp-down constraints
- minimum up/down time constraints
- reserve margin constraints
- optional DC network constraints

The cost model is linear in this first version. Quadratic costs can be added later either as MIQP or by piecewise-linearization.

## Synthetic vs IEEE cases

The first version uses synthetic UC cases by default because they are guaranteed to run and include all UC-specific fields.

Example:

```yaml
uc:
  case_name: synthetic_10
  case_source: synthetic
  n_generators: 10
  n_buses: 5
```

For IEEE/MATPOWER-style network data, set:

```yaml
uc:
  case_name: case118
  case_source: pandapower
```

Important: standard power-flow cases usually do not include full UC parameters such as startup cost, shutdown cost, ramp limits, or minimum up/down time. This repository assigns those missing UC parameters reproducibly for experimentation, and they should be replaced with validated values before final publication claims.

## Project structure

```text
qcnn_uc_project/
├── data/
│   ├── raw/
│   ├── processed/
│   └── results/
├── config/
│   └── config.yaml
├── src/
│   ├── data_generation/
│   │   ├── load_ieee_case.py
│   │   ├── uc_milp_model.py
│   │   ├── generate_scenarios.py
│   │   └── generate_dataset.py
│   ├── preprocessing/
│   │   └── prepare_dataset.py
│   ├── models/
│   │   ├── classical_cnn.py
│   │   ├── henderson_quanv.py
│   │   ├── trainable_pqc_qcnn.py
│   │   └── model_utils.py
│   ├── training/
│   │   ├── train_model.py
│   │   └── evaluate_model.py
│   ├── feasibility/
│   │   └── check_uc_feasibility.py
│   ├── milp_acceleration/
│   │   ├── warm_start_milp.py
│   │   ├── partial_fixing_milp.py
│   │   └── compare_speedup.py
│   ├── plotting/
│   │   └── plot_results.py
│   └── utils/
│       ├── config_loader.py
│       ├── logging_utils.py
│       ├── metrics.py
│       └── seed.py
├── experiments/
│   ├── run_dataset_generation.py
│   ├── run_training.py
│   ├── run_feasibility_check.py
│   ├── run_milp_acceleration.py
│   └── run_scalability_study.py
├── requirements.txt
├── README.md
└── main.py
```

## Recommended first run

Use a small test first:

```yaml
uc:
  case_name: synthetic_10
  n_generators: 10
  n_buses: 5
  time_horizon: 12
  n_scenarios: 20
  enable_network_constraints: false
```

Then scale gradually:

1. `synthetic_10`, 12-hour horizon
2. `synthetic_10`, 24-hour horizon
3. `synthetic_24`, 24-hour horizon
4. `case118` via pandapower with network constraints enabled

## Next implementation stages

The next stage should implement `src/preprocessing/prepare_dataset.py`, then the CNN baseline, then the quantum models. The order is intentional: the ML/QCNN models need stable feature tensors and leakage-safe scenario splits before training.
