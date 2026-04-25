# QCNN-Assisted Unit Commitment Research Pipeline

This repository implements a complete Python pipeline for using classical CNN and quantum/QCNN-style models to predict Unit Commitment (UC) binary schedules, then using those predictions to accelerate MILP solving.

The publication idea is:

> Train ML/QML models to predict generator on/off decisions from UC scenario features, then use the predicted binaries as warm starts or confidence-based partial fixing inside the MILP. The final MILP preserves feasibility and gives objective-quality comparisons against the full MILP baseline.

---

## 1. Project structure

```text
qcnn_uc_project/
├── config/config.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── results/
├── src/
│   ├── data_generation/
│   ├── preprocessing/
│   ├── models/
│   ├── training/
│   ├── feasibility/
│   ├── milp_acceleration/
│   ├── plotting/
│   └── utils/
├── experiments/
├── requirements.txt
├── README.md
└── results_summary_template.md
```

---

## 2. Installation

Create and activate an environment, then install dependencies:

```bash
pip install -r requirements.txt
```

You also need a working Gurobi installation and license for MILP solving.

Check Gurobi:

```bash
python -c "import gurobipy as gp; print(gp.gurobi.version())"
```

---

## 3. UC formulation implemented

The MILP includes:

- generator on/off binaries
- dispatch variables
- startup/shutdown variables
- generation minimum and maximum limits
- demand balance
- reserve margin
- ramp-up/ramp-down constraints
- startup and shutdown costs
- no-load cost
- marginal generation cost
- minimum up/down time constraints
- reduced network skeleton files for future PTDF/network extension

Objective:

```text
min sum_t sum_g (
    marginal_cost_g * p_g,t
  + no_load_cost_g * u_g,t
  + startup_cost_g * startup_g,t
  + shutdown_cost_g * shutdown_g,t
)
```

The current default is a single-bus UC formulation. The `case118_reduced` case keeps a 118-bus-inspired skeleton but does not activate network constraints by default, so large-scale UC experiments remain tractable.

---

## 4. Dataset generation

Run:

```bash
python experiments/run_dataset_generation.py
```

Or specify case and sample count:

```bash
python experiments/run_dataset_generation.py --case case10 --n-scenarios 100
python experiments/run_dataset_generation.py --case case24 --n-scenarios 60
python experiments/run_dataset_generation.py --case case118_reduced --n-scenarios 20
```

Outputs are saved to:

```text
data/processed/<case_name>/
├── features.csv
├── labels_commitment.csv
├── dispatch.csv
├── milp_summary.csv
├── generators.csv
├── buses_reduced.csv
└── lines_reduced.csv
```

---

## 5. Preprocessing

Run:

```bash
python experiments/run_preprocessing.py --case case10
```

This creates:

```text
data/processed/<case_name>/
├── train.npz
├── val.npz
├── test.npz
├── scaler.joblib
└── preprocessing_metadata.json
```

Important: the split is by `scenario_id`, not by random rows, to avoid leakage.

Label tensor shape:

```text
Y = samples × num_generators × time_horizon
```

---

## 6. Train models

### Classical CNN

```bash
python experiments/run_training_cnn.py --case case10
```

### Henderson-style quanvolution

Fixed random quantum filters:

```bash
python experiments/run_training_henderson_quanv.py --case case10
```

Trainable quantum filters:

```bash
python experiments/run_training_henderson_quanv.py --case case10 --trainable
```

### Trainable PQC/QCNN

```bash
python experiments/run_training_pqc_qcnn.py --case case10
```

Quantum models are slower than the CNN. For development, the config includes:

```yaml
max_train_samples: 256
max_val_samples: 128
```

Increase these for final publication experiments.

---

## 7. Evaluation outputs

For each model:

```text
data/results/<case_name>/<model_name>/
├── best_model.pt
├── history.csv
├── test_metrics.csv
├── predictions_test.csv
├── targets_test.csv
├── per_generator_accuracy.csv
└── per_time_accuracy.csv
```

Metrics include:

- train/validation loss
- bitwise accuracy
- per-generator accuracy
- per-time-step accuracy
- exact schedule match accuracy
- F1 score
- training time
- prediction time

---

## 8. Feasibility checking

Run:

```bash
python experiments/run_feasibility_check.py --case case10 --model cnn
```

Outputs:

```text
data/results/<case_name>/<model_name>/feasibility/
├── feasible_predictions.csv
├── infeasible_predictions.csv
├── feasibility_summary.csv
├── feasibility_by_scenario.csv
└── violation_breakdown.csv
```

Checks include:

- demand capacity coverage
- reserve margin
- minimum generation exceeding demand
- approximate dispatch feasibility
- ramp feasibility
- minimum up/down violations

---

## 9. QCNN-assisted MILP acceleration

Run:

```bash
python experiments/run_milp_acceleration.py --case case10 --model cnn
```

Modes compared:

1. full MILP from scratch
2. MILP with predicted binary warm start
3. full binary fixing only if predicted schedule is feasible
4. confidence-based partial fixing

Confidence:

```text
confidence = abs(probability - 0.5) * 2
```

For partial fixing:

```text
fix u_g,t only if confidence_g,t >= confidence_threshold
```

Outputs:

```text
data/results/<case_name>/<model_name>/milp_acceleration/
├── acceleration_results.csv
└── acceleration_summary.csv
```

Metrics include:

- solve time
- objective cost
- solver gap
- solver status
- feasibility status
- number and percentage of binaries fixed
- speedup vs full MILP
- cost deviation vs full MILP

---

## 10. Scalability study

Run:

```bash
python experiments/run_scalability_study.py --model cnn
```

This loops through:

- `case10`
- `case24`
- `case118_reduced`

Outputs:

```text
data/results/scalability/
├── scalability_results.csv
└── *.png
```

---

## 11. Run the full pipeline

For quick smoke testing:

```bash
python experiments/run_all.py --case case10 --n-scenarios 20 --skip-quantum
```

For a complete run using the configured models:

```bash
python experiments/run_all.py
```

---

## 12. Notes for publication experiments

Recommended comparisons:

- CNN vs Henderson fixed quanvolution vs Henderson trainable quanvolution vs trainable PQC/QCNN
- random quantum filters vs trainable quantum filters
- different encodings and qubit counts
- prediction accuracy vs feasibility rate
- feasibility rate vs confidence threshold
- full MILP vs warm start vs full fixing vs partial fixing
- speedup vs objective deviation
- scalability across 10-generator, 24-generator, and reduced 118-bus-inspired cases

Recommended ablation table:

| Experiment | Metric |
|---|---|
| CNN vs QML models | bitwise accuracy, exact schedule accuracy, F1 |
| Prediction quality vs feasibility | feasibility rate, violation breakdown |
| MILP acceleration | solve time, speedup, objective deviation |
| Confidence threshold sweep | fixed binaries %, feasibility, speedup |
| Scalability | runtime and feasibility trends |

---

## 13. Common issues

### Gurobi unavailable

If dataset generation returns `solver_unavailable`, check that Gurobi is installed and licensed.

### Quantum training too slow

Reduce:

```yaml
models:
  henderson_quanv:
    max_train_samples: 64
    max_val_samples: 32
```

Or run only CNN first:

```bash
python experiments/run_all.py --skip-quantum
```
Or run each of the steps manually with
```bash
python experiments/run_dataset_generation.py --case case10 --n-scenarios 100
python experiments/run_preprocessing.py --case case10
python experiments/run_training_cnn.py --case case10
python experiments/run_training_henderson_quanv.py --case case10
python experiments/run_training_henderson_quanv.py --case case10 --trainable
python experiments/run_training_pqc_qcnn.py --case case10
python experiments/run_feasibility_check.py --case case10 --model cnn
python experiments/run_milp_acceleration.py --case case10 --model cnn
python experiments/run_scalability_study.py --model cnn
```

### MILP infeasible scenarios

Synthetic demand is scaled to remain below available capacity, but infeasibilities may still appear because of ramping and minimum up/down constraints. These are logged in `milp_summary.csv`, and preprocessing uses feasible solved scenarios only by default.
