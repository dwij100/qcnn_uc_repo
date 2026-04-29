"""
Generate IEEE IAS digest figures from qcnn_uc_repo result files.

Run from repository root:
    python ias_digest_plots.py --repo-root . --out-dir data/results/ias_digest_figures

The script expects the project structure used in qcnn_uc_repo:
    data/results/<case>/<model>/test_metrics.csv
    data/results/<case>/<model>/feasibility/feasibility_summary.csv
    data/results/<case>/<model>/milp_acceleration/acceleration_summary.csv

It is robust to missing files. Missing cases/models are skipped.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

MODELS = ["cnn", "henderson_quanv", "pqc_qcnn"]
MODEL_LABELS = {
    "cnn": "CNN",
    "henderson_quanv": "Quanvolutional NN",
    "pqc_qcnn": "Parameterized QCNN",
}
CASES = ["case10", "case24", "case118_reduced"]
CASE_LABELS = {
    "case10": "10-gen",
    "case24": "24-gen",
    "case118_reduced": "118-reduced",
}


def read_csv_if_exists(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return None


def collect_metrics(repo_root: Path) -> pd.DataFrame:
    rows = []
    for case in CASES:
        for model in MODELS:
            p = repo_root / "data" / "results" / case / model / "test_metrics.csv"
            df = read_csv_if_exists(p)
            if df is not None and len(df):
                row = df.iloc[0].to_dict()
                row["case"] = case
                row["model"] = model
                rows.append(row)
    return pd.DataFrame(rows)


def collect_feasibility(repo_root: Path) -> pd.DataFrame:
    rows = []
    for case in CASES:
        for model in MODELS:
            p = repo_root / "data" / "results" / case / model / "feasibility" / "feasibility_summary.csv"
            df = read_csv_if_exists(p)
            if df is not None and len(df):
                row = df.iloc[0].to_dict()
                row["case"] = case
                row["model"] = model
                rows.append(row)
    return pd.DataFrame(rows)


def collect_acceleration(repo_root: Path) -> pd.DataFrame:
    rows = []
    for case in CASES:
        for model in MODELS:
            p = repo_root / "data" / "results" / case / model / "milp_acceleration" / "acceleration_summary.csv"
            df = read_csv_if_exists(p)
            if df is not None and len(df):
                df["case"] = case
                df["model"] = model
                rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def fig_pipeline(out_dir: Path):
    fig, ax = plt.subplots(figsize=(8.5, 2.2))
    ax.axis("off")
    boxes = [
        "UC scenarios\nload + renewables",
        "Full MILP\nGurobi labels",
        "CNN / QCNN\nprobabilities",
        "Feasibility\nfilter",
        "Confidence\npartial fixing",
        "Assisted MILP\nfeasible solution",
    ]
    xs = [0.05, 0.22, 0.39, 0.56, 0.72, 0.88]
    for i, (x, text) in enumerate(zip(xs, boxes)):
        ax.text(x, 0.55, text, ha="center", va="center", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="black", lw=1.0))
        if i < len(xs) - 1:
            ax.annotate("", xy=(xs[i+1]-0.07, 0.55), xytext=(x+0.07, 0.55),
                        arrowprops=dict(arrowstyle="->", lw=1.0))
    ax.text(0.5, 0.14, "The learning model does not replace UC optimization; it selects reliable binary guidance for the MILP.",
            ha="center", va="center", fontsize=8)
    out = out_dir / "fig1_pipeline.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_model_accuracy(metrics: pd.DataFrame, out_dir: Path):
    if metrics.empty:
        return None
    df = metrics.copy()
    df["label"] = df["model"].map(MODEL_LABELS)
    df["case_label"] = df["case"].map(CASE_LABELS)
    pivot = df.pivot(index="label", columns="case_label", values="bitwise_accuracy") * 100
    pivot = pivot.reindex([MODEL_LABELS[m] for m in MODELS if MODEL_LABELS[m] in pivot.index])
    ax = pivot.plot(kind="bar", figsize=(6.4, 3.0), rot=20)
    ax.set_ylabel("Bitwise accuracy (%)")
    ax.set_xlabel("Model")
    ax.set_ylim(90, 100)
    ax.set_title("Commitment prediction accuracy")
    ax.legend(title="Case")
    ax.grid(axis="y", alpha=0.3)
    out = out_dir / "fig2_model_accuracy.png"
    ax.figure.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(ax.figure)
    return out


def fig_accuracy_feasibility(metrics: pd.DataFrame, feas: pd.DataFrame, out_dir: Path):
    if metrics.empty or feas.empty:
        return None
    m = metrics[["case", "model", "bitwise_accuracy"]].copy()
    f = feas[["case", "model", "partial_feasibility_rate", "feasibility_rate"]].copy()
    df = m.merge(f, on=["case", "model"], how="inner")
    if df.empty:
        return None
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    for _, r in df.iterrows():
        label = f"{MODEL_LABELS.get(r['model'], r['model'])}, {CASE_LABELS.get(r['case'], r['case'])}"
        ax.scatter(r["bitwise_accuracy"] * 100, r["partial_feasibility_rate"] * 100)
        ax.text(r["bitwise_accuracy"] * 100 + 0.03, r["partial_feasibility_rate"] * 100 + 0.3, label, fontsize=7)
    ax.set_xlabel("Bitwise accuracy (%)")
    ax.set_ylabel("Partial feasibility rate (%)")
    ax.set_title("Accuracy vs Constraint Violations")
    ax.set_xlim(max(90, df["bitwise_accuracy"].min()*100 - 1), 100)
    ax.set_ylim(0, max(40, df["partial_feasibility_rate"].max()*100 + 10))
    ax.grid(alpha=0.3)
    out = out_dir / "fig3_accuracy_feasibility.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_speedup(acc: pd.DataFrame, out_dir: Path, case: str = "case10"):
    if acc.empty:
        return None
    df = acc[(acc["case"] == case) & (acc["mode"].isin(["warm_start", "partial_fix_confident"]))].copy()
    if df.empty:
        return None
    df["label"] = df["model"].map(MODEL_LABELS)
    pivot = df.pivot(index="label", columns="mode", values="mean_speedup")
    pivot = pivot.reindex([MODEL_LABELS[m] for m in MODELS if MODEL_LABELS[m] in pivot.index])
    ax = pivot.plot(kind="bar", figsize=(6.4, 3.0), rot=20)
    ax.set_ylabel("Speedup relative to full MILP")
    ax.set_xlabel("Model")
    ax.set_title(f"MILP acceleration on {CASE_LABELS.get(case, case)}")
    ax.legend(["Partial fixing","Warm start"], title="Assistance mode")
    ax.axhline(1.0, linestyle="--", linewidth=1)
    ax.grid(axis="y", alpha=0.3)
    out = out_dir / "fig4_speedup.png"
    ax.figure.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(ax.figure)
    return out


def fig_scalability(metrics: pd.DataFrame, out_dir: Path):
    if metrics.empty:
        return None
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    for model in MODELS:
        df = metrics[metrics["model"] == model].copy()
        if df.empty:
            continue
        df["case_order"] = df["case"].map({"case10": 10, "case24": 24, "case118_reduced": 118})
        df = df.sort_values("case_order")
        ax.plot(df["case_order"], df["prediction_time_per_sample"], marker="o", label=MODEL_LABELS[model])
    ax.set_xlabel("System size indicator")
    ax.set_ylabel("Prediction time per sample (s)")
    ax.set_title("Inference scalability")
    ax.set_yscale("log")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    out = out_dir / "fig5_scalability_prediction_time.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--out-dir", default="data/results/ias_digest_figures")
    args = ap.parse_args()
    repo_root = Path(args.repo_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = collect_metrics(repo_root)
    feas = collect_feasibility(repo_root)
    acc = collect_acceleration(repo_root)

    paths = []#[fig_pipeline(out_dir)]
    for fn in [fig_model_accuracy, fig_accuracy_feasibility, fig_speedup, fig_scalability]:
        if fn.__name__ == "fig_model_accuracy":
            p = fn(metrics, out_dir)
        elif fn.__name__ == "fig_accuracy_feasibility":
            p = fn(metrics, feas, out_dir)
        elif fn.__name__ == "fig_speedup":
            p = fn(acc, out_dir)
        else:
            p = fn(metrics, out_dir)
        if p is not None:
            paths.append(p)
    print("Generated figures:")
    for p in paths:
        print(f"  {p}")


if __name__ == "__main__":
    main()
