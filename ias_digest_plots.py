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
import numpy as np
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

MODELS = ["cnn", "henderson_quanv", "pqc_qcnn"]
MODEL_LABELS = {
    "cnn": "CNN",
    "henderson_quanv": "QNN",
    "pqc_qcnn": "PQCNN",
}
CASES = ["case10", "case24", "case118_reduced"]
CASE_LABELS = {
    "case10": "10-gen",
    "case24": "24-gen",
    "case118_reduced": "54-gen",
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
            p = repo_root / "data" / "results" / case / model / "feasibility" / "feasibility_summary_1.csv"
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
    print(df)
    df["label"] = df["model"].map(MODEL_LABELS)
    df["case_label"] = df["case"].map(CASE_LABELS)
    pivot = df.pivot(index="label", columns="case_label", values="bitwise_accuracy") * 100
    pivot = pivot.reindex([MODEL_LABELS[m] for m in MODELS if MODEL_LABELS[m] in pivot.index])
    ax = pivot.plot(kind="bar", figsize=(6.4, 3.0), rot=0)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f", fontsize=8, padding=2)
    ax.set_ylabel("Bitwise accuracy (%)")
    ax.set_xlabel("Model")
    ax.set_ylim(90, 97)
    ax.set_title("Bitwise commitment prediction accuracy")
    ax.legend(title="Case", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    out = out_dir / "fig1_model_accuracy.png"
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
        
        sc = ax.scatter(r["bitwise_accuracy"] * 100,
                        r["partial_feasibility_rate"] * 100)
        
        color = sc.get_facecolor()[0]  # extract RGBA color
        
        ax.text(r["bitwise_accuracy"] * 100 +0.1,
                r["partial_feasibility_rate"] * 100-0.8 ,
                label,
                fontsize=6.7,
                color=color)
    ax.set_xlabel("Bitwise accuracy (%)")
    ax.set_ylabel("Partial feasibility rate (%)")
    ax.set_title("Accuracy vs Constraint Violations")
    ax.set_xlim(max(90, df["bitwise_accuracy"].min()*100 - 1), 97)
    ax.set_ylim(0, max(50, df["partial_feasibility_rate"].max()*100 + 10))
    ax.grid(alpha=0.7)
    out = out_dir / "fig2_accuracy_feasibility.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


# def fig_speedup(acc: pd.DataFrame, out_dir: Path, case: str = "case118_reduced"):
#     if acc.empty:
#         return None
#     df = acc[(acc["case"] == case) & (acc["mode"].isin(["warm_start", "partial_fix_confident"]))].copy()
#     if df.empty:
#         return None
#     df["label"] = df["model"].map(MODEL_LABELS)
#     pivot = df.pivot(index="label", columns="mode", values="mean_speedup")
#     pivot = pivot.reindex([MODEL_LABELS[m] for m in MODELS if MODEL_LABELS[m] in pivot.index])
#     ax = pivot.plot(kind="bar", figsize=(6.4, 3.0), rot=20)
#     ax.set_ylabel("Speedup relative to full MILP")
#     ax.set_xlabel("Model")
#     ax.set_title(f"MILP acceleration on {CASE_LABELS.get(case, case)}")
#     ax.legend(["Partial fixing","Warm start"], title="Assistance mode")
#     ax.axhline(1.0, linestyle="--", linewidth=1)
#     ax.grid(axis="y", alpha=0.3)
#     out = out_dir / "fig4_speedup.png"
#     ax.figure.savefig(out, dpi=300, bbox_inches="tight")
#     plt.close(ax.figure)
#     return out

def collect_policy_total_times(repo_root: Path) -> pd.DataFrame:
    rows = []

    for case in CASES:
        for model in MODELS:
            p = (
                repo_root
                / "data"
                / "results"
                / case
                / model
                / "milp_acceleration"
                / "feasibility_policy_total_times_1.csv"
            )

            df = read_csv_if_exists(p)

            if df is not None and len(df):
                row = df.iloc[0].to_dict()
                row["case"] = case
                row["model"] = model
                rows.append(row)

    return pd.DataFrame(rows)


# def fig_speedup(policy: pd.DataFrame, out_dir: Path):
#     """
#     Figure 4 for IAS digest:
#     normalized total runtime over all test scenarios.

#     Baseline full MILP is 1.0 for every case.
#     Each model's bar is:
#         total assisted policy time / total full MILP time
#     """

#     if policy.empty:
#         return None

#     df = policy.copy()
#     df["case_label"] = df["case"].map(CASE_LABELS)
#     df["model_label"] = df["model"].map(MODEL_LABELS)

#     # Model bars.
#     pivot = df.pivot(
#         index="case_label",
#         columns="model_label",
#         values="normalized_policy_time_vs_full_milp",
#     )

#     case_order = [
#         CASE_LABELS[c]
#         for c in CASES
#         if CASE_LABELS[c] in pivot.index
#     ]

#     model_order = [
#         MODEL_LABELS[m]
#         for m in MODELS
#         if MODEL_LABELS[m] in pivot.columns
#     ]

#     pivot = pivot.reindex(case_order)[model_order]*100

#     # Add full MILP baseline = 1.0.
#     pivot.insert(0, "Full MILP", 100)

#     ax = pivot.plot(
#         kind="bar",
#         figsize=(6.6, 3.1),
#         rot=0,
#         width=0.78,
#     )

#     ax.set_ylabel("Normalized total runtime")
#     ax.set_xlabel("UC test case")
#     ax.set_title("Total MILP runtime with feasibility-guided assistance")
#     # ax.axhline(1.0, linestyle="--", linewidth=1)
#     ax.grid(axis="y", alpha=0.3)
#     ax.legend(title="Method", fontsize=7)
#     ax.set_ylim(0,110)
#     # # Lower is better.
#     # ax.text(
#     #     0.01,
#     #     0.95,
#     #     "Lower is better",
#     #     transform=ax.transAxes,
#     #     fontsize=8,
#     #     va="top",
#     # )
#     for container in ax.containers:
#         ax.bar_label(container, fmt="%.1f", fontsize=7, padding=2)
#     out = out_dir / "fig4_normalized_total_runtime.png"
#     ax.figure.savefig(out, dpi=300, bbox_inches="tight")
#     plt.close(ax.figure)

#     return out

def fig_speedup(policy: pd.DataFrame, out_dir: Path):
    """
    Figure 4 for IAS digest:
    normalized total runtime over all test scenarios.

    Layout:
    - 3 subplots, one per case
    - one shared legend
    - subplot titles include total baseline MILP time
    - no x-axis values/tick labels shown
    - custom legend labels:
        Full MILP        -> Unassisted MILP
        cnn              -> CNN Assisted
        henderson_quanv  -> QNN Assisted
        pqc_qcnn         -> PQC-QCNN Assisted
    """

    if policy.empty:
        return None

    df = policy.copy()
    df["case_label"] = df["case"].map(CASE_LABELS)
    df["model_label"] = df["model"].map(MODEL_LABELS)

    case_order = [c for c in CASES if c in df["case"].unique()]
    model_order = [m for m in MODELS if m in df["model"].unique()]

    # Internal keys used for plotting and color consistency
    method_keys = ["Full MILP"] + model_order

    # Clean display labels used only in the legend
    display_labels = {
        "Full MILP": "Unassisted MILP",
        "cnn": "CNN Assisted MILP",
        "henderson_quanv": "QNN Assisted MILP",
        "pqc_qcnn": "PQCNN Assisted MILP",
    }

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(6.6, 2.4),
        sharey=True,
    )

    if not isinstance(axes, (list, np.ndarray)):
        axes = [axes]

    # Consistent colors using internal method keys
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    method_colors = {
        method: color_cycle[i % len(color_cycle)]
        for i, method in enumerate(method_keys)
    }

    legend_handles = []
    legend_labels = []

    for ax_idx, ax in enumerate(axes):
        if ax_idx >= len(case_order):
            ax.axis("off")
            continue

        case = case_order[ax_idx]
        case_df = df[df["case"] == case].copy()

        # Baseline total MILP time for subplot title
        baseline_total_time = np.nan
        if not case_df.empty and "baseline_total_full_milp_time" in case_df.columns:
            baseline_total_time = float(case_df.iloc[0]["baseline_total_full_milp_time"])

        # Build bars using internal keys
        method_keys_case = ["Full MILP"]
        values = [100.0]

        for model in model_order:
            row = case_df[case_df["model"] == model]
            if not row.empty:
                val = float(row.iloc[0]["normalized_policy_time_vs_full_milp"]) * 100.0
            else:
                val = np.nan

            method_keys_case.append(model)
            values.append(val)

        x = np.arange(len(method_keys_case))

        bars = ax.bar(
            x,
            values,
            width=0.72,
            color=[method_colors[k] for k in method_keys_case],
        )

        # Subplot title with baseline MILP total time
        if np.isfinite(baseline_total_time):
            title = f"{CASE_LABELS[case]} (MILP total = {baseline_total_time:.1f} s)"
        else:
            title = f"{CASE_LABELS[case]}"

        ax.set_title(title, fontsize=7.5)

        # Remove all x-axis labels/ticks/values
        ax.set_xticks([])
        ax.tick_params(axis="x", which="both", length=0)

        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(0, 110)

        # Bar value labels
        for bar in bars:
            h = bar.get_height()
            if np.isfinite(h):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    h + 1.5,
                    f"{h:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=6,
                )

        # Collect legend once using clean display labels
        if ax_idx == 0:
            for key in method_keys_case:
                handle = plt.Rectangle((0, 0), 1, 1, color=method_colors[key])
                legend_handles.append(handle)
                legend_labels.append(display_labels.get(key, key))

    fig.supylabel("Normalized total runtime (%)", fontsize=9)

    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        ncol=len(legend_labels),
        fontsize=7,
        title="Method",
        title_fontsize=8,
        bbox_to_anchor=(0.5, 0.95),
        frameon=False,
    )

    fig.suptitle(
        "Total MILP runtime with feasibility-guided assistance",
        fontsize=10,
        y=1.0,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.93])

    out = out_dir / "fig4_normalized_total_runtime.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)

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
    policy = collect_policy_total_times(repo_root)
    metrics = collect_metrics(repo_root)
    feas = collect_feasibility(repo_root)
    acc = collect_acceleration(repo_root)

    paths = []#[fig_pipeline(out_dir)]
    for fn in [fig_model_accuracy, fig_accuracy_feasibility, fig_speedup]:
        if fn.__name__ == "fig_model_accuracy":
            p = fn(metrics, out_dir)
        elif fn.__name__ == "fig_accuracy_feasibility":
            p = fn(metrics, feas, out_dir)
        elif fn.__name__ == "fig_speedup":
            p = fn(policy, out_dir)
        else:
            p = fn(metrics, out_dir)
        if p is not None:
            paths.append(p)
    print("Generated figures:")
    for p in paths:
        print(f"  {p}")


if __name__ == "__main__":
    main()
