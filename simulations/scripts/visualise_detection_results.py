"""
Aggregate summary JSON files and plot recombination detection performance.

This produces a 3-by-3 figure: rows are k (num_mismatches), and columns are at_frequency.
Each panel is a heatmap of detection performance metric over mutation rates and recombination rates.
"""

import ast
import json
import re
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RUN_RE = re.compile(r"^(?P<sim>.+)_rep(?P<rep>\d+)_k(?P<k>\d+)\.json$")
SIM_RE = re.compile(r"^seq_m(?P<mut>.+)_r(?P<rec>.+)_f(?P<frq>\d+)$")


def sort_rates(rates, reverse):
    return sorted({str(r) for r in rates}, key=float, reverse=reverse)


def parse_metrics(path: Path) -> dict:
    text = path.read_text()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return ast.literal_eval(text)


def compute_rates(d: dict) -> dict:
    tp, tn, fp, fn = d["tp"], d["tn"], d["fp"], d["fn"]
    total = tp + tn + fp + fn
    precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    if np.isnan(precision) or np.isnan(recall) or (precision + recall) == 0:
        f1 = np.nan
    else:
        f1 = 2 * precision * recall / (precision + recall)
    accuracy = (tp + tn) / total if total > 0 else np.nan
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }


def load_results(results_dir: Path) -> pd.DataFrame:
    records = []
    for path in sorted(results_dir.glob("*.json")):
        m = RUN_RE.match(path.name)
        if m is None:
            continue
        sim_parts = SIM_RE.match(m.group("sim"))
        if sim_parts is None:
            continue
        metrics = parse_metrics(path)
        rates = compute_rates(metrics)
        records.append(
            {
                "rep": int(m.group("rep")),
                "k": int(m.group("k")),
                "mutation_rate": sim_parts.group("mut"),
                "recombination_rate": sim_parts.group("rec"),
                "at_frequency": int(sim_parts.group("frq")),
                **metrics,
                **rates,
            }
        )
    if not records:
        raise click.ClickException(f"No summary JSON files found in {results_dir}.")
    return pd.DataFrame(records)


def metric_pivot(df, k, at_frequency, metric):
    """Mean metric over replicates for one (k, at_frequency) pair."""
    sub_df = df[(df["k"] == k) & (df["at_frequency"] == at_frequency)]
    mut_rates = sort_rates(df["mutation_rate"].unique(), reverse=True)
    rec_rates = sort_rates(df["recombination_rate"].unique(), reverse=False)
    if sub_df.empty:
        return pd.DataFrame(np.nan, index=mut_rates, columns=rec_rates)

    agg = (
        sub_df.groupby(["mutation_rate", "recombination_rate"], as_index=False)[metric]
        .mean()
    )
    pivot = agg.pivot(
        index="mutation_rate",
        columns="recombination_rate",
        values=metric,
    )
    return pivot.reindex(index=mut_rates, columns=rec_rates)


def metric_pivot_with_fallback(df, k, at_frequency, metric):
    pivot = metric_pivot(df, k, at_frequency, metric)
    if pivot.isna().all().all() and metric != "accuracy":
        pivot = metric_pivot(df, k, at_frequency, "accuracy")
    return pivot


def plot_heatmap(ax, pivot, title, vmin=0, vmax=1, show_ylabel=True, show_xlabel=True):
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="#cccccc")
    data = pivot.to_numpy(dtype=float)
    im = ax.imshow(data, aspect="equal", vmin=vmin, vmax=vmax, cmap=cmap)
    nrows, ncols = data.shape
    ax.set_title(title, fontsize=9)
    ax.set_xticks(range(ncols))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(nrows))
    ax.set_yticklabels(pivot.index, fontsize=7)
    ax.set_xticks(np.arange(-0.5, ncols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, nrows, 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.5)
    ax.tick_params(which="minor", size=0)
    if show_xlabel:
        ax.set_xlabel("Recombination rate")
    if show_ylabel:
        ax.set_ylabel("Mutation rate")
    for i in range(nrows):
        for j in range(ncols):
            val = data[i, j]
            if np.isnan(val):
                label = "n/a"
                color = "black"
            else:
                label = f"{val:.2f}"
                color = "white" if val < 0.5 * (vmin + vmax) else "black"
            ax.text(j, i, label, ha="center", va="center", color=color, fontsize=7)
    return im


@click.command()
@click.argument(
    "results_dir",
    type=click.Path(exists=True, dir_okay=True, file_okay=False),
)
@click.option(
    "--output",
    required=True,
    type=click.Path(dir_okay=False, file_okay=True),
    help="Output file path.",
)
@click.option(
    "--metric",
    default="accuracy",
    show_default=True,
    type=click.Choice(["accuracy", "f1", "recall", "precision"]),
    help="Detection performance metric to plot in each heatmap.",
)
def main(results_dir, output, metric):
    results_dir = Path(results_dir)
    df = load_results(results_dir)

    k_vals = sorted(df["k"].unique())
    at_freqs = sorted(df["at_frequency"].unique())
    if len(k_vals) != 3 or len(at_freqs) != 3:
        raise click.ClickException(
            f"Expected 3 k values and 3 at_frequency values, "
            f"found k={k_vals}, at_frequency={at_freqs}."
        )

    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    ims = []
    metric_label = {
        "accuracy": "Accuracy",
        "f1": "F1",
        "recall": "Recall",
        "precision": "Precision",
    }[metric]

    for i, k in enumerate(k_vals):
        for j, frq in enumerate(at_freqs):
            ax = axes[i, j]
            pivot = metric_pivot_with_fallback(df, k, frq, metric)
            title = f"k={k}, at_frequency={frq}"
            im = plot_heatmap(
                ax,
                pivot,
                title,
                show_ylabel=(j == 0),
                show_xlabel=(i == len(k_vals) - 1),
            )
            ims.append(im)

    fig.suptitle(
        f"{metric_label} (mean over replicates)",
        fontsize=12,
    )
    fig.tight_layout()
    fig.colorbar(ims[0], ax=axes.ravel().tolist(), label=metric_label)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
