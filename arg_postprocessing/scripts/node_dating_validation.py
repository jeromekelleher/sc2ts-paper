"""
Check the internal consistency of node dating. We choose a set of sample
nodes with children, matched to the non-sample (internal) nodes on time and
average number of children, and mark them as non-samples. The resulting ARG
is then dated with run_nonsample_dating.py (re-dating the chosen nodes along
with all the internal nodes), and the re-estimated times are compared with
the original sample dates.
"""

import json

import click
import numpy as np
import tskit
import tszip

NUM_BINS = 20


def average_children(ts):
    """
    Return the number of children of each node, averaged over the genome.
    """
    span = ts.edges_right - ts.edges_left
    total = np.bincount(ts.edges_parent, weights=span, minlength=ts.num_nodes)
    return total / ts.sequence_length


def choose_nodes(ts, rng):
    """
    Return as many sample nodes with children as there are internal nodes,
    chosen so that the distribution of (time, log average children) matches
    that of the internal nodes.
    """
    num_children = average_children(ts)
    is_sample = (ts.nodes_flags & tskit.NODE_IS_SAMPLE) != 0
    internal = np.where(~is_sample & (num_children > 0))[0]
    eligible = np.where(is_sample & (num_children > 0))[0]

    time_bins = np.histogram_bin_edges(ts.nodes_time[internal], NUM_BINS)
    children_bins = np.histogram_bin_edges(np.log(num_children[internal]), NUM_BINS)

    def bin_index(nodes):
        x = np.digitize(ts.nodes_time[nodes], time_bins) - 1
        y = np.digitize(np.log(num_children[nodes]), children_bins) - 1
        x = np.clip(x, 0, NUM_BINS - 1)
        y = np.clip(y, 0, NUM_BINS - 1)
        return x * NUM_BINS + y

    target = np.bincount(bin_index(internal), minlength=NUM_BINS**2)
    eligible_bins = bin_index(eligible)
    available = np.bincount(eligible_bins, minlength=NUM_BINS**2)
    # Importance weights: target density over the density of eligible nodes
    weights = target[eligible_bins] / available[eligible_bins]
    return rng.choice(
        eligible, size=len(internal), replace=False, p=weights / weights.sum()
    )


@click.command()
@click.argument("ts_in")
@click.argument("ts_out")
@click.option("--seed", default=42, show_default=True)
def mask(ts_in, ts_out, seed):
    """
    Mark a matched set of sample nodes as non-samples so they get re-dated.
    """
    ts = tszip.load(ts_in)
    chosen = choose_nodes(ts, np.random.default_rng(seed))
    tables = ts.dump_tables()
    flags = tables.nodes.flags
    flags[chosen] &= ~np.uint32(tskit.NODE_IS_SAMPLE)
    tables.nodes.flags = flags
    tables.dump(ts_out)
    print(f"Marked {len(chosen)} sample nodes as non-samples")


def histogram(values, edges):
    counts, _ = np.histogram(values, edges)
    return counts.tolist()


@click.command()
@click.argument("original_ts")
@click.argument("redated_ts")
@click.argument("json_out")
def compare(original_ts, redated_ts, json_out):
    """
    Summarise the re-estimated times of the re-dated sample nodes as
    histograms, for plotting.
    """
    orig = tszip.load(original_ts)
    redated = tszip.load(redated_ts)
    orig_sample = (orig.nodes_flags & tskit.NODE_IS_SAMPLE) != 0
    redated_sample = (redated.nodes_flags & tskit.NODE_IS_SAMPLE) != 0
    assert np.all(orig_sample[redated_sample])
    # Samples that were not chosen should keep their dates, except where
    # tsdate's final constraint step pushes a sample above a node dated older
    # than it.
    sample_change = np.abs(
        redated.nodes_time[redated_sample] - orig.nodes_time[redated_sample]
    )
    nodes = np.where(orig_sample & ~redated_sample)[0]
    time = orig.nodes_time[nodes]
    redated_time = redated.nodes_time[nodes]
    difference = redated_time - time

    with np.errstate(divide="ignore"):
        log_children = np.log10(average_children(orig))
    internal = np.where(~orig_sample & np.isfinite(log_children))[0]
    time_edges = np.histogram_bin_edges(orig.nodes_time[internal], NUM_BINS)
    children_edges = np.histogram_bin_edges(log_children[internal], NUM_BINS)
    difference_edges = np.histogram_bin_edges(difference, 80)
    heatmap_edges = np.linspace(
        min(time.min(), redated_time.min()), max(time.max(), redated_time.max()), 51
    )
    heatmap, _, _ = np.histogram2d(time, redated_time, bins=heatmap_edges)

    summary = {
        "num_redated": len(nodes),
        "mean_difference": difference.mean(),
        "median_difference": np.median(difference),
        # Comparable to the accuracy measure used for Chronumental
        "fraction_within_30_days": np.mean(np.abs(difference) <= 30),
        "num_moved_samples": int(np.sum(~np.isclose(sample_change, 0))),
        "max_sample_change": sample_change.max(),
        "time_hist": {
            "edges": time_edges.tolist(),
            "internal": histogram(orig.nodes_time[internal], time_edges),
            "redated": histogram(time, time_edges),
        },
        "children_hist": {
            "edges": children_edges.tolist(),
            "internal": histogram(log_children[internal], children_edges),
            "redated": histogram(log_children[nodes], children_edges),
        },
        "difference_hist": {
            "edges": difference_edges.tolist(),
            "counts": histogram(difference, difference_edges),
        },
        "heatmap": {
            "edges": heatmap_edges.tolist(),
            "counts": heatmap.astype(int).tolist(),
        },
    }
    with open(json_out, "w") as f:
        json.dump(summary, f, indent=1)
    print(
        f"{summary['num_moved_samples']} of {np.sum(redated_sample)} other samples "
        f"changed date (max change {summary['max_sample_change']:.2f}). "
        f"Mean difference for {len(nodes)} re-dated nodes: "
        f"{summary['mean_difference']:.1f}"
    )


@click.group()
def cli():
    pass


cli.add_command(mask)
cli.add_command(compare)

if __name__ == "__main__":
    cli()
