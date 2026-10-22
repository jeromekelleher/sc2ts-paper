"""
Check the internal consistency of node dating. We choose a set of sample
nodes with children, matched to the non-sample (internal) nodes on time and
average number of children, and mark them as non-samples. The resulting ARG
is then dated with run_nonsample_dating.py (re-dating the chosen nodes along
with all the internal nodes), and the re-estimated times are compared with
the original sample dates.
"""

import click
import numpy as np
import pandas as pd
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


@click.command()
@click.argument("original_ts")
@click.argument("redated_ts")
@click.argument("csv_out")
def compare(original_ts, redated_ts, csv_out):
    """
    Write the original and re-estimated times of the re-dated sample nodes.
    """
    orig = tszip.load(original_ts)
    redated = tszip.load(redated_ts)
    orig_sample = (orig.nodes_flags & tskit.NODE_IS_SAMPLE) != 0
    redated_sample = (redated.nodes_flags & tskit.NODE_IS_SAMPLE) != 0
    assert np.all(orig_sample[redated_sample])
    # Samples that were not chosen must keep their dates
    assert np.allclose(
        orig.nodes_time[redated_sample], redated.nodes_time[redated_sample]
    )
    nodes = np.where(orig_sample & ~redated_sample)[0]
    df = pd.DataFrame(
        {
            "node_id": nodes,
            "time": orig.nodes_time[nodes],
            "redated_time": redated.nodes_time[nodes],
        }
    )
    df.to_csv(csv_out, index=False, float_format="%.2f")
    print(f"Wrote {len(df)} re-dated nodes to {csv_out}")


@click.group()
def cli():
    pass


cli.add_command(mask)
cli.add_command(compare)

if __name__ == "__main__":
    cli()
