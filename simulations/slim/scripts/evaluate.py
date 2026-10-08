"""
Score an sc2ts ARG inferred from simulated sequences against the true ARG:
overall accuracy with tscompare, and recombinant detection for each sample
against the pedigree.
"""
import click
import numpy as np
import pandas as pd
import tscompare
import tskit


def true_strains(ts):
    """
    Return the strain names of the samples in a true ARG from sample_slim.py.
    """
    return [
        f"seq_{ts.individual(ts.node(u).individual).metadata['pedigree_id']}"
        for u in ts.samples()
    ]


def inferred_strains(ts):
    """
    Return the strain names of the samples in an sc2ts ARG. The reference node
    isn't flagged as a sample, so isn't included.
    """
    return [ts.node(u).metadata["strain"] for u in ts.samples()]


def prepare_for_comparison(true_ts, inferred_ts):
    """
    Return copies of the true and inferred ARGs simplified to the samples that
    sc2ts placed, in the same order, on the same coordinates and timescale, so
    that tscompare can match sample node i in one to sample node i in the other.
    """
    # sc2ts prepends a dummy base to the reference so site positions are
    # 1-based, giving a sequence length one longer than SLiM's.
    L = inferred_ts.sequence_length
    if L != true_ts.sequence_length + 1:
        raise ValueError("Inferred sequence length must be one more than true.")

    true_nodes = dict(zip(true_strains(true_ts), true_ts.samples()))
    inferred_nodes = dict(zip(inferred_strains(inferred_ts), inferred_ts.samples()))
    strains = [s for s in true_nodes if s in inferred_nodes]

    tables = inferred_ts.dump_tables()
    tables.reference_sequence.clear()
    tables.keep_intervals([[1, L]], simplify=False)
    tables.ltrim()
    inferred_ts = tables.tree_sequence()

    true_ts = true_ts.simplify([true_nodes[s] for s in strains])
    inferred_ts = inferred_ts.simplify([inferred_nodes[s] for s in strains])

    # Both timescales are 1 per generation (day), but the inferred times are
    # relative to the last sample date, and the true ones to the end of the
    # simulation.
    offset = np.median(
        true_ts.nodes_time[true_ts.samples()]
        - inferred_ts.nodes_time[inferred_ts.samples()]
    )
    tables = inferred_ts.dump_tables()
    tables.nodes.time += offset
    tables.time_units = true_ts.time_units
    return true_ts, tables.tree_sequence()


def classify_samples(inferred_ts, truth):
    """
    Return the truth table with the sc2ts match of each sample added. A sample
    is inferred to be a recombinant if its HMM match has more than one parent.
    Breakpoints are converted to SLiM's 0-based coordinates, where the
    breakpoint is the first position inherited from the second parent.
    """
    matches = {}
    for u in inferred_ts.samples():
        md = inferred_ts.node(u).metadata
        path = md["sc2ts"]["hmm_match"]["path"]
        row = {
            "placed": True,
            "num_inferred_parents": len(path),
            "inferred_recombinant": len(path) > 1,
            "inferred_breakpoint": -1,
            "breakpoint_interval_left": -1,
            "breakpoint_interval_right": -1,
        }
        if len(path) > 1:
            row["inferred_breakpoint"] = path[1]["left"] - 1
            left, right = md["sc2ts"]["breakpoint_intervals"][0]
            row["breakpoint_interval_left"] = left - 1
            row["breakpoint_interval_right"] = right - 1
        matches[md["strain"]] = row
    df = truth.copy()
    placed = pd.DataFrame.from_dict(matches, orient="index")
    df = df.join(placed, on="strain")
    df["placed"] = df["placed"].astype("boolean").fillna(False).astype(bool)
    df["inferred_recombinant"] = (
        df["inferred_recombinant"].astype("boolean").fillna(False).astype(bool)
    )
    for col in ["num_inferred_parents", "inferred_breakpoint",
                "breakpoint_interval_left", "breakpoint_interval_right"]:
        df[col] = df[col].fillna(-1).astype(int)
    return df


def score_recombinants(samples):
    """
    Summarise per-sample recombinant classification, over placed samples only.
    """
    df = samples[samples.placed]
    true = df.is_recombinant
    inferred = df.inferred_recombinant
    tp = int(np.sum(true & inferred))
    fp = int(np.sum(~true & inferred))
    fn = int(np.sum(true & ~inferred))
    # Single breakpoint recombinants found with a single breakpoint.
    found = df[true & inferred & (df.num_inferred_parents == 2)]
    return {
        "num_samples": len(samples),
        "num_placed": len(df),
        "num_true_recombinants": int(np.sum(true)),
        "num_detectable_recombinants": int(np.sum(df.detectable)),
        "num_inferred_recombinants": int(np.sum(inferred)),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": tp / (tp + fp) if tp + fp > 0 else np.nan,
        "recall": tp / (tp + fn) if tp + fn > 0 else np.nan,
        "recall_detectable": (
            np.sum(df.detectable & inferred) / np.sum(df.detectable)
            if np.sum(df.detectable) > 0
            else np.nan
        ),
        "mean_abs_breakpoint_error": (
            np.mean(np.abs(found.inferred_breakpoint - found.breakpoint))
            if len(found) > 0
            else np.nan
        ),
        "breakpoint_in_interval": (
            np.mean(
                (found.breakpoint_interval_left <= found.breakpoint)
                & (found.breakpoint <= found.breakpoint_interval_right)
            )
            if len(found) > 0
            else np.nan
        ),
    }


def num_recombinant_nodes(ts):
    """
    Return the number of nodes with more than one parent. This counts
    recombination events rather than recombinant samples.
    """
    edges = pd.DataFrame({"child": ts.edges_child, "parent": ts.edges_parent})
    num_parents = edges.drop_duplicates().child.value_counts()
    return int(np.sum(num_parents > 1))


def score_arg(true_ts, inferred_ts):
    result = tscompare.haplotype_arf(inferred_ts, true_ts)
    return {"arf": result.arf, "tpr": result.tpr, "rmse": result.rmse}


@click.command()
@click.argument("true_ts", type=click.Path(exists=True, dir_okay=False))
@click.argument("inferred_ts", type=click.Path(exists=True, dir_okay=False))
@click.argument("truth", type=click.Path(exists=True, dir_okay=False))
@click.argument("output", type=click.Path(dir_okay=False))
@click.argument("output_samples", type=click.Path(dir_okay=False))
@click.option("--pathogen", required=True)
@click.option("--rep", type=int, required=True)
@click.option("--p", "probability", type=float, required=True)
@click.option("--k", type=int, required=True)
def run(
    true_ts, inferred_ts, truth, output, output_samples, pathogen, rep,
    probability, k
):
    true_ts = tskit.load(true_ts)
    inferred_ts = tskit.load(inferred_ts)
    samples = classify_samples(inferred_ts, pd.read_csv(truth))
    samples.to_csv(output_samples, index=False)

    row = {"pathogen": pathogen, "rep": rep, "p": probability, "k": k}
    row.update(score_recombinants(samples))
    row["num_recombinant_nodes"] = num_recombinant_nodes(inferred_ts)
    row.update(score_arg(*prepare_for_comparison(true_ts, inferred_ts)))
    pd.DataFrame([row]).to_csv(output, index=False)


if __name__ == "__main__":
    run()
