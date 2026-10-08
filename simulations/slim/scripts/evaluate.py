"""
Score an sc2ts ARG inferred from simulated sequences against the true ARG:
overall accuracy with tscompare, and recombinant detection for each sample
against the pedigree.
"""
import pathlib

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
    Return the truth table with the sc2ts match of each sample, and whether it
    is expected to be a recombinant, added. A sample is inferred to be a
    recombinant if its HMM match has more than one parent.
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
    # A sample should be inferred to be a recombinant if it has a recombinant
    # ancestor on its clonal line (or is one), and nothing placed in the ARG
    # in between already accounts for that recombination.
    placed_strains = set(df.strain[df.placed])
    df["expected_recombinant"] = [
        ancestor >= 0 and placed_strains.isdisjoint(between.split())
        for ancestor, between in zip(
            df.recombinant_ancestor, df.sampled_between.fillna("")
        )
    ]
    df["inferred_recombinant"] = (
        df["inferred_recombinant"].astype("boolean").fillna(False).astype(bool)
    )
    for col in ["num_inferred_parents", "inferred_breakpoint",
                "breakpoint_interval_left", "breakpoint_interval_right"]:
        df[col] = df[col].fillna(-1).astype(int)
    return df


def event_table(samples):
    """
    Return one row for each recombination event that a placed sample is
    expected to carry, keyed by the recombinant ancestor.

    Several samples can share an expected recombinant ancestor, and once sc2ts
    has inferred that recombination in one of them the others correctly match
    it with a single parent. So an event is detected if any sample expected to
    carry it is inferred to be a recombinant. The breakpoint interval and error
    are from the earliest sample in which the event is found with a single
    breakpoint, compared with the recombinant ancestor's breakpoint.
    """
    df = samples[samples.placed & samples.expected_recombinant]
    events = df.groupby("recombinant_ancestor").agg(
        num_samples=("strain", "size"),
        sampled=("is_recombinant", "any"),
        detectable=("ancestor_detectable", "first"),
        detected=("inferred_recombinant", "any"),
        breakpoint=("ancestor_breakpoint", "first"),
    )
    found = (
        df[df.inferred_recombinant & (df.num_inferred_parents == 2)]
        .sort_values("gen", kind="stable")
        .groupby("recombinant_ancestor")
        .first()
    )
    events["interval_width"] = (
        found.breakpoint_interval_right - found.breakpoint_interval_left
    )
    events["breakpoint_error"] = found.inferred_breakpoint - found.ancestor_breakpoint
    events["breakpoint_in_interval"] = (
        (found.breakpoint_interval_left <= found.ancestor_breakpoint)
        & (found.ancestor_breakpoint <= found.breakpoint_interval_right)
    ).astype("boolean")
    return events.reset_index()


def placement_table(samples):
    """
    Return the number of samples, and the number placed in the ARG, in each
    generation.
    """
    return (
        samples.groupby("gen")
        .agg(num_samples=("strain", "size"), num_placed=("placed", "sum"))
        .reset_index()
    )


def score_recombinants(samples, events):
    """
    Summarise recombinant detection, over placed samples only. Precision is per
    sample: the fraction of samples inferred to be recombinants that are
    expected to be. Recall and breakpoint accuracy are per event (see
    event_table).
    """
    df = samples[samples.placed]
    expected = df.expected_recombinant
    inferred = df.inferred_recombinant
    tp = int(np.sum(expected & inferred))
    fp = int(np.sum(~expected & inferred))
    detectable = events[events.detectable]
    found = events.dropna(subset=["interval_width"])
    return {
        "num_samples": len(samples),
        "num_placed": len(df),
        "num_sampled_recombinants": int(np.sum(df.is_recombinant)),
        "num_expected_recombinants": int(np.sum(expected)),
        "num_inferred_recombinants": int(np.sum(inferred)),
        "true_positives": tp,
        "false_positives": fp,
        "precision": tp / (tp + fp) if tp + fp > 0 else np.nan,
        "num_events": len(events),
        "num_detectable_events": len(detectable),
        "events_detected": int(events.detected.sum()),
        "recall": events.detected.mean() if len(events) > 0 else np.nan,
        "recall_detectable": (
            detectable.detected.mean() if len(detectable) > 0 else np.nan
        ),
        "mean_abs_breakpoint_error": (
            found.breakpoint_error.abs().mean() if len(found) > 0 else np.nan
        ),
        "breakpoint_in_interval": (
            found.breakpoint_in_interval.astype(float).mean()
            if len(found) > 0
            else np.nan
        ),
        "median_interval_width": (
            found.interval_width.median() if len(found) > 0 else np.nan
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
@click.argument("output_dir", type=click.Path(file_okay=False))
@click.option("--pathogen", required=True)
@click.option("--rep", type=int, required=True)
@click.option("--p", "probability", type=float, required=True)
@click.option("--k", type=int, required=True)
def run(true_ts, inferred_ts, truth, output_dir, pathogen, rep, probability, k):
    """
    Write evaluation.csv (one row), events.csv, placement.csv and samples.csv
    to OUTPUT_DIR. All but samples.csv start with the run's parameters, so they
    can be concatenated across runs.
    """
    output_dir = pathlib.Path(output_dir)
    true_ts = tskit.load(true_ts)
    inferred_ts = tskit.load(inferred_ts)
    samples = classify_samples(inferred_ts, pd.read_csv(truth))
    samples.to_csv(output_dir / "samples.csv", index=False)
    events = event_table(samples)
    placement = placement_table(samples)

    run_params = {"pathogen": pathogen, "rep": rep, "p": probability, "k": k}
    row = dict(run_params)
    row.update(score_recombinants(samples, events))
    row["num_recombinant_nodes"] = num_recombinant_nodes(inferred_ts)
    row.update(score_arg(*prepare_for_comparison(true_ts, inferred_ts)))
    pd.DataFrame([row]).to_csv(output_dir / "evaluation.csv", index=False)
    for name, df in [("events", events), ("placement", placement)]:
        for j, (column, value) in enumerate(run_params.items()):
            df.insert(j, column, value)
        df.to_csv(output_dir / f"{name}.csv", index=False)


if __name__ == "__main__":
    run()
