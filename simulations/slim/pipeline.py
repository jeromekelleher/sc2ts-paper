"""
The steps of the SLiM simulation pipeline after running SLiM itself, as click
subcommands: sample, export-samples, make-truth, write-sc2ts-config and
evaluate. See README.md.

Run from simulations/slim/, e.g.:
    python pipeline.py sample sim.slim.ts true.trees --probability 0.1 --seed 1
"""
import datetime
import json
import pathlib

import click
import numpy as np
import pandas as pd
import pyslim
import sc2ts.core
import tscompare
import tskit


# Generation 0 (the founder) is collected on this date, and each subsequent
# generation one day later. sc2ts's reference date must be before it.
START_DATE = datetime.date(2026, 1, 1)


def find_founder(ts):
    # The single founder is the only individual with no pedigree parents.
    founders = [
        ind.id
        for ind in ts.individuals()
        if ind.metadata["pedigree_p1"] == -1 and ind.metadata["pedigree_p2"] == -1
    ]
    if len(founders) != 1:
        raise ValueError("Expected exactly one founder in the SLiM genealogy.")
    return founders[0]


def sample_individuals(ts, probability, seed):
    """
    Return the genealogy of individuals sampled with the given probability,
    simplified so the founder is sample node 0. The founder is always sampled.
    Draws are made for every individual regardless of probability, so for a
    given seed the samples are nested as probability increases.
    """
    if not 0 <= probability <= 1:
        raise ValueError("Probability must be in [0, 1].")
    founder = find_founder(ts)
    rng = np.random.default_rng(seed)
    selected = rng.random(ts.num_individuals) < probability
    selected[founder] = True
    # Haploid individuals still have a second, vacant, node in SLiM. These have
    # no ancestry, so drop them as samples and keep one node per individual.
    ts = pyslim.remove_vacant(ts)
    samples = ts.samples()
    samples = samples[selected[ts.nodes_individual[samples]]]
    is_founder = ts.nodes_individual[samples] == founder
    samples = np.concatenate((samples[is_founder], samples[~is_founder]))
    return ts.simplify(samples=samples, filter_individuals=True)


def strain_name(pedigree_id):
    return f"seq_{pedigree_id}"


def sample_pedigree_ids(ts):
    return [
        ts.individual(ts.node(u).individual).metadata["pedigree_id"]
        for u in ts.samples()
    ]


def read_fasta(path, names):
    """
    Return a dict of the sequences in the FASTA file with the given names.
    """
    names = set(names)
    sequences = {}
    name = None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                name = line[1:]
            elif name in names:
                sequences[name] = sequences.get(name, "") + line
    missing = names - set(sequences)
    if len(missing) > 0:
        raise ValueError(f"{len(missing)} sequences missing from {path}")
    return sequences


def recombinant_ancestors(pedigree, ids):
    """
    Return, for each sampled individual, its nearest recombinant ancestor on
    its clonal line (itself, if it is a recombinant), or -1 if there is none,
    and the sampled individuals strictly between the two, along with the
    ancestor itself if it is sampled and not the individual. If any of those
    are in the inferred ARG, they account for the recombination; otherwise the
    individual should be inferred to be a recombinant.
    """
    parent = dict(zip(pedigree.child, pedigree.parent1))
    is_recombinant = dict(zip(pedigree.child, pedigree.is_recombinant.astype(bool)))
    sampled = set(ids)
    ancestors = []
    between = []
    for child in ids:
        ancestor = -1
        path = []
        u = child
        while True:
            if is_recombinant[u]:
                ancestor = u
                break
            # Non-recombinants have parent1 == parent2.
            u = parent[u]
            if u < 0:
                break
            if u in sampled:
                path.append(u)
        ancestors.append(ancestor)
        between.append(path if ancestor >= 0 else [])
    return ancestors, between


def required_sequences(pedigree, ids):
    """
    Return the pedigree IDs whose sequences truth_table needs: every
    recombinant ancestor and its parents.
    """
    df = pedigree.set_index("child")
    ancestors = {x for x in recombinant_ancestors(pedigree, ids)[0] if x >= 0}
    return (
        ancestors | set(df.parent1[list(ancestors)]) | set(df.parent2[list(ancestors)])
    )


def truth_table(pedigree, ids, sequences):
    """
    Return the true recombination history of the sampled individuals.

    is_recombinant and breakpoint describe the sample itself. The
    recombinant_ancestor is its nearest recombinant ancestor on its clonal
    line, and sampled_between the sampled individuals that could account for
    that recombination in the inferred ARG (see recombinant_ancestors), as
    space-separated strains. ancestor_breakpoint is the ancestor's breakpoint,
    and ancestor_detectable whether the ancestor's sequence differs from both
    of its parents', without which the recombination leaves no trace.
    """
    df = pedigree.set_index("child", drop=False)

    def detectable(u):
        row = df.loc[u]
        seq = sequences[strain_name(u)]
        return (
            seq != sequences[strain_name(row.parent1)]
            and seq != sequences[strain_name(row.parent2)]
        )

    def breakpoint(u):
        return int(df.breakpoints[u])

    ancestors, between = recombinant_ancestors(pedigree, ids)
    rows = df.loc[ids]
    out = pd.DataFrame({
        "strain": [strain_name(c) for c in ids],
        "gen": rows.gen.values,
        "parent1": rows.parent1.values,
        "parent2": rows.parent2.values,
        "is_recombinant": rows.is_recombinant.astype(bool).values,
    })
    out["breakpoint"] = [
        breakpoint(c) if rec else -1 for c, rec in zip(ids, out.is_recombinant)
    ]
    out["recombinant_ancestor"] = ancestors
    out["ancestor_breakpoint"] = [breakpoint(a) if a >= 0 else -1 for a in ancestors]
    out["ancestor_detectable"] = [detectable(a) if a >= 0 else False for a in ancestors]
    out["sampled_between"] = [" ".join(strain_name(u) for u in path) for path in between]
    return out


def toml_value(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, list):
        return "[" + ", ".join(toml_value(v) for v in value) + "]"
    raise ValueError(f"Unsupported value: {value!r}")


def make_config(
    *, dataset, reference, run_id, results_dir, log_dir, matches_dir,
    num_mismatches, num_threads, extend_parameters
):
    top = {
        "dataset": dataset,
        "run_id": run_id,
        "results_dir": results_dir,
        "log_dir": log_dir,
        "matches_dir": matches_dir,
        "log_level": 2,
        "exclude_dates": [],
        "exclude_sites": [],
        "date_field": "date",
        "reference_fasta": reference,
        # The day before the founder (the reference) is sampled on START_DATE:
        # the reference must predate every sample.
        "reference_date": "2025-12-31",
    }
    extend = {
        **extend_parameters,
        "num_mismatches": num_mismatches,
        "num_threads": num_threads,
    }
    lines = [f"{k} = {toml_value(v)}" for k, v in top.items()]
    lines.append("")
    lines.append("[extend_parameters]")
    lines.extend(f"{k} = {toml_value(v)}" for k, v in extend.items())
    return "\n".join(lines) + "\n"


def true_strains(ts):
    """
    Return the strain names of the samples in a true ARG from sample_individuals.
    """
    return [strain_name(x) for x in sample_pedigree_ids(ts)]


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
    is expected to be a recombinant, added.

    A sample is placed if it has a node in the inferred ARG. Samples identical
    to a node already in the ARG are exact matches, added by sc2ts postprocess;
    samples without a node were held back by sc2ts. A sample is inferred to be a
    recombinant if its HMM match has more than one parent. Breakpoints are
    converted to SLiM's 0-based coordinates, where the breakpoint is the first
    position inherited from the second parent.
    """
    nodes = {}
    for u in inferred_ts.samples():
        md = inferred_ts.node(u).metadata
        path = md["sc2ts"]["hmm_match"]["path"]
        row = {
            "placed": True,
            "exact_match": bool(
                inferred_ts.nodes_flags[u] & sc2ts.core.NODE_IS_EXACT_MATCH
            ),
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
        nodes[md["strain"]] = row
    df = truth.copy()
    placed = pd.DataFrame.from_dict(nodes, orient="index")
    df = df.join(placed, on="strain")
    for col in ["placed", "exact_match", "inferred_recombinant"]:
        df[col] = df[col].astype("boolean").fillna(False).astype(bool)
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
    Return the number of samples in each generation, and how many were placed
    in the ARG and were exact matches.
    """
    return (
        samples.groupby("gen")
        .agg(
            num_samples=("strain", "size"),
            num_placed=("placed", "sum"),
            num_exact_matches=("exact_match", "sum"),
        )
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
        "num_exact_matches": int(np.sum(df.exact_match)),
        "num_held_back": len(samples) - len(df),
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


@click.group()
def cli():
    pass


@cli.command()
@click.argument("input_ts", type=click.Path(exists=True, dir_okay=False))
@click.argument("output_ts", type=click.Path(dir_okay=False))
@click.option("--probability", type=float, required=True)
@click.option("--seed", type=int, required=True)
def sample(input_ts, output_ts, probability, seed):
    """
    Sample individuals from all generations of a SLiM simulation and simplify
    their genealogy to give the true ARG.
    """
    ts = tskit.load(input_ts)
    sample_individuals(ts, probability, seed).dump(output_ts)


@cli.command()
@click.argument("sampled_ts", type=click.Path(exists=True, dir_okay=False))
@click.argument("fasta", type=click.Path(exists=True, dir_okay=False))
@click.argument("pedigree", type=click.Path(exists=True, dir_okay=False))
@click.argument("output_fasta", type=click.Path(dir_okay=False))
@click.argument("output_metadata", type=click.Path(dir_okay=False))
def export_samples(sampled_ts, fasta, pedigree, output_fasta, output_metadata):
    """
    Export the sequences and sc2ts metadata of the individuals sampled in a
    simplified SLiM tree sequence.
    """
    ids = sample_pedigree_ids(tskit.load(sampled_ts))
    df = pd.read_csv(pedigree, sep="\t").set_index("child").loc[ids]
    sequences = read_fasta(fasta, [strain_name(x) for x in ids])

    with open(output_fasta, "w") as f:
        for pedigree_id in ids:
            name = strain_name(pedigree_id)
            print(f">{name}\n{sequences[name]}", file=f)

    metadata = pd.DataFrame({
        "Run": [strain_name(c) for c in ids],
        "date": [str(START_DATE + datetime.timedelta(days=int(g))) for g in df.gen],
    })
    metadata.to_csv(output_metadata, sep="\t", index=False)


@cli.command()
@click.argument("sampled_ts", type=click.Path(exists=True, dir_okay=False))
@click.argument("fasta", type=click.Path(exists=True, dir_okay=False))
@click.argument("pedigree", type=click.Path(exists=True, dir_okay=False))
@click.argument("output", type=click.Path(dir_okay=False))
def make_truth(sampled_ts, fasta, pedigree, output):
    """
    Record the true recombination history of each sampled individual from the
    SLiM pedigree.
    """
    ids = sample_pedigree_ids(tskit.load(sampled_ts))
    pedigree = pd.read_csv(pedigree, sep="\t")
    names = [strain_name(x) for x in required_sequences(pedigree, ids)]
    sequences = read_fasta(fasta, names)
    truth_table(pedigree, ids, sequences).to_csv(output, index=False)


@cli.command()
@click.argument("output", type=click.Path(dir_okay=False))
@click.option("--dataset", required=True)
@click.option("--reference", required=True)
@click.option("--run-id", required=True)
@click.option("--results-dir", required=True)
@click.option("--log-dir", required=True)
@click.option("--matches-dir", required=True)
@click.option("--num-mismatches", type=int, required=True)
@click.option("--num-threads", type=int, required=True)
@click.option(
    "--extend-parameters",
    default="{}",
    help="JSON object of other sc2ts extend parameters",
)
def write_sc2ts_config(
    output, dataset, reference, run_id, results_dir, log_dir, matches_dir,
    num_mismatches, num_threads, extend_parameters
):
    """
    Write the TOML config file for an sc2ts inference run.
    """
    config = make_config(
        dataset=dataset,
        reference=reference,
        run_id=run_id,
        results_dir=results_dir,
        log_dir=log_dir,
        matches_dir=matches_dir,
        num_mismatches=num_mismatches,
        num_threads=num_threads,
        extend_parameters=json.loads(extend_parameters),
    )
    with open(output, "w") as f:
        f.write(config)


@cli.command()
@click.argument("true_ts", type=click.Path(exists=True, dir_okay=False))
@click.argument("inferred_ts", type=click.Path(exists=True, dir_okay=False))
@click.argument("truth", type=click.Path(exists=True, dir_okay=False))
@click.argument("output_dir", type=click.Path(file_okay=False))
@click.option("--pathogen", required=True)
@click.option("--rep", type=int, required=True)
@click.option("--p", "probability", type=float, required=True)
@click.option("--k", type=int, required=True)
def evaluate(true_ts, inferred_ts, truth, output_dir, pathogen, rep, probability, k):
    """
    Score an sc2ts ARG inferred from simulated sequences against the true ARG:
    overall accuracy with tscompare, and recombinant detection against the
    pedigree.

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
    cli()
