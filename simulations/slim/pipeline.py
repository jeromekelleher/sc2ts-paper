"""
The steps of the SLiM simulation pipeline after running SLiM itself, as click
subcommands: sample, export-samples, make-truth, write-sc2ts-config and
evaluate. See README.md.

Run from simulations/slim/, e.g.:
    python pipeline.py sample sim.slim.ts sim.slim.samples.tsv true.trees \
        --samples-per-day 100
"""
import collections
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


def read_candidates(path):
    """
    Return the candidate samples SLiM wrote, indexed by pedigree ID.
    """
    return pd.read_csv(path, sep="\t").set_index("id")


def sample_individuals(ts, candidates, samples_per_day):
    """
    Return the genealogy of samples_per_day individuals from each generation,
    simplified so the founder is sample node 0. SLiM puts each generation in a
    random order and keeps the first few as candidates, so these are the
    candidates of rank below samples_per_day: a uniform random sample of each
    generation, nested as samples_per_day increases. The founder is the only
    individual in its generation, so is always sampled.
    """
    if samples_per_day < 1:
        raise ValueError("samples_per_day must be at least 1.")
    founder = find_founder(ts)
    selected = set(candidates.index[candidates["rank"] < samples_per_day])
    # Haploid individuals still have a second, vacant, node in SLiM. These have
    # no ancestry, so drop them as samples and keep one node per individual.
    ts = pyslim.remove_vacant(ts)
    samples = ts.samples()
    pedigree_ids = np.array(
        [ind.metadata["pedigree_id"] for ind in ts.individuals()], dtype=int
    )
    samples = samples[
        np.isin(pedigree_ids[ts.nodes_individual[samples]], list(selected))
    ]
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


def truth_table(ids, candidates, recombinants):
    """
    Return the true recombination history of the sampled individuals, from the
    ancestry SLiM tracks for each candidate.

    is_recombinant and breakpoint describe the sample itself. The
    recombinant_ancestor is its nearest recombinant ancestor on its clonal line
    (itself, if it is a recombinant), or -1 if there is none. sampled_between
    lists, as space-separated strains, the candidates on the clonal line between
    the sample and that ancestor, including the ancestor if it is a candidate,
    found by following prev_candidate. If any of those are in the inferred ARG
    they account for the recombination; otherwise the sample should be inferred
    to be a recombinant. ancestor_breakpoint is the ancestor's breakpoint, and
    ancestor_detectable whether the ancestor's sequence differs from both of
    its parents', without which the recombination leaves no trace.
    """
    rows = candidates.loc[ids]
    recombinants = recombinants.set_index("id")
    ancestors = rows.recombinant_ancestor.values
    between = []
    for u, ancestor in zip(ids, ancestors):
        path = []
        if ancestor >= 0:
            u = candidates.prev_candidate[u]
            while u >= 0:
                path.append(u)
                u = candidates.prev_candidate[u]
        between.append(" ".join(strain_name(v) for v in path))
    has_ancestor = ancestors >= 0
    out = pd.DataFrame({
        "strain": [strain_name(c) for c in ids],
        "gen": rows.gen.values,
        "is_recombinant": rows.is_recombinant.astype(bool).values,
        "recombinant_ancestor": ancestors,
    })
    ancestor_breakpoint = np.full(len(ids), -1)
    ancestor_breakpoint[has_ancestor] = recombinants.breakpoint[ancestors[has_ancestor]]
    out["breakpoint"] = np.where(out.is_recombinant, ancestor_breakpoint, -1)
    out["ancestor_breakpoint"] = ancestor_breakpoint
    ancestor_detectable = np.zeros(len(ids), dtype=bool)
    ancestor_detectable[has_ancestor] = recombinants.detectable[
        ancestors[has_ancestor]
    ].astype(bool)
    out["ancestor_detectable"] = ancestor_detectable
    out["sampled_between"] = between
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


def make_samples_leaves(ts):
    """
    Return a copy of the ARG in which every sample is a leaf. Each sample with
    children is replaced by a new non-sample node just above it, which takes
    its parents, mutations and children, and the sample becomes the new node's
    child over the whole sequence. Whether a sample is an ancestor, or a
    sibling of its descendants under an identical ancestor, can't be told from
    sequences, and tscompare matches samples only to themselves, so this
    puts both ARGs on the same footing. Mutation times are dropped, as moved
    mutations may no longer fit them. Sample IDs don't change.
    """
    tables = ts.dump_tables()
    tables.mutations.time = np.full(tables.mutations.num_rows, tskit.UNKNOWN_TIME)
    edges_parent = ts.edges_parent.copy()
    edges_child = ts.edges_child.copy()
    mutations_node = ts.mutations_node.copy()
    has_children = np.zeros(ts.num_nodes, dtype=bool)
    has_children[ts.edges_parent] = True
    leaf_edges = []
    for sample in ts.samples()[has_children[ts.samples()]]:
        is_parent_edge = ts.edges_child == sample
        time = ts.nodes_time[sample]
        # The new node must be younger than all of the sample's parents.
        offset = 1e-6
        if np.any(is_parent_edge):
            gap = np.min(ts.nodes_time[ts.edges_parent[is_parent_edge]]) - time
            offset = min(offset, gap / 2)
        node = tables.nodes.add_row(time=time + offset)
        edges_child[is_parent_edge] = node
        edges_parent[ts.edges_parent == sample] = node
        mutations_node[ts.mutations_node == sample] = node
        leaf_edges.append((0, ts.sequence_length, node, sample))
    tables.edges.parent = edges_parent
    tables.edges.child = edges_child
    for edge in leaf_edges:
        tables.edges.add_row(*edge)
    tables.mutations.node = mutations_node
    tables.sort()
    tables.build_index()
    tables.compute_mutation_parents()
    return tables.tree_sequence()


def collapse_unsupported(ts):
    """
    Return a copy of the ARG with the non-sample nodes that sequences can't
    resolve removed. A node with no mutations and only one parent has the same
    haplotype as its parent, so whether its children descend from it or from
    the parent can't be told apart. The children of each such node are
    attached to its parent over each interval where it has one, and left where
    they are elsewhere (where the node is a root). Nodes with more than one
    parent are kept, as their mosaic haplotypes differ from each parent's.
    Samples are kept, so should first be made leaves with make_samples_leaves.
    Sample IDs don't change.
    """
    num_mutations = np.bincount(ts.mutations_node, minlength=ts.num_nodes)
    by_parent = collections.defaultdict(list)
    by_child = collections.defaultdict(list)
    for edge in ts.edges():
        row = (edge.left, edge.right, edge.parent, edge.child)
        by_parent[edge.parent].append(row)
        by_child[edge.child].append(row)
    num_parents = {u: len({e[2] for e in edges}) for u, edges in by_child.items()}

    # Youngest first, so the children attached to a removed node's parent are
    # moved again if it's removed too.
    for u in np.argsort(ts.nodes_time, kind="stable"):
        if (
            ts.node(u).is_sample()
            or num_mutations[u] > 0
            or num_parents.get(u, 0) > 1
        ):
            continue
        parent_edges = by_child[u]
        for left, right, _, child in by_parent.pop(u, []):
            by_child[child].remove((left, right, u, child))
            new_edges = []
            # Each piece of [left, right) is attached to u's parent there, if
            # it has one, and otherwise stays on u.
            for p_left, p_right, parent, _ in sorted(parent_edges):
                if p_right <= left or p_left >= right:
                    continue
                if p_left > left:
                    new_edges.append((left, p_left, u, child))
                new_edges.append((max(left, p_left), min(right, p_right), parent, child))
                left = min(right, p_right)
            if left < right:
                new_edges.append((left, right, u, child))
            for edge in new_edges:
                by_parent[edge[2]].append(edge)
                by_child[child].append(edge)

    tables = ts.dump_tables()
    tables.edges.clear()
    for edges in by_parent.values():
        for left, right, parent, child in edges:
            tables.edges.add_row(left, right, parent, child)
    tables.edges.squash()
    tables.sort()
    tables.simplify(ts.samples())
    return tables.tree_sequence()


def score_arg(true_ts, inferred_ts):
    """
    Return tscompare's ARF, TPR and RMSE for the inferred ARG against the true
    one, along with ARF and TPR over non-sample nodes only. tscompare matches
    each sample only to itself, which for samples without descendants is
    always correct, so samples inflate TPR and deflate ARF. For the "internal"
    versions, each non-sample node is credited with the span it shares with
    its best match in the other ARG (as tscompare does for ARF), weighted by
    its span; they are NaN if an ARG has no non-sample nodes.
    """
    result = tscompare.haplotype_arf(inferred_ts, true_ts)

    def matched_fraction(ts, other):
        is_internal = np.ones(ts.num_nodes, dtype=bool)
        is_internal[ts.samples()] = False
        span = tscompare.node_spans(ts, include_missing=True)[is_internal]
        matched = tscompare.match_node_ages(ts, other)[1][is_internal]
        return np.sum(matched) / np.sum(span) if np.sum(span) > 0 else np.nan

    return {
        "arf": result.arf,
        "tpr": result.tpr,
        "rmse": result.rmse,
        "arf_internal": 1 - matched_fraction(inferred_ts, true_ts),
        "tpr_internal": matched_fraction(true_ts, inferred_ts),
    }


@click.group()
def cli():
    pass


@cli.command()
@click.argument("input_ts", type=click.Path(exists=True, dir_okay=False))
@click.argument("candidates", type=click.Path(exists=True, dir_okay=False))
@click.argument("output_ts", type=click.Path(dir_okay=False))
@click.option("--samples-per-day", type=int, required=True)
def sample(input_ts, candidates, output_ts, samples_per_day):
    """
    Sample individuals from each generation of a SLiM simulation and simplify
    their genealogy to give the true ARG.
    """
    ts = tskit.load(input_ts)
    sample_individuals(ts, read_candidates(candidates), samples_per_day).dump(output_ts)


@cli.command()
@click.argument("sampled_ts", type=click.Path(exists=True, dir_okay=False))
@click.argument("fasta", type=click.Path(exists=True, dir_okay=False))
@click.argument("candidates", type=click.Path(exists=True, dir_okay=False))
@click.argument("output_fasta", type=click.Path(dir_okay=False))
@click.argument("output_metadata", type=click.Path(dir_okay=False))
def export_samples(sampled_ts, fasta, candidates, output_fasta, output_metadata):
    """
    Export the sequences and sc2ts metadata of the individuals sampled in a
    simplified SLiM tree sequence.
    """
    ids = sample_pedigree_ids(tskit.load(sampled_ts))
    gens = read_candidates(candidates).gen[ids]
    sequences = read_fasta(fasta, [strain_name(x) for x in ids])

    with open(output_fasta, "w") as f:
        for pedigree_id in ids:
            name = strain_name(pedigree_id)
            print(f">{name}\n{sequences[name]}", file=f)

    metadata = pd.DataFrame({
        "Run": [strain_name(c) for c in ids],
        "date": [str(START_DATE + datetime.timedelta(days=int(g))) for g in gens],
    })
    metadata.to_csv(output_metadata, sep="\t", index=False)


@cli.command()
@click.argument("sampled_ts", type=click.Path(exists=True, dir_okay=False))
@click.argument("candidates", type=click.Path(exists=True, dir_okay=False))
@click.argument("recombinants", type=click.Path(exists=True, dir_okay=False))
@click.argument("output", type=click.Path(dir_okay=False))
def make_truth(sampled_ts, candidates, recombinants, output):
    """
    Record the true recombination history of each sampled individual, from the
    ancestry SLiM tracked.
    """
    ids = sample_pedigree_ids(tskit.load(sampled_ts))
    truth = truth_table(
        ids, read_candidates(candidates), pd.read_csv(recombinants, sep="\t")
    )
    truth.to_csv(output, index=False)


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
@click.option("--samples-per-day", type=int, required=True)
@click.option("--k", type=int, required=True)
def evaluate(
    true_ts, inferred_ts, truth, output_dir, pathogen, rep, samples_per_day, k
):
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

    run_params = {
        "pathogen": pathogen,
        "rep": rep,
        "samples_per_day": samples_per_day,
        "k": k,
    }
    row = dict(run_params)
    row.update(score_recombinants(samples, events))
    row["num_recombinant_nodes"] = num_recombinant_nodes(inferred_ts)
    true_ts, inferred_ts = prepare_for_comparison(true_ts, inferred_ts)
    row.update(score_arg(true_ts, inferred_ts))
    resolved = score_arg(
        *[collapse_unsupported(make_samples_leaves(ts)) for ts in (true_ts, inferred_ts)]
    )
    row.update({f"{key}_resolved": value for key, value in resolved.items()})
    pd.DataFrame([row]).to_csv(output_dir / "evaluation.csv", index=False)
    for name, df in [("events", events), ("placement", placement)]:
        for j, (column, value) in enumerate(run_params.items()):
            df.insert(j, column, value)
        df.to_csv(output_dir / f"{name}.csv", index=False)


if __name__ == "__main__":
    cli()
