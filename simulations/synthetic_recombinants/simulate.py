"""
Estimate the false positive and detection rates of sc2ts recombinant matching
as a function of the recombination penalty k.

Synthetic single-breakpoint recombinants are spliced from pairs of real
sequences collected on one day, and matched against the raw inference ARG for
the day before. Real sequences held out of the parent pool act as
non-recombinant controls, so a control matched to more than one parent is a
false positive.

Recombinants are characterised with the same quantities the paper reports:
breakpoint intervals from sc2ts's own characterise_recombinants, net supporting
loci on each flank, and the pangonet distance between the parents.
"""

import collections
import json
import sys
from pathlib import Path

import click
import numpy as np
import pandas as pd
import tskit
import tszip
import zarr

import sc2ts
from sc2ts.dataset import decode_alleles
from sc2ts.inference import (
    HmmMatch,
    Sample,
    characterise_recombinants,
)

# Unambiguous bases come first in core.IUPAC_ALLELES; -1 is missing data. Only
# sites where both parents are unambiguous and differ can distinguish them.
NUM_ACGT = 4

# The paper clusters sites 3 or fewer bases apart into a single "locus" and
# gates recombinants on at least 4 net supporting loci on both flanks.
ADJACENT_DISTANCE = 3
NET_SUPPORTING_LOCI_CUTOFF = 4

# Majority lineage of a matched parent node is taken over at most this many of
# its descendant samples.
MAX_DESCENDANTS = 1000


def is_acgt(a):
    return (a >= 0) & (a < NUM_ACGT)


def classify_scorpio(scorpio):
    """
    Alpha/Delta label from a Viridian scorpio call. Missing is the literal
    string ".".

    The Alpha test is a containment rather than a prefix because a handful of
    samples are called "B.1.1.7-like+E484K", with no WHO name in front.
    """
    if "B.1.1.7-like" in scorpio:
        return "Alpha"
    if scorpio.startswith("Delta ("):
        return "Delta"
    return "Other"


def load_pangonet(pangonet_data):
    """
    Build the pango network from the pinned copies in arg_postprocessing, and
    return it with the repo's own distance function.
    """
    pangonet_data = Path(pangonet_data)
    sys.path.insert(0, str(pangonet_data.parent / "scripts"))
    from add_pangonet_distance_to_csv import get_pangonet_distance
    from pangonet.pangonet import PangoNet

    pango = PangoNet().build(
        alias_key=str(pangonet_data / "alias_key.json"),
        lineage_notes=str(pangonet_data / "lineage_notes.txt"),
    )
    return pango, get_pangonet_distance


@click.group()
def cli():
    pass


@cli.command()
@click.argument("dataset", type=click.Path(exists=True))
@click.argument("reference_arg", type=click.Path(exists=True))
@click.argument("output", type=click.Path(dir_okay=False))
@click.option("--date", required=True)
@click.option("--country", required=True)
def make_pool(dataset, reference_arg, output, date, country):
    """
    Build the pool of real samples that recombinant parents and controls are
    drawn from: those collected in one country on one date, labelled Alpha or
    Delta, and accepted by the published inference.
    """
    root = zarr.open(dataset, mode="r")
    fields = ["id", "Date_tree", "Country", "Viridian_pangolin", "Viridian_scorpio"]
    df = pd.DataFrame({name: root[f"sample_{name}"][:] for name in fields})
    df = df[(df.Country == country) & (df.Date_tree == date)]
    df["lineage"] = df.Viridian_scorpio.astype(str).map(classify_scorpio)
    df = df[df.lineage != "Other"]

    ts = tszip.load(reference_arg)
    accepted = {ts.node(u).metadata["sample_id"] for u in ts.samples()}
    df = df[df.id.isin(accepted)]

    df = df[["id", "lineage", "Viridian_pangolin", "Viridian_scorpio"]].rename(
        columns={"id": "strain"}
    )
    df = df.sort_values("strain")
    df.to_csv(output, index=False)
    print(df.lineage.value_counts().to_string())


@cli.command()
@click.argument("pool", type=click.Path(exists=True, dir_okay=False))
@click.argument("dataset", type=click.Path(exists=True))
@click.argument("arg", type=click.Path(exists=True, dir_okay=False))
@click.option("--out-prefix", required=True)
@click.option("--date", required=True)
@click.option("--crosses", required=True, help="e.g. 'Alpha:Delta,Alpha:Alpha'")
@click.option("--num-recombinants", type=int, required=True, help="per cross")
@click.option("--num-controls", type=int, default=-1, help="-1 for all")
@click.option("--pangonet-data", type=click.Path(exists=True, file_okay=False))
@click.option("--seed", type=int, required=True)
@click.option("--max-attempts", type=int, default=100)
def generate(
    pool,
    dataset,
    arg,
    out_prefix,
    date,
    crosses,
    num_recombinants,
    num_controls,
    pangonet_data,
    seed,
    max_attempts,
):
    """
    Make synthetic recombinants and controls, and write them out for import as
    a sc2ts dataset.

    The pool is split in half within each lineage: one half supplies parents,
    the other controls, so no control is a parent of any recombinant.
    """
    rng = np.random.default_rng(seed)
    df_pool = pd.read_csv(pool)
    crosses = [tuple(cross.split(":")) for cross in crosses.split(",")]
    pango, pangonet_distance = load_pangonet(pangonet_data)

    # Only sites in the ARG are visible to the HMM.
    arg_sites = tszip.load(arg).sites_position.astype(int)
    arg_index = arg_sites - 1

    # Each alignment lookup decompresses a whole chunk of samples, so fetch in
    # index order with a cache big enough to hold every chunk the pool touches.
    ds = sc2ts.Dataset(dataset, chunk_cache_size=16)
    order = sorted(df_pool.strain, key=lambda strain: ds.metadata.sample_id_map[strain])
    alignments = {strain: ds.alignment[strain] for strain in order}
    sequence_length = len(next(iter(alignments.values())))

    parents = {}
    controls = []
    for lineage, group in df_pool.groupby("lineage"):
        strains = rng.permutation(group.strain.to_numpy())
        split = len(strains) // 2
        parents[lineage] = strains[:split]
        controls.extend(strains[split:])
    controls = sorted(controls)
    if num_controls >= 0:
        controls = sorted(rng.choice(controls, size=num_controls, replace=False))

    pango_of = dict(zip(df_pool.strain, df_pool.Viridian_pangolin))
    lineage_of = dict(zip(df_pool.strain, df_pool.lineage))
    scorpio_of = dict(zip(df_pool.strain, df_pool.Viridian_scorpio))

    def informative_sites(left, right):
        a = alignments[left][arg_index]
        b = alignments[right][arg_index]
        return arg_sites[(a != b) & is_acgt(a) & is_acgt(b)]

    records = []
    sequences = {}
    for cross in crosses:
        for j in range(num_recombinants):
            # Draw a usable parent pair: distinct samples with at least two
            # sites telling them apart, so that some breakpoint is detectable.
            for attempt in range(max_attempts):
                left_lineage, right_lineage = (
                    cross if rng.random() < 0.5 else cross[::-1]
                )
                left = rng.choice(parents[left_lineage])
                right = rng.choice(parents[right_lineage])
                if left == right:
                    continue
                sites = informative_sites(left, right)
                if len(sites) >= 2:
                    break
            else:
                raise ValueError(
                    f"No usable parent pair for {cross} after {max_attempts} attempts"
                )

            # A breakpoint is detectable exactly when it leaves at least one
            # distinguishing site on each flank, i.e. sites[0] < bp <= sites[-1].
            # Drawing uniformly from that window is the same as drawing over the
            # genome and rejecting, without the loop.
            breakpoint = int(rng.integers(sites[0] + 1, sites[-1] + 1))

            left_alignment = alignments[left]
            right_alignment = alignments[right]
            sequence = np.concatenate(
                [left_alignment[: breakpoint - 1], right_alignment[breakpoint - 1 :]]
            )
            strain = f"rec_{len(records):05d}"
            sequences[strain] = sequence

            records.append(
                {
                    "strain": strain,
                    "type": "recombinant",
                    "cross": "x".join(cross),
                    "left_parent": left,
                    "right_parent": right,
                    "left_lineage": left_lineage,
                    "right_lineage": right_lineage,
                    "left_pango": pango_of[left],
                    "right_pango": pango_of[right],
                    "parent_pangonet_distance": pangonet_distance(
                        pango, pango_of[left], pango_of[right]
                    ),
                    "breakpoint": breakpoint,
                    "num_informative_left": int(np.sum(sites < breakpoint)),
                    "num_informative_right": int(np.sum(sites >= breakpoint)),
                    # The span of breakpoints that are detectable at all, which
                    # the draw above was conditioned on.
                    "detectable_window": int(sites[-1] - sites[0]),
                    "num_missing": int(np.sum(sequence[arg_index] < 0)),
                }
            )

    for strain in controls:
        sequences[strain] = alignments[strain]
        records.append(
            {
                "strain": strain,
                "type": "control",
                "cross": "",
                "left_parent": "",
                "right_parent": "",
                "left_lineage": lineage_of[strain],
                "right_lineage": lineage_of[strain],
                "left_pango": pango_of[strain],
                "right_pango": pango_of[strain],
                "parent_pangonet_distance": -1,
                "breakpoint": -1,
                "num_informative_left": -1,
                "num_informative_right": -1,
                "detectable_window": -1,
                "num_missing": int(np.sum(alignments[strain][arg_index] < 0)),
            }
        )

    df_truth = pd.DataFrame(records)
    df_truth.to_csv(f"{out_prefix}.truth.csv", index=False)

    with open(f"{out_prefix}.fasta", "w") as f:
        for strain, sequence in sequences.items():
            print(f">{strain}", file=f)
            print("".join(decode_alleles(sequence)), file=f)

    # run-hmm does not use the metadata, but storing it keeps the dataset
    # self-describing. Blanks would be read back as NaN and rejected by the
    # Zarr string encoding, so controls get an explicit "none" cross.
    df_truth.assign(
        Run=df_truth.strain, date=date, cross=df_truth.cross.replace("", "none")
    )[
        ["Run", "date", "type", "cross", "left_lineage", "right_lineage", "breakpoint"]
    ].to_csv(f"{out_prefix}.metadata.tsv", sep="\t", index=False)

    with open(f"{out_prefix}.strains.txt", "w") as f:
        for strain in sequences:
            print(strain, file=f)

    counts = df_truth[df_truth.type == "recombinant"].cross.value_counts()
    print(f"{counts.to_dict()} recombinants, {len(controls)} controls")


def node_allele_chars(ts, nodes):
    """
    The allele at every site in the ARG, as a character, for each of the given
    nodes. Characters rather than genotype indices so that parents can be
    compared against a recombinant's own IUPAC-coded sequence.
    """
    chars = np.empty((len(nodes), ts.num_sites), dtype="U1")
    for var in ts.variants(samples=list(nodes), isolated_as_missing=False):
        alleles = np.array(
            [allele if allele is not None else "N" for allele in var.alleles],
            dtype="U1",
        )
        genotypes = var.genotypes
        chars[:, var.site.id] = np.where(
            genotypes < 0, "N", alleles[np.maximum(genotypes, 0)]
        )
    return chars


def net_supporting_loci(recombinant, parents, positions, path):
    """
    Net number of loci supporting the assigned parent on each flank.

    A port of recombinant_supporting_locations in
    arg_postprocessing/scripts/add_recombinant_minlength_to_csv.py, which works
    on recombinant nodes already in an ARG. Here the recombinant is not in the
    ARG, so the same scoring is applied to its own sequence against the
    haplotypes of the parents the HMM assigned it.

    Sites where the parents agree carry no information. Among the rest, sites
    within ADJACENT_DISTANCE of the previous one on the same flank are folded
    into the same locus and counted once. Each locus scores +1 if the
    recombinant carries the assigned parent's allele, -1 if it carries the
    other parent's, and 0 if it carries neither (a de novo mutation).
    """
    counts = np.zeros(len(path), dtype=int)
    last_position = np.full(len(path), -np.inf)
    informative = np.where(parents[0] != parents[1])[0]
    segment = 0
    for site_index in informative:
        position = positions[site_index]
        while path[segment].right <= position:
            segment += 1
        assigned = parents[segment][site_index]
        other = parents[1 - segment][site_index]
        allele = recombinant[site_index]
        if position - last_position[segment] > ADJACENT_DISTANCE:
            if allele == assigned:
                counts[segment] += 1
            elif allele == other:
                counts[segment] -= 1
        last_position[segment] = position
    return counts


def parent_lineages(ts, queries):
    """
    Label each (position, node) query with the majority Alpha/Delta lineage
    among the node's descendant samples in the tree covering that position.
    """
    sample_lineage = {}
    for u in ts.samples():
        scorpio = ts.node(u).metadata.get("Viridian_scorpio")
        if scorpio is not None:
            sample_lineage[u] = classify_scorpio(scorpio)

    result = {}
    tree = tskit.Tree(ts, sample_lists=True)
    for position, node in sorted(queries):
        tree.seek(position)
        counts = collections.Counter()
        for j, u in enumerate(tree.samples(node)):
            if j == MAX_DESCENDANTS:
                break
            counts[sample_lineage.get(u, "Other")] += 1
        result[(position, node)] = (
            counts.most_common(1)[0][0] if len(counts) > 0 else "Other",
            tree.num_samples(node),
        )
    return result


@cli.command()
@click.argument("truth", type=click.Path(exists=True, dir_okay=False))
@click.argument("dataset", type=click.Path(exists=True))
@click.argument("arg", type=click.Path(exists=True, dir_okay=False))
@click.argument("output", type=click.Path(dir_okay=False))
@click.argument("hmm_files", nargs=-1, type=click.Path(exists=True, dir_okay=False))
def summarise(truth, dataset, arg, output, hmm_files):
    """
    Join the run-hmm output for each k with the simulation truth, giving one
    row per (strain, k).
    """
    df_truth = pd.read_csv(truth, keep_default_na=False).set_index("strain")
    ts = tszip.load(arg)
    positions = ts.sites_position.astype(int)
    arg_index = positions - 1
    ds = sc2ts.Dataset(dataset)

    runs = []
    for path in hmm_files:
        with open(path) as f:
            for line in f:
                run = json.loads(line)
                run["match"] = HmmMatch.fromdict(run["match"])
                runs.append(run)

    # Breakpoint intervals, using the function the inference pipeline itself
    # calls (sc2ts.inference._extend). It derives both edges of the interval
    # from the parents' haplotypes, and asserts that the forward HMM landed on
    # an informative site.
    for run in runs:
        sample = Sample(run["strain"])
        sample.hmm_match = run["match"]
        run["sample"] = sample
    characterise_recombinants(ts, [run["sample"] for run in runs])

    recombinant_nodes = sorted(
        {segment.parent for run in runs for segment in run["match"].path}
    )
    node_index = {node: j for j, node in enumerate(recombinant_nodes)}
    chars = node_allele_chars(ts, recombinant_nodes)

    queries = {
        ((segment.left + segment.right) // 2, segment.parent)
        for run in runs
        for segment in run["match"].path
    }
    lineages = parent_lineages(ts, queries)

    rows = []
    for run in runs:
        match = run["match"]
        path = match.path
        k = int(run["num_mismatches"])
        match.compute_cost(k)
        truth_row = df_truth.loc[run["strain"]]
        labels = [
            lineages[((segment.left + segment.right) // 2, segment.parent)]
            for segment in path
        ]

        row = {
            "strain": run["strain"],
            "k": k,
            "type": truth_row.type,
            "cross": truth_row.cross,
            "num_parents": len(path),
            "detected": len(path) > 1,
            "num_mutations": len(match.mutations),
            "hmm_cost": int(match.cost),
            "parents": "|".join(str(segment.parent) for segment in path),
            "parent_left_scorpio": labels[0][0],
            "parent_right_scorpio": labels[-1][0],
            "parent_num_samples": "|".join(str(n) for _, n in labels),
        }

        if len(path) == 2:
            interval_left, interval_right = run["sample"].breakpoint_intervals[0]
            row["interval_left"] = interval_left
            row["interval_right"] = interval_right
            row["interval_width"] = interval_right - interval_left
            # The breakpoint cannot be resolved more finely than this interval,
            # so containment is the right notion of a correct breakpoint.
            row["breakpoint_in_interval"] = bool(
                interval_left <= truth_row.breakpoint <= interval_right
            )
            sequence = ds.alignment[run["strain"]][arg_index]
            counts = net_supporting_loci(
                decode_alleles(sequence),
                [chars[node_index[segment.parent]] for segment in path],
                positions,
                path,
            )
            row["net_min_supporting_loci_lft"] = int(counts[0])
            row["net_min_supporting_loci_rgt"] = int(counts[1])
            row[f"net_min_supporting_loci_lft_rgt_ge_{NET_SUPPORTING_LOCI_CUTOFF}"] = (
                bool(min(counts) >= NET_SUPPORTING_LOCI_CUTOFF)
            )
        rows.append(row)

    df = pd.DataFrame(rows).merge(
        df_truth.reset_index().drop(columns=["type", "cross"]), on="strain", how="left"
    )
    df = df.sort_values(["k", "type", "cross", "strain"])
    df.to_csv(output, index=False)

    qc_column = f"net_min_supporting_loci_lft_rgt_ge_{NET_SUPPORTING_LOCI_CUTOFF}"
    for (k, group_type), group in df.groupby(["k", "type"]):
        if group_type == "control":
            print(
                f"k={k} control: false positives "
                f"{group.detected.sum()}/{len(group)}, "
                f"passing QC {group[qc_column].fillna(False).sum()}"
            )
        else:
            for cross, sub in group.groupby("cross"):
                print(
                    f"k={k} {cross}: detected "
                    f"{sub.detected.sum()}/{len(sub)} "
                    f"({sub.detected.mean():.2f}), "
                    f"passing QC {sub[qc_column].fillna(False).sum()}"
                )


if __name__ == "__main__":
    cli()
