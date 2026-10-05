"""
Join the run-hmm output for each value of k with the simulation truth, giving
one row per (strain, k).
"""

import collections
import json

import click
import numpy as np
import pandas as pd
import tskit
import tszip

from lineage import classify

# Majority lineage of a parent node is taken over at most this many of its
# descendant samples. The traversal order is not random, so for a node high in
# the tree the label is indicative only; parent_num_samples says when that is
# the case.
MAX_DESCENDANTS = 1000


def node_lineages(ts, queries):
    """
    Label each (position, node) query with the majority Alpha/Delta lineage
    among the node's descendant samples in the tree covering that position.
    """
    sample_lineage = {}
    for u in ts.samples():
        pango = ts.node(u).metadata.get("Viridian_pangolin")
        if pango is not None:
            sample_lineage[u] = classify(pango)

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


@click.command()
@click.argument("truth", type=click.Path(exists=True, dir_okay=False))
@click.argument("arg", type=click.Path(exists=True, dir_okay=False))
@click.argument("output", type=click.Path(dir_okay=False))
@click.argument("hmm_files", nargs=-1, type=click.Path(exists=True, dir_okay=False))
def main(truth, arg, output, hmm_files):
    df_truth = pd.read_csv(truth, keep_default_na=False).set_index("strain")
    ts = tszip.load(arg)

    runs = []
    queries = set()
    for path in hmm_files:
        with open(path) as f:
            for line in f:
                run = json.loads(line)
                runs.append(run)
                for segment in run["match"]["path"]:
                    midpoint = (segment["left"] + segment["right"]) // 2
                    queries.add((midpoint, segment["parent"]))

    lineages = node_lineages(ts, queries)

    rows = []
    for run in runs:
        match = run["match"]
        path = match["path"]
        k = int(run["num_mismatches"])
        num_mutations = len(match["mutations"])
        labels = [lineages[((s["left"] + s["right"]) // 2, s["parent"])] for s in path]
        parent_lineages = [label for label, _ in labels]
        # Switch positions between consecutive segments.
        inferred = [s["left"] for s in path[1:]]
        truth_row = df_truth.loc[run["strain"]]

        row = {
            "strain": run["strain"],
            "k": k,
            "type": truth_row.type,
            "num_parents": len(path),
            "detected": len(path) > 1,
            "num_mutations": num_mutations,
            # Matches sc2ts.HmmMatch.compute_cost
            "hmm_cost": k * (len(path) - 1) + num_mutations,
            "parents": "|".join(str(s["parent"]) for s in path),
            "parent_lineages": "|".join(parent_lineages),
            "parent_num_samples": "|".join(str(n) for _, n in labels),
            "inferred_breakpoints": "|".join(str(b) for b in inferred),
        }
        if truth_row.type == "recombinant":
            row["lineages_correct"] = sorted(set(parent_lineages)) == ["Alpha", "Delta"]
            # A single inferred breakpoint is correct if it falls in the gap
            # between the informative sites flanking the true one.
            row["breakpoint_correct"] = len(inferred) == 1 and (
                truth_row.interval_left < inferred[0] <= truth_row.interval_right
            )
            row["breakpoint_error"] = (
                min(abs(b - truth_row.breakpoint) for b in inferred)
                if len(inferred) > 0
                else -1
            )
        rows.append(row)

    df = pd.DataFrame(rows).merge(
        df_truth.reset_index().drop(columns=["type"]), on="strain", how="left"
    )
    df = df.sort_values(["k", "type", "strain"])
    df.to_csv(output, index=False)

    for k, group in df.groupby("k"):
        controls = group[group.type == "control"]
        recombinants = group[group.type == "recombinant"]
        detectable = recombinants[
            np.minimum(
                recombinants.num_informative_left, recombinants.num_informative_right
            )
            > 0
        ]
        print(
            f"k={k}: "
            f"FP {controls.detected.sum()}/{len(controls)} "
            f"({controls.detected.mean():.3f}), "
            f"detected {recombinants.detected.sum()}/{len(recombinants)} "
            f"({recombinants.detected.mean():.3f}), "
            f"detectable {detectable.detected.mean():.3f}"
        )


if __name__ == "__main__":
    main()
