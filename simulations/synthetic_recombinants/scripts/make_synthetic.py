"""
Make synthetic single-breakpoint Alpha x Delta recombinants, plus a set of real
non-recombinant controls, and write them out for import as a sc2ts dataset.

The pool is split in half per lineage: one half supplies recombinant parents,
the other supplies controls, so no control is a parent of any recombinant.
"""

import click
import numpy as np
import pandas as pd
import tszip

import sc2ts
from sc2ts.dataset import decode_alleles

# Unambiguous bases come first in the core.IUPAC_ALLELES encoding; -1 is
# missing data. Sites where both parents are unambiguous and differ are the
# only ones the HMM can use to tell the parents apart.
NUM_ACGT = 4


def is_acgt(a):
    return (a >= 0) & (a < NUM_ACGT)


@click.command()
@click.argument("pool", type=click.Path(exists=True, dir_okay=False))
@click.argument("dataset", type=click.Path(exists=True))
@click.argument("arg", type=click.Path(exists=True, dir_okay=False))
@click.option("--out-prefix", required=True)
@click.option("--date", required=True)
@click.option("--num-recombinants", type=int, required=True)
@click.option("--seed", type=int, required=True)
def main(pool, dataset, arg, out_prefix, date, num_recombinants, seed):
    rng = np.random.default_rng(seed)
    df_pool = pd.read_csv(pool)

    # Only sites in the ARG are visible to the HMM.
    arg_sites = tszip.load(arg).sites_position.astype(int)

    # Each alignment lookup decompresses a whole chunk of samples, so fetch in
    # index order with a cache big enough to hold every chunk the pool touches.
    ds = sc2ts.Dataset(dataset, chunk_cache_size=16)
    order = sorted(df_pool.strain, key=lambda strain: ds.metadata.sample_id_map[strain])
    alignments = {strain: ds.alignment[strain] for strain in order}
    sequence_length = len(next(iter(alignments.values())))
    arg_index = arg_sites - 1

    # Split each lineage in half: parents and controls.
    parents = {}
    controls = []
    for lineage, group in df_pool.groupby("lineage"):
        strains = rng.permutation(group.strain.to_numpy())
        split = len(strains) // 2
        parents[lineage] = strains[:split]
        controls.extend(strains[split:])
    controls = sorted(controls)

    records = []
    sequences = {}
    for j in range(num_recombinants):
        alpha_first = rng.random() < 0.5
        left_lineage = "Alpha" if alpha_first else "Delta"
        right_lineage = "Delta" if alpha_first else "Alpha"
        left = rng.choice(parents[left_lineage])
        right = rng.choice(parents[right_lineage])
        # Positions < breakpoint come from the left parent, the rest from the
        # right parent.
        breakpoint = int(rng.integers(2, sequence_length + 1))

        left_alignment = alignments[left]
        right_alignment = alignments[right]
        sequence = np.concatenate(
            [left_alignment[: breakpoint - 1], right_alignment[breakpoint - 1 :]]
        )
        strain = f"rec_{j:05d}"
        sequences[strain] = sequence

        # Count the sites either side of the breakpoint where the parents are
        # distinguishable. A recombinant with none on one side cannot be
        # detected, however small k is.
        a = left_alignment[arg_index]
        b = right_alignment[arg_index]
        informative = (a != b) & is_acgt(a) & is_acgt(b)
        # The true breakpoint is only identifiable up to the gap between the
        # informative sites flanking it, so record that gap to score inferred
        # breakpoints against.
        informative_sites = arg_sites[informative]
        before = informative_sites[informative_sites < breakpoint]
        after = informative_sites[informative_sites >= breakpoint]
        records.append(
            {
                "strain": strain,
                "type": "recombinant",
                "left_parent": left,
                "right_parent": right,
                "left_lineage": left_lineage,
                "right_lineage": right_lineage,
                "breakpoint": breakpoint,
                "num_informative_left": len(before),
                "num_informative_right": len(after),
                "interval_left": int(before[-1]) if len(before) > 0 else 1,
                "interval_right": (
                    int(after[0]) if len(after) > 0 else sequence_length + 1
                ),
                "num_missing": int(np.sum(sequence[arg_index] < 0)),
            }
        )

    lineage_of = dict(zip(df_pool.strain, df_pool.lineage))
    for strain in controls:
        sequences[strain] = alignments[strain]
        records.append(
            {
                "strain": strain,
                "type": "control",
                "left_parent": "",
                "right_parent": "",
                "left_lineage": lineage_of[strain],
                "right_lineage": lineage_of[strain],
                "breakpoint": -1,
                "num_informative_left": -1,
                "num_informative_right": -1,
                "interval_left": -1,
                "interval_right": -1,
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
    # self-describing.
    df_truth.assign(Run=df_truth.strain, date=date)[
        ["Run", "date", "type", "left_lineage", "right_lineage", "breakpoint"]
    ].to_csv(f"{out_prefix}.metadata.tsv", sep="\t", index=False)

    with open(f"{out_prefix}.strains.txt", "w") as f:
        for strain in sequences:
            print(strain, file=f)

    print(f"{num_recombinants} recombinants, {len(controls)} controls")


if __name__ == "__main__":
    main()
