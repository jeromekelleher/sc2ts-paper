"""
Export the sequences, sc2ts metadata and true recombination status of the
individuals sampled in a simplified SLiM tree sequence.
"""
import datetime

import click
import numpy as np
import pandas as pd
import tskit


# Generation 0 (the founder) is collected on this date, and each subsequent
# generation one day later. sc2ts's reference date must be before it.
START_DATE = datetime.date(2026, 1, 1)


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


def truth_table(pedigree, sequences):
    """
    Return the pedigree rows for the sampled individuals, with a "detectable"
    column: a recombinant is only detectable if its sequence differs from both
    of its parents'.
    """
    df = pedigree.copy()
    df.insert(0, "strain", [strain_name(c) for c in df.child])
    df["is_recombinant"] = df.is_recombinant.astype(bool)
    df["breakpoint"] = np.where(
        df.is_recombinant, pd.to_numeric(df.breakpoints, errors="coerce"), -1
    ).astype(int)
    detectable = []
    for row in df.itertuples():
        child = sequences[row.strain]
        detectable.append(
            row.is_recombinant
            and child != sequences[strain_name(row.parent1)]
            and child != sequences[strain_name(row.parent2)]
        )
    df["detectable"] = detectable
    return df[
        ["strain", "gen", "parent1", "parent2", "is_recombinant", "breakpoint",
         "detectable"]
    ]


@click.command()
@click.argument("sampled_ts", type=click.Path(exists=True, dir_okay=False))
@click.argument("fasta", type=click.Path(exists=True, dir_okay=False))
@click.argument("pedigree", type=click.Path(exists=True, dir_okay=False))
@click.argument("output_fasta", type=click.Path(dir_okay=False))
@click.argument("output_metadata", type=click.Path(dir_okay=False))
@click.argument("output_truth", type=click.Path(dir_okay=False))
def run(sampled_ts, fasta, pedigree, output_fasta, output_metadata, output_truth):
    ts = tskit.load(sampled_ts)
    ids = sample_pedigree_ids(ts)
    df = pd.read_csv(pedigree, sep="\t").set_index("child", drop=False).loc[ids]
    # Parents are needed to tell whether a recombinant is detectable; the
    # founder has none.
    related = set(ids) | set(df.parent1[df.parent1 >= 0]) | set(df.parent2[df.parent2 >= 0])
    sequences = read_fasta(fasta, [strain_name(x) for x in related])

    with open(output_fasta, "w") as f:
        for pedigree_id in ids:
            name = strain_name(pedigree_id)
            print(f">{name}\n{sequences[name]}", file=f)

    metadata = pd.DataFrame({
        "Run": [strain_name(c) for c in df.child],
        "date": [str(START_DATE + datetime.timedelta(days=int(g))) for g in df.gen],
    })
    metadata.to_csv(output_metadata, sep="\t", index=False)
    truth_table(df, sequences).to_csv(output_truth, index=False)


if __name__ == "__main__":
    run()
