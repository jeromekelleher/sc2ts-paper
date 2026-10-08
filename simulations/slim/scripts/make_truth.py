"""
Record the true recombination history of each sampled individual from the
SLiM pedigree.
"""
import click
import pandas as pd
import tskit

from scripts.export_samples import read_fasta, sample_pedigree_ids, strain_name


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


@click.command()
@click.argument("sampled_ts", type=click.Path(exists=True, dir_okay=False))
@click.argument("fasta", type=click.Path(exists=True, dir_okay=False))
@click.argument("pedigree", type=click.Path(exists=True, dir_okay=False))
@click.argument("output", type=click.Path(dir_okay=False))
def run(sampled_ts, fasta, pedigree, output):
    ids = sample_pedigree_ids(tskit.load(sampled_ts))
    pedigree = pd.read_csv(pedigree, sep="\t")
    names = [strain_name(x) for x in required_sequences(pedigree, ids)]
    sequences = read_fasta(fasta, names)
    truth_table(pedigree, ids, sequences).to_csv(output, index=False)


if __name__ == "__main__":
    run()
