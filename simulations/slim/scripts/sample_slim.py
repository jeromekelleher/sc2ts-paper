"""
Sample individuals from all generations of a SLiM simulation and simplify their
genealogy to give the true ARG.
"""
import click
import numpy as np
import pyslim
import tskit


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


def sample(ts, probability, seed):
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


@click.command()
@click.argument("input_ts", type=click.Path(exists=True, dir_okay=False))
@click.argument("output_ts", type=click.Path(dir_okay=False))
@click.option("--probability", type=float, required=True)
@click.option("--seed", type=int, required=True)
def run(input_ts, output_ts, probability, seed):
    ts = tskit.load(input_ts)
    sample(ts, probability, seed).dump(output_ts)


if __name__ == "__main__":
    run()
