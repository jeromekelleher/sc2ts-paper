import tszip
import click
import sc2ts
import numpy as np
import pandas as pd


@click.command()
@click.argument("ts")
@click.argument("output")
def run(ts, output):
    ts = tszip.load(ts)

    recomb_nodes = np.where(ts.nodes_flags & sc2ts.NODE_IS_RECOMBINANT > 0)[0]
    node_mutations = np.bincount(ts.mutations_node)
    df = pd.DataFrame(
        {"recombinant": recomb_nodes, "mutations": node_mutations[recomb_nodes]}
    )

    df.to_csv(output, index=False)


if __name__ == "__main__":
    run()
