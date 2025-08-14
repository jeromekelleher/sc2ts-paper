import click
import tszip
import tskit
import numpy as np


@click.command()
@click.argument("ts_path")
@click.argument("output")
def run(ts_path, output):

    # Set all nodes to be sample nodes.
    ts = tszip.load(ts_path)
    tables = ts.dump_tables()
    node_flags = tables.nodes.flags
    node_flags[:] = tskit.NODE_IS_SAMPLE
    tables.nodes.flags = node_flags
    ts = tables.tree_sequence()
    if not np.all(ts.nodes_flags == tskit.NODE_IS_SAMPLE):
        raise ValueError("Not all the nodes are samples.")
    samples = np.arange(ts.num_nodes)
    with open(f"{output}", "w") as f:
        # NOTE: this takes about 80G of RAM to do in one go. Can split
        # into chunks of nodes to make it more managable
        for j, entry in enumerate(ts.alignments(samples=samples, left=1)):
            f.write(f">n{samples[j]}\n")
            f.write(entry + "\n")


if __name__ == "__main__":
    run()
