import tqdm
import click
import tszip
import tskit
import pandas as pd
import numpy as np
import pathlib
import subprocess
import tempfile
import os.path


def run_single_3seq(fasta_file):
    exe = os.path.abspath("./tmp/3seq/3seq")
    fasta_file = os.path.abspath(fasta_file)
    with tempfile.TemporaryDirectory() as tempdir:
        # NOTE: I can't get 3seq to run in -triplet mode whatever I do,
        # but full seems to work. Same thing, ultimately?
        subprocess.check_output(
            f"yes | {exe} -full {fasta_file}", shell=True, cwd=tempdir
        )
        output = pd.read_csv(pathlib.Path(tempdir) / "3s.rec.csv")
    return output


@click.command()
@click.argument("recombinants_csv")
@click.argument("fasta_dir")
@click.argument("output")
def run_3seq(recombinants_csv, fasta_dir, output):
    fasta_dir = pathlib.Path(fasta_dir)
    dfr = pd.read_csv(recombinants_csv)

    data = []
    for recombinant in tqdm.tqdm(dfr["recombinant"].values):
        fasta = fasta_dir / f"{recombinant}.fasta"
        out = run_single_3seq(fasta)
        if len(out) > 0:
            out["recombinant"] = recombinant
            data.append(out)
    # print(data)
    df = pd.concat(data)
    df.to_csv(output, index=False)


@click.command()
@click.argument("ts")
@click.argument("recombinants_csv")
@click.argument("output_dir")
def generate_fasta(ts, recombinants_csv, output_dir):
    output_dir = pathlib.Path(output_dir)
    ts = tszip.load(ts)
    dfr = pd.read_csv(recombinants_csv)

    # Set all nodes to be sample nodes.
    tables = ts.dump_tables()
    node_flags = tables.nodes.flags
    node_flags[:] = tskit.NODE_IS_SAMPLE
    tables.nodes.flags = node_flags
    ts = tables.tree_sequence()
    if not np.all(ts.nodes_flags == tskit.NODE_IS_SAMPLE):
        raise ValueError("Not all the nodes are samples.")

    labels = ["sample", "parent_left", "parent_right"]
    samples = []
    for _, row in dfr.iterrows():
        samples.extend([row[label] for label in labels])

    print(f"Computing alignments for {len(samples)} nodes")
    unique_samples = sorted(set(samples))
    alignments = dict(
        zip(unique_samples, ts.alignments(samples=unique_samples, left=1))
    )
    print("done")

    for _, row in dfr.iterrows():
        d = {label: alignments[row[label]] for label in labels}
        output = output_dir / f"{row['recombinant']}.fasta"
        with open(f"{output}", "w") as f:
            for k, v in d.items():
                f.write(f">{k}\n")
                f.write(v + "\n")


@click.group()
def cli():
    pass


cli.add_command(run_3seq)
cli.add_command(generate_fasta)
cli()
