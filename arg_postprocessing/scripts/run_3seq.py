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
import dataclasses


def run_single_3seq(child_fasta, parent_fastas):

    exe = os.path.abspath("./3seq/3seq")
    # if True:
    #     tempdir = pathlib.Path("tmp/")
    with tempfile.TemporaryDirectory() as tempdir:
        tempdir = pathlib.Path(tempdir)
        parents_file = (tempdir / "parents.fasta").absolute()
        with open(parents_file, "w") as fw:
            for j, parent_file in enumerate(parent_fastas):
                with open(parent_file) as fr:
                    fw.write(fr.read())
        cmd = f"{exe} -full {parents_file} {child_fasta}"
        subprocess.check_output(f"yes | {cmd}", shell=True, cwd=tempdir)
        # Note: this emits a parser warning when 3SEQ suggests multiple
        # breakpoints, as it uses commas to separate them (sigh). So, we
        # thrown away that information here. This is better than the
        # alternative, which gets column alignment completely wrong.
        output = pd.read_csv(tempdir / "3s.rec.csv", index_col=False)
    return output


@click.command()
@click.argument("recombinants_csv")
@click.argument("fasta_dir")
@click.argument("output")
def run_sc2ts_3seq(recombinants_csv, fasta_dir, output):
    fasta_dir = pathlib.Path(fasta_dir)
    dfr = pd.read_csv(recombinants_csv)

    def fasta_file(node_id):
        return (fasta_dir / f"{node_id}.fasta").absolute()

    data = []
    for _, row in tqdm.tqdm(list(dfr.iterrows())):
        child_fasta = fasta_file(row["recombinant"])
        parent_fastas = [fasta_file(row[p]) for p in ["parent_left", "parent_right"]]
        out = run_single_3seq(child_fasta.absolute(), parent_fastas)
        if len(out) > 0:
            out["recombinant"] = row["recombinant"]
            data.append(out)
    # print(data)
    df = pd.concat(data)
    df.to_csv(output, index=False)


@click.command()
@click.argument("ts")
@click.argument("recombinants_csv")
@click.argument("output_dir")
def generate_sc2ts_fasta(ts, recombinants_csv, output_dir):
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

    labels = ["recombinant", "parent_left", "parent_right"]
    samples = []
    for _, row in dfr.iterrows():
        samples.extend([row[label] for label in labels])

    print(f"Computing alignments for {len(samples)} nodes")
    unique_samples = sorted(set(samples))
    alignments = dict(
        zip(unique_samples, ts.alignments(samples=unique_samples, left=1))
    )
    print("done")

    for k, v in alignments.items():
        output = output_dir / f"{k}.fasta"
        with open(f"{output}", "w") as f:
            f.write(f">{k}\n")
            f.write(v + "\n")


@click.command()
@click.argument("ripples_file")
@click.argument("output")
def generate_ripples_sample_list(ripples_file, output):
    df = pd.read_csv(ripples_file)
    samples = set(df["recombinant"]) | set(df["parent1"]) | set(df["parent2"])
    samples = np.array(sorted(list(samples)))
    np.savetxt(f"{output}", samples, fmt="%s")


@click.command()
@click.argument("ripples_file")
@click.argument("fasta_dir")
@click.argument("output")
def run_ripples_3seq(ripples_file, fasta_dir, output):

    df = pd.read_csv(ripples_file)

    def fasta_file(name):
        return os.path.abspath(pathlib.Path(fasta_dir) / f"{name}_NC_045512:0.fa")

    data = []
    for _, row in tqdm.tqdm(list(df.iterrows())):
        out = run_single_3seq(
            fasta_file(row["recombinant"]),
            [fasta_file(row["parent1"]), fasta_file(row["parent2"])],
        )
        if len(out) > 0:
            out["recombinant"] = row["recombinant"]
            data.append(out)
    df = pd.concat(data)

    df.to_csv(output, index=False)
    print(df)


@click.group()
def cli():
    pass


cli.add_command(run_sc2ts_3seq)
cli.add_command(generate_sc2ts_fasta)
cli.add_command(generate_ripples_sample_list)
cli.add_command(run_ripples_3seq)
cli()
