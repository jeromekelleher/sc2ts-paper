import click
import pandas as pd
import sc2ts


@click.command()
@click.argument("dataset")
@click.argument("samples_csv")
@click.argument("output")
def run(dataset, samples_csv, output):

    ds = sc2ts.Dataset(dataset)
    df = pd.read_csv(samples_csv)
    with open(f"{output}", "w") as f:
        ds.write_fasta(f, sample_id=df["sample_id"])


if __name__ == "__main__":
    run()
