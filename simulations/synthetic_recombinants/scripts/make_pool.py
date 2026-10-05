"""
Build the pool of real samples that synthetic recombinants and controls are
drawn from.

Takes the samples collected in one country on one date, labels them Alpha or
Delta, and keeps only those accepted by the published sc2ts inference.
"""

import click
import pandas as pd
import tszip
import zarr

from lineage import classify


@click.command()
@click.argument("dataset", type=click.Path(exists=True))
@click.argument("reference_arg", type=click.Path(exists=True))
@click.argument("output", type=click.Path(dir_okay=False))
@click.option("--date", required=True)
@click.option("--country", required=True)
def main(dataset, reference_arg, output, date, country):
    root = zarr.open(dataset, mode="r")
    df = pd.DataFrame(
        {
            name: root[f"sample_{name}"][:]
            for name in ["id", "Date_tree", "Country", "Viridian_pangolin"]
        }
    )
    df = df[(df.Country == country) & (df.Date_tree == date)]
    df["lineage"] = df.Viridian_pangolin.astype(str).map(classify)
    df = df[df.lineage != "Other"]

    ts = tszip.load(reference_arg)
    accepted = {ts.node(u).metadata["sample_id"] for u in ts.samples()}
    df = df[df.id.isin(accepted)]

    df = df[["id", "lineage", "Viridian_pangolin"]].rename(columns={"id": "strain"})
    df = df.sort_values("strain")
    df.to_csv(output, index=False)
    print(df.lineage.value_counts().to_string())


if __name__ == "__main__":
    main()
