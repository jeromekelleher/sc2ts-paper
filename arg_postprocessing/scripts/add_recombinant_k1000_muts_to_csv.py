import click
import pandas as pd
import json
import numpy as np


@click.command()
@click.argument("k1000_muts_json")
@click.argument("recombinants_csv")
@click.argument("output")
def run(k1000_muts_json, recombinants_csv, output):
    dfr = pd.read_csv(recombinants_csv).set_index("recombinant")
    with open(k1000_muts_json) as f:
        # JSON keys are strings, convert back to node IDs
        k1000_muts = {int(k): v for k, v in json.load(f).items()}

    dfr["k1000_muts"] = k1000_muts
    missing = np.sum(dfr["k1000_muts"].isna())
    if missing > 0:
        print(f"WARNING!!! Missing {missing}/{dfr.shape[0]} recombinants from matches")

    dfr.reset_index().to_csv(output, index=False)


if __name__ == "__main__":
    run()
