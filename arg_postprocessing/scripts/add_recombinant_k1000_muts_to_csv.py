import click
import pandas as pd
import json
import numpy as np


@click.command()
@click.argument("rematches_json")
@click.argument("recombinants_csv")
@click.argument("output")
def run(rematches_json, recombinants_csv, output):
    dfr = pd.read_csv(recombinants_csv).set_index("recombinant")
    no_recomb_match_mutations = {}
    with open(rematches_json) as f:
        for r in json.load(f):
            nrm = r["no_recomb_match"]
            assert len(nrm["path"]) == 1
            nrm_muts = len(nrm["mutations"])
            no_recomb_match_mutations[r["recombinant"]] = nrm_muts

    dfr["k1000_muts"] = no_recomb_match_mutations
    missing = np.sum(dfr["k1000_muts"].isna())
    if missing > 0:
        print(f"WARNING!!! Missing {missing}/{dfr.shape[0]} recombinants from matches")

    dfr.reset_index().to_csv(output, index=False)


if __name__ == "__main__":
    run()
