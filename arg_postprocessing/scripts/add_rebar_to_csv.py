import click
import pandas as pd
import numpy as np


@click.command()
@click.argument("recombinants_csv")
@click.argument("rebar_tsv")
@click.argument("output")
def run(recombinants_csv, rebar_tsv, output):
    dfr = pd.read_csv(recombinants_csv, index_col=0).set_index("sample_id")
    df_rebar = pd.read_csv(rebar_tsv, sep="\t").set_index("strain")
    dfr["is_rebar_recombinant"] = ~df_rebar["recombinant"].isna()
    print(np.sum(dfr["is_rebar_recombinant"]), "recombinants as per rebar")
    dfr.reset_index().to_csv(output, index=False)


if __name__ == "__main__":
    run()
