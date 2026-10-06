"""
Export the samples from Pango X lineages that are present in the dataset but
absent from the ARG, along with the HMM match that sc2ts recorded for each one
during primary inference (if it got as far as matching).

A Pango X lineage is grouped with its sublineages (e.g. XBC.1.2 is counted as
XBC) and is absent if no sample in the ARG is assigned to any lineage in the
group, using the Viridian_pangolin_1.29 assignments. Samples that are not in the
match DB failed sample QC (or were outside the inference date range).
"""
import bz2
import pickle
import sqlite3

import click
import pandas as pd
import sc2ts
import tszip


def absent_pango_x_samples(dataset_path, ts):
    ds = sc2ts.Dataset(dataset_path, date_field="Date_tree")
    df = ds.metadata.as_dataframe(["Viridian_pangolin_1.29", "Date_tree"])
    df = df.rename(columns={"Viridian_pangolin_1.29": "pango", "Date_tree": "date"})
    df_node = sc2ts.node_data(ts, inheritance_stats=False)
    arg_samples = df_node[df_node["is_sample"]]
    last_date = str(arg_samples["date"].max().date())
    df = df[(df["date"] != ".") & (df["date"] != "2020-12-31") & (df["date"] <= last_date)]
    df = df[df["pango"].str.startswith("X")].copy()
    df["family"] = df["pango"].str.split(".").str[0]
    in_arg = df.index.isin(arg_samples["sample_id"])
    present = set(df["family"][in_arg])
    df = df[~df["family"].isin(present)]
    df.index.name = "sample_id"
    return df.sort_values(["family", "pango", "date"])


def get_matches(match_db_path, strains):
    """
    Return the HMM match for each strain in the match DB (strains are unique in
    the DB, so there is at most one).
    """
    strains = list(strains)
    conn = sqlite3.connect(f"file:{match_db_path}?mode=ro", uri=True)
    sql = (
        "SELECT strain, match_date, hmm_cost, pickle FROM samples "
        f"WHERE strain IN ({', '.join('?' * len(strains))})"
    )
    rows = {}
    for strain, match_date, hmm_cost, pkl in conn.execute(sql, strains):
        sample = pickle.loads(bz2.decompress(pkl))
        hmm_match = sample.hmm_match
        rows[strain] = {
            "match_date": match_date,
            "hmm_cost": hmm_cost,
            "num_missing_sites": sample.num_missing_sites,
            "num_parents": len(hmm_match.path),
            "num_mutations": len(hmm_match.mutations),
            "path": " ".join(
                f"{seg.left}:{seg.right}:{seg.parent}" for seg in hmm_match.path
            ),
            "mutations": hmm_match.mutation_summary(),
        }
    conn.close()
    return pd.DataFrame.from_dict(rows, orient="index")


@click.command()
@click.argument("dataset")
@click.argument("ts_path")
@click.argument("output")
@click.option("--match-db", default=None, help="Inference match DB to get HMM matches from")
def run(dataset, ts_path, output, match_db):
    ts = tszip.load(ts_path)
    df = absent_pango_x_samples(dataset, ts)
    print(df.groupby("family").size())
    if match_db is not None:
        df_match = get_matches(match_db, df.index)
        df["in_match_db"] = df.index.isin(df_match.index)
        df = df.join(df_match)
        print(df.groupby("family")["in_match_db"].sum())
    df.to_csv(output)


if __name__ == "__main__":
    run()
