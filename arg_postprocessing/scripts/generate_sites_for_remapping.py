import click
import pandas as pd
import numpy as np
import json
import tszip


@click.command()
@click.argument("base_ts")
@click.argument("del_data")
@click.argument("lineage_data")
@click.argument("output_csv")
@click.argument("output_txt")
def run(base_ts, del_data, lineage_data, output_csv, output_txt):

    base_ts = tszip.load(base_ts)
    ts_sites = set(base_ts.sites_position.astype(int))
    print(f"Have {len(ts_sites)} in base ts")

    with open(lineage_data) as f:
        doc = json.load(f)

    lineage_sites = set()
    for lineage, mut_list in doc.items():
        for mut in mut_list:
            site = int(mut[1:-1])
            lineage_sites.add(site)

    missing_lineage_sites = lineage_sites - ts_sites

    print(
        f"Got {len(lineage_sites)} lineage defining sites: "
        f" {len(missing_lineage_sites)} missing from ts"
    )

    df = pd.read_excel(del_data)
    # Denominator here reported in the Text of Li et al.
    df["frequency"] = df["Count"] / 9_149_680
    df_1pc = df[df["frequency"] >= 0.01].sort_values("Start")
    print(df_1pc)
    del_sites = set()
    for _, row in df_1pc.iterrows():
        start = row["Start"]
        for j in range(start, start + row["Length"]):
            del_sites.add(j)

    missing_del_sites = del_sites - ts_sites
    print(
        f"Got {len(del_sites)} deletion sites"
        f" {len(missing_del_sites)} missing from ts"
    )

    # There are two mutually exclusive sets
    assert len(missing_lineage_sites & del_sites) == 0
    data = []
    for pos in missing_lineage_sites:
        data.append({"position": pos, "reason": "pango"})
    for pos in del_sites:
        data.append({"position": pos, "reason": "deletion"})
    df = pd.DataFrame(data).sort_values("position")

    df.to_csv(output_csv, index=False)
    # We save to CSV and text here because sc2ts wants a text
    # file as input and it's not worth changing that.
    np.savetxt(df.position.values, output_txt)


if __name__ == "__main__":
    run()
