import click
import pandas as pd
import numpy as np
from pangonet.pangonet import PangoNet

def get_pangonet_distance(pango, label_1, label_2):
    label_1 = pango.compress(label_1)
    label_2 = pango.compress(label_2)
    if label_1 == label_2:
        return 0

    mrcas = pango.get_mrca([label_1, label_2])
    assert len(mrcas) == 1
    mrca = mrcas[0]
    parent_paths = {}
    for name, parent_pango in [("left", label_1), ("right", label_2)]:
        paths = pango.get_paths(start=parent_pango, end=mrca)
        # We can have multiple paths because of recombinant lineages. Taking
        # the minimum seems simplest.
        min_len = min(len(path) for path in paths)
        for path in paths:
            if len(path) == min_len:
                parent_paths[name] = path
                break
    left_path = parent_paths["left"]
    right_path = parent_paths["right"]
    assert left_path[-1] == right_path[-1]
    return len(left_path) + len(right_path) - 2



@click.command()
@click.argument("recombinants_csv")
@click.argument("output")
def run(recombinants_csv, output):
    dfr = pd.read_csv(recombinants_csv)

    # Pangonet downloads these files by default, uncomment to work around rate limit
    # problems
    # pango = PangoNet().build(alias_key="alias_key.json", lineage_notes="lineage_notes.txt")
    pango = PangoNet().build()

    parent_path_len = []
    for _, row in dfr.iterrows():
        left_pp = row["parent_left_pango"]
        right_pp = row["parent_right_pango"]
        distance = get_pangonet_distance(pango, left_pp, right_pp)
        parent_path_len.append(distance)

    dfr["parent_pangonet_distance"] = parent_path_len
    dfr.to_csv(output, index=False)


if __name__ == "__main__":
    run()
