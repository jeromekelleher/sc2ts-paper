import click
import pandas as pd
import numpy as np
from pangonet.pangonet import PangoNet

def get_pangonet_distance(pango, label_1, label_2):
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


#     """Get the number of edges separating two Pango labels on a reference phylogeny."""
#     # Pangonet sometimes returns empty paths between uncompressed labels.
#     # So, it is better to work with compressed labels instead.
#     label_1_c = pangonet.compress(label_1)
#     label_2_c = pangonet.compress(label_2)
#     # Special case
#     if label_1_c == label_2_c:
#         return 0    # Distance
#     # Check ancestor-descendant relationship
#     label_1_anc = [pangonet.uncompress(p) for p in pangonet.get_ancestors(label_1_c)]
#     label_2_anc = [pangonet.uncompress(p) for p in pangonet.get_ancestors(label_2_c)]
#     if (label_1_c in label_2_anc) or (label_2_c in label_1_anc):
#         # Paths include the focal nodes
#         anc_desc_path = pangonet.get_paths(start=label_1_c, end=label_2_c)
#         if len(anc_desc_path) != 1:
#             raise ValueError("pangonet returns unexpected number of paths.")
#         distance = len(anc_desc_path[0]) - 1
#     else:
#         mrca = pangonet.get_mrca([label_1_c, label_2_c])
#         if len(mrca) != 1:
#             raise ValueError("pangonet returns unexpected number of MRCAs.")
#         # Paths include the focal nodes
#         mrca_pango_1_path = pangonet.get_paths(start=label_1_c, end=mrca[0])
#         mrca_pango_2_path = pangonet.get_paths(start=label_2_c, end=mrca[0])
#         if (len(mrca_pango_1_path) != 1) or (len(mrca_pango_2_path) != 1):
#             raise ValueError("pangonet returns unexpected number of paths.")
#         mrca_pango_1_distance = len(mrca_pango_1_path[0]) - 1
#         mrca_pango_2_distance = len(mrca_pango_2_path[0]) - 1
#         distance = mrca_pango_1_distance + mrca_pango_2_distance
#     return distance



@click.command()
@click.argument("recombinants_csv")
@click.argument("output")
def run(recombinants_csv, output):
    dfr = pd.read_csv(recombinants_csv)
    print(dfr)

    pango = PangoNet().build()

    parent_path_len = []
    for _, row in dfr.iterrows():
        left_pp = row["parent_left_pango"]
        right_pp = row["parent_right_pango"]
        distance = get_pangonet_distance(pango, left_pp, right_pp)
        parent_path_len.append(distance)

    print(parent_path_len)

    # df_rebar = pd.read_csv(rebar_tsv, sep="\t").set_index("strain")
    # dfr["is_rebar_recombinant"] = ~df_rebar["recombinant"].isna()
    # print(np.sum(dfr["is_rebar_recombinant"]), "recombinants as per rebar")
    # dfr.reset_index().to_csv(output, index=False)


if __name__ == "__main__":
    run()
