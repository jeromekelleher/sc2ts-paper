import click
import collections
import tszip
import pandas as pd
import numpy as np
import sc2ts
from pangonet.pangonet import PangoNet


def pango_x_events(ts, df_node, df_pango):
    """
    Returns a list of dictionaries describing the ARG events for the specified
    dataframe of nodes (representing all the nodes with a given root pango assignment).
    """
    tree = ts.first()

    # print(pango_x_descendants[pango_x])
    # print(df_pango)
    events = []
    pango_nodes = set(df_pango.index.values)

    while len(pango_nodes) > 0:
        # Each iteration of this loop corresponds to an event
        descendants = set()
        pairs = sorted([(-ts.nodes_time[u], u) for u in pango_nodes])
        assert len(pairs) == 1 or pairs[0][0] != pairs[1][0]
        first = df_node.loc[pairs[0][1]]
        pango_samples = 0
        non_pango_samples = collections.Counter()
        root = first.name
        for v in tree.nodes(root):
            e = tree.edge(v)
            if ts.edges_left[e] != 0 and ts.edges_right[e] != ts.sequence_length:
                raise ValueError("Recombinant in subgraph")
            descendants.add(v)

            if tree.is_sample(v):
                if v in pango_nodes:
                    pango_samples += 1
                else:
                    non_pango_samples[df_node.loc[v]["pango"]] += 1

        remaining_pango = pango_nodes - descendants
        non_pango_descendants = descendants - pango_nodes
        # print(df_node.loc[non_pango_descendants]["pango"].value_counts())
        if first.is_sample:
            root_type = "S"
        elif ts.nodes_flags[first.name] & sc2ts.NODE_IS_RECOMBINANT:
            root_type = "R"
        else:
            root_type = "I"

        u = root
        while u != -1 and (ts.nodes_flags[u] & sc2ts.NODE_IS_RECOMBINANT) == 0:
            u = tree.parent(u)
        closest_recombinant = u

        closest_recombinant_path_len = np.inf
        closest_recombinant_time = np.inf
        averted_muts = -1
        if closest_recombinant != -1:
            recomb_record = df_node.loc[closest_recombinant]
            closest_recombinant_path_len = tree.path_length(root, closest_recombinant)
            closest_recombinant_time = (
                ts.nodes_time[closest_recombinant] - ts.nodes_time[root]
            )
            # recomb_info = df_recombinants.loc[closest_recombinant]
            # averted_muts = recomb_info["k1000_muts"] - recomb_info["num_mutations"]

        events.append(
            {
                # "pango": pango_x,
                "root": root,
                "root_pango": first.pango,
                "root_mutations": first.num_mutations,
                "root_type": root_type,
                "pango_samples": pango_samples,
                "non_pango_samples": dict(non_pango_samples),
                # "arg_count": pango_x_arg_counts[pango_x],
                # "ds_count": pango_x_ds_counts[pango_x],
                "closest_recombinant": closest_recombinant,
                "closest_recombinant_path_len": closest_recombinant_path_len,
                "closest_recombinant_time": closest_recombinant_time,
                # "closest_recombinant_averted_mutations": averted_muts,
            }
        )

        pango_nodes = remaining_pango

    return events


@click.command()
@click.argument("ts_path")
@click.argument("output")
def run(ts_path, output):

    ts = tszip.load(ts_path)
    # dfr = pd.read_csv(recombinants_csv)
    print(ts)

    df_node = sc2ts.node_data(ts, inheritance_stats=False).set_index("node_id")

    # Pangonet downloads these files by default if you use
    # pango = PangoNet().build()
    # We fix on specific files here to make this reproducible.
    pn = PangoNet().build(
        alias_key="pangonet_data/alias_key.json",
        lineage_notes="pangonet_data/lineage_notes.txt",
    )

    df_samples = df_node[df_node.is_sample]

    lineage_counts = collections.Counter(df_samples.pango)
    pango_x_counts = collections.Counter()
    pango_xs = set(
        x.split(".")[0] for x in df_samples.pango.unique() if x.startswith("X")
    )

    data = []
    for pango_x in pango_xs:
        descendants = pn.get_descendants(pango_x)
        for pango in [pango_x] + descendants:
            pango_x_counts[pango_x] += lineage_counts[pango]
        df_pango = df_node[df_node.pango.isin([pango_x] + descendants)]

        for event in pango_x_events(ts, df_node, df_pango):
            event["pango"] = pango_x
            print(
                pango_x,
                event["root_pango"],
                event["pango_samples"],
                event["non_pango_samples"],
            )
            data.append(event)
    df = pd.DataFrame(data)
    print(df)
    df.to_csv(output, index=False)


if __name__ == "__main__":
    run()
