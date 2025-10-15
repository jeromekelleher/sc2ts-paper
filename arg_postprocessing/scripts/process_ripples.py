import click
import pandas as pd
import numpy as np
import sc2ts
import tszip
import dataclasses
import collections
import tqdm
import bte


@dataclasses.dataclass
class UsherEvent:
    recombinant: str
    parent1: str
    parent2: str
    recombinant_date: str
    parent1_date: str
    parent2_date: str
    descendants: list[str]
    pango_counts: dict
    sc2ts_mrca: int = -1
    sc2ts_num_descendants: int = 0
    sc2ts_closest_recombinant: int = -1
    sc2ts_closest_recombinant_path_len: int = 2**31
    sc2ts_num_mutations: int = -1
    x_lineage_involved: bool = False

    def asdict(self):
        d = dataclasses.asdict(self)
        del d["descendants"]
        d["usher_num_descendants"] = len(self.descendants)
        return d


@click.command()
@click.argument("ripples_dir")
@click.argument("usher_tree")
@click.argument("sc2ts_ts")
@click.argument("output")
def run(ripples_dir, usher_tree, sc2ts_ts, output):

    mt = bte.MATree(usher_tree)

    descendants = {}
    with open(f"{ripples_dir}/descendants.tsv") as f:
        for line in f:
            if not line.startswith("#"):
                split = line.strip().split("\t")
                node_id = split[0]
                descs = split[1].split(",")
                if descs[-1] == "":
                    descs = descs[:-1]
                descendants[node_id] = descs
                # Check that we're looking at the same things here
                assert mt.count_leaves(node_id) == len(descs)

    len("Read {len(descendants)} events")

    df = pd.read_csv(f"{ripples_dir}/recombination.tsv", sep="\t")
    df["recomb_node_id"] = df["#recomb_node_id"]
    print(f"Staring with {df.shape[0]} records")
    # The dataframe can contain multiple events for each recombination node. We
    # pick the one with the "maximum" parsimony (i.e., the one with the lowest
    # parsimony score). If there's several, we pick one arbitrarily.
    df_events = df.loc[df.groupby(["recomb_node_id"])["recomb_parsimony"].idxmin()]
    df_events = df_events.set_index("recomb_node_id")
    print(f"Have {df_events.shape[0]} unique recombination events")

    ts = tszip.load(sc2ts_ts)
    dfn = sc2ts.node_data(ts, inheritance_stats=False)
    df_sample = dfn[dfn.is_sample].set_index("sample_id")

    print("Computing mappings")
    sample_id_to_node = df_sample["node_id"].to_dict()
    # Note: possibly we should use the original source metadata
    # for sample Pango mappings here, but it really shouldn't
    # matter.
    sample_id_to_pango = df_sample["pango"].to_dict()
    sample_id_to_date = df_sample["date"].to_dict()
    node_mutations = np.bincount(ts.mutations_node)
    tree = ts.first()

    def get_date(usher_node):
        if "node" in usher_node:
            date = min(sample_id_to_date[x] for x in mt.get_leaves_ids(usher_node))
        else:
            date = sample_id_to_date[usher_node]
        return str(date.date())

    seen_descendants = set()
    events = []
    for usher_node, desc in tqdm.tqdm(descendants.items()):
        desc = frozenset(desc)
        if desc in seen_descendants:
            # We just pick the first one if we see repeated descendant sets
            print(f"Skipping {len(desc)} descendant set already seen")
            continue
        seen_descendants.add(desc)
        row = df_events.loc[usher_node]
        parent1 = row["acceptor_node_id"]
        parent2 = row["donor_node_id"]
        event = UsherEvent(
            usher_node,
            parent1,
            parent2,
            get_date(usher_node),
            get_date(parent1),
            get_date(parent2),
            desc,
            {"unclassified": -1},
        )
        if len(desc) < 100_000:
            nodes = [sample_id_to_node[sid] for sid in desc]
            if len(nodes) == 1:
                mrca = int(nodes[0])
            else:
                mrca = tree.mrca(*nodes)
            event.sc2ts_mrca = mrca
            event.sc2ts_num_descendants = tree.num_samples(mrca)
            event.pango_counts = dict(
                collections.Counter([sample_id_to_pango[sid] for sid in desc])
            )
            event.x_lineage_involved = any(
                "X" in lin for lin in event.pango_counts.keys()
            )
            u = mrca
            while u != -1 and (ts.nodes_flags[u] & sc2ts.NODE_IS_RECOMBINANT) == 0:
                u = tree.parent(u)
            event.sc2ts_closest_recombinant = u
            if u != -1:
                event.sc2ts_closest_recombinant_path_len = tree.path_length(mrca, u)
            event.sc2ts_num_mutations = node_mutations[mrca]
        else:
            print(f"Skipping classification for node with {len(desc)} descendants")

        events.append(event)

    data = []
    for e in events:
        data.append(e.asdict())
    df = pd.DataFrame(data)
    df.to_csv(output, index=False)


if __name__ == "__main__":
    run()
