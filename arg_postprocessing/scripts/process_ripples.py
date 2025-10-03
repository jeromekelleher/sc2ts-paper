import click
import pandas as pd
import numpy as np
import sc2ts
import tszip
import dataclasses
import collections
import tqdm


@dataclasses.dataclass
class UsherEvent:
    usher_node: str
    descendants: list[str]
    pango_counts: dict
    sc2ts_mrca: int = -1
    sc2ts_num_descendants: int = 0
    sc2ts_closest_recombinant: int = -1
    sc2ts_closest_recombinant_path_len: int = 2**31

    def asdict(self):
        d = dataclasses.asdict(self)
        del d["descendants"]
        d["usher_num_descendants"] = len(self.descendants)
        return d


@click.command()
@click.argument("ripples_dir")
@click.argument("sc2ts_ts")
@click.argument("output")
def run(ripples_dir, sc2ts_ts, output):

    descendants = {}
    with open(f"{ripples_dir}/descendants.tsv") as f:
        for line in f:
            if not line.startswith("#"):
                split = line.strip().split("\t")
                descs = split[1].split(",")
                if descs[-1] == "":
                    descs = descs[:-1]
                descendants[split[0]] = descs
    len("Read {len(descendants)} events")

    ts = tszip.load(sc2ts_ts)
    dfn = sc2ts.node_data(ts, inheritance_stats=False)
    df_sample = dfn[dfn.is_sample].set_index("sample_id")

    print("Computing mappings")
    sample_id_to_node = df_sample["node_id"].to_dict()
    sample_id_to_pango = df_sample["pango"].to_dict()
    tree = ts.first()

    seen_descendants = set()
    events = []
    for usher_node, desc in tqdm.tqdm(descendants.items()):
        desc = frozenset(desc)
        if desc in seen_descendants:
            # We just pick the first one if we see repeated descendant sets
            print(f"Skipping {len(desc)} descendant set already seen")
            continue
        seen_descendants.add(desc)
        event = UsherEvent(usher_node, desc, {"unclassified": -1})
        if len(desc) < 10_000:
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
            u = mrca
            while u != -1 and (ts.nodes_flags[u] & sc2ts.NODE_IS_RECOMBINANT) == 0:
                u = tree.parent(u)
            event.sc2ts_closest_recombinant = u
            if u != -1:
                event.sc2ts_closest_recombinant_path_len = tree.path_length(mrca, u)

        events.append(event)

    data = []
    for e in events:
        data.append(e.asdict())
    df = pd.DataFrame(data)
    df.reset_index().to_csv(output, index=False)


if __name__ == "__main__":
    run()
