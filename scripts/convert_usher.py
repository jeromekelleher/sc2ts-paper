"""
Convert an Usher tree in JSON format to sc2ts-like tskit.
"""

import sys
import json
import dataclasses

import numpy as np
import tskit
import tqdm


def set_tree_time(tables, unit_scale=False):
    # Add times using max number of hops from leaves
    pi = np.full(len(tables.nodes), -1, dtype=int)
    tau = np.full(len(tables.nodes), -1, dtype=float)
    pi[tables.edges.child] = tables.edges.parent
    samples = np.where(tables.nodes.flags == tskit.NODE_IS_SAMPLE)[0]
    for sample in tqdm.tqdm(samples, desc="Time"):
        t = 0
        u = sample
        while u != -1:
            tau[u] = max(tau[u], t)
            t += 1
            u = pi[u]
    if unit_scale:
        tau /= max(1, np.max(tau))
    tables.nodes.time = tau


@dataclasses.dataclass
class TranslatedMutation:
    site: int
    derived_state: str
    metadata: dict


def main(in_path, out_path):
    L = 29_904
    tables = tskit.TableCollection(L)
    tables.nodes.metadata_schema = tskit.MetadataSchema.permissive_json()
    tables.mutations.metadata_schema = tskit.MetadataSchema.permissive_json()
    meta_prefix = "meta_"
    with open(in_path) as f:
        header = json.loads(next(f))
        mutations = []
        site_id_map = {}
        for mut in header["mutations"]:
            if mut["type"] == "aa":
                pos = mut["nuc_for_codon"]
            else:
                pos = mut["residue_pos"]
            if pos not in site_id_map:
                site_id = tables.sites.add_row(pos, ancestral_state="X")
                site_id_map[pos] = site_id
            mutations.append(
                TranslatedMutation(
                    site=site_id, derived_state=mut["new_residue"], metadata=mut
                )
            )
        pi = np.zeros(header["total_nodes"]) - 1
        for line in tqdm.tqdm(f, total=header["total_nodes"], desc="Parse"):
            node = json.loads(line)
            name = node["name"]
            node_id = node["node_id"]
            flags = 0 if name.startswith("node_") else tskit.NODE_IS_SAMPLE
            u = tables.nodes.add_row(
                flags,
                time=-1,
                metadata={
                    "name": name,
                    "num_tips": node["num_tips"],  # including for validation
                    **{
                        k[len(meta_prefix) :]: v
                        for k, v in node.items()
                        if k.startswith(meta_prefix)
                    },
                },
            )
            assert u == node["node_id"]
            parent = node["parent_id"]
            assert pi[u] == -1
            assert pi[u] != u
            pi[u] = parent
            tables.edges.add_row(0, L, parent=parent, child=u)
            for mut_id in node["mutations"]:
                mut = mutations[mut_id]
                tables.mutations.add_row(
                    node=u,
                    site=mut.site,
                    derived_state=mut.derived_state,
                    metadata=mut.metadata,
                )
            # if u == 1000:
            #     break

    # set_tree_time(tables)
    tables.dump("intermediate.trees")
    tables.sort()
    tables.build_index()
    tables.compute_mutation_parents()
    ts = tables.tree_sequence()
    ts.dump(out_path)
    # print(node)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise ValueError("Usage: [usher in] [tskit out]")
    main(sys.argv[1], sys.argv[2])
