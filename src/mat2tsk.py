"""
Convert an Usher tree in JSON format to sc2ts-like tskit.

https://figshare.com/articles/dataset/Global_Viridian_tree/27194547
"""

import sys
import json
import dataclasses

import numpy as np
import tskit
import tqdm
import click

def set_tree_time(tables, unit_scale=False):
    # Add times using max number of hops from leaves.
    pi = np.full(len(tables.nodes), -1, dtype=int)
    tau = np.full(len(tables.nodes), -1, dtype=float)
    pi[tables.edges.child] = tables.edges.parent
    samples = np.where(tables.nodes.flags == tskit.NODE_IS_SAMPLE)[0]
    for sample in tqdm.tqdm(samples, desc="Time"):
        t = 0
        u = sample
        # This is an inefficient algorithm - we could do better.
        while u != -1:
            tau[u] = max(tau[u], t)
            t += 1
            u = pi[u]
    if unit_scale:
        tau /= max(1, np.max(tau))
    tables.nodes.time = tau


def convert_root_mutations(tables, root):
    # Convert mutations over the root to ancestral state values
    mutations = tables.mutations
    root_mutations = mutations.node == root
    ancestral_state = ["N" for _ in range(len(tables.sites))]
    for mutation in mutations[root_mutations]:
        ancestral_state[mutation.site] = mutation.derived_state
    mutations.parent = np.full_like(mutations.parent, -1)
    mutations.keep_rows(~root_mutations)
    tables.sites.ancestral_state = [ord(a) for a in ancestral_state]


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
    tables.sites.metadata_schema = tskit.MetadataSchema.permissive_json()
    meta_prefix = "meta_"
    with open(in_path) as f:
        header = json.loads(next(f))
        mutations = {}
        site_id_map = {}
        for mut in header["mutations"]:
            if mut["type"] == "aa":
                pos = mut["nuc_for_codon"]
            else:
                pos = mut["residue_pos"]
            if pos not in site_id_map:
                site_id = tables.sites.add_row(pos, ancestral_state="X")
                site_id_map[pos] = site_id
            mutations[mut["mutation_id"]] = TranslatedMutation(
                site=site_id_map[pos], derived_state=mut["new_residue"], metadata=mut
            )

        assert len(mutations) == len(header["mutations"])
        assert set(mutations.keys()) == set(range(len(mutations)))
        print(f"Got mutations for {len(site_id_map)} sites")
        # for k, v in mutations.items():
        #     print(k, site_id_map[v.site], v, sep="\t")

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
            # The root has a loop.
            if u != parent:
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

        # tables.dump("intermediate.trees")
    # tables = tskit.TableCollection.load("intermediate.trees")
    # pi = np.full(len(tables.nodes), -1, dtype=int)
    # pi[tables.edges.child] = tables.edges.parent

    root = np.where(pi == -1)[0][0]
    convert_root_mutations(tables, root)
    set_tree_time(tables)
    # Set the time of mutations to their node so that they sort appropriately.
    tables.mutations.time = tables.nodes.time[tables.mutations.node]
    tables.sort()
    tables.build_index()
    tables.compute_mutation_parents()
    ts = tables.tree_sequence()
    ts.dump(out_path)


@click.command()
@click.argument("usher_json", type=click.Path(dir_okay=False, file_okay=True))
@click.argument("tsk", type=click.Path(dir_okay=False, file_okay=True))
def convert_topology(usher_json, tsk):
    print("hello")

@click.group()
def cli():
    pass

cli.add_command(convert_topology)
# cli.add_command(convert_mutations)
cli()
