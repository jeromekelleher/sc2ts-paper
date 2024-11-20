"""
Convert an Usher tree in JSON format to sc2ts-like tskit.

https://figshare.com/articles/dataset/Global_Viridian_tree/27194547
"""

import sys
import gzip
import json
import dataclasses

import pandas as pd
import numpy as np
import tskit
import tqdm
import click
import pyfaidx


def set_mutations(ts, ref, mutations_per_node):
    tables = ts.dump_tables()
    mutations = tables.mutations
    mutations.clear()
    sites = tables.sites
    sites.clear()

    for j in range(int(ts.sequence_length)):
        sites.add_row(j, ref[j])

    for node_id, node_mutations in tqdm.tqdm(mutations_per_node.items()):
        for muts in node_mutations:
            for mutation_str in muts.split(","):
                inherited = mutation_str[0]
                derived = mutation_str[-1]
                pos = int(mutation_str[1:-1])
                mutations.add_row(
                    site=pos,
                    derived_state=derived,
                    node=node_id,
                    time=ts.nodes_time[node_id],
                )
    mutations_per_site = np.bincount(mutations.site, minlength=len(sites))
    zero_mutation_sites = np.where(mutations_per_site == 0)[0]
    print(f"Deleting {len(zero_mutation_sites)} sites with zero mutations")
    tables.delete_sites(zero_mutation_sites)
    tables.sort()
    tables.build_index()
    tables.compute_mutation_parents()
    return tables.tree_sequence()


@click.command()
@click.argument("tsk_in", type=click.Path(dir_okay=False, file_okay=True))
@click.argument("usher_tsv", type=click.Path(dir_okay=False, file_okay=True))
@click.argument("reference_fa", type=click.Path(dir_okay=False, file_okay=True))
@click.argument("tsk_out", type=click.Path(dir_okay=False, file_okay=True))
def convert_mutations(tsk_in, usher_tsv, reference_fa, tsk_out):

    # Convert to 0-based coordinate
    reference = "X" + str(pyfaidx.Fasta(reference_fa)["Wuhan/Hu-1/2019"])
    df = pd.read_csv(usher_tsv, sep="\t")
    ts = tskit.load(tsk_in)
    print("loaded tskit file")
    node_id_map = {}
    for node in tqdm.tqdm(ts.nodes(), total=ts.num_nodes):
        node_id_map[node.metadata["strain"]] = node.id
        # if node.id == 10000:
        #     break

    nt_mutations = {}
    for _, row in tqdm.tqdm(df.iterrows(), total=df.shape[0]):
        tsk_id = node_id_map[row.node_id]
        assert tsk_id not in nt_mutations
        nt_mutations[tsk_id] = set(row.nt_mutations.split(";"))

    ts_muts = set_mutations(ts, reference, nt_mutations)
    ts_muts.dump(tsk_out)


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


@click.command()
@click.argument("usher_json", type=click.Path(dir_okay=False, file_okay=True))
@click.argument("tsk", type=click.Path(dir_okay=False, file_okay=True))
def convert_topology(usher_json, tsk):
    L = 29_904

    tables = tskit.TableCollection(L)
    tables.metadata_schema = tskit.MetadataSchema.permissive_json()
    tables.nodes.metadata_schema = tskit.MetadataSchema.permissive_json()
    tables.mutations.metadata_schema = tskit.MetadataSchema.permissive_json()
    tables.sites.metadata_schema = tskit.MetadataSchema.permissive_json()

    meta_prefix = "meta_"
    with gzip.open(usher_json) as f:
        header = json.loads(next(f))
        samples_strain = []
        pi = np.zeros(header["total_nodes"]) - 1
        for line in tqdm.tqdm(f, total=header["total_nodes"], desc="Parse"):
            node = json.loads(line)
            name = node["name"]
            node_id = node["node_id"]
            flags = 0
            if not name.startswith("node_"):
                flags = tskit.NODE_IS_SAMPLE
                samples_strain.append(name)
            u = tables.nodes.add_row(
                flags,
                time=-1,
                metadata={
                    "strain": name,
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

    # Store the mapping samples strain in the same format as sc2ts for simplicity
    tables.metadata = {"sc2ts": {"samples_strain": samples_strain}}

    set_tree_time(tables)
    tables.sort()
    tables.build_index()
    ts = tables.tree_sequence()
    ts.dump(tsk)


@click.group()
def cli():
    """
    Utilities to convert Usher MAT files to sc2ts-like tskit format.
    """
    pass


cli.add_command(convert_topology)
cli.add_command(convert_mutations)
cli()
