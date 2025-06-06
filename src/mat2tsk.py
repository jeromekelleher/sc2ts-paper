"""
Convert an Usher tree in JSON format to sc2ts-like tskit.

https://figshare.com/articles/dataset/Global_Viridian_tree/27194547

Also requires the mutations to be converted into nucleotide format using
matutils.
"""

import gzip
import json

import pandas as pd
import numpy as np
import tskit
import tszip
import tqdm
import click
import pyfaidx
import numpy.testing as nt


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
                # inherited = mutation_str[0]
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
    fa_map = dict(pyfaidx.Fasta(reference_fa))
    assert len(fa_map) == 1
    key = "MN908947"
    # Convert to 0-based coordinate
    reference = "X" + str(fa_map[key])
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
    sample_date = []
    with gzip.open(usher_json) as f:
        header = json.loads(next(f))
        pi = np.zeros(header["total_nodes"]) - 1
        for line in tqdm.tqdm(f, total=header["total_nodes"], desc="Parse"):
            node = json.loads(line)
            name = node["name"]
            flags = 0
            if not name.startswith("node_"):
                flags = tskit.NODE_IS_SAMPLE
                sample_date.append(node["meta_Date_tree"])
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

    # This version does just enough to dates that fit tskit rules
    set_tree_time(tables)
    tables.sort()
    tables.build_index()
    ts = tables.tree_sequence()
    ts.dump(tsk)


@click.command()
@click.argument("tsk_in", type=click.Path(dir_okay=False, file_okay=True))
@click.argument("tsk_out", type=click.Path(dir_okay=False, file_okay=True))
def date_samples(tsk_in, tsk_out):
    """
    Generate reasonable dates for the nodes in the specified tree by
    keeping only samples with full-precision dates, and setting the
    time of each node based on that. Internal nodes are currently
    dated as 1+ the maximum of the dates of children.
    """
    ts = tskit.load(tsk_in)

    sample_date = {}
    for u in ts.samples():
        node = ts.node(u)
        date = node.metadata["Date_tree"]
        if len(date) == 10:
            sample_date[u] = date

    print(f"Dropping {ts.num_samples - len(sample_date)} samples without exact dates")

    samples = np.array(list(sample_date.keys()))
    dates = np.array(list(sample_date.values()), dtype="datetime64[D]")

    ts = ts.simplify(samples)

    time_zero = dates.max()
    time = (time_zero - dates).astype(int)
    sample_time = dict(zip(ts.samples(), time))

    node_time = np.zeros(ts.num_nodes)
    tree = ts.first()
    for u in tree.nodes(order="postorder"):
        if u in sample_time:
            assert tree.num_children(u) == 0
            node_time[u] = sample_time[u]
        else:
            # Internal node
            assert tree.num_children(u) > 0
            node_time[u] = max(node_time[v] + 1 for v in tree.children(u))

    tables = ts.dump_tables()
    tables.metadata = {"sc2ts": {"date": str(time_zero)}}
    tables.nodes.time = node_time
    tables.mutations.time = node_time[tables.mutations.node]
    tables.sort()
    tables.time_units = "days"
    tables.build_index()
    ts = tables.tree_sequence()
    ts.dump(tsk_out)


@click.command()
@click.argument("tsk_in", type=click.Path(dir_okay=False, file_okay=True))
@click.argument("tsk_out", type=click.Path(dir_okay=False, file_okay=True))
def date_internal(tsk_in, tsk_out):
    """
    Generate reasonable dates for the internal nodes in the specified
    sc2ts tree using tsdate without rescaling.
    """
    import tsdate  # We do this here because it takes a long time to import
    ts = tskit.load(tsk_in)

   # assume to first order approximation that the mutation rate is constant for all muts
    edge_times = ts.nodes_time[ts.edges_parent] - ts.nodes_time[ts.edges_child]
    av_mu = ts.num_mutations / ((ts.edges_right - ts.edges_left) * edge_times).sum()

    dated_ts = tsdate.date(
        ts,
        mutation_rate=av_mu,
        rescaling_intervals=0,
        max_iterations=1,  # single tree, so only one round needed
        time_units=ts.time_units,
        allow_unary=True,
        progress=True,
        set_metadata=False,
    )
    dated_ts.dump(tsk_out)


@click.command()
@click.argument("usher_in", type=click.Path(dir_okay=False, file_okay=True))
@click.argument("sc2ts_in", type=click.Path(dir_okay=False, file_okay=True))
@click.argument("usher_out", type=click.Path(dir_okay=False, file_okay=True))
@click.argument("sc2ts_out", type=click.Path(dir_okay=False, file_okay=True))
@click.option(
    "--intersect-sites/--no-intersect-sites",
    default=True,
    help="Reduce sites to the intersection of both (assuming sc2ts is superset)",
    show_default=True,
)
def intersect(usher_in, sc2ts_in, usher_out, sc2ts_out, intersect_sites):
    """
    Compute the intersection of the samples and sites in the specified usher
    and sc2ts ARGs. The output user and sc2ts files will contain the same
    set of samples in the same order.
    """
    ts_usher_in = tszip.load(usher_in)
    ts_sc2ts_in = tszip.load(sc2ts_in)

    samples_strain_usher = ts_usher_in.metadata["sc2ts"]["samples_strain"]
    samples_strain_sc2ts = ts_sc2ts_in.metadata["sc2ts"]["samples_strain"]
    usher_lookup = dict(zip(samples_strain_usher, ts_usher_in.samples()))
    sc2ts_lookup = dict(zip(samples_strain_sc2ts, ts_sc2ts_in.samples()))

    intersection = set(usher_lookup.keys()) & set(sc2ts_lookup.keys())
    print(
        f"Usher: {len(usher_lookup)}; sc2ts: {len(sc2ts_lookup)} "
        f"inter = {len(intersection)}"
    )
    intersection = sorted(intersection)

    sc2ts_samples = [sc2ts_lookup[k] for k in intersection]
    tables = ts_sc2ts_in.dump_tables()
    tables.simplify(sc2ts_samples, filter_sites=False)
    tables.metadata = {"sc2ts": {"samples_strain": intersection}}
    ts_sc2ts_out = tables.tree_sequence()
    del tables

    usher_samples = [usher_lookup[k] for k in intersection]
    tables = ts_usher_in.dump_tables()
    tables.simplify(usher_samples, filter_sites=False)
    tables.metadata = {"sc2ts": {"samples_strain": intersection}}
    ts_usher_out = tables.tree_sequence()
    del tables

    if intersect_sites:
        assert set(ts_sc2ts_out.sites_position) >= set(ts_usher_out.sites_position)
        missing_positions = set(ts_sc2ts_out.sites_position) - set(
            ts_usher_out.sites_position
        )
        missing_sites = np.searchsorted(
            ts_sc2ts_out.sites_position, list(missing_positions)
        )
        ts_sc2ts_out = ts_sc2ts_out.delete_sites(missing_sites)

    ts_sc2ts_out.dump(sc2ts_out)
    ts_usher_out.dump(usher_out)


@click.command()
@click.argument("usher_path", type=click.Path(dir_okay=False, file_okay=True))
@click.argument("sc2ts_path", type=click.Path(dir_okay=False, file_okay=True))
def validate(usher_path, sc2ts_path):
    """
    Checks that the output of the intersect command above has the expected properties.
    """
    ts_usher = tszip.load(usher_path)
    ts_sc2ts = tszip.load(sc2ts_path)
    num_samples = ts_usher.num_samples
    assert num_samples == ts_sc2ts.num_samples
    num_sites = ts_usher.num_sites
    assert num_sites == ts_sc2ts.num_sites
    samples_strain = ts_sc2ts.metadata["sc2ts"]["samples_strain"]
    assert samples_strain == ts_usher.metadata["sc2ts"]["samples_strain"]
    nt.assert_array_equal(ts_usher.samples(), np.arange(num_samples))
    nt.assert_array_equal(ts_sc2ts.samples(), np.arange(num_samples))

    print(
        f"nodes: {ts_usher.num_nodes} {ts_sc2ts.num_nodes} "
        f"{ts_sc2ts.num_nodes / ts_usher.num_nodes * 100 : .2f}%"
    )
    print(
        f"mutations: {ts_usher.num_mutations} {ts_sc2ts.num_mutations} "
        f"{ts_sc2ts.num_mutations / ts_usher.num_mutations * 100 : .2f}%"
    )

    for u in tqdm.tqdm(range(num_samples), desc="Samples"):
        sc2ts_node = ts_sc2ts.node(u)
        usher_node = ts_usher.node(u)
        assert sc2ts_node.metadata["strain"] == samples_strain[u]
        assert usher_node.metadata["strain"] == samples_strain[u]

    var_sc2ts = tskit.Variant(ts_sc2ts, alleles=tuple("ACGT"))
    var_usher = tskit.Variant(ts_usher, alleles=tuple("ACGT"))

    identical_sites = 0
    total_diffs = 0
    for j in tqdm.tqdm(range(num_sites), desc="Sites"):
        var_sc2ts.decode(j)
        var_usher.decode(j)
        assert var_sc2ts.site.ancestral_state == var_usher.site.ancestral_state
        assert var_sc2ts.alleles == var_usher.alleles
        g_sc2ts = var_sc2ts.genotypes
        g_usher = var_usher.genotypes
        assert np.all(g_sc2ts >= 0)
        assert np.all(g_usher >= 0)
        diff = np.sum(g_sc2ts != g_usher)
        if diff == 0:
            identical_sites += 1
        total_diffs += diff
    print(f"identical_sites = {identical_sites} ({identical_sites / num_sites:.2f})")
    print(f"total differences = {total_diffs}")
    print(
        f"fraction of genotypes different = {total_diffs / (num_sites * num_samples): .2g}"
    )


@click.group()
def cli():
    """
    Utilities to convert Usher MAT files to sc2ts-like tskit format.
    """
    pass


cli.add_command(convert_topology)
cli.add_command(convert_mutations)
cli.add_command(date_samples)
cli.add_command(date_internal)
cli.add_command(intersect)
cli.add_command(validate)
cli()
