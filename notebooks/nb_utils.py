import collections
from datetime import datetime, timedelta
import json
import os
import requests
import fileinput

import sc2ts
import tskit
import tszip
import numpy as np
from tqdm.auto import tqdm
from matplotlib import pyplot as plt
from matplotlib.colors import rgb2hex


NODE_REPORT_KEYS = ", ".join([
    "title",
    "metadata",
    "node_title",
    "parents",
    "edge_title",
    "edges",
    "copying_title",
    "copying_pattern",
    "lpath_title",
    "closest_lrecomb",
    "lft_path",
    "rpath_title",
    "closest_rrecomb",
    "rgt_path",
    "children_title",
    "children",
    "mutations_title",
    "mutations"]
)

PARENT_COLOURS = [  # Chose to be light enough that black text on top is readable
    "#8D8",  # First parent: light green
    "#6AD",  # Second parent: light blue
    "#B9D",  # Third parent (if any): light purple
    "#A88",  # Fourth parent (if any): light brown
]

TSDIR = "../data"
 
def load(filename = "v1-beta1_2023-02-21.pp.md.bpshift.ts.dated.il.tsz"):
    ts = tszip.decompress(os.path.join(TSDIR, filename))
    print(
        f"Loaded {ts.nbytes/1e6:0.1f} megabyte SARS-CoV2 genealogy of {ts.num_samples} strains",
        f"({ts.num_trees} trees, {ts.num_mutations} mutations over {ts.sequence_length} basepairs).",
        f"Last collection date is {ts.node(ts.samples()[-1]).metadata['date']}",
    )
    return ts

def load_dataset(filename = "viridian_mafft_2024-10-14_v1.vcz"):
    return sc2ts.Dataset(os.path.join(TSDIR, filename), date_field="Date_tree")

def date(ts, node_id):
    return (
        datetime.fromisoformat(ts.node(1).metadata["date"]) + 
        timedelta(days=int(ts.node(1).time) - ts.node(node_id).time)
    )

def list_of_months(start_date, end_date):
    first_days = []
    current_year = start_date.year
    current_month = start_date.month if start_date.day == 1 else start_date.month + 1
    
    # Handle case where we need to increment the year
    if current_month > 12:
        current_year += 1
        current_month = 1
    
    while datetime(current_year, current_month, 1) < end_date:
        first_days.append(datetime(current_year, current_month, 1))
        
        # Move to next month
        current_month += 1
        if current_month > 12:
            current_year += 1
            current_month = 1
    return first_days

def remove_single_descendant_re_nodes(ts):
    re_nodes = np.where(ts.nodes_flags & sc2ts.NODE_IS_RECOMBINANT)[0]
    single_sample_re_nodes = []
    for u in re_nodes:
        children = np.unique(ts.edges_child[ts.edges_parent==u])
        if len(children) == 1 and children[0] in ts.samples():
            single_sample_re_nodes.append(u)
    tables = ts.dump_tables()
    nodes_flags = tables.nodes.flags
    nodes_flags[single_sample_re_nodes] = 0
    tables.nodes.flags = nodes_flags
    tables.simplify(list(set(ts.samples()) - set(ts.edges_child[np.isin(ts.edges_parent, single_sample_re_nodes)])), filter_nodes=False, keep_unary=True)
    return tables.tree_sequence()

def oldest_imputed(ts):
    oldest_imputed = collections.defaultdict(lambda: tskit.Node(-1, 0, 0, 0, 0, b""))
    for nd in tqdm(ts.nodes(), desc="Find oldest node for imputed Pangos"):
        pango = nd.metadata["Imputed_Viridian_pangolin"]
        if nd.time > oldest_imputed[pango].time:
            oldest_imputed[pango] = nd
    return oldest_imputed

def fetch_genbank_comment(accession):
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=nucleotide&rettype=gb&retmode=text"
    url += f"&id={accession}"
    response = requests.get(url)
    for line in response.text.split('\n'):
        if line.strip().startswith('COMMENT'):
            return line.strip()        
    return ""

def cumulative_branch_length(tree):
    """
    Calculate the cumulative branch length in the tree going from oldest to youngest
    (unique) node times in the tree. This is equivalent to the
    area available for mutations. The last value in the returned `cumulative_lengths`
    vector is the total branch length of this tree.

    Note that if there are any mutations above local roots, these do
    not occur on edges, and hence are not relevant to this function.
    
    Parameters:
    tree: tskit.Tree object
    
    Returns:
    tuple: (times, cumulative_lengths)
        times: array of unique node times where branches start or end, sorted descending
        cumulative_lengths: cumulative branch lengths from an indefinitely long time ago to
            each timepoint in the `times` array
    """
    used_ids = tree.edge_array[tree.edge_array != tskit.NULL]
    ts = tree.tree_sequence
    starts = -ts.nodes_time[ts.edges_parent[used_ids]]
    ends = -ts.nodes_time[ts.edges_child[used_ids]]
    
    # Create event arrays: each event has a position and a num lineage change
    # Start events add lineages, end events subtract lineages
    event_times = np.concatenate([starts, ends])
    event_lineage_count = np.concatenate([np.ones(len(used_ids)), -np.ones(len(used_ids))])
    times = np.unique(event_times)
    dt = np.diff(times)
    assert np.all(dt) > 0
    event_ind = np.searchsorted(times, event_times)
    cumulative_areas = np.cumsum(dt * np.cumsum(np.bincount(event_ind, weights=event_lineage_count))[:-1])
    return -times, np.concatenate([[0.0], cumulative_areas])

def mutation_p_values(ts, min_time=1, progress=True):
    """
    Calculate the p-value for each mutation in the tree sequence, based on the
    probability of the closest similar mutation in time (regardless of topological position)

    min_time (in days) gives the minimum time difference between mutations used to
    calculate a p-value (e.g. if the nearest similar mutation is 0.01 days, assume 1 day)
    """
    mut_p_val = np.ones(ts.num_mutations)

    def prob_closest_mut_within_timedelta(t, cdf_t, focal_times, time_deltas, num_muts):
        """
        Given mutational area cdf defined by t and cdf_t, and a number of mutations
        at a site, what is the probability of observing at least one of those mutations
        at a time difference of time_delta from a focal time, assuming mutations are
        randomly placed according to the cdf.

        """
        if cdf_t[-1] != 1:
            cdf_t = cdf_t/cdf_t[-1]
        if np.all(np.diff(t) <= 0): # make times go from 0 .. max_time
            t = t[::-1]
            cdf_t = cdf_t[::-1]
        assert np.all(np.diff(t) >= 0)
        # note the CDF is the wrong way around from what is conventional
        assert cdf_t[0] == 1
        assert cdf_t[-1] == 0
        prob_gt_time_deltas = np.interp(focal_times + time_deltas, t, cdf_t)
        prob_lt_time_deltas = 1-np.interp(focal_times - time_deltas, t, cdf_t)
        prob_outside = prob_gt_time_deltas + prob_lt_time_deltas
    
        return 1 - prob_outside**num_muts


    with tqdm(total=ts.num_mutations, disable=not progress) as pbar:
        for tree in ts.trees():
            # Null model is from the mutational areas
            times, lengths = cumulative_branch_length(tree)
            for site in tree.sites():
                mut_classes = collections.defaultdict(list)
                ds = {m.id: m.derived_state for m in site.mutations}
                ds[-1] = site.ancestral_state
                for m in site.mutations:
                    mut_classes[frozenset((m.derived_state, ds[m.parent]))].append((m.id, m.time))
                for mutations in mut_classes.values():
                    if len(mutations) != 1:
                        mut_ids = np.array([m[0] for m in mutations])
                        mut_times = np.array([m[1] for m in mutations])
                        nearest = np.zeros(len(mutations))
                        for i, focal_time in enumerate(mut_times):
                            nearest[i] = np.min(np.abs(focal_time - np.concatenate((mut_times[:i], mut_times[i+1:]))))
                        # e.g. if less than 1 day apart, assume a day's worth of difference (avoids p==0.0)
                        nearest = np.where(nearest< min_time, min_time, nearest)
                        mut_p_val[mut_ids] = prob_closest_mut_within_timedelta(
                            times, lengths, mut_times, nearest, len(mutations)-1
                        )
                    pbar.update(len(mutations))
    return mut_p_val


def set_sc2ts_labels_and_styles(d3arg, ts, add_strain_names=True):
    # Set node labels to Pango lineage + strain, if it exists
    # A questin mark at the end of a pango lineage indicates that the lineage is imputed
    def label_lines(md):
        s = md.get('strain', '')
        if s == "Vestigial_ignore":
            return([""])
        imputed = md.get("Imputed_Viridian_pangolin", "")
        if not imputed.startswith("Unknown"):
            imputed += "?"  # show this label is imputed using a question mark
        if add_strain_names:
            return [md.get("Viridian_pangolin", imputed), f"({s})"]
        else:
            return [md.get("Viridian_pangolin", imputed)]

    nodes = set(d3arg.nodes.id)
    d3arg.set_node_labels({
        u: "\n".join([s for s in label_lines(ts.node(u).metadata) if s not in ("()", "?")])
        for u in tqdm(range(ts.num_nodes), desc="Setting all labels")
        if u in nodes
    })
    # Mark recombination nodes in white and samples as squares
    d3arg.nodes.loc[:, "size"] = 50
    d3arg.nodes.loc[:, "fill"] = "darkgrey"
    d3arg.nodes.loc[:, "stroke_width"] = 1
    is_sample = np.isin(d3arg.nodes["id"], ts.samples())
    d3arg.nodes.loc[is_sample, "symbol"] = 'd3.symbolSquare'
    d3arg.nodes.loc[is_sample, "size"] = 100
    d3arg.nodes.loc[is_sample, "fill"] = "lightgrey"
    d3arg.set_node_styles([{"id": u, "fill": "white", "size": 150} for u in np.where(ts.nodes_flags & sc2ts.NODE_IS_RECOMBINANT)[0]])
    

def plot_sc2ts_subgraph(
    d3arg,
    nodes,
    title=None,
    parent_levels=20,
    child_levels=1,
    *,
    height=1000,
    width=800,
    cmap=plt.cm.tab10,
    y_axis_scale="time", # use "rank" to highlight topology more
    y_axis_labels=None,
    condense_mutations=False,
    return_included_nodes=None,
    colour_recurrent_mutations=True,
    include_mutation_labels=False,
    highlight_mutations=None,
    highlight_nodes=True,  # Can also be a mapping of colour to node IDs
    highlight_colour="plum",
    oldest_y_label=None,
    node_1_date=datetime(2019, 12, 26),  # date of Wuhan, node #1
):
    """
    Display a subset of the sc2ts arg, with mutations that are recurrent or reverted
    in the subgraph coloured by position (reversions have a black outline).

    highlight_mutations can be a list of mutation IDs to highlight in pinkish outline,
    or a PosAlt named tuple with positions and derived states (e.g. from
    `MutationContainer.get_mutations(Pango)` )
    """
    select_nodes = np.isin(d3arg.nodes.id, nodes)
    # Find mutations with duplicate position values and the same alleles (could be parallel eg. A->T & A->T, or reversions, e.g. A->T, T->A)
    # Create a composite key for basic duplicates
    if colour_recurrent_mutations:
        df = d3arg.subset_graph(nodes, (parent_levels, child_levels)).mutations.copy()
        df['fill'] = "lightgrey"
        df['stroke'] = "grey" # default stroke color
        df['duplicate_key'] = df.apply(lambda row: f"{row['position']}_{sorted([row['inherited'], row['derived']])}", axis=1)
        # Create a polarization key to identify the polarization of the allele arrangements
        df['polarization_key'] = df.apply(lambda row: f"{row['position']}_{row['inherited']}_{row['derived']}", axis=1)
        
        # Identify which rows are duplicates
        duplicate_mask = df.duplicated(subset=['duplicate_key'], keep=False)
            
        # Get only the duplicate keys that appear multiple times
        duplicate_keys = df[duplicate_mask]['duplicate_key'].unique()
        colors = {key: rgb2hex(cmap(i))
            for i, key in enumerate(duplicate_keys)
        }
        
        # Process only the duplicate groups
        for duplicate_key in duplicate_keys:
            group = df[df['duplicate_key'] == duplicate_key]
            polarization_keys = group['polarization_key'].unique()
                
            # Assign fill color based on duplicate_key
            fill = colors[duplicate_key]
            # If there are multiple direction keys, assign different strokes
            has_multiple_directions = len(polarization_keys) > 1
            for idx in group.index:
                df.loc[idx, 'fill'] = fill
                df.loc[idx, 'stroke'] = (
                    'grey' if not has_multiple_directions else 
                    ('grey' if group.loc[idx, 'polarization_key'] == polarization_keys[0] else 'black')
                )
        
        d3arg.mutations.loc[df.index, 'fill'] = df['fill']
        d3arg.mutations.loc[df.index, 'stroke'] = df['stroke']

    # For deletions, swap the stroke for the fill colour, and fill in black
    is_deletion = d3arg.mutations.derived == "-"
    stroke = d3arg.mutations.loc[is_deletion, 'fill']
    stroke = np.where(stroke == "white", "lightgrey", stroke)
    d3arg.mutations.loc[is_deletion, 'stroke'] = stroke
    d3arg.mutations.loc[is_deletion, 'fill'] = "black"

    # For going from a deletion back to the original state (insertion: very unexpected),
    # swap the stroke for the fill colour, and fill in magenta to highlight
    is_insertion = d3arg.mutations.inherited == "-"
    d3arg.mutations.loc[is_insertion, 'stroke'] = d3arg.mutations.loc[is_insertion, 'fill']
    d3arg.mutations.loc[is_insertion, 'fill'] = "magenta"

    if highlight_mutations is not None:
        if hasattr(highlight_mutations, "positions"):
            use = np.char.add(
                np.array(highlight_mutations.positions, dtype=str),
                highlight_mutations.derived_states
            )
            match = np.char.add(
                d3arg.mutations.position.astype(int).astype(str),
                d3arg.mutations.derived
            )
            d3arg.mutations.loc[np.isin(match, use), 'stroke'] = highlight_colour
        else:
            raise NotImplementedError("Cannot list mutations by ID yet")

    if highlight_nodes:
        d3arg.nodes.loc[select_nodes, 'fill'] = highlight_colour
        if isinstance(highlight_nodes, dict):
            for colour, nodelist in highlight_nodes.items():
                d3arg.nodes.loc[np.isin(d3arg.nodes.id, nodelist), 'fill'] = colour

    if y_axis_labels is None:
        times = d3arg.nodes.loc[select_nodes, 'time']
        zero_date = node_1_date + timedelta(days=int(d3arg.nodes.loc[d3arg.nodes.id == 1, 'time'].values[0]))
        if oldest_y_label is None:
            # Can't just use the oldest of select_nodes here, because that doesn't include parents
            oldest_y_label = node_1_date
        else:
            try:
                if len(oldest_y_label) == 7:
                    oldest_y_label += "-01"
                oldest_y_label = datetime.fromisoformat(oldest_y_label)
            except TypeError:
                pass
        y_axis_labels = {
            (zero_date - d).days: str(d)[:7]
            for d in list_of_months(oldest_y_label, zero_date-timedelta(days=times.min()))
        }

    shown_nodes = d3arg.draw_nodes(
        nodes,
        title=title,
        degree=(parent_levels, child_levels),
        height=height,
        width=width,
        show_mutations=True,
        y_axis_scale=y_axis_scale,
        y_axis_labels=y_axis_labels,
        include_mutation_labels=include_mutation_labels,
        condense_mutations=condense_mutations,
        return_included_nodes=return_included_nodes
    )
    if return_included_nodes:
        return shown_nodes

def set_x_01_from_json(d3arg, json_file):
    default = 0.5
    with open(json_file) as f:
        j = json.load(f)
        x_pos = {nd["id"]: nd.get(("x"), nd.get("fx", default)) for nd in j["data"]["nodes"]}
        mx, mn = max(x_pos.values()), min(x_pos.values())
        x_pos_01 = {k: (v-mn)/(mx-mn) for k, v in x_pos.items()}
        d3arg.nodes["x_pos_01"] = d3arg.nodes.id.map(x_pos_01)
        # Unspecified nodes in the middle
        d3arg.nodes.fillna({"x_pos_01": default}, inplace=True)
        return x_pos  # Just for debugging
    
def clear_x_01(d3arg):
    d3arg.nodes.drop(columns=["x_pos_01"], errors="ignore", inplace=True)

## from run_lineage_imputation.py
class MutationContainer:

    def __init__(self):
        self.names = {}
        self.positions = []
        self.alts = []
        self.size = 0
        self.all_positions = {}

    def add_root(self, root_lineage_name):
        self.names[root_lineage_name] = self.size
        self.size += 1
        self.positions.append([])
        self.alts.append([])

    def add_item(self, item, position, alt):
        if item not in self.names:
            self.names[item] = self.size
            self.positions.append([position])
            self.alts.append([alt])
            self.size += 1
        else:
            index = self.names[item]
            self.positions[index].append(position)
            self.alts[index].append(alt)
        # map each position to a set of alt alleles
        if position in self.all_positions:
            self.all_positions[position].add(alt)
        else:
            self.all_positions[position] = {alt}
 
    def get_mutations(self, pango):
        index = self.names[pango]
        return np.rec.fromarrays(
            [self.positions[index], self.alts[index]],
            names='positions,derived_states',
        )

    def get_unique_mutations(self, pango, parent_pangos):
        # return the mutations in item that are not in any of the parents
        if isinstance(parent_pangos, str):
            parent_pangos = [parent_pangos]
        idx = self.names[pango]
        use = np.rec.fromarrays([self.positions[idx], self.alts[idx]], names='positions,derived_states')
        indexes = [self.names[p] for p in parent_pangos]
        omit = np.concatenate([
            np.rec.fromarrays([self.positions[i], self.alts[i]], names='positions,derived_states') for i in indexes
        ])
        return np.setdiff1d(use, omit)
    
## from run_lineage_imputation.py
def read_in_mutations(
    json_filepath,
    verbose=False,
    exclude_positions=None,
):
    """
    Read in lineage-defining mutations from COVIDCG input json file.
    Assumes root lineage is B.
    """
    if exclude_positions is None:
        exclude_positions = set()
    else:
        exclude_positions = set(exclude_positions)
    with fileinput.hook_compressed(json_filepath, "r") as file:
        linmuts = json.load(file)

    # Read in lineage defining mutations
    linmuts_dict = MutationContainer()
    linmuts_dict.add_root("B")
    if verbose:
        check_multiallelic_sites = collections.defaultdict(
            set
        )  # will check how many multi-allelic sites there are

    excluded_pos = collections.defaultdict(int)
    excluded_del = collections.defaultdict(int)
    for item in linmuts:
        if item["pos"] in exclude_positions:
            excluded_pos[item["pos"]] += 1
        elif item["ref"] == "-" or item["alt"] == "-":
            excluded_del[item["pos"]] += 1
        else:
            linmuts_dict.add_item(item["name"], item["pos"], item["alt"])
            if verbose:
                check_multiallelic_sites[item["pos"]].add(item["ref"])
            if verbose:
                check_multiallelic_sites[item["pos"]].add(item["alt"])

    if verbose:
        multiallelic_sites_count = 0
        for value in check_multiallelic_sites.values():
            if len(value) > 2:
                multiallelic_sites_count += 1
        print(
            "Multiallelic sites:",
            multiallelic_sites_count,
            "out of",
            len(check_multiallelic_sites),
        )
        print("Number of lineages:", linmuts_dict.size)
        if len(excluded_pos) > 0:
            print(
                f"Excluded {len(excluded_pos)} positions not in ts:",
                f"{list(excluded_pos.keys())}"
            )
        if len(excluded_del) > 0:
            print("Excluded deletions at positions", list(excluded_del.keys()))

    return linmuts_dict


##### Code below ported from sc2ts, see https://github.com/jeromekelleher/sc2ts/issues/445 


def node_path_to_samples(
    nodes, ts, rootwards=True, ignore_initial=True, stop_at_recombination=False
):
    """
    Given a list of nodes, traverse rootwards (if rootwards is True) or
    tipwards (if rootwards is False) to the nearest sample nodes,
    returning all nodes on the path, including the nodes passed in.
    Note that this does not account for genomic intervals, so parent
    or child edges can be followed even if no genetic material links
    them to the passed-in nodes.

    :param rootwards bool: If True, ascend rootwards, otherwise descend tipwards.
    :param ignore_initial bool: If True, the initial nodes passed in are not considered
        as samples for the purposes of stopping the traversal.
    :param stop_at_recombination bool: If True, stop the traversal at recombination nodes.
    """
    nodes = np.array(list(nodes))
    ret = {n: True for n in nodes}  # Use a dict not a set, to maintain order
    if not ignore_initial:
        nodes = nodes[(ts.nodes_flags[nodes] & tskit.NODE_IS_SAMPLE) == 0]
        if stop_at_recombination:
            nodes = nodes[(ts.nodes_flags[nodes] & sc2ts.NODE_IS_RECOMBINANT) == 0]
    while len(nodes) > 0:
        if rootwards:
            nodes = ts.edges_parent[np.isin(ts.edges_child, nodes)]
        else:
            nodes = ts.edges_child[np.isin(ts.edges_parent, nodes)]
        ret.update({n: True for n in nodes})
        nodes = nodes[(ts.nodes_flags[nodes] & tskit.NODE_IS_SAMPLE) == 0]
        if stop_at_recombination:
            nodes = nodes[(ts.nodes_flags[nodes] & sc2ts.NODE_IS_RECOMBINANT) == 0]
    return np.array(list(ret.keys()), dtype=ts.edges_child.dtype)


def edges_for_nodes(ts, nodes, include_external=False):
    """
    Returns the edges that connect the specified numpy array of nodes in the ts.
    """
    edges = np.logical_and(
        np.isin(ts.edges_child, nodes),
        np.isin(ts.edges_parent, nodes),
    )
    return np.flatnonzero(edges)


def to_nx_subgraph(ts, nodes, return_external_edges=False):
    """
    Return a networkx graph relating the specified nodes.
    If return_external_edges is true, also return a tuple
    of (parent_edge_list, child_edge_list) giving the edges
    from the nodes that are *not* in the graph (because they
    connect to nodes not in ``nodes``)
    """
    G = nx.DiGraph()
    for u in nodes:
        G.add_node(u)
    edges = edges_for_nodes(ts, nodes)
    for parent, child in zip(ts.edges_parent[edges], ts.edges_child[edges]):
        G.add_edge(parent, child)
    if return_external_edges:
        parent_e = np.setdiff1d(np.flatnonzero(np.isin(ts.edges_child, nodes)), edges)
        parent_e = parent_e[np.argsort(ts.edges_child[parent_e])]  # Sort by child id
        child_e = np.setdiff1d(np.flatnonzero(np.isin(ts.edges_parent, nodes)), edges)
        child_e = child_e[np.argsort(ts.edges_parent[child_e])]  # Sort by parent id

        return G, (parent_e, child_e)
    return G


def plot_subgraph(
    nodes,
    ts,
    ti=None,
    show_mutation_positions=None,  # NB - can pass linmuts.all_positions
    filepath=None,
    *,
    ax=None,
    node_size=None,
    exterior_edge_len=None,
    node_colours=None,
    colour_metadata_key=None,
    ts_id_labels=None,
    node_metadata_labels=None,
    sample_metadata_labels=None,
    show_descendant_samples=None,
    edge_labels=None,
    edge_font_size=None,
    node_font_size=None,
    label_replace=None,
    node_positions=None,
):
    """
    Draws out a subgraph of the ARG defined by the provided node ids and the
    edges connecting them.

    :param list nodes: A list of node ids used in the subgraph. Only edges connecting
        these nodes will be drawn.
    :param tskit.TreeSequence ts: The tree sequence to use.
    :param TreeInfo ti: The TreeInfo instance associated with the tree sequence. If
        ``None`` calculate the TreeInfo within this function. However, as
        calculating the TreeInfo class takes some time, if you have it calculated
        already, it is far more efficient to pass it in here.
    :param str show_mutation_positions: A set of integer positions (only relevant
        if ``edge_labels`` is ``None``). If provided, only mutations in this file will
        be listed on edges of the plot, with others shown as "+N mutations". If ``None``
        (default), show all mutations. If the empty set, only plot the number of mutations.
    :param str filepath: If given, save the plot to this file path.
    :param plt.Axes ax: a matplotlib axis object on which to plot the graph.
        This allows the graph to be placed as a subplot or the size and aspect ratio
        to be adjusted. If ``None`` (default) plot to the current axis with some
        sensible figsize defaults, calling ``plt.show()`` once done.
    :param int node_size: The size of the node circles. Default:
        ``None``, treated as 2800.
    :param bool exterior_edge_len: The relative length of the short dotted lines,
        representing missing edges to nodes that we have not drawn. If ``0``,
        do not plot such lines. Default: ``None``, treated as ``0.4``.
    :param bool ts_id_labels: Should we label nodes with their tskit node ID? If
        ``None``, show the node ID only for sample nodes. If ``True``, show
        it for all nodes. If ``False``, do not show. Default: ``None``.
    :param str node_metadata_labels: Should we label all nodes with a value from their
        metadata: Default: ``None``, treated as ``"Imputed_GISAID_lineage"``. If ``""``,
        do not plot any all-node metadata.
    :param str sample_metadata_labels: Should we additionally label sample nodes with a
        value from their metadata: Default: ``None``, treated as ``"gisaid_epi_isl"``.
    :param str show_descendant_samples: Should we label nodes with the maximum number
        of samples descending from them in any tree (in the format "+XXX samples").
        If ``"samples"``, only label sample nodes. If "tips", label all tip nodes.
        If ``"sample_tips"` label all tips that are also samples. If ``"all"``, label
        all nodes. If ``""`` or False, do not show labels. Default: ``None``, treated
        as ``"sample_tips"``. If a node has no descendant samples, a label is not placed.
    :param dict edge_labels: a mapping of {(parent_id, child_id): "label")} with which
        to label the edges. If ``None``, label with mutations or (if above a
        recombination node) with the edge interval. If ``{}``, do not plot
        edge labels.
    :param float edge_font_size: The font size for edge labels.
    :param float node_font_size: The font size for node labels.
    :param dict label_replace: A dict of ``{key: value}`` such that node or edge
        labels containing the string ``key`` have that string replaced with
        ``value``. For example, the word "Unknown" can be removed from
        the plot, by specifying ``{"Unknown": "", "Unknown ": ""}``.
    :param dict node_colours: A dict mapping nodes to colour values. The keys of the
        dictionary can be integer node IDs, strings, or None. If the key is a string,
        it is compared to the value of ``node.metadata[colour_metadata_key]`` (see
        below). If no relevant key exists, the fill colour is set to the value of
        ``node_colours[None]``, or is set to empty if there is no key of ``None``.
        However, if ``node_colours`` is itself ``None``, use the default colourscheme
        which distinguishes between sample nodes, recombination nodes, and all others.
    :param dict colour_metadata_key: A key in the metadata, to use when specifying
        bespoke node colours. Default: ``None``, treated as "strain".
    :param dict node_positions: A dictionary of ``node_id: [x, y]`` positions, for
        example .


    :return: The networkx Digraph
    :rtype:  nx.DiGraph

    """

    def sort_mutation_label(s):
        """
        Mutation labels are like "A123T", "+1 mutation", or "3",
        """
        try:
            return float(s)
        except ValueError:
            if s[0] == "$":
                # matplotlib mathtext - remove the $ and the formatting
                s = (
                    s.replace("$", "")
                    .replace(r"\bf", "")
                    .replace(r"\it", "")
                    .replace("{", "")
                    .replace("}", "")
                )
            try:
                return float(s[1:-1])
            except ValueError:
                return np.inf  # put at the end

    if ti is None:
        ti = sc2ts.info.TreeInfo(ts)
    if node_size is None:
        node_size = 2800
    if edge_font_size is None:
        edge_font_size = 5
    if node_font_size is None:
        node_font_size = 6
    if node_metadata_labels is None:
        node_metadata_labels = "Imputed_Viridian_pangolin"
    if sample_metadata_labels is None:
        sample_metadata_labels = "gisaid_epi_isl"
    if show_descendant_samples is None:
        show_descendant_samples = "sample_tips"
    if colour_metadata_key is None:
        colour_metadata_key = "strain"
    if exterior_edge_len is None:
        exterior_edge_len = 0.4

    if show_descendant_samples not in {
        "samples",
        "tips",
        "sample_tips",
        "all",
        "",
        False,
    }:
        raise ValueError(
            "show_descendant_samples must be one of 'samples', 'tips', 'sample_tips', 'all', or '' / False"
        )

    exterior_edges = None
    if exterior_edge_len != 0:
        G, exterior_edges = to_nx_subgraph(ts, nodes, return_external_edges=True)
    else:
        G = to_nx_subgraph(ts, nodes)

    nodelabels = collections.defaultdict(list)
    shown_tips = []
    for u, out_deg in G.out_degree():
        node = ts.node(u)
        if node_metadata_labels:
            nodelabels[u].append(node.metadata[node_metadata_labels])
        if ts_id_labels or (ts_id_labels is None and node.is_sample()):
            nodelabels[u].append(f"tsk{node.id}")
        if node.is_sample():
            if sample_metadata_labels:
                nodelabels[u].append(node.metadata[sample_metadata_labels])
        if show_descendant_samples:
            show = True if show_descendant_samples == "all" else False
            is_tip = out_deg == 0
            if show_descendant_samples == "tips" and is_tip:
                show = True
            elif node.is_sample():
                if show_descendant_samples == "samples":
                    show = True
                elif show_descendant_samples == "sample_tips" and is_tip:
                    show = True
            if show:
                s = ti.nodes_max_descendant_samples[u]
                if node.is_sample():
                    s -= 1  # don't count self
                if s > 0:
                    nodelabels[u].append(f"+{s} {'samples' if s > 1 else 'sample'}")

    nodelabels = {k: "\n".join(v) for k, v in nodelabels.items()}

    interval_labels = {k: collections.defaultdict(str) for k in ("lft", "mid", "rgt")}
    mutation_labels = collections.defaultdict(set)

    ## Details for mutations (labels etc)
    mut_nodes = set()
    mutation_suffix = collections.defaultdict(set)
    used_edges = set(edges_for_nodes(ts, nodes))
    for m in ts.mutations():
        if m.edge in used_edges:
            mut_nodes.add(m.node)
            if edge_labels is None:
                edge = ts.edge(m.edge)
                pos = int(ts.site(m.site).position)
                includemut = False
                if m.parent == tskit.NULL:
                    inherited_state = ts.site(m.site).ancestral_state
                else:
                    inherited_state = ts.mutation(m.parent).derived_state

                if ti.mutations_is_reversion[m.id]:
                    mutstr = f"$\\bf{{{inherited_state.lower()}{pos}{m.derived_state.lower()}}}$"
                elif ts.mutations_parent[m.id] != tskit.NULL:
                    mutstr = f"$\\bf{{{inherited_state.upper()}{pos}{m.derived_state.upper()}}}$"
                else:
                    mutstr = f"{inherited_state.upper()}{pos}{m.derived_state.upper()}"
                if show_mutation_positions is None or pos in show_mutation_positions:
                    includemut = True
                if includemut:
                    mutation_labels[(edge.parent, edge.child)].add(mutstr)
                else:
                    mutation_suffix[(edge.parent, edge.child)].add(mutstr)
    for key, value in mutation_suffix.items():
        mutation_labels[key].add(
            ("" if len(mutation_labels[key]) == 0 else "+")
            + f"{len(value)} mutation{'s' if len(value) > 1 else ''}"
        )

    multiline_mutation_labels = False
    for key, value in mutation_labels.items():
        mutation_labels[key] = "\n".join(sorted(value, key=sort_mutation_label))
        if len(value) > 1:
            multiline_mutation_labels = True

    if edge_labels is None:
        for pc in G.edges():
            if ts.node(pc[1]).flags & sc2ts.NODE_IS_RECOMBINANT:
                for e in edges_for_nodes(ts, pc):
                    edge = ts.edge(e)
                    lpos = "mid"
                    if edge.left == 0 and edge.right < ts.sequence_length:
                        lpos = "lft"
                    elif edge.left > 0 and edge.right == ts.sequence_length:
                        lpos = "rgt"
                    # Add spaces between or in front of labels if
                    # multiple lft or rgt labels (i.e. intervals) exist for an edge
                    if interval_labels[lpos][pc]:  # between same side labels
                        interval_labels[lpos][pc] += "  "
                    if (
                        lpos == "rgt" and interval_labels["lft"][pc]
                    ):  # in front of rgt label
                        interval_labels[lpos][pc] = "  " + interval_labels[lpos][pc]
                    interval_labels[lpos][pc] += f"{int(edge.left)}…{int(edge.right)}"
                    if (
                        lpos == "lft" and interval_labels["rgt"][pc]
                    ):  # at end of lft label
                        interval_labels[lpos][pc] += "  "

    if label_replace is not None:
        for search, replace in label_replace.items():
            for k, v in nodelabels.items():
                nodelabels[k] = v.replace(search, replace)
            for k, v in mutation_labels.items():
                mutation_labels[k] = v.replace(search, replace)
            for key in interval_labels.keys():
                for k, v in interval_labels[key].items():
                    interval_labels[key][k] = v.replace(search, replace)

    # Shouldn't need this once https://github.com/jeromekelleher/sc2ts/issues/132 fixed
    unary_nodes_to_remove = set()
    for (k, in_deg), (k2, out_deg) in zip(G.in_degree(), G.out_degree()):
        assert k == k2
        flags = ts.node(k).flags
        if (
            in_deg == 1
            and out_deg == 1
            and k not in mut_nodes
            and not (flags & sc2ts.NODE_IS_RECOMBINANT)
        ):
            G.add_edge(*G.predecessors(k), *G.successors(k))
            for d in [mutation_labels, *list(interval_labels.values()), edge_labels]:
                if d is not None and (k, *G.successors(k)) in d:
                    d[(*G.predecessors(k), *G.successors(k))] = d.pop(
                        (k, *G.successors(k))
                    )
            unary_nodes_to_remove.add(k)
    [G.remove_node(k) for k in unary_nodes_to_remove]
    nodelabels = {k: v for k, v in nodelabels.items() if k not in unary_nodes_to_remove}

    if node_positions is None:
        node_positions = nx.nx_agraph.graphviz_layout(G, prog="dot")
    if ax is None:
        dim_x = len(set(x for x, y in node_positions.values()))
        dim_y = len(set(y for x, y in node_positions.values()))
        fig, ax = plt.subplots(1, 1, figsize=(dim_x * 1.5, dim_y * 1.1))

    if exterior_edges is not None:
        # Draw a short dotted line above nodes with extra parent edges to show that more
        # topology exists above them. For simplicity we assume when calculating how to
        # space the lines that no other parent edges from this node have been plotted.
        # Parent edges are sorted by child id, so we can use this to groupby

        # parent-child dist
        av_y = np.mean(
            [node_positions[u][1] - node_positions[v][1] for u, v in G.edges()]
        )
        # aspect_ratio = np.divide(*np.ptp([[x, y] for x, y in node_positions.values()], axis=0))
        aspect_ratio = 1.0
        for child, edges in itertools.groupby(
            exterior_edges[0], key=lambda e: ts.edge(e).child
        ):
            edges = list(edges)[:6]  # limit to 6 lines, otherwise it gets messy
            for x in [0] if len(edges) < 2 else np.linspace(-1, 1, len(edges)):
                dx = x * aspect_ratio * av_y * exterior_edge_len
                dy = av_y * exterior_edge_len
                # make lines the same length
                hypotenuse = np.sqrt(dx**2 + dy**2)
                dx *= dy / hypotenuse
                dy *= dy / hypotenuse
                ax.plot(
                    [node_positions[child][0], node_positions[child][0] + dx],
                    [node_positions[child][1], node_positions[child][1] + dy],
                    marker="",
                    linestyle=":",
                    color="gray",
                    zorder=-1,
                )

        # Draw a short dotted line below nodes with extra child edges to show that more
        # topology exists below them. For simplicity we assume when calculating how to
        # space the lines that no other child edges from this node have been plotted.
        # Child edges are sorted by child id, so we can use this to groupby
        for parent, edges in itertools.groupby(
            exterior_edges[1], key=lambda e: ts.edge(e).parent
        ):
            edges = list(edges)[:6]  # limit to 6 lines, otherwise it gets messy
            for x in [0] if len(edges) < 2 else np.linspace(-1, 1, len(edges)):
                dx = x * aspect_ratio * av_y * exterior_edge_len
                dy = av_y * exterior_edge_len
                # make lines the same length
                hypotenuse = np.sqrt(dx**2 + dy**2)
                dx *= dy / hypotenuse
                dy *= dy / hypotenuse
                ax.plot(
                    [node_positions[parent][0], node_positions[parent][0] + dx],
                    [node_positions[parent][1], node_positions[parent][1] - dy],
                    marker="",
                    linestyle=":",
                    color="gray",
                    zorder=-1,
                )

    fill_cols = []
    if node_colours is None:
        for u in G.nodes:
            fill_cols.append(
                "k" if ts.node(u).flags & sc2ts.NODE_IS_RECOMBINANT else "white"
            )
    else:
        default_colour = node_colours.get(None, "None")
        for u in G.nodes:
            try:
                fill_cols.append(node_colours[u])
            except KeyError:
                md_val = ts.node(u).metadata.get(colour_metadata_key, None)
                fill_cols.append(node_colours.get(md_val, default_colour))

    # Put a line around the point if white or transparent
    stroke_cols = [
        "black"
        if col == "None" or np.mean(colors.ColorConverter.to_rgb(col)) > 0.99
        else col
        for col in fill_cols
    ]
    fill_cols = np.array(fill_cols)
    stroke_cols = np.array(stroke_cols)

    is_sample = np.array([ts.node(u).is_sample() for u in G.nodes])
    # Use a loop so allow possiblity of different shapes for samples and non-samples
    for use_sample, shape, size in zip(
        [True, False], ["o", "o"], [node_size, node_size / 3]
    ):
        node_list = np.array(list(G.nodes))
        use = is_sample == use_sample
        nx.draw_networkx_nodes(
            G,
            node_positions,
            nodelist=node_list[use],
            ax=ax,
            node_color=fill_cols[use],
            edgecolors=stroke_cols[use],
            node_size=size,
            node_shape=shape,
        )
    nx.draw_networkx_edges(
        G,
        node_positions,
        ax=ax,
        node_size=np.where(is_sample, node_size, node_size / 3),
        arrowstyle="-",
    )

    black_labels = {}
    white_labels = {}
    for node, col in zip(list(G), fill_cols):
        if node in nodelabels:
            if col == "None" or np.mean(colors.ColorConverter.to_rgb(col)) > 0.2:
                black_labels[node] = nodelabels[node]
            else:
                white_labels[node] = nodelabels[node]
    if black_labels:
        nx.draw_networkx_labels(
            G,
            node_positions,
            ax=ax,
            labels=black_labels,
            font_size=node_font_size,
            font_color="k",
        )
    if white_labels:
        nx.draw_networkx_labels(
            G,
            node_positions,
            ax=ax,
            labels=white_labels,
            font_size=node_font_size,
            font_color="w",
        )
    av_dy = np.median(
        [
            # We could use the minimum y diff here, but then could be susceptible to
            # pathological cases where the y diff is very small.
            np.abs(node_positions[u][1] - node_positions[v][1])
            for u, v in G.edges
        ]
    )
    ax_height = np.diff(ax.get_ylim())
    height_pts = (
        ax.get_position().transformed(ax.get_figure().transFigure).height
        * 72
        / ax.get_figure().dpi
    )
    node_height = np.sqrt(node_size) / height_pts * ax_height
    if multiline_mutation_labels:
        # Bottom align mutations: useful when there are multiple lines of mutations
        mut_pos = node_height / 2 / av_dy
        mut_v_align = "bottom"
    else:
        # Center align mutations, still placed near the child if possible
        font_height = edge_font_size / height_pts * ax_height
        mut_pos = (node_height / 2 + font_height / 2) / av_dy
        if mut_pos > 0.5:
            # Never go further up the line than the middle
            mut_pos = 0.5
        mut_v_align = "center"

    for name, (labels, position, valign, halign) in {
        "mutations": [mutation_labels, mut_pos, mut_v_align, "center"],
        "user": [edge_labels, 0.5, "center", "center"],
        "intervals_l": [interval_labels["lft"], 0.6, "top", "right"],
        "intervals_m": [interval_labels["mid"], 0.6, "top", "center"],
        "intervals_r": [interval_labels["rgt"], 0.6, "top", "left"],
    }.items():
        if labels:
            font_color = "darkred" if name == "mutations" else "k"
            nx.draw_networkx_edge_labels(
                G,
                node_positions,
                ax=ax,
                edge_labels=labels,
                label_pos=position,
                verticalalignment=valign,
                horizontalalignment=halign,
                font_color=font_color,
                rotate=False,
                font_size=edge_font_size,
                bbox={"facecolor": "white", "pad": 0.5, "edgecolor": "none"},
            )
    if filepath:
        plt.savefig(filepath)
    elif ax is None:
        plt.show()
    return G, node_positions


def sample_subgraph(sample_node, ts, ti=None, **kwargs):
    """
    Returns a subgraph of the tree sequence containing the specified nodes.
    """
    # Ascend up from input node
    up_nodes = node_path_to_samples([sample_node], ts)
    # Descend from these
    nodes = sc2ts.node_path_to_samples(
        up_nodes, ts, rootwards=False, ignore_initial=False
    )
    # Ascend again, to get parents of downward nonsamples
    up_nodes = sc2ts.node_path_to_samples(nodes, ts, ignore_initial=False)
    nodes = np.append(nodes, up_nodes)
    # Remove duplicates
    _, idx = np.unique(nodes, return_index=True)
    nodes = nodes[np.sort(idx)]

    return plot_subgraph(nodes, ts, ti, **kwargs)


### Routines to facilitate analysis of novel recombinants.
# Use pangonet to compute node distances between Pango labels.
def initialise_pangonet(alias_key_file, lineage_notes_file):
    from pangonet.pangonet import PangoNet
    pangonet = PangoNet().build(alias_key=alias_key_file, lineage_notes=lineage_notes_file)
    return pangonet


def get_pangonet_distance(pangonet, *, label_1, label_2):
    """Get the number of edges separating two Pango labels on a reference phylogeny."""
    # Pangonet sometimes returns empty paths between uncompressed labels.
    # So, it is better to work with compressed labels instead.
    label_1_c = pangonet.compress(label_1)
    label_2_c = pangonet.compress(label_2)
    # Special case
    if label_1_c == label_2_c:
        return 0    # Distance
    # Check ancestor-descendant relationship
    label_1_anc = [pangonet.uncompress(p) for p in pangonet.get_ancestors(label_1_c)]
    label_2_anc = [pangonet.uncompress(p) for p in pangonet.get_ancestors(label_2_c)]
    if (label_1_c in label_2_anc) or (label_2_c in label_1_anc):
        # Paths include the focal nodes
        anc_desc_path = pangonet.get_paths(start=label_1_c, end=label_2_c)
        if len(anc_desc_path) != 1:
            raise ValueError("pangonet returns unexpected number of paths.")
        distance = len(anc_desc_path[0]) - 1
    else:
        mrca = pangonet.get_mrca([label_1_c, label_2_c])
        if len(mrca) != 1:
            raise ValueError("pangonet returns unexpected number of MRCAs.")
        # Paths include the focal nodes
        mrca_pango_1_path = pangonet.get_paths(start=label_1_c, end=mrca[0])
        mrca_pango_2_path = pangonet.get_paths(start=label_2_c, end=mrca[0])
        if (len(mrca_pango_1_path) != 1) or (len(mrca_pango_2_path) != 1):
            raise ValueError("pangonet returns unexpected number of paths.")
        mrca_pango_1_distance = len(mrca_pango_1_path[0]) - 1
        mrca_pango_2_distance = len(mrca_pango_2_path[0]) - 1
        distance = mrca_pango_1_distance + mrca_pango_2_distance
    return distance


def draw_stacked_histogram(
    a,
    b,
    ax,
    *,
    alegend,
    blegend,
    xlabel,
    ylabel,
    xlim,
    show_legend=True,
):
    bin_edges = np.arange(xlim[0], xlim[1])
    hist_a, _ = np.histogram(a, bins=bin_edges)
    hist_b, _ = np.histogram(b, bins=bin_edges)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2 - 0.5
    bar_width = 0.8
    _ = ax.bar(
        bin_centers,
        hist_a,
        width=bar_width,
        label=alegend,
    )
    _ = ax.bar(
        bin_centers,
        hist_b,
        bottom=hist_a,
        width=bar_width,
        label=blegend,
    )
    ax.set_xticks(bin_centers.astype(int))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if show_legend:
        ax.legend();
