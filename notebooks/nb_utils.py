import collections
from datetime import datetime, timedelta
import json
import os
import requests
import fileinput
from pathlib import Path

import msprime  # only for the NODE_IS_RE_EVENT flag
import pandas as pd
import sc2ts
import tskit
import tszip
import numpy as np
from tqdm.auto import tqdm
from matplotlib import pyplot as plt
from matplotlib.colors import rgb2hex
import tskit_arg_visualizer as argviz

DATA_DIR = Path("../data")

def print_info(ts):
    time_zero_date = ts.metadata.get("time_zero_date", "Unknown date")
    print(
        f"Using a {ts.nbytes/1e6:0.1f} megabyte ARG up to "
        f"{time_zero_date}, with {ts.num_samples} sampled SARS-CoV2 sequences"
    )
    n_RE = np.sum(ts.nodes_flags & sc2ts.NODE_IS_RECOMBINANT != 0)
    print(
        f"({ts.num_trees} trees, {ts.num_mutations} mutations "
        f"over {ts.sequence_length}bp with {n_RE} recomb. events)"
    )
    return ts

def load_dataset(filename="viridian_mafft_2024-10-14_v1.vcz.zip"):
    return sc2ts.Dataset(os.path.join(DATA_DIR, filename), date_field="Date_tree")



def standard_recombinant_labels(ts, pango_x_events_file):
    """
    Return a standard set of labels for "known" recombination nodes
    (Pango Xs plus hord-coded Jackson recombinants). The
    pango_x_events_file should be a path to pango_x_events.csv
    """
    df = sc2ts.node_data(ts).set_index("sample_id")
    pango_x_events = pd.read_csv(pango_x_events_file)
    tree = ts.first()  # Assume the first tree gives  areasonable number of descendant samples for a RE node
    # First keep all the standard recombinants
    use = pango_x_events.root_type == "R"
    # Cases where one PangoX has multiple RE nodes (only XM), use the one with the most descendants
    for p in np.unique(pango_x_events[use].root_pango):
        match = pango_x_events.root_pango == p
        if sum(match) > 1:
            use[match] = False
            sorted_indexes = sorted(
                np.where(match)[0],
                key=lambda i: tree.num_samples(pango_x_events.closest_recombinant[i])
            )
            use[sorted_indexes[-1]] = True
    
    labels = {row.closest_recombinant: row.root_pango for row in pango_x_events[use].itertuples()}
    assert len(labels) == np.sum(use)
    
    labels = {row.closest_recombinant: row.root_pango for row in pango_x_events[use].itertuples()}
    for re_node, rows in pango_x_events[pango_x_events.root_type != "R"].groupby("closest_recombinant"):
        if re_node >=0:
            # exclude those with > 100,000 descendants (to exclude BA.5 as a RE node)
            if re_node not in labels and tree.num_samples(re_node) < 1e5:
                if len(rows) == 1:
                    labels[re_node] = rows.iloc[0].root_pango
                else:
                    labels[re_node] = "/".join(sorted(rows.root_pango.values))
    
    # Tweak the "XBB.1" RE node label, which should have a bespoke label because it's not
    # reflective of the majority of XBB.1 samples
    XBB_1 = [k for k, v in labels.items() if v == "XBB.1"][0]
    labels[XBB_1] = "XBB.x"
    Xx = [k for k, v in labels.items() if "XZ" in v][0]
    labels[Xx] = "Xx"
    
    # Add particular pangos that are not X but are recombinants
    
    #for re_pango in ["BQ.1.21"]:  # BQ.1.21 is not robust, so skip it
    #    potential_re_node = ts.first().mrca(*df.loc[df.pango == re_pango, "node_id"])
    #    assert ts.node(potential_re_node).flags & sc2ts.NODE_IS_RECOMBINANT
    #    labels[potential_re_node] = re_pango
    
    # Add the Jackson recombinants
    XA = [k for k, v in labels.items() if v == "XA"][0]
    labels[XA] = "XA(JA)"
    
    jackson_recombs = {
        "JB": "ERR5058070",
        "JC": "ERR5232711",
        "JD": "ERR5335088",
        "J1": "ERR5054123", # CAMC-CBA018
        "J2": "ERR5304348", # MILK-103C712
        "J3": "ERR5238288", # QEUH-1067DEF
    }
    
    for label, sample_id in jackson_recombs.items():
        u = df.loc[sample_id, "node_id"]
        it = tree.ancestors(u)
        while not(ts.nodes_flags[u] & sc2ts.NODE_IS_RECOMBINANT):
            u = next(it)
        assert u != -1
        labels[int(u)] = label
    
    return labels


def date(ts, node_id):
    return date.fromisoformat(ts.metadata["time_zero_date"]) - timedelta(
        days=ts.node(node_id).time
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
        children = np.unique(ts.edges_child[ts.edges_parent == u])
        if len(children) == 1 and children[0] in ts.samples():
            single_sample_re_nodes.append(u)
    tables = ts.dump_tables()
    nodes_flags = tables.nodes.flags
    nodes_flags[single_sample_re_nodes] = 0
    tables.nodes.flags = nodes_flags
    tables.simplify(
        list(
            set(ts.samples())
            - set(ts.edges_child[np.isin(ts.edges_parent, single_sample_re_nodes)])
        ),
        filter_nodes=False,
        keep_unary=True,
    )
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
    for line in response.text.split("\n"):
        if line.strip().startswith("COMMENT"):
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
    event_lineage_count = np.concatenate(
        [np.ones(len(used_ids)), -np.ones(len(used_ids))]
    )
    times = np.unique(event_times)
    dt = np.diff(times)
    assert np.all(dt) > 0
    event_ind = np.searchsorted(times, event_times)
    cumulative_areas = np.cumsum(
        dt * np.cumsum(np.bincount(event_ind, weights=event_lineage_count))[:-1]
    )
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
            cdf_t = cdf_t / cdf_t[-1]
        if np.all(np.diff(t) <= 0):  # make times go from 0 .. max_time
            t = t[::-1]
            cdf_t = cdf_t[::-1]
        assert np.all(np.diff(t) >= 0)
        # note the CDF is the wrong way around from what is conventional
        assert cdf_t[0] == 1
        assert cdf_t[-1] == 0
        prob_gt_time_deltas = np.interp(focal_times + time_deltas, t, cdf_t)
        prob_lt_time_deltas = 1 - np.interp(focal_times - time_deltas, t, cdf_t)
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
                    mut_classes[frozenset((m.derived_state, ds[m.parent]))].append(
                        (m.id, m.time)
                    )
                for mutations in mut_classes.values():
                    if len(mutations) != 1:
                        mut_ids = np.array([m[0] for m in mutations])
                        mut_times = np.array([m[1] for m in mutations])
                        nearest = np.zeros(len(mutations))
                        for i, focal_time in enumerate(mut_times):
                            nearest[i] = np.min(
                                np.abs(
                                    focal_time
                                    - np.concatenate(
                                        (mut_times[:i], mut_times[i + 1 :])
                                    )
                                )
                            )
                        # e.g. if less than 1 day apart, assume a day's worth of difference (avoids p==0.0)
                        nearest = np.where(nearest < min_time, min_time, nearest)
                        mut_p_val[mut_ids] = prob_closest_mut_within_timedelta(
                            times, lengths, mut_times, nearest, len(mutations) - 1
                        )
                    pbar.update(len(mutations))
    return mut_p_val


class D3ARG_viz:
    highlight_colour = "plum"
    colours = ['#332288', '#88CCEE', '#44AA99', '#117733', '#999933', '#DDCC77']  # from https://personal.sron.nl/~pault/
    def __init__(self, ts, df, lineage_consensus_muts=None, pangolin_field="pango", progress=True):
        self.ts = ts
        self.df = df
        self.pangolin_field = pangolin_field
        self.lineage_consensus_muts = lineage_consensus_muts
        self.d3arg = argviz.D3ARG.from_ts(ts, progress=progress)
        self.d3arg.nodes.loc[self.d3arg.nodes.id == 1, "label"] = "Wuhan"
        self.pango_lineage_samples = (
            df[df.is_sample].groupby(pangolin_field)["node_id"].apply(list).to_dict()
        )

    def set_sc2ts_node_labels(self, add_strain_names=True, progress=True):
        # Set node labels to Pango lineage + strain, if it exists
        def label_lines(row):
            if getattr(row, "sample_id", "") == "Vestigial_ignore":
                return [""]
            lab = getattr(row, self.pangolin_field)
            return [lab, f"({row.Index})"] if add_strain_names else [lab]

        node_labels = {}
        for row in tqdm(
            self.df.itertuples(), total=len(self.df), desc="Setting all labels",
            disable=not progress,
        ):
            node_labels[row.node_id] = "\n".join(
                [s for s in label_lines(row) if s not in ("()", "?")]
            )
        boldnumbers = {str(i): s for i, s in enumerate("𝟎𝟏𝟐𝟑𝟒𝟓𝟔𝟕𝟖𝟗")}
        for u in np.where(self.ts.nodes_flags & sc2ts.NODE_IS_RECOMBINANT)[0]:
            ec = self.ts.edges_child == u
            breaks = set(self.ts.edges_left[ec]) | set(self.ts.edges_right[ec])
            breaks = {int(b) for b in breaks} - {0, int(self.ts.sequence_length)}
            breaks = ["".join(boldnumbers[i] for i in str(x)) for x in sorted(breaks)]
            node_labels[u] += "\n/" + ("/".join(breaks)) + "/"
        self.d3arg.set_node_labels(node_labels)

    def set_sc2ts_node_styles(self):
        # Mark recombination nodes in white and samples as squares
        self.d3arg.nodes["size"] = 50
        self.d3arg.nodes["fill"] = "darkgrey"
        self.d3arg.nodes["stroke_width"] = 1
        is_sample = np.isin(self.d3arg.nodes["id"], self.ts.samples())
        self.d3arg.nodes.loc[is_sample, "symbol"] = "d3.symbolSquare"
        self.d3arg.nodes.loc[is_sample, "size"] = 100
        self.d3arg.nodes.loc[is_sample, "fill"] = "lightgrey"
        is_recombinant = (
            self.d3arg.nodes.ts_flags.values & sc2ts.NODE_IS_RECOMBINANT
        ).astype(bool)
        self.d3arg.nodes.loc[
            is_recombinant, "ts_flags"
        ] |= (
            msprime.NODE_IS_RE_EVENT
        )  # tskit_arg_visualizer hack to put label in centre
        self.d3arg.nodes.loc[is_recombinant, "fill"] = "white"
        self.d3arg.nodes.loc[is_recombinant, "size"] = 150

    def plot_pango_subgraph(
        self,
        pangos,
        extra_html=None,
        *,
        show_title=True,
        width=750,
        height=950,
        parent_levels=20,
        child_levels=1,
        restrict_to_first=None,  # restrict every pango to the first N samples
        exclude=None,
        include=None,
        highlight_nodes=True,
        highlight_mutations=None,
        parent_pangos=None,
        positions_file=None,
        **kwargs,
    ):
        """
        A specific routine to plot sc2ts subgraphs for a specific Pango lineage or
        list of Pango lineages (designed for recombiant pangos, e.g. "XA")

        If positions_file is None, look for a file called "'-'.join(sorted(pangos)) + '.json'", e.g. XA-XB.json
        """
        pangos = [pangos] if isinstance(pangos, str) else pangos
        exclude = set() if exclude is None else set(list(exclude))
        include = set() if include is None else set(list(include))
        if positions_file is None:
            positions_file = f"layout_data/{'-'.join(pangos)}.json"
        try:
            self.d3arg.set_node_x_positions(
                pos=argviz.extract_x_positions_from_json(
                    json.loads(Path(positions_file).read_text())
                )
            )
        except FileNotFoundError:
            pass
            # self.clear_x_01(self.d3arg)

        used_pango_samples = set()
        pango_samples = set()
        for p in pangos:
            # only the first N samples of each lineage are shown
            shown_pango = set(self.pango_lineage_samples[p][:restrict_to_first])
            # plus any samples in that lineage that are specifically included
            ps = set(self.pango_lineage_samples[p])
            shown_pango |= ps & include
            # minus any excluded
            shown_pango = shown_pango - set(exclude)
            used_pango_samples.update(shown_pango)
            pango_samples |= ps

        nodes = list((used_pango_samples | include) - exclude)
        used = self.d3arg.subset_graph(nodes, depth=(parent_levels, child_levels))
        title = (
            f'Subgraph of {self.pangolin_field.capitalize()} {"/".join(pangos)}: '
            f'({len(pango_samples)} sample{"" if len(pango_samples) == 1 else "s"},'
            f" {len(used_pango_samples)} shown)"
        ) if show_title else None
        if highlight_mutations is None:
            if parent_pangos is not None and self.lineage_consensus_muts is not None:
                # Could replace with https://github.com/andersen-lab/Freyja/blob/main/freyja/data/lineage_mutations.json
                highlight_mutations = self.lineage_consensus_muts.get_unique_mutations(p, parent_pangos)

        return self.plot_sc2ts_subgraph(
            list(nodes),
            preamble=extra_html,
            width=width,
            height=height,
            title=title,
            highlight_mutations=highlight_mutations,
            highlight_nodes=highlight_nodes,
            parent_levels=parent_levels,
            child_levels=child_levels,
            save_filename="-".join(pangos),
            **kwargs,
        )

    def plot_sc2ts_subgraph(
        self,
        nodes,
        parent_levels=20,
        child_levels=1,
        *,
        cmap=plt.cm.tab10,
        y_axis_scale="time",  # use "rank" to highlight topology more
        y_axis_labels=None,
        colour_recurrent_mutations=True,
        label_mutations=False,
        highlight_mutations=None,
        highlight_nodes=True,  # Can also be a mapping of colour to node IDs
        highlight_colour=None,
        positions_file=None,
        oldest_y_label=None,
        node_1_date=datetime(2019, 12, 26),  # date of Wuhan, node #1
        save_filename=None,
        **kwargs,
    ):
        """
        Display a subset of the sc2ts arg, with mutations that are recurrent or reverted
        in the subgraph coloured by position (reversions have a black outline).

        highlight_mutations can be a positions,derived_states named numpy recarray with
        positions and derived states (e.g. from `MutationContainer.get_mutations(Pango)` )
        which will be highlighted in the default (pink) colour, or a mapping of colours to
        a set of such recarrays.
        """
        if highlight_colour is None:
            highlight_colour = self.highlight_colour
        if positions_file is not None:
            try:
                self.d3arg.set_node_x_positions(
                    pos=argviz.extract_x_positions_from_json(json.loads(Path(positions_file).read_text()))
                )
            except FileNotFoundError:
                pass
                # self.clear_x_01(self.d3arg)

        select_nodes = np.isin(self.d3arg.nodes.id, nodes)
        # Find mutations with duplicate position values and the same alleles (could be parallel eg. A->T & A->T, or reversions, e.g. A->T, T->A)
        # Create a composite key for basic duplicates
        if colour_recurrent_mutations:
            df = self.d3arg.subset_graph(
                nodes, (parent_levels, child_levels)
            ).mutations.copy()
            df["fill"] = "lightgrey"
            df["stroke"] = "grey"  # default stroke color
            df["duplicate_key"] = df.apply(
                lambda row: f"{row['position']}_{sorted([row['inherited'], row['derived']])}",
                axis=1,
            )
            # Create a polarization key to identify the polarization of the allele arrangements
            df["polarization_key"] = df.apply(
                lambda row: f"{row['position']}_{row['inherited']}_{row['derived']}",
                axis=1,
            )

            # Identify which rows are duplicates
            duplicate_mask = df.duplicated(subset=["duplicate_key"], keep=False)

            # Get only the duplicate keys that appear multiple times
            duplicate_keys = df[duplicate_mask]["duplicate_key"].unique()
            colors = {key: rgb2hex(cmap(i)) for i, key in enumerate(duplicate_keys)}

            # Process only the duplicate groups
            for duplicate_key in duplicate_keys:
                group = df[df["duplicate_key"] == duplicate_key]
                polarization_keys = group["polarization_key"].unique()

                # Assign fill color based on duplicate_key
                fill = colors[duplicate_key]
                # If there are multiple direction keys, assign different strokes
                has_multiple_directions = len(polarization_keys) > 1
                for idx in group.index:
                    df.loc[idx, "fill"] = fill
                    df.loc[idx, "stroke"] = (
                        "grey"
                        if not has_multiple_directions
                        else (
                            "grey"
                            if group.loc[idx, "polarization_key"]
                            == polarization_keys[0]
                            else "black"
                        )
                    )

            self.d3arg.mutations.loc[df.index, "fill"] = df["fill"]
            self.d3arg.mutations.loc[df.index, "stroke"] = df["stroke"]

        # For deletions, swap the stroke for the fill colour, and fill in black
        is_deletion = self.d3arg.mutations.derived == "-"
        stroke = self.d3arg.mutations.loc[is_deletion, "fill"]
        stroke = np.where(stroke == "white", "lightgrey", stroke)
        self.d3arg.mutations.loc[is_deletion, "stroke"] = stroke
        self.d3arg.mutations.loc[is_deletion, "fill"] = "black"

        # For going from a deletion back to the original state (insertion: very unexpected),
        # swap the stroke for the fill colour, and fill in magenta to highlight
        is_insertion = self.d3arg.mutations.inherited == "-"
        self.d3arg.mutations.loc[is_insertion, "stroke"] = self.d3arg.mutations.loc[
            is_insertion, "fill"
        ]
        self.d3arg.mutations.loc[is_insertion, "fill"] = "magenta"

        if highlight_mutations is not None:
            if hasattr(highlight_mutations, "positions"):
                highlight_mutations = {highlight_colour: highlight_mutations}
            for colour, data in highlight_mutations.items():
                if not hasattr(data, "positions"):
                    raise NotImplementedError("Cannot list mutations by ID yet")
                use = np.char.add(
                    np.array(data.positions, dtype=str), data.derived_states,
                )
                match = np.char.add(
                    self.d3arg.mutations.position.astype(int).astype(str),
                    self.d3arg.mutations.derived,
                )
                self.d3arg.mutations.loc[np.isin(match, use), "stroke"] = (
                    colour
                )
                

        if highlight_nodes:
            self.d3arg.nodes.loc[select_nodes, "fill"] = highlight_colour
            if isinstance(highlight_nodes, dict):
                for colour, nodelist in highlight_nodes.items():
                    self.d3arg.nodes.loc[
                        np.isin(self.d3arg.nodes.id, nodelist), "fill"
                    ] = colour

        if y_axis_labels is None:
            times = self.d3arg.nodes.loc[select_nodes, "time"]
            zero_date = node_1_date + timedelta(
                days=int(
                    self.d3arg.nodes.loc[self.d3arg.nodes.id == 1, "time"].values[0]
                )
            )
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
                for d in list_of_months(
                    oldest_y_label, zero_date - timedelta(days=times.min())
                )
            }
        try:
            return self.d3arg.draw_nodes(
                nodes,
                depth=(parent_levels, child_levels),
                show_mutations=True,
                y_axis_scale=y_axis_scale,
                y_axis_labels=y_axis_labels,
                y_axis_title="Estimated date of occurrence (YYYY-MM)",
                label_mutations=label_mutations,
                save_filename=save_filename,
                **kwargs,
            )
        except TypeError:
            return self.d3arg.draw_nodes(
                nodes,
                depth=(parent_levels, child_levels),
                show_mutations=True,
                y_axis_scale=y_axis_scale,
                y_axis_labels=y_axis_labels,
                label_mutations=label_mutations,
                **kwargs,
            )


def make_joint_ts(ts1, ts2, metadata_name1, metadata_name2):
    """
    Combine two tree sequences of one tree each into a single tree sequence of
    2 trees on the basis of the metadata names in each ts.
    """
    if ts1.num_trees > 1 or ts2.num_trees > 1:
        raise ValueError(
            f"Found {ts1.num_trees} trees in the first ts and {ts2.num_trees} trees in the second ts. Expected exactly one tree in each."
        )
    ts1_node_map = {
        nd.metadata[metadata_name1]: nd.id for nd in ts1.nodes() if nd.is_sample()
    }
    ts2_to_ts1_node_map = [
        ts1_node_map.get(nd.metadata.get(metadata_name2), tskit.NULL)
        for nd in ts2.nodes()
    ]
    tables1 = ts1.dump_tables()
    tables1.reference_sequence.clear()
    ts1 = tables1.tree_sequence()
    tables2 = ts2.dump_tables()
    tables2.reference_sequence.clear()
    ts2 = tables2.tree_sequence()

    ts = ts1.concatenate(ts2, node_mappings=[ts2_to_ts1_node_map])
    return ts


def tanglegram(
    ts,
    tree_indexes=None,
    titles=None,
    *,
    size=None,
    order=None,
    separation=None,
    line_gap=None,
    node_labels=None,
    style=None,
    x_axis=None,
    x_label=None,
    x_ticks=None,
    x_gridlines=None,
    y_axis=None,
    y_label=None,
    y_ticks=None,
    y_gridlines=None,
    tweak_rh_lab=0,
    **kwargs,
):
    r"""
    Create an SvgTree object describing a "tanglegram" that compares the topology leading
    to leaf nodes on two trees in a tree sequence, by drawing them facing each other, and
    plotting lines between the same leaves in each tree. By default, compare the first
    and last non-empty tree, and plot titles based on the tree indexes. This does not
    "untangle" the trees to minimise the number of line crossings, but simply
    plots them using the default "minlex" order. If you wish to do untangle them,
    you will need to use an external program to calculate the leaf orders, and
    pass them in through the `order` parameter. The separate plots can be styled
    as the left and right SVGs are given `lft_tree` and ``rgt_tree` classes.

    .. note::
        If you have two trees from separate tree sequences, you can concatenate them
        together using :meth:`tskit.TreeSequence.concatenate()`, providing a
        node_mapping to match leaf or sample nodes between the two trees, and
        then pass the resulting tree sequence to this function.

    :param TreeSequence ts: The tree sequence from which to take trees
    :param tuple tree_indexes: A tuple of two integers, the indexes of the two trees to
        compare. By default take the first and last non-empty trees in the tree sequence.
    :param tuple titles: A tuple of two strings, the titles to use for the left and right
        trees. By default show "Tree X". To show no titles, provide titles=(None, None).
    :param tuple size: A tuple of two integers, the width and height of the SVG image.
        By default, use the default (square) size per tree, giving (400, 200) in total.
    :param tuple order: A tuple of two lists of integers, the order in which to plot the
        leaves. Either list can be `None`, meaning default to ``minlex_postorder``.
        If either is a list, it must contain unique integers corresponding to the IDs
        of the leaves in the corresponding tree, with the length of each list matching
        the number of leaves in the tree. Default: ``None`` treated as ``(None, None)``.
    :param float separation: The distance between the base of each tree. Default:
        ``None`` treated as a standard distance of 64px.
    :param float line_gap: The distance between tangle_lines and each leaf on the tree.
        If None, draw tangle lines of equal length in the middle of the plot
    :param str path: A path to which the SVG will be saved.
    :param dict node_labels: A dictionary mapping node IDs to label to plot. See
        :meth:`Tree.draw_svg()` for details.
    :param str style: a string of CSS styles. See :meth:`Tree.draw_svg()` for details.
    :param bool x_axis: Should we plot two horizontal axes showing time underneath
        the trees. Note that this corresponds to the y-axis in a conventional
        SVG plot, as tanglegrams are rotated 90°.
    :param str x_label: X axis label (equivalent of ``y_label`` in ``draw_svg()``).
    :param tuple[Union[list, dict]] x_ticks: Location of the tick marks on the two
        time axes. This is a tuple of two values, each of which is
        equivalent the the ``y_ticks`` value in ``draw_svg()``.
    :param bool x_gridlines: Whether to plot vertical lines behind the tree
        at each y tickmark (equivalent of ``y_gridlines`` in ``draw_svg()``).
    :param bool y_axis: Equivalent of `x_axis` parameter in ``draw_svg()``.
        Probably not what you want.
    :param bool y_label: Equivalent of `y_label` parameter in ``draw_svg()``
        Probably not what you want.
    :param bool y_ticks: Dummy option: has no effect
    :param bool y_gridlines: Dummy option: has no effect
    :param \**kwargs: Additional keyword arguments to pass to :meth:`Tree.draw_svg()`,
        such as `time_scale` ("log_time", "rank", etc)

    :returns:
        A tuple of an SvgTree object (that can be plotted by calling obj.draw()) and a left
        and a right node mapping.
    """

    def node_positions(svgtree):
        x = svgtree.node_x_coord
        node_times = svgtree.ts.nodes_time
        if svgtree.time_scale == "rank":
            within_tree_node_times = np.unique(node_times[svgtree.tree.preorder()])
            node_times = np.searchsorted(within_tree_node_times, node_times)
        y = svgtree.timescaling.transform(node_times)
        return x, y

    def make_reverse_map(node_map):
        reverse_map = np.zeros_like(node_map)
        kept = node_map != tskit.NULL
        reverse_map[node_map[kept]] = np.arange(len(node_map))[kept]
        return reverse_map

    def reorder_tree_nodes(tree, node_order):
        # given a node order and a tree, make a new tree and return that and the order
        node_map = np.arange(tree.tree_sequence.num_nodes)
        node_map[np.sort(node_order)] = node_order
        ts = tree.tree_sequence.subset(node_map, False, False, False)
        return node_map, ts.at_index(tree.index)

    def get_valid_leaf_order(tree, node_order):
        # take a node ordering and return a leaf ordering on the reordered-node tree
        if len(np.unique(node_order)) != len(node_order):
            raise ValueError("Order must contain unique integers")
        node_map, tree = reorder_tree_nodes(tree, node_order)
        leaves = np.array(
            [u for u in tree.nodes(order="minlex_postorder") if tree.is_leaf(u)]
        )
        return node_map[leaves]

    if y_ticks is not None:
        raise ValueError("Invalid option")
    if y_gridlines is not None:
        raise ValueError("Invalid option")
    if order is None:
        order = (None, None)
    if tree_indexes is None:
        tree_indexes = (
            1 if ts.first().num_edges == 0 else 0,
            -2 if ts.last().num_edges == 0 else -1,
        )
    lft = ts.at_index(tree_indexes[0])
    rgt = ts.at_index(tree_indexes[1])
    if titles is None:
        titles = (f"Tree {lft.index}", f"Tree {rgt.index}")

    if separation is None:
        extra_sep = 0
    else:
        extra_sep = separation - 64
    if size is None:
        # Note that the width is twice the default tree height, as these are rotated
        size = (200 * 2, 200)
    w = (
        size[0] / 2
    )  # width (tree height) of one of the plotted trees, after 90° rotation
    height = size[1]
    style = (
        ".lft_tree > g.tree, .lft_tree > g.tangle_lines {transform: translate(0, "
        + str(height)
        + "px) rotate(-90deg);}"
        + ".lft_tree > g.tree .node > .lab {text-anchor: start; transform: rotate(90deg) translate(4px);}"
        ".lft_tree > .title {transform: translate(" + str(w / 2) + "px);}"
        ".rgt_tree > g.tree {transform: translate(" + str(w) + "px, 0) rotate(90deg);}"
        ".rgt_tree > g.tree .node > .lab {text-anchor: end; transform: rotate(-90deg) translate("
        + str(-4 - tweak_rh_lab)
        + "px);}"
        ".rgt_tree > .title {transform: translate(" + str(w / 2) + "px);}"
        ".lft_tree .axes .y-axis .title text {transform: translate(11px) rotate(90deg);}"
        ".rgt_tree .axes .y-axis .title text {transform: translate(-11px) rotate(-90deg);}"
        ".rgt_tree .axes .y-axis .ticks .lab {text-anchor: end; transform: rotate(180deg);}"
    ) + (style or "")

    # For tree 1 we need to reverse the plotting order of leaves, so the leftmost
    # tip appears at the top when the tree is rotated 90° anticlockwise. We do this
    # by reordering using `subset()`, so minlex order will reverse the current order
    # This also means we need to re-adjust the node labels, as the sample IDs will change
    if node_labels is None:
        node_labels = {u: str(u) for u in np.arange(ts.num_nodes)}

    if order[0] is None:
        leaves = np.array(
            [u for u in lft.nodes(order="minlex_postorder") if lft.is_leaf(u)]
        )[::-1]
    else:
        leaves = get_valid_leaf_order(lft, order[0][::-1])

    lft_node_map, lft = reorder_tree_nodes(lft, leaves)
    lft_rev_map = make_reverse_map(lft_node_map)
    # Have to change the node labels, because even provided ones will be targetting the wrong IDs
    lft_node_labels = {
        u: node_labels[v] for u, v in enumerate(lft_node_map) if v in node_labels
    }
    if order[1] is None:
        # We do not reorder the RH tree, so the node IDs should stay as-is
        # TODO - we could check the leaf IDs match here
        rgt_node_labels = node_labels
        rgt_node_map = rgt_rev_map = np.arange(ts.num_nodes)
    else:
        rleaves = get_valid_leaf_order(rgt, order[1])
        rgt_node_map, rgt = reorder_tree_nodes(rgt, rleaves)
        if set(rleaves) != set(leaves):
            raise ValueError("Leaf IDs in the two trees are not the same")
        rgt_node_labels = {
            u: node_labels[v] for u, v in enumerate(rgt_node_map) if v in node_labels
        }
        rgt_rev_map = make_reverse_map(rgt_node_map)
    kwargs["size"] = (height, w)
    kwargs["order"] = "minlex_postorder"
    kwargs["y_label"] = x_label  # Swapped because of 90° rotation
    kwargs["y_gridlines"] = x_gridlines  # Swapped because of 90° rotation
    kwargs["x_axis"] = y_axis  # Swapped because of 90° rotation
    kwargs["x_label"] = y_label  # Swapped because of 90° rotation
    # We'll embed the right tree and the tangle lines within the left tree later, via the preamble
    svgtree_lft = tskit.drawing.SvgTree(
        lft,
        title=titles[0],
        canvas_size=(w * 2 + extra_sep, height),
        node_labels=lft_node_labels,
        root_svg_attributes={"class": "lft_tree"},
        y_axis=x_axis,
        y_ticks=None if x_ticks is None else x_ticks[0],
        **kwargs,
    )
    svgtree_rgt = tskit.drawing.SvgTree(
        rgt,
        title=titles[1],
        canvas_size=(w, height),
        node_labels=rgt_node_labels,
        root_svg_attributes={"class": "rgt_tree", "x": w + extra_sep},
        y_axis="right" if x_axis else x_axis,  # Swapped because of 90° rotation
        y_ticks=None if x_ticks is None else x_ticks[1],
        style=style,
        **kwargs,
    )
    x_lft, y_lft = node_positions(svgtree_lft)

    # Here we just need any list of leaves (order doesn't matter, as long as it's consistent)
    tip_h_lft = [x_lft[u] for u in lft_rev_map[leaves]]
    tip_w_lft = y_lft[lft_rev_map[leaves]]
    tip_w_lft = (
        np.full_like(tip_w_lft, w - 10) if line_gap is None else (tip_w_lft + line_gap)
    )

    x_rgt, y_rgt = node_positions(svgtree_rgt)
    tip_h_rgt = [x_rgt[u] for u in rgt_rev_map[leaves]]
    tip_w_rgt = y_rgt[rgt_rev_map[leaves]]
    tip_w_rgt = (
        np.full_like(tip_w_rgt, w - 10) if line_gap is None else (tip_w_rgt + line_gap)
    )

    lines = [
        f'<line stroke="blue" x1="{x1}" y1="{y1}" x2="{height-x2}" y2="{w*2 + extra_sep - y2}" />'
        for x1, y1, x2, y2 in zip(tip_h_lft, tip_w_lft, tip_h_rgt, tip_w_rgt)
    ]

    svgtree_lft.preamble = (  # Add the RH tree plus lines as the preamble
        '<g class="tangle_lines">' + "".join(lines) + "</g>" + svgtree_rgt.draw()
    )
    return svgtree_lft, lft_rev_map, rgt_rev_map


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
            names="positions,derived_states",
        )

    def get_unique_mutations(self, pango, parent_pangos):
        # return the mutations in item that are not in any of the parents
        if isinstance(parent_pangos, str):
            parent_pangos = [parent_pangos]
        idx = self.names[pango]
        use = np.rec.fromarrays(
            [self.positions[idx], self.alts[idx]], names="positions,derived_states"
        )
        indexes = [self.names[p] for p in parent_pangos]
        omit = np.concatenate(
            [
                np.rec.fromarrays(
                    [self.positions[i], self.alts[i]], names="positions,derived_states"
                )
                for i in indexes
            ]
        )
        return np.setdiff1d(use, omit)


## from run_lineage_imputation.py
def read_in_mutations(
    json_filepath,
    verbose=False,
    exclude_positions=None,
):
    """
    Read in lineage-consensus mutations from COVIDCG input json file.
    Assumes root lineage is B.
    """
    if exclude_positions is None:
        exclude_positions = set()
    else:
        exclude_positions = set(exclude_positions)
    with fileinput.hook_compressed(json_filepath, "r") as file:
        linmuts = json.load(file)

    # Read in lineage consensus mutations
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
                f"{list(excluded_pos.keys())}",
            )
        if len(excluded_del) > 0:
            print("Excluded deletions at positions", list(excluded_del.keys()))

    return linmuts_dict
