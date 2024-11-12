import argparse
import collections
from datetime import datetime, timedelta
import hashlib
import itertools
import re
import json
import os
import tempfile
import subprocess
import logging
import gzip
import shutil

import matplotlib as mpl

mpl.use("Agg")  # NOQA
from matplotlib.colors import rgb2hex
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib import collections as mc
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
import tqdm
import pandas as pd

import tsconvert  # Not on pip. Install with python -m pip install git+http://github.com/tskit-dev/tsconvert
import tszip
#import sc2ts
#import sc2ts.utils

#import utils

# Redefine the path to your local dendroscope Java app & chromium app here
chromium_binary = "/usr/local/bin/chromium"

genes = {
    "ORF1ab": (266, 21555),
    "S": (21563, 25384),
    "ORF3a": (25393, 26220),
    "E": (26245, 26472),
    "M": (26523, 27191),
    "ORF6": (27202, 27387),
    "ORF7a": (27394, 27759),
    "ORF7b": (27756, 27887),
    "ORF8": (27894, 28259),
    "N": (28274, 29533),
    "ORF10": (29558, 29674),
}


class SingleTreeTs:
    """Convenience class to access a single focal tree in a tree sequence"""

    def __init__(self, ts, basetime=None):
        # .tree is the first non-empty tree
        self.tree = next(t for t in ts.trees() if t.num_edges > 0)
        self.basetime = basetime

    @property
    def ts(self):
        return self.tree.tree_sequence

    @property
    def samples(self):
        return self.tree.tree_sequence.samples()

    def timediff(self, isodate):
        return getattr(
            self.basetime - datetime.fromisoformat(isodate), self.ts.time_units
        )

    def strain(self, u):
        return self.tree.tree_sequence.node(u).metadata.get("strain", "")

    def hash_samples_under_node(self, u):
        b2b = hashlib.blake2b(
            " ".join(sorted([self.strain(s) for s in self.tree.samples(u)])).encode(),
            digest_size=20,
        )
        return b2b.digest()


class Nextstrain:
    def __init__(self, filename, span, prefix="data"):
        """
        Load from a nextstrain nexus file.
        Note that NextClade also produces a tree with more samples but no branch lengths
        e.g. at
            https://github.com/nextstrain/nextclade_data/tree/
            release/data/datasets/sars-cov-2/references/MN908947/versions
        It is possible to load this using
            nextclade_json_ts = sc2ts.load_nextclade_json("../results/tree.json")
        """
        ts = sc2ts.newick_from_nextstrain_with_comments(
            sc2ts.extract_newick_from_nextstrain_nexus(os.path.join(prefix, filename)),
            min_edge_length=0.0001 * 1 / 365,
            span=span,
        )
        # Remove "samples" without names
        keep = [n.id for n in ts.nodes() if n.is_sample() and "strain" in n.metadata]
        self.ts = ts.simplify(keep)

    @staticmethod
    def pango_names(ts):
        # This is relevant to any nextstrain tree seq, not just the stored one
        return {
            n.metadata.get("comment", {}).get("pango_lineage", "") for n in ts.nodes()
        }


def variant_name(pango):
    """
    Classification from the following site
    https://www.cdc.gov/coronavirus/2019-ncov/variants/variant-classifications.html
    Alpha (B.1.1.7 and Q lineages)
    Beta (B.1.351 and descendent lineages)
    Gamma (P.1 and descendent lineages)
    Delta (B.1.617.2 and AY lineages)
    Epsilon (B.1.427 and B.1.429)
    Eta (B.1.525)
    Iota (B.1.526)
    Kappa (B.1.617.1)
    Mu (B.1.621, B.1.621.1)
    Zeta (P.2)
    Omicron (B.1.1.529, BA.1, BA.1.1, BA.2, BA.3, BA.4 and BA.5 lineages)
    """
    if pango == "B.1.1.7" or pango.startswith("Q"):
        return "Alpha"
    if pango.startswith("B.1.351"):
        return "Beta"
    if pango == "P.1" or pango.startswith("P.1."):
        return "Gamma"
    if pango.startswith("AY") or pango == "B.1.617.2":
        return "Delta"
    if pango == "B.1.427" or pango == "B.1.429":
        return "Epsilon"
    if pango == "B.1.526":
        return "Iota"
    if pango == "B.1.617.1":
        return "Kappa"
    if pango == "B.1.621" or pango == "B.1.621.1":
        return "Mu"
    if pango == "P.2" or pango.startswith("P.2."):
        return "Zeta"
    if pango == "B.1.1.529" or pango.startswith("BA."):
        return "Omicron"
    return "__other__"


class Figure:
    """
    Superclass for creating figures. Each figure is a subclass
    """

    name = None
    ts_dir = "data"
    wide_fn = "upgma-full-md-30-mm-3-2021-06-30-recinfo2-gisaid-il.ts.tsz"
    long_fn = "upgma-mds-1000-md-30-mm-3-2022-06-30-recinfo2-gisaid-il.ts.tsz"

    def plot(self, args):
        raise NotImplementedError()


def isomonths_iter(start):
    """
    start as 2020-01, 2021-07
    yield each month as a YYYY-MM string
    """
    year, month = [int(x) for x in start.split("-")]
    while True:
        yield f"{year}-{month:02}"
        month += 1
        if month > 12:
            month = 1
            year += 1

class Cophylogeny(Figure):
    """
    Plot a cophylogeny of a sc2ts tree and a nextstrain tree
    requires .tsz files produced by running s2.2_cophylo_recombinants.ipynb
    """
    name = None
    file_prefix = None

    # Utility functions
    @staticmethod
    def strain_order(focal_tree_ts):
        """
        Map strain name to order of leaf node in a tree
        """
        tree = focal_tree_ts.tree
        # Can't use the tree.leaves iterator as we need to specify order
        leaves = [u for u in tree.nodes(order="minlex_postorder") if tree.is_leaf(u)]
        return {focal_tree_ts.strain(v): i for i, v in enumerate(leaves)}

    def __init__(self):
        """
        Defines two simplified tree sequences, focussed on a specific tree. These are
        stored in self.sc2ts and self.nxstr
        """
        plotted_sc2ts_ts, basetime = utils.load_tsz(self.ts_dir, self.file_prefix + "_untangled.tsz")
        plotted_nxstr_ts = tszip.decompress(os.path.join(self.ts_dir, self.file_prefix + "_nxstr_untangled.tsz"))

        # Align the time in the nextstrain tree to the sc2ts tree
        ns_sc2_time_difference = []
        sc2ts_map = {plotted_sc2ts_ts.node(u).metadata["strain"]: u for u in plotted_sc2ts_ts.samples()}
        for u in plotted_nxstr_ts.samples():
            nxstr_nd = plotted_nxstr_ts.node(u)
            sc2ts_id = sc2ts_map.get(nxstr_nd.metadata["strain"])
            if sc2ts_id is not None:
                ns_sc2_time_difference.append(plotted_sc2ts_ts.node(sc2ts_id).time - nxstr_nd.time)
        assert len(ns_sc2_time_difference) > 1
        dt = timedelta(**{plotted_nxstr_ts.time_units: np.median(ns_sc2_time_difference)})

        self.sc2ts = SingleTreeTs(plotted_sc2ts_ts, basetime)
        self.nxstr = SingleTreeTs(plotted_nxstr_ts, basetime - dt)

        logging.info(
            f"{self.sc2ts.ts.num_trees} trees in the simplified 'backbone' ARG. Using the one "
            + f"between pos {self.sc2ts.tree.interval.left} and {self.sc2ts.tree.interval.right}."
        )

    def plot(self, args):
        prefix = os.path.join("figures", self.name)
        strain_id_map = {
            self.sc2ts.strain(n): n
            for n in self.sc2ts.samples
            if self.sc2ts.strain(n) != ""
        }

        # A few color schemes to try
        cmap = plt.get_cmap("tab20b", 50)
        pango = Nextstrain.pango_names(self.nxstr.ts)
        colours = {
            # Name in ns comment metadata, colour scheme
            "Pango": {"md_key": "pango_lineage", "scheme": sc2ts.pango_colours},
            "Nextclade": {
                "md_key": "clade_membership",
                "scheme": sc2ts.ns_clade_colours,
            },
            "PangoMpl": {
                "md_key": "pango_lineage",
                "scheme": {k: rgb2hex(cmap(i)) for i, k in enumerate(pango)},
            },
            "PangoB.1.1": {
                "md_key": "pango_lineage",
                "scheme": {
                    k: ("#FF0000" if k == ("B.1.1") else "#000000")
                    for i, k in enumerate(pango)
                },
            },
        }

        # NB - lots of the graphics parameters below such as pixel translations etc are
        # hard-coded to work for this size of plot (800 x 400) and number of tips.
        # This is a hack: ideally we should work out the formulae required.

        global_styles = [".lab {font-size: 9px}"]
        left_tree_styles = [
            ".tree .node > .lab {text-anchor: end; transform: rotate(90deg) translate(-10px, 10px); font-size: 12px}",
            ".tree .leaf > .lab {text-anchor: start; transform: rotate(90deg) translate(6px)}",
            ".y-axis {transform: translateX(-10px)}",
            ".y-axis .title text {transform: translateX(30px) rotate(90deg)}",
            ".y-axis .lab {transform: translateX(-4px) rotate(90deg); text-anchor: middle}",
        ]
        right_tree_styles = [
            ".tree .node > .lab {text-anchor: start; transform: rotate(-90deg) translate(10px, 10px); font-size: 12px}",
            ".tree .leaf > .lab {text-anchor: end; transform: rotate(-90deg) translate(-6px)}",
            ".y-axis {transform: translateX(734px)}",
            ".y-axis .title text {transform: translateY(40px) translateX(85px) rotate(-90deg)}",
            ".y-axis .ticks {transform: translateX(5px)}",
            ".y-axis .lab {transform: translateX(11px) rotate(-90deg); text-anchor: middle}",
        ]
        global_styles.extend([".left_tree " + style for style in left_tree_styles])
        global_styles.extend([".right_tree " + style for style in right_tree_styles])

        # Assign colours
        col = colours[self.use_colour]
        nxstr_styles = []
        sc2ts_styles = []
        legend = {}
        for n in self.nxstr.ts.nodes():
            clade = n.metadata.get("comment", {}).get(col["md_key"], None)
            if clade is not None:
                if clade in col["scheme"]:
                    legend[clade] = col["scheme"][clade]
                    nxstr_styles.append(
                        f".nxstr .n{n.id} .edge {{stroke: {col['scheme'][clade]}}}"
                    )
                    s = self.nxstr.strain(n.id)
                    if s in strain_id_map:
                        sc2ts_styles.append(
                            f".sc2ts .n{strain_id_map[s]} .edge {{stroke: {col['scheme'][clade]}}}"
                        )

        # Find shared splits to plot as solid circular nodes
        # uses a hash to summarise the samples under a node, otherwise the sets get big
        nxstr_hashes = {
            self.nxstr.hash_samples_under_node(u): u
            for u in self.nxstr.tree.nodes()
            if not self.nxstr.tree.is_sample(u)
        }
        sc2ts_hashes = {
            self.sc2ts.hash_samples_under_node(u): u
            for u in self.sc2ts.tree.nodes()
            if not self.sc2ts.tree.is_sample(u)
        }

        shared_split_keys = set(nxstr_hashes.keys()).intersection(
            set(sc2ts_hashes.keys())
        )
        for key in shared_split_keys:
            nxstr_styles.append(f".nxstr .n{nxstr_hashes[key]} > .sym {{r: 3px}}")
            sc2ts_styles.append(f".sc2ts .n{sc2ts_hashes[key]} > .sym {{r: 3px}}")

        focal_nodes = {"Delta": {}, "Alpha": {}}
        for nm, tree in zip(("sc2ts", "nxstr"), (self.sc2ts.tree, self.nxstr.tree)):
            delta = []
            alpha = []
            for node in tree.tree_sequence.nodes():
                if node.is_sample():
                    if nm == "nxstr":
                        pango = node.metadata.get("comment", {}).get(
                            "pango_lineage", ""
                        )
                    else:
                        pango = node.metadata.get("Nextclade_pango", "")
                    if pango.startswith("AY") or pango == "B.1.617.2":
                        delta.append(node.id)
                    if pango == "B.1.1.7":
                        alpha.append(node.id)
            focal_nodes["Delta"][nm] = tree.mrca(*delta)
            focal_nodes["Alpha"][nm] = tree.mrca(*alpha)

        node_labels = {}
        for nm, focal_ts in [("sc2ts", self.sc2ts), ("nxstr", self.nxstr)]:
            node_labels[nm] = {u: focal_ts.strain(u) for u in focal_ts.tree.nodes()}
            node_labels[nm].update({focal_nodes[k][nm]: k for k in focal_nodes})

        svg1 = self.sc2ts.tree.draw_svg(
            size=(800, 400),
            canvas_size=(800, 800),
            node_labels=node_labels["sc2ts"],
            root_svg_attributes={"class": "sc2ts"},
            mutation_labels={},
            omit_sites=True,
            symbol_size=1,
            y_axis=True,
            y_ticks={
                self.sc2ts.timediff(date+"-01"): (date if show else "")
                for date, show in self.x_axis_ticks.items()
            },
            y_label=" ",
        )

        svg2 = self.nxstr.tree.draw_svg(
            size=(800, 400),
            canvas_size=(900, 800),  # Allow for time axis at the other side of the tree
            node_labels=node_labels["nxstr"],
            root_svg_attributes={"class": "nxstr"},
            mutation_labels={},
            omit_sites=True,
            symbol_size=1,
            y_axis=True,
            y_ticks={
                self.nxstr.timediff(date+"-01"): (date if show else "")
                for date, show in self.x_axis_ticks.items()
            },
            y_label=" ",
        )

        names_lft = self.strain_order(self.sc2ts)
        names_rgt = self.strain_order(self.nxstr)
        min_lft_time = self.sc2ts.ts.nodes_time[self.sc2ts.samples].min()
        min_rgt_time = self.nxstr.ts.nodes_time[self.nxstr.samples].min()

        loc = {}
        for nm in names_lft.keys():
            lft_node = names_lft[nm]
            lft_rel_time = (self.sc2ts.tree.time(lft_node) - min_lft_time) / (
                self.sc2ts.tree.time(self.sc2ts.tree.root) - min_lft_time
            )
            rgt_node = names_rgt[nm]
            rgt_rel_time = (self.nxstr.tree.time(rgt_node) - min_rgt_time) / (
                self.nxstr.tree.time(self.nxstr.tree.root) - min_rgt_time
            )
            loc[nm] = {
                "lft": (
                    370 - lft_rel_time * 340,
                    763 - lft_node * ((800 - 77) / self.sc2ts.ts.num_samples) - 22,
                ),
                "rgt": (
                    430 + rgt_rel_time * 340,
                    rgt_node * ((800 - 77) / self.nxstr.ts.num_samples) + 22,
                ),
            }

        global_styles += [
            # hide node labels by default
            "#main .node > .sym ~ .lab {display: none}"
            # Unless the adjacent node or the label is hovered over
            "#main .node > .sym:hover ~ .lab {display: inherit}"
            "#main .node > .sym ~ .lab:hover {display: inherit}"
        ]

        global_styles += [
            # hide mutation labels by default
            "#main .mut .sym ~ .lab {display: none}"
            # Unless the adjacent node or the label is hovered over
            "#main .mut .sym:hover ~ .lab {display: inherit}"
            "#main .mut .sym ~ .lab:hover {display: inherit}"
        ]

        global_styles += [
            # These are optional, but setting the label text to bold with grey stroke and
            # black fill serves to make black text readable against a black tree
            "svg#main {background-color: white}",
            "#main .tree .plotbox .lab {stroke: #CCC; fill: black; font-weight: bold}",
            "#main .tree .mut .lab {stroke: #FCC; fill: red; font-weight: bold}",
        ]

        # override the labels for Delta and Alpha
        global_styles += [
            f"#main .{nm} .n{u} > .sym ~ .lab {{stroke: none; fill: black; font-weight: normal; display: inherit}}"
            for v in focal_nodes.values()
            for nm, u in v.items()
        ]

        global_styles += nxstr_styles
        global_styles += sc2ts_styles
        sc2ts_str = f"Sc2ts {self.name[-4:].capitalize()} ARG: genomic positions "
        sc2ts_str += f"{self.sc2ts.tree.interval.left:.0f} to {self.sc2ts.tree.interval.right:.0f}"
        w, h = 900, 800
        mar_in = 0.05
        w_in = w / 96 + mar_in * 2
        h_in = h / 96 + mar_in * 2
        svg_string = (
            f'<svg baseProfile="full" width="{w}" height="{h}" version="1.1" id="main"'
            + ' xmlns="http://www.w3.org/2000/svg" '
            + 'xmlns:ev="http://www.w3.org/2001/xml-events" xmlns:xlink="http://www.w3.org/1999/xlink">'
            + f"<defs><style>@page {{margin: {mar_in}in; padding: 0; size: {w_in:.2f}in {h_in:.2f}in;}} "
            + f'{"".join(global_styles)}</style></defs>'
            + f'<text text-anchor="middle" transform="translate(200, 12)">{sc2ts_str}</text>'
            + '<text text-anchor="middle" transform="translate(600, 12)">Nextstrain tree</text>'
            + "<g>"
            + "".join(
                [
                    f'<line x1="{v["lft"][0]}" y1="{v["lft"][1]}" x2="{v["rgt"][0]}" y2="{v["rgt"][1]}" stroke="#CCCCCC" />'
                    for v in loc.values()
                ]
            )
            + "</g>"
            + '<g class="left_tree" transform="translate(0 800) rotate(-90)">'
            + svg1
            + '</g><g class="right_tree" transform="translate(800 -37) rotate(90)">'
            + svg2
            + "</g>"
            + '<g class="legend" transform="translate(800 30)">'
            + f"<text>{self.use_colour} lineage</text>"
            + "".join(
                f'<line x1="0" y1="{25+i*15}" x2="15" y2="{25+i*15}" stroke-width="2" stroke="{legend[nm]}" /><text font-size="10pt" x="20" y="{30+i*15}">{nm}</text>'
                for i, nm in enumerate(sorted(legend))
            )
            + "</g>"
            + "</svg>"
        )

        with open(f"{prefix}.svg", "wt") as file:
            file.write(svg_string)
        if args.outtype == "pdf":
            subprocess.run(
                [
                    chromium_binary,
                    "--headless",
                    "--disable-gpu",
                    "--run-all-compositor-stages-before-draw",
                    "--print-to-pdf-no-header",
                    f"--print-to-pdf={prefix}.pdf",
                    f"{prefix}.svg",
                ]
            )


class CophylogenyWide(Cophylogeny):
    name = "cophylogeny_wide"
    file_prefix = "upgma-full-md-30-mm-3-2021-06-30-recinfo2-gisaid-il"
    use_colour = "Pango"
    x_axis_ticks = {
        # show ticks for 19 months (up to 2021-07), with labels every 3 months
        date: i % 3 == 0 for date, i in zip(isomonths_iter("2020-01"), range(19))
    }


class CophylogenyLong(Cophylogeny):
    name = "supp_cophylogeny_long"
    file_prefix = "upgma-mds-1000-md-30-mm-3-2022-06-30-recinfo2-gisaid-il"
    use_colour = "Pango"
    x_axis_ticks = {
        # show ticks for 31 months (up to 2022-07), with labels every 6 months
        date: i % 4 == 0 for date, i in zip(isomonths_iter("2020-01"), range(29))
    }


class RecombinationNodeMrcas(Figure):
    name = None
    sc2ts_filename = "upgma-mds-1000-md-30-mm-3-2022-06-30-recinfo2-gisaid-il.ts.tsz"
    csv_fn = "long_arg_recombinants.csv"
    data_dir = "data"

    def __init__(self):
        self.ts, self.basetime = utils.load_tsz(self.data_dir, self.sc2ts_filename)

        prefix = utils.snip_tsz_suffix(self.sc2ts_filename)
        df = pd.read_csv(
            os.path.join(self.data_dir, self.csv_fn.format(prefix)),
            parse_dates=["causal_date", "mrca_date"])
        df[f"parents_dist_{self.ts.time_units}"] = (
            self.ts.nodes_time[df["mrca"]] - self.ts.nodes_time[df["node"]]
        )
        logging.info(f"{len(df)} breakpoints | {len(np.unique(df.node))} re nodes read")
        # Remove potential contaminents
        self.df = df[df.max_descendant_samples > 1]
        logging.info(
            f"{len(self.df)} breakpoints in the ARG | "
            + f"{len(np.unique(self.df.node))} re nodes initially retained"
        )

    @staticmethod
    def add_common_lines(ax, num, ts, common_proportions):
        v_pos = {k: v for v, k in enumerate(common_proportions.keys())}
        for i, (u, (pango, prop, date)) in enumerate(common_proportions.items()):
            n_children = len(np.unique(ts.edges_child[ts.edges_parent == u]))
            logging.info(
                f"{ordinal(i+1)} most freq. parent MRCA has id {u} (imputed: {pango}) "
                f"@ date={date}; "
                f"num_children={n_children}"
            )
            # Find all samples of the focal lineage
            ax.axhline(date, ls=":", c="grey", lw=1)
            sep = "\n" if v_pos[u] == 0 else " "
            most = "Most" if v_pos[u] == 0 else ordinal(i + 1) + " most"
            ax.text(
                (ts.node(u).time + 5) / 7,  # Hand tweaked to get nice label positions
                date if i !=2 else date - timedelta(days=7),  # Push apart the 2nd and 3rd labels
                f"{most} frequent{sep}MRCA,{sep}{n_children} children,{sep}{prop * 100:.1f} % of {pango}",
                fontsize=10,
                va="center",
                bbox=dict(facecolor="white", edgecolor="none", pad=0),
            )

    def do_plot(
        self,
        main_ax,
        hist_ax,
        df,
        title,
        label_tweak,
        xlab=True,
        ylab=True,
        filter_pass_size=20,
        filter_fail_size=5,
        filter_num_desc = 5
    ):

        def filter_breaks(df_in, inverse=False):
            # NB previously we used HMM consistency, commented out below for reference
            # return np.logical_and(df.fwd_bck_parents_max_dist == 0, df.is_hmm_mutation_consistent)
            use = df_in.max_descendant_samples >= 5
            return df_in[~use] if inverse else df_in[use]

        logging.info(f"Plotting {title}")
        filter_pass = (
            filter_pass_size,
            filter_breaks(df),
            f"descendant samples $\\geq$ {filter_num_desc}")
        filter_fail = (
            filter_fail_size,
            filter_breaks(df, inverse=True),
            f"1 $<$ descendant samples $<$ {filter_num_desc}")
        self.hist_legend_elements = []
        for size, _df, label in (filter_pass, filter_fail):
            self.hist_legend_elements.append(
                Patch(
                    facecolor='tab:blue',
                    edgecolor='none',
                    alpha=0.5 if size == filter_fail_size else 0.75,
                    label=label,
                )
            )
            main_ax.scatter(
                _df.parents_dist_days / 7,
                _df.mrca_date,
                alpha=0.2,
                s=size,
                color="tab:blue",
                label=label,
            )
        delta_T = df.parents_dist_days
        logging.info(f" {len(df)} points ({len(filter_pass[1])} filtered)")
        logging.info(f" {len(np.unique(df.node))} rec nodes ({len(np.unique(filter_pass[1].node))} filtered)")
        logging.info(f" T diff: min={min(delta_T)}, max={max(delta_T)} days")
        df_X = df[df.causal_lineage.str.startswith("X")]

        filter_fail = (filter_fail_size, filter_breaks(df_X, inverse=True), "< 10")
        filter_pass = (filter_pass_size, filter_breaks(df_X), "$geq$ 10")
        for size, _df, label in (filter_fail, filter_pass):
            for row in _df.itertuples():
                if size == filter_pass_size:
                    lab = f" {row.causal_lineage}"
                    ha = "left"
                    sz=8
                else:
                    lab = f"{row.causal_lineage} "
                    ha = "right"
                    sz = 6
                pos = [row.parents_dist_days / 7 + label_tweak[0], row.mrca_date]
                va = "center_baseline"
                main_ax.annotate(
                    lab, pos, size=sz, ha=ha, va=va, rotation=70, rotation_mode='anchor')
            main_ax.scatter(_df.parents_dist_days / 7, _df.mrca_date, c="orange", s=size, alpha=0.8)
        if xlab:
            hist_ax.set_xlabel("Estimated divergence between parents (weeks)")
        if ylab:
            main_ax.set_ylabel(f"Estimated MRCA date")
            hist_ax.set_ylabel("# breakpoints")
        main_ax.set_title(title)
        hist_ax.spines["top"].set_visible(False)
        hist_ax.spines["right"].set_visible(False)
        hist_ax.yaxis.set_tick_params(labelleft=True)
        hist_params = dict(density=False, alpha=0.5, color="tab:blue")
        _, bins, _ = hist_ax.hist(df.parents_dist_days / 7, bins=60, **hist_params)
        hist_ax.hist(filter_breaks(df).parents_dist_days / 7, bins=bins, **hist_params)



class RecombinationNodeMrcas_all(RecombinationNodeMrcas):
    name = "recombination_node_mrcas"
    num_common_lines = 5

    def plot(self, args):
        prefix = os.path.join("figures", self.name)
        gridspec = {"height_ratios": [3.5, 1]}
        _, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True, gridspec_kw=gridspec)

        mrca_counts = collections.Counter(self.df.mrca)
        common_mrcas = mrca_counts.most_common(self.num_common_lines)
        logging.info(
            "Calculating proportions of descendants for "
            f"{['mrca: {} ({} counts)'.format(id, c) for id, c in common_mrcas]}"
        )
        focal_node_map = {c[0]: None for c in common_mrcas}
        focal_node_map[519829] = "BA.1"
        # 5868 is a common MRCA almost exactly contemporary with the most common node
        focal_node_map.pop(5868, None)
        dates_for_mrcas = {r.mrca: r.mrca_date for r in self.df.itertuples()}
        proportions = descendant_proportion(self.ts, focal_node_map, dates_for_mrcas)
        self.add_common_lines(axes[0], 5, self.ts, proportions)
        title = ""  # "Breakpoints from nonsingleton recombination nodes in the Long ARG"
        self.do_plot(*axes, self.df, title, label_tweak=[0.1, 0])
        axes[0].legend()
        axes[1].legend(handles=self.hist_legend_elements, loc='upper right')
        plt.subplots_adjust(hspace=0.1)
        plt.savefig(prefix + f".{args.outtype}", bbox_inches="tight")


class RecombinationNodeMrcas_subset(RecombinationNodeMrcas):
    name = "supp_recombination_node_mrcas"

    def subplot(self, ax_main, ax_hist, restrict, parent_variants, labs=None, **kwargs):
        if labs is not None:
            kwargs["xlab"] = kwargs["ylab"] = labs
        self.do_plot(
            ax_main,
            ax_hist,
            self.df[[v == set(restrict) for v in parent_variants]],
            f"{'+'.join(restrict)} breakpoints",
            label_tweak=[0.2, 0],
            **kwargs,
        )

    def plot(self, args):
        prefix = os.path.join("figures", self.name)

        fig, axes = plt.subplots(
            nrows=5,  # include an extra row to hack some space
            ncols=3,
            figsize=(18, 12),
            sharex=True,
            sharey="row",
            gridspec_kw={"height_ratios": [3.5, 1, 0.2, 3.5, 1]},
        )
        # Make sure the main plots have the same y-axis limits on different rows
        for row in (0, 3):
            axes[row, 0].set_ylim(self.df.mrca_date.min(), self.df.mrca_date.max())

        parent_variants = [
            frozenset(
                (
                    variant_name(row.left_parent_imputed_lineage),
                    variant_name(row.right_parent_imputed_lineage),
                )
            )
            for row in self.df.itertuples()
        ]

        logging.info(collections.Counter(parent_variants))

        # mrca_counts = collections.Counter(self.df.parents_mrca)
        # self.add_common_lines(axes[0], mrca_counts, 5, self.ts)
        self.subplot(
            axes[0][0], axes[1][0], ["Alpha", "Alpha"], parent_variants, xlab=False
        )
        self.subplot(
            axes[0][1], axes[1][1], ["Delta", "Delta"], parent_variants, labs=False
        )
        self.subplot(
            axes[0][2], axes[1][2], ["Omicron", "Omicron"], parent_variants, labs=False
        )
        axes[2][0].axis('off')
        axes[2][1].axis('off')
        axes[2][2].axis('off')

        self.subplot(
            axes[3][0], axes[4][0], ["Alpha", "Delta"], parent_variants)

        self.subplot(
            axes[3][1], axes[4][1], ["Alpha", "Omicron"], parent_variants, ylab=False
        )
        self.subplot(
            axes[3][2], axes[4][2], ["Delta", "Omicron"], parent_variants, ylab=False
        )
        axes[0][0].legend()
        axes[1][0].legend(handles=self.hist_legend_elements, loc='upper right')
        plt.subplots_adjust(hspace=0.1)

        plt.savefig(prefix + f".{args.outtype}", bbox_inches="tight")


def change_pos(pos, key, dx=None, dy=None, x=None, y=None, xy_from=None):
    """
    Helper function to change the position of a node in a graph layout. `pos`
    is a dictionary of node positions, keyed by node ID, as returned by networkx
    """
    pos[key] = (
        pos[xy_from or key][0] + (dx or 0) if x is None else x,
        pos[xy_from or key][1] + (dy or 0) if y is None else y,
    )


class Pango_X_graph(Figure):
    sample_metadata_labels = ""  # Don't show the strain name
    node_colours = {
        # B lineages in blue/green (AY.100 is B.1.617.2.100)
        # Basic in blue
        "B.1": "#212fe1",
        "B.1.384": "#4ac7ff",
        "B.1.177.18": "#93e8ff",
        "B.1.631": "#add9e5",
        "B.1.634": "#2092e1",
        "B.1.627": "#7b96cc",
        "B.1.1": "#aea4ff",
        # Alpha: purple / lilac
        "B.1.1.7": "#ae9cff",
        # Delta: yellow/brown
        "B.1.617.2": "#e2da91",  # Delta
        "AY.1": "#ffe44a",  # Delta
        "AY.4": "#b78c19",  # Delta
        # Omicron (B.1.1.529...) in green-ish
        "BA.1": "#74b058",  # Omicron
        "BA.1.17": "#57de3c",  # Omicron
        "BA.1.15": "#248f0f",  # Omicron
        "BA.2": "#68ffa5",  # Omicron
        "BA.2.9": "#21e1ab",  # Omicron
        # Pango X-lineages in red/orange/pink
        "XA": "#ff0055",
        "XB": "#fc31fb",
        "XD": "#bb0000",
        "XQ": "#db5884",
        "XU": "#894a4e",
        "XAA": "#ffc9a5",
        "XAB": "#fdacfe",
        "XAG": "#fe812e",
        None: "lightgray",  # Default
        "Unknown (R)": "k",
        "Unknown": "None",
    }

    @staticmethod
    def grow_graph(ts, input_nodes):
        # Ascend up from input nodes
        up_nodes = sc2ts.utils.node_path_to_samples(
            input_nodes, ts, ignore_initial=True, stop_at_recombination=True
        )
        # Descend from these
        nodes = sc2ts.utils.node_path_to_samples(
            up_nodes, ts, rootwards=False, ignore_initial=False
        )
        # Ascend again, to get parents of downward nonsamples
        up_nodes = sc2ts.utils.node_path_to_samples(nodes, ts, ignore_initial=False)
        nodes = np.append(nodes, up_nodes)
        # Add parents of recombination nodes up to the nearest sample
        re_nodes = nodes[ts.nodes_flags[nodes] & sc2ts.NODE_IS_RECOMBINANT > 0]
        re_parents = np.unique(ts.edges_parent[np.isin(ts.edges_child, re_nodes)])
        re_ancestors = sc2ts.utils.node_path_to_samples(re_parents, ts, ignore_initial=False)
        nodes = np.append(nodes, re_ancestors)
        # Remove duplicates
        _, idx = np.unique(nodes, return_index=True)
        return nodes[np.sort(idx)]

    def __init__(self):
        ts, self.basetime = utils.load_tsz(self.ts_dir, self.long_fn)
        self.ts = sc2ts.utils.detach_singleton_recombinants(ts)
        logging.info(
            f"Removed {ts.num_samples - self.ts.num_samples} singleton recombinants"
        )
        self.node_positions_fn = os.path.join(self.ts_dir, self.name + "_nodepos.json")
        try:
            with open(self.node_positions_fn) as f:
                self.node_positions = {
                    int(k): v for k, v in json.loads("".join(f.readlines())).items()
                }
            logging.info(f"Loaded positions from {self.node_positions_fn}")
        except FileNotFoundError:
            self.node_positions = None

        if self.show_metadata:
            self.node_metadata_label_key = "Imputed_" + self.imputed_lineage
        else:
            self.node_metadata_label_key = ""
        self.fn_prefix = os.path.join("figures", self.name)


class Pango_X_tight_graph(Pango_X_graph):
    edge_font_size = 10
    node_font_size = 7.5
    node_size = 250
    show_ids = False
    show_metadata = False
    mutations_fn = ""
    exterior_edge_len = 0.5
    legend = None
    node_def_lineage = (
        "Nextclade_pango"  # Always use this when defining which nodes to pick
    )
    show_descendant_samples = "sample_tips"

    def __init__(self):
        super().__init__()
        if not hasattr(self, "label_replace"):
            self.label_replace = {
                "Unknown": "",
                "Unknown ": "",
                " samples": "",
                " sample": "",
                " mutations": "",
                " mutation": "",
                f"…{int(self.ts.sequence_length)}": "─┤",
                "0…": "├─",
                "…": "─",
            }

    @classmethod
    def define_nodes(cls, ts):
        raise NotImplementedError()

    @classmethod
    def used_pango_colours(cls, used_nodes, ts):
        used_pango = set(
            [
                ts.node(n).metadata.get("Imputed_" + cls.imputed_lineage, None)
                for n in used_nodes
            ]
        )
        return {k: v for k, v in cls.node_colours.items() if k in used_pango}

    @classmethod
    def save_adjusted_positions(cls, pos, fn):
        cls.adjust_positions(pos)
        logging.info("Saving positions")
        with open(fn, "wt") as f:
            # json.dumps doesn't like np.int's
            f.write(json.dumps({int(k): v for k, v in pos.items()}))

    @classmethod
    def post_process(cls, ax):
        pass

    @staticmethod
    def adjust_positions(pos):
        # Override this to adjust the node positions by altering `pos`
        # e.g. by using change_position
        pass

    def plot_subgraph(
        self,
        nodes,
        ax,
        treeinfo=None,
        node_positions=None,
        sample_metadata_labels=None,
        label_replace=None,
    ):
        if sample_metadata_labels is None:
            sample_metadata_labels = self.sample_metadata_labels
        if label_replace is None:
            label_replace = self.label_replace
        return sc2ts.utils.plot_subgraph(
            nodes,
            self.ts,
            ti=treeinfo,
            mutations_json_filepath=self.mutations_fn,
            ax=ax,
            exterior_edge_len=self.exterior_edge_len,
            ts_id_labels=self.show_ids,
            node_colours=self.node_colours,
            node_metadata_labels=self.node_metadata_label_key,
            sample_metadata_labels=sample_metadata_labels,
            node_size=self.node_size,
            node_font_size=self.node_font_size,
            label_replace=label_replace,
            edge_font_size=self.edge_font_size,
            colour_metadata_key="Imputed_" + self.imputed_lineage,
            node_positions=node_positions,
            show_descendant_samples=self.show_descendant_samples
        )

    def plot(self, args):
        fig, ax = plt.subplots(1, 1, figsize=self.figsize)
        self.nodes = self.define_nodes(self.ts)
        G, pos = self.plot_subgraph(
            self.nodes, ax=ax, node_positions=self.node_positions
        )
        self.legend_key(ax, self.nodes, self.ts)
        self.post_process(ax)

        if self.node_positions is None:
            self.save_adjusted_positions(pos, self.node_positions_fn)
            fn = self.fn_prefix + "_defaultpos"
        else:
            fn = self.fn_prefix
        plt.savefig(fn + f".{args.outtype}", bbox_inches="tight")

    @classmethod
    def legend_key(cls, ax, nodes, ts):
        # Plot or otherwise output a legend or key. Define in the subcless
        pass

    @classmethod
    def make_legend_elements(cls, used_colours, sizes):
        legend_elements = []
        for (k, v), size in zip(used_colours.items(), itertools.cycle(sizes)):
            if k is None:
                k = "Other"
            elif k == "Unknown (R)":
                k = "Recombination node"
            elif k == "Unknown":
                k = "Not imputed"
            else:
                voc = variant_name(k)
                if not voc.startswith("_"):
                    # Add Greek VoC name in braces
                    k += f" ({voc})"
            stroke = "k" if v == "None" else v
            legend_elements.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    label=k,
                    linestyle="None",
                    markerfacecolor=v,
                    markeredgecolor=stroke,
                    markersize=size,
                )
            )
        return legend_elements


class Pango_XA_nxcld_tight_graph(Pango_X_tight_graph):
    name = "Pango_XA_nxcld_tight_graph"
    imputed_lineage = "Nextclade_pango"
    sample_metadata_labels = (
        "Imputed_" + imputed_lineage
    )  # For the first plot, use labels
    figsize = (5, 5)

    def __init__(self):
        super().__init__()
        # If this is the base class, don't abbreviate the "mutation" labels
        if self.__class__.__name__ == "Pango_XA_nxcld_tight_graph":
            self.label_replace = self.no_replace_mutation_labels(self.label_replace)

    @staticmethod
    def no_replace_mutation_labels(label_replace):
        return {k: v for k, v in label_replace.items() if "mutation" not in k}

    @classmethod
    def define_nodes(cls, ts):
        XA = [n.id for n in ts.nodes() if n.metadata.get(cls.node_def_lineage) == "XA"]
        nodes = Pango_X_graph.grow_graph(ts, XA)
        XA = [n.id for n in ts.nodes() if n.metadata.get(cls.imputed_lineage) == "XA"]
        if not set(XA).issubset(set(nodes)):
            logging.warning("Not all XA nodes included in graph")
        return nodes

    @staticmethod
    def adjust_positions(pos):
        dx = pos[147032][0] - pos[154551][0]
        change_pos(pos, 227648, dx=-1.5 * dx)
        change_pos(pos, 228656, dx=-dx)

    @classmethod
    def post_process(cls, ax):
        # Make room for the legend
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        ax.set_xlim(x_min - (x_max - x_min) * 0.02)
        ax.set_ylim(y_min, y_max + (y_max - y_min) * 0.1)
        ax.text(
            0.98,
            0.05,
            "(A) XA subgraph",
            fontsize=18,
            transform=ax.transAxes,
            ha="right",
        )

    @classmethod
    def legend_key(cls, ax, nodes, ts):
        lin = cls.imputed_lineage.split("_")[0]
        legend_pts = ax.legend(
            title=f"$\\bf{{Nodes}}$ (labelled/coloured\nby {lin} Pango lineage)",
            handles=cls.make_legend_elements(
                {
                    "Inserted node": "lightgray",
                    "Recombination node": "k",
                    "Sample node": "lightgray",
                },
                sizes=np.sqrt(np.array([1 / 3, 1 / 3, 1]) * cls.node_size),
            ),
            loc="center left",
            labelspacing=0.9,
            alignment="left",
            borderpad=0.6,
            handletextpad=0.5,
        )
        # extra legend above the previous one, without points, showing the edge intervals

        legend_intervals = ax.legend(
            title="$\\bf{Inheritance\ above\ recombination\ nodes}$",
            handles=[
                Patch(
                    facecolor="none",
                    edgecolor="none",
                    label="from genomic position $0..N$",
                ),
                Patch(
                    facecolor="none",
                    edgecolor="none",
                    label="from genomic position $N$ to end",
                ),
            ],
            loc="upper left",
            labelspacing=0.9,
            alignment="left",
            borderpad=0.6,
            handletextpad=3,
        )
        ax.add_artist(legend_pts)
        ax.add_artist(legend_intervals)


class Pango_XAG_nxcld_tight_graph(Pango_X_tight_graph):
    name = "Pango_XAG_nxcld_tight_graph"
    imputed_lineage = "Nextclade_pango"
    legend = "upper left"
    figsize = (11, 4.5)

    @classmethod
    def legend_key(cls, ax, nodes, ts):
        used_colours = cls.used_pango_colours(nodes, ts)
        del used_colours["Unknown (R)"]
        lin = cls.imputed_lineage.split("_")[0]
        ax.legend(
            title=f"$\\bf{{{lin}\ Pango\ lineage}}$",
            handles=cls.make_legend_elements(
                used_colours, sizes=[np.sqrt(cls.node_size) * 0.8]
            ),
            loc="upper left",
            labelspacing=0.75,
            alignment="left",
            borderpad=0.6,
            handletextpad=0.5,
        )

    @classmethod
    def define_nodes(cls, ts):
        nodes = Pango_X_graph.grow_graph(ts, [712029])  # This gives a nice layout
        XAG = [n.id for n in ts.nodes() if n.metadata.get(cls.imputed_lineage) == "XAG"]
        if not set(XAG).issubset(set(nodes)):
            logging.warning("Not all XAG nodes included in the graph")
        return nodes

    @classmethod
    def post_process(cls, ax: plt.Axes) -> None:
        x_min, x_max = ax.get_xlim()
        ax.set_xlim(x_min + (x_max - x_min) * 0.06, x_max - (x_max - x_min) * 0.06)
        ax.text(
            0.98,
            0.05,
            "(B) XAG subgraph",
            fontsize=18,
            transform=ax.transAxes,
            ha="right",
        )

    @staticmethod
    def adjust_positions(pos):
        a = pos[734769][0]
        b = pos[716314][0]
        dx = a - b
        change_pos(pos, 675573, dx=dx * 5)
        change_pos(pos, 635895, dx=dx)
        change_pos(pos, 583065, dx=dx)
        change_pos(pos, 635896, dx=dx / 2)
        change_pos(pos, 673653, dx=-dx / 2)
        change_pos(pos, 780910, dx=-dx / 2)


class Pango_XD_nxcld_tight_graph(Pango_X_tight_graph):
    name = "Pango_XD_nxcld_tight_graph"
    imputed_lineage = "Nextclade_pango"

    figsize = (7, 6)

    @classmethod
    def define_nodes(cls, ts):
        XD = [n.id for n in ts.nodes() if n.metadata.get(cls.node_def_lineage) == "XD"]
        # Add the immediate descendants of XD too
        XD.extend(ts.edges_child[np.isin(ts.edges_parent, XD)])
        nodes = Pango_X_graph.grow_graph(ts, XD)
        XD = [n.id for n in ts.nodes() if n.metadata.get(cls.imputed_lineage) == "XD"]
        if not set(XD).issubset(set(nodes)):
            logging.warning("Not all XD nodes included in the graph")
        return nodes

    @classmethod
    def post_process(cls, ax):
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        # ax.set_xlim(x_min - (x_max - x_min) * 0.01, x_max + (x_max - x_min) * 0.01)
        ax.set_ylim(y_min - (y_max - y_min) * 0.01, y_max)
        ax.text(
            0.98,
            0.96,
            "(C) XD subgraphs",
            fontsize=18,
            transform=ax.transAxes,
            ha="right",
            va="top",
        )

    @staticmethod
    def adjust_positions(pos):
        dx = pos[644359][0] - pos[638774][0]
        dy = pos[665231][1] - pos[644359][1]

        # All adjustments below found by tedious trial and error
        # Arranged so that the recombination edge intervals
        # are in L->R order
        change_pos(pos, 517356, dy=dy, dx=dx)
        change_pos(pos, 518483, xy_from=630772, dy=2 * dy, dx=-dx)
        change_pos(pos, 361569, xy_from=518483, dy=2 * dy)
        change_pos(pos, 630772, xy_from=638775, dy=1.5 * dy)

        change_pos(pos, 396442, x=pos[557733][0] - dx)
        change_pos(pos, 557733, xy_from=416354, dy=dy)
        change_pos(pos, 416354, x=pos[400974][0] + dx)
        change_pos(pos, 400974, x=pos[396442][0])
        change_pos(pos, 375328, x=pos[396442][0])

        # Shift positions of all nodes in the LH subgraph
        shift = dx * 8
        change_pos(pos, 396442, dx=shift, dy=dy * 2.5)
        change_pos(pos, 557733, dx=shift, dy=dy * 2.5)
        change_pos(pos, 416354, dx=shift, dy=dy * 2.5)
        change_pos(pos, 400974, dx=shift, dy=dy * 2.5)
        change_pos(pos, 375328, dx=shift, dy=dy * 2.5)
        change_pos(pos, 573905, dx=shift, dy=dy * 2.5)
        change_pos(pos, 573904, dx=shift, dy=dy * 2.5)
        change_pos(pos, 575079, dx=shift, dy=dy * 2.3)

    @classmethod
    def legend_key(cls, ax, nodes, ts):
        used_colours = cls.used_pango_colours(nodes, ts)
        del used_colours["Unknown (R)"]
        lin = cls.imputed_lineage.split("_")[0]
        ax.legend(
            title=f"$\\bf{{{lin}\ Pango\ lineage}}$",
            handles=cls.make_legend_elements(
                used_colours, sizes=[np.sqrt(cls.node_size) * 0.8]
            ),
            loc="lower right",
            labelspacing=0.75,
            alignment="left",
            borderpad=0.6,
            ncol=2,
            handletextpad=0.5,
        )


class Pango_XB_nxcld_tight_graph(Pango_X_tight_graph):
    name = "Pango_XB_nxcld_tight_graph"
    imputed_lineage = "Nextclade_pango"
    figsize = (16, 6)

    @classmethod
    def define_nodes(cls, ts):
        # Order of nodes here matters. This is found by trial-and-error :(
        basic_nodes = [
            285180,  # lone AB on the side
            335592,  #
            280287,
            206465,
            334261,
            337869,
            345330,
            # 394058, # a singleton recombinant, prob worth not showing
        ]
        basic_nodes += [n.id
               for n in ts.nodes()
               if n.metadata.get(cls.node_def_lineage) == "XB" and n.is_sample() and n.id not in basic_nodes]
        for n in basic_nodes:
            pango = ts.node(n).metadata.get("Imputed_" + cls.imputed_lineage)
            if pango != "XB":
                logging.info(f"XB plot: input node {n} is {pango} not XB")
        nodes = Pango_X_graph.grow_graph(ts, basic_nodes)
        if logging.getLogger().isEnabledFor(logging.INFO):
            XB = [
                n.id for n in ts.nodes() if n.metadata.get(cls.imputed_lineage) == "XB"
            ]
            logging.info(
                f"XB nodes not shown (perhaps below '+N' nodes): {set(XB) - set(nodes)}"
            )
        return nodes

    @classmethod
    def post_process(cls, ax):
        x_min, x_max = ax.get_xlim()
        ax.set_xlim(x_min + (x_max - x_min) * 0.06, x_max - (x_max - x_min) * 0.06)
        ax.text(
            0.98, 0.05, "XB subgraph", fontsize=18, transform=ax.transAxes, ha="right"
        )

    @staticmethod
    def adjust_positions(pos):
        dx = pos[350705][0] - pos[341400][0]
        dy = pos[351074][1] - pos[341400][1]

        # All adjustments below found by tedious trial and error
        # Root rec node
        change_pos(pos, 206466, dx=3/4 * dx)
        change_pos(pos, 200603, dx=2.5 * dx)
        change_pos(pos, 12108, dx=-1 * dx)

        change_pos(pos, 274176, dx=dx)
        change_pos(pos, 325792, dx=2 * dx)
        change_pos(pos, 273897, dx=3 * dx)
        #change_pos(pos, 295321, dx=dx)
        change_pos(pos, 282861, dx= -3 * dx)
        change_pos(pos, 282727, dx= -2 * dx)
        #change_pos(pos, 265975, dx=3 * dx)

        change_pos(pos, 309252, dx=-2 * dx)  # Rec node
        change_pos(pos, 289479, dx=-dx / 2, dy=dy / 2)
        change_pos(pos, 320601, dx=2 * dx) # Far left B.1.634 parent
        change_pos(pos, 339521, dx=dx / 2)
        change_pos(pos, 320394, dx=2.5 * dx)
        change_pos(pos, 179, dx=-2 * dx, dy=dy)
        change_pos(pos, 3940, dx=-3.5 * dx, dy=dy)
        change_pos(pos, 1165, dx=-dx, dy=dy)
        change_pos(pos, 5731, dx=dx/3)


        change_pos(pos, 285180, dx=2.5 * dx)
        change_pos(pos, 285181, dx=2 * dx)
        change_pos(pos, 300560, dx=dx)
        change_pos(pos, 328838, dx=dx)


    @classmethod
    def legend_key(cls, ax, nodes, ts):
        used_colours = cls.used_pango_colours(nodes, ts)
        del used_colours["Unknown (R)"]
        lin = cls.imputed_lineage.split("_")[0]
        ax.legend(
            title=f"$\\bf{{{lin}\ Pango\ lineage}}$",
            handles=cls.make_legend_elements(
                used_colours, sizes=[np.sqrt(cls.node_size) * 0.8]
            ),
            loc="upper right",
            labelspacing=0.75,
            alignment="left",
            borderpad=0.6,
            ncol=3,
            handletextpad=0.5,
        )


class Pango_XA_XAG_XD_nxcld_tight_graph(Pango_X_tight_graph):
    name = "Pango_XA_XAG_XD_nxcld_tight_graph"
    imputed_lineage = "Nextclade_pango"
    figsize = (14, 9.5)

    def __init__(self):
        super().__init__()
        assert self.node_positions is None
        self.node_positions = {}
        self.node_positions_fn = {}
        for Xlin, cls in [
            ("XA", Pango_XA_nxcld_tight_graph),
            ("XAG", Pango_XAG_nxcld_tight_graph),
            ("XD", Pango_XD_nxcld_tight_graph),
        ]:
            fn = os.path.join(self.ts_dir, cls.name + "_nodepos.json")
            self.node_positions_fn[Xlin] = fn
            try:
                with open(fn) as f:
                    self.node_positions[Xlin] = {
                        int(k): v for k, v in json.loads("".join(f.readlines())).items()
                    }
                logging.info(f"Loaded positions from {fn}")
            except FileNotFoundError:
                self.node_positions[Xlin] = None

    def plot(self, args):
        fig = plt.figure(figsize=self.figsize)
        width_ratios = [0.8, 1]
        height_ratios = [1, 0.95]
        gs = fig.add_gridspec(
            2,
            2,
            width_ratios=width_ratios,
            height_ratios=height_ratios,
            wspace=0.05,
            hspace=0.05,
        )
        axC = fig.add_subplot(gs[0, 1])
        axA = fig.add_subplot(gs[0, 0])
        axB = fig.add_subplot(gs[1, :])

        treeinfo = sc2ts.utils.TreeInfo(self.ts)

        fn = self.fn_prefix
        for Xlin, ax, cls in [
            ("XA", axA, Pango_XA_nxcld_tight_graph),
            ("XAG", axB, Pango_XAG_nxcld_tight_graph),
            ("XD", axC, Pango_XD_nxcld_tight_graph),
        ]:
            logging.info(f"Plotting {Xlin}")
            nodes = cls.define_nodes(self.ts)
            G, pos = self.plot_subgraph(
                nodes,
                ax,
                treeinfo=treeinfo,
                node_positions=self.node_positions[Xlin],
                sample_metadata_labels=cls.sample_metadata_labels,
                label_replace=cls.no_replace_mutation_labels(self.label_replace)
                if Xlin == "XA"
                else None,
            )
            cls.legend_key(ax, nodes, self.ts)
            cls.post_process(ax)

            if self.node_positions[Xlin] is None:
                cls.save_adjusted_positions(pos, self.node_positions_fn[Xlin])
                fn = fn + f"_no{Xlin}pos"

        plt.savefig(fn + f".{args.outtype}", bbox_inches="tight")


large_replace = {
    "Unknown (R)": "Rec\nnode",
    "Unknown": "",
    # Make all the Pango lineages bold
    "XB": r"$\bf XB$",
    "XA": r"$\bf XA$",
    "XD": r"$\bf XD$",
    "XQ": r"$\bf XQ$",
    "XU": r"$\bf XU$",
    "XA$G": "XAG$",  # This is a hack because replacement has been already done
    "XA$A": "XAA$",
    "XA$B": "XAB$",
    "BA.2": r"$\bf BA.2$",
    "BA.2$.9": "BA.2.9$",
    "BA.1": r"$\bf BA.1$",
    "BA.1$.17": "BA.1.17$",
    "BA.1$.15": "BA.1.15$",
    "B.1": r"$\bf B.1$",
    "B.1$.1": "B.1.1$",
    "B.1.1$.7": "B.1.1.7$",
    "B.1.1$77.18": "B.1.177.18$",
    "B.1$.384": "B.1.384$",
    "B.1$.631": "B.1.631$",
    "B.1$.634": "B.1.634$",
    "B.1$.627": "B.1.637$",
    "B.1$.617.2": "B.1.617.2$",
    "AY.1": r"$\bf AY.1$",
    "AY.1$18": "AY.118$",
    "AY.4": r"$\bf AY.4$",
    "AY.4$3": "AY.43$",
}


def print_sample_map(ts, nodes):
    isl_map = {}
    for u in nodes:
        if ts.node(u).is_sample():
            md = ts.node(u).metadata
            isl_map[u] = (
                md["gisaid_epi_isl"] if "gisaid_epi_isl" in md else "EPI_ISL_unknown",
                md["strain"] if "strain" in md else "Strain_unknown",
            )
    # Find the nodes that descend from a recombinant node
    ts, node_map = ts.simplify(nodes, map_nodes=True)
    new_nodes = node_map[nodes]
    re_nodes = np.flatnonzero(ts.nodes_flags & sc2ts.NODE_IS_RECOMBINANT)
    recombinant_samples = set()
    for tree in ts.trees():
        for u in re_nodes:
            recombinant_samples.update(tree.samples(u))
    for k in sorted(isl_map.keys()):
        prefix = "*tsk" if node_map[k] in recombinant_samples else "tsk"
        print(f"{prefix}{k}: {isl_map[k][0]}, {isl_map[k][1]}")


class Pango_XA_gisaid_large_graph(Pango_XA_nxcld_tight_graph):
    name = "Pango_XA_gisaid_large_graph"
    imputed_lineage = "GISAID_lineage"
    figsize = (12, 10)
    node_size = 2000
    show_ids = None  # Only show for sample nodes
    edge_font_size = 6
    node_font_size = 7.5
    mutations_fn = None
    show_metadata = True
    sample_metadata_labels = ""
    label_replace = large_replace

    @classmethod
    def legend_key(cls, ax, nodes, ts):
        print_sample_map(ts, nodes)

    @classmethod
    def post_process(cls, ax):
        pass


class Pango_XAG_gisaid_large_graph(Pango_XAG_nxcld_tight_graph):
    name = "Pango_XAG_gisaid_large_graph"
    imputed_lineage = "GISAID_lineage"
    figsize = (35, 16)
    node_size = 2000
    show_ids = None  # Only show for sample nodes
    edge_font_size = 6
    node_font_size = 7.5
    mutations_fn = None
    show_metadata = True
    label_replace = large_replace

    def legend_key(cls, ax, nodes, ts):
        print_sample_map(ts, nodes)

    @classmethod
    def post_process(cls, ax):
        x_min, x_max = ax.get_xlim()
        ax.set_xlim(x_min + (x_max - x_min) * 0.06, x_max - (x_max - x_min) * 0.06)


class Pango_XD_gisaid_large_graph(Pango_XD_nxcld_tight_graph):
    name = "Pango_XD_gisaid_large_graph"
    imputed_lineage = "GISAID_lineage"
    figsize = (16, 16)
    node_size = 2000
    show_ids = None  # Only show for sample nodes
    edge_font_size = 6
    node_font_size = 7.5
    mutations_fn = None
    show_metadata = True
    label_replace = large_replace

    @classmethod
    def legend_key(cls, ax, nodes, ts):
        print_sample_map(ts, nodes)

    @classmethod
    def post_process(cls, ax):
        x_min, x_max = ax.get_xlim()
        ax.set_xlim(x_min - (x_max - x_min) * 0.01, x_max + (x_max - x_min) * 0.01)


class Pango_XB_gisaid_large_graph(Pango_XB_nxcld_tight_graph):
    name = "Pango_XB_gisaid_large_graph"
    imputed_lineage = "GISAID_lineage"
    figsize = (40, 20)
    node_size = 2000
    show_ids = None  # Only show for sample nodes
    edge_font_size = 6
    node_font_size = 7.5
    mutations_fn = None
    show_metadata = True
    label_replace = large_replace

    @classmethod
    def legend_key(cls, ax, nodes, ts):
        print_sample_map(ts, nodes)

    @classmethod
    def post_process(cls, ax):
        x_min, x_max = ax.get_xlim()
        ax.set_xlim(x_min + (x_max - x_min) * 0.06, x_max - (x_max - x_min) * 0.06)

class LongTopTwoFalsePositiveGraph(Pango_X_tight_graph):
    name = "false_positive_top2_nxcld_large_graph"
    focal_nodes = [229998, 235293]
    main_focal_node_parents = [21605, 204945]
    main_focal_child = 248215
    main_focal_grandchild = 243149
    main_delta_source = 261771
    node_colours = {
        **{
            main_delta_source: "gold",
            "B.1.617.2": "#e2da91",
            "AY.100": "#e2da91",
            "AY.112": "#e2da91",
            None: "lightgray",
            "Unknown (R)": "k",
            "Unknown": "white",
        },
        **{u: "darkred" for u in focal_nodes}
    }
    imputed_lineage = "Nextclade_pango"
    label_replace = {"Unknown (R)": "Rec node"}
    show_ids = True # Show IDs for all nodes
    figsize = (25, 30)
    node_size = 5000
    edge_font_size = 7
    node_font_size = 7
    mutations_fn = None
    show_metadata = True
    show_descendant_samples = "tips"

    @classmethod
    def define_nodes(cls, ts):
        to_plot = set()
        for u in set(cls.focal_nodes) | {cls.main_focal_child, cls.main_focal_grandchild, cls.main_focal_node_parents[1]}:
            to_plot.add(u)
            for child in np.unique(ts.edges_child[ts.edges_parent == u]):
                to_plot.add(child)

        # Add the parents of each focal recombination node
        for u in cls.focal_nodes:
            to_plot.update(ts.edges_parent[ts.edges_child == u])

        # Add the main Delta polytomy
        to_plot.add(cls.main_delta_source)
        logging.info(
            f"Node {cls.main_delta_source} (coloured {cls.node_colours[cls.main_delta_source]}" +
            f", Pango {ts.node(cls.main_delta_source).metadata['Imputed_Nextclade_pango']})" +
            f" has {len(np.unique(ts.edges_child[ts.edges_parent == 261771]))} children"
        )

        # Add path between the two major parents and their MRCA
        tree = ts.at(21400)
        mrca = tree.mrca(*cls.main_focal_node_parents)
        for u in cls.main_focal_node_parents:
            while True:
                u = tree.parent(u)
                if u < 0:
                    break
                to_plot.add(u)
                if u == mrca:
                    break
        return list(to_plot)

    @classmethod
    def legend_key(cls, ax, nodes, ts):
        print_sample_map(ts, nodes)
        labels = {
            cls.focal_nodes[0]: "Focal recombination node",
            "Unknown (R)": "Additional recombination node",
            "B.1.617.2": "Nextclade Pango Delta VoC",
            cls.main_delta_source: "Majority Delta ancestor\n(root of 90% of Delta samples, >100 children)",
            "Unknown": "Unimputed Pango lineage",
        }
        ax.legend(
            title=f"$\\bf{{Highlighted\ nodes}}$",
            handles=[
                Line2D(
                    [0], [0],
                    label=v,
                    markerfacecolor=cls.node_colours[k],
                    markeredgecolor="k" if cls.node_colours[k] == "white" else cls.node_colours[k],
                    linestyle="None",
                    marker="o",
                    markersize=np.sqrt(cls.node_size) * 0.55,
                )
                for k, v in labels.items()
            ],
            loc="upper left",
            labelspacing=1.4,
            alignment="center",
            borderpad=0.6,
            handletextpad=1,
            title_fontsize=20,
            prop={'size': 20}
        )

    @staticmethod
    def adjust_positions(pos):
        pos[179] = (pos[179][0] + 90, pos[179][1]) # shift to make room for mutation labels
        pos[5184] = (pos[5184][0] + 80, pos[5184][1]) # shift
        pos[21656] = (pos[21656][0] + 10, pos[21656][1]) # shift
        pos[5868] = (pos[5868][0] + 80, pos[5868][1]) # shift
        pos[205995] = (pos[205995][0] + 70, pos[205995][1]) # shift
        pos[274163] = (pos[274163][0] + 100, pos[274163][1] - 10) # shift
        pos[294002] = (pos[294002][0] + 100, pos[294002][1]) # shift
        pos[21605] = (pos[21605][0] - 60, pos[21605][1] + 85) # shift
        pos[357852] = (pos[357852][0] + 140, pos[357852][1]) # shift
        pos[240637] = (pos[240637][0] - 70, pos[240637][1]) # shift
        pos[204945] = (pos[204945][0] + 60, pos[204945][1]) # shift
        p = pos[232088][0]
        pos[232088] = (pos[317538][0], pos[232088][1]) # shift
        pos[317538] = (pos[209245][0], pos[317538][1]) # shift
        pos[209245] = (pos[315067][0], pos[209245][1]) # shift
        pos[315067] = (p, pos[315067][1]) # shift
        pos[235293] = (pos[235293][0] - 20, pos[235293][1]) # shift

    @classmethod
    def post_process(cls, ax):
        y_min, y_max = ax.get_ylim()
        ax.set_ylim(y_min + (y_max - y_min) * 0.04, y_max - (y_max - y_min) * 0.04)


######################################
#
# Utility functions
#
######################################


def get_subclasses(cls):
    for subclass in cls.__subclasses__():
        yield from get_subclasses(subclass)
        yield subclass


def ordinal(n):
    return ["First", "Second", "Third", "Fourth", "Fifth", "Sixth", "Seventh"][n - 1]


def descendant_proportion(ts, focal_nodes_map, dates):
    """
    Take each focal node which maps to an imputed lineage, and work out how many
    samples of that lineage type have the focal node in their ancestry. If
    it maps to None, use the ts to get the imputed lineage of that node.
    Return a dict mapping focal node to (lineage, proportion, date) tuples.
    """
    focal_lin = {
        u: ts.node(u).metadata["Imputed_Nextclade_pango"] if lin is None else lin
        for u, lin in focal_nodes_map.items()
    }
    sample_lists = {u: [] for u in focal_nodes_map}
    # The time consuming step is finding the samples of each lineage type
    # this would probably be quicker is we could be bothered to do it via TreeInfo
    # but we don't have that calculated in the plotting code
    ret = {}
    for nd in tqdm.tqdm(ts.nodes(), desc="Finding sample lists"):
        if nd.is_sample():
            lineage = nd.metadata.get("Nextclade_pango", "")
            for k, v in focal_lin.items():
                if lineage == v:
                    sample_lists[k].append(nd.id)
    for ancestor, samples in sample_lists.items():
        sts, node_map = ts.simplify(samples, map_nodes=True, keep_unary=True)
        ok = np.zeros(sts.num_samples, dtype=bool)
        assert sts.samples().max() == sts.num_samples - 1
        for tree in sts.trees():
            for u in tree.samples(node_map[ancestor]):
                ok[u] = True
        ret[ancestor] = focal_lin[ancestor], np.sum(ok) / len(ok), dates[ancestor]
    return ret


class MutationalSpectra(Figure):
    name = "mutational_spectra"

    def plot(self, args):
        df = pd.read_csv("data/mutational_spectra.csv")

        counter1 = dict(zip(df["mutation"], df["sc2ts"]))
        counter2 = dict(zip(df["mutation"], df["yi"]))

        def _convert_to_percentages(counter):
            # Skip entries involving gaps
            total_count = sum([v for k, v in counter.items() if "-" not in k])
            counter = {k: (v / total_count) * 100 for k, v in counter.items()}
            return counter

        counter1 = _convert_to_percentages(counter1)
        counter2 = _convert_to_percentages(counter2)

        width = 0.4  # bar width
        fig, ax = plt.subplots(1, 1)
        fig.set_size_inches(6.5, 7)
        ax.tick_params(axis="both", which="major", labelsize=16)
        types = ["C>T", "G>A", "G>T", "G>C", "C>A", "T>A"]
        rev_types = [t[::-1] for t in types]
        x = np.arange(len(types))

        # FIXME standardise on colour schemes, and font sizes etc
        ax.set_ylabel("Percentage", fontsize=20)
        ax.bar(x, [counter1[t] for t in types], width=width, color="royalblue")
        ax.bar(
            x,
            [-counter1[t] for t in rev_types],
            width=width,
            color="lightsteelblue",
            bottom=0,
        )
        ax.bar(x + width, [counter2[t] for t in types], width=width, color="orange")
        ax.bar(
            x + width,
            [-counter2[t] for t in rev_types],
            width=width,
            color="moccasin",
            bottom=0,
        )

        ax2 = ax.secondary_xaxis("top")
        ax2.tick_params(axis="x")
        ax2.set_xticks(x + width / 2)

        def _replace_T_with_U(s):
            return s.replace("T", "U")

        types = [_replace_T_with_U(t) for t in types]
        rev_types = [_replace_T_with_U(t) for t in rev_types]
        ax2.set_xticklabels(types, fontsize=18)
        ax.set_xticks(x + width / 2)
        ax.set_xticklabels(rev_types, fontsize=18)

        labels = ["Wide ARG (min inheritor = 2)", "Yi et al. (2021), shared SBS"]
        ax.legend(
            [labels[0], labels[0], labels[1], labels[1]],
            loc="best",
            frameon=False,
            fontsize=15,
        )
        prefix = os.path.join("figures", self.name)
        plt.savefig(prefix + f".{args.outtype}", bbox_inches="tight")


class RecombinationIntervals(Figure):
    def plot_breakpoints(self, df, df_sites, unique_only=False):
        dfs = df.sort_values(["breakpoint_interval_left", "breakpoint_interval_right"])
        length = dfs.breakpoint_interval_right - dfs.breakpoint_interval_left
        intervals = [
            (row.breakpoint_interval_left, row.breakpoint_interval_right)
            for (_, row) in dfs.iterrows()
        ]
        if unique_only:
            unique = set(intervals)
            intervals = sorted(unique)

        norm = mpl.colors.Normalize(vmin=length.min(), vmax=length.max())
        cmap = mpl.colormaps["viridis"]

        lines = []
        colours = []
        for j, (left, right) in enumerate(intervals):
            lines.append(((left, j), (right, j)))
            colours.append(cmap(norm(right - left)))
        lc = mc.LineCollection(lines, colors=colours)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8))
        ax1.add_collection(lc)
        ax1.autoscale()
        ax1.set_yticks([])
        axins1 = inset_axes(
            ax1,
            width=0.1,
            height=1.5,
            bbox_to_anchor=(0.01, 1.0),
            bbox_transform=ax1.transAxes,
            loc="upper left",
        )
        fig.colorbar(
            ScalarMappable(norm=norm, cmap=cmap), cax=axins1, orientation="vertical"
        )

        axins2 = inset_axes(
            ax1,
            width=2.7,
            height=1.5,
            bbox_to_anchor=(0.13, 1.0),
            bbox_transform=ax1.transAxes,
            loc="upper left",
        )
        y, x, _ = axins2.hist(length, bins=np.arange(0, 10_000, 500))
        axins2.set_xlabel("Width of interval")
        axins2.set_ylabel("Interval count (100s)")
        axins2.yaxis.set_ticks(np.arange(0, y.max() + 20, 100))
        axins2.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda y, pos: int(y/100)))

        print("Recombination nodes = ", len(df.node.unique()))
        print("Num intervals:", len(intervals))
        print("Unique intervals:", len(set(intervals)))
        print(length.describe())

        covers = np.zeros(df_sites.position.max())
        for left, right in intervals:
            covers[left:right] += 1

        color = "tab:orange"
        ax2.set_ylabel("Intersecting intervals", color=color)
        ax2.tick_params(axis="y", labelcolor=color)
        ax2.plot(covers, color=color)

        count = df_sites.num_mutations
        pos = df_sites.position
        color = "tab:blue"
        ax3 = ax2.twinx()
        ax3.plot(pos, count, color=color, alpha=0.7)
        ax3.tick_params(axis="y", labelcolor=color)
        ax3.set_ylabel("Mutations per site", color=color)

        ax2.set_xlabel("Genome position")

        top_count = 10
        top_sites = np.argsort(count)[-top_count:]
        for site in top_sites:
            ax3.annotate(
                f"{int(pos[site])}", xy=(pos[site], count[site]), xycoords="data"
            )

        j = 0
        mids = []
        for gene, (left, right) in genes.items():
            mids.append(left + (right - left) / 2)
            j += 1
            for ax in [ax1, ax2, ax3]:
                if j % 2 == 1:
                    ax.axvspan(left, right, color="black", alpha=0.1, zorder=0)
                else:
                    ax.axvspan(left, right, color="green", alpha=0.1, zorder=0)

        for ax in reversed([ax1, ax2]):
            ax.set_xlim(-10, 30_000)
            axs = ax.secondary_xaxis("top")
            axs.tick_params(axis="x")
            axs.set_xticks(mids, minor=False)
            axs.set_xticklabels(["" for _ in genes.keys()], rotation="vertical")
        axs.set_xticklabels(list(genes.keys()), rotation="vertical")


class WideRecombinationIntervals(RecombinationIntervals):
    name = "wide_arg_recombination_intervals"

    def plot(self, args):
        df_sites = pd.read_csv("data/wide_arg_site_info.csv")
        df_recombs = pd.read_csv("data/wide_arg_recombinants.csv")
        df_recombs = df_recombs[df_recombs.max_descendant_samples > 1]
        self.plot_breakpoints(df_recombs, df_sites, unique_only=False)
        prefix = os.path.join("figures", self.name)
        plt.savefig(prefix + f".{args.outtype}", bbox_inches="tight")


class LongRecombinationIntervals(RecombinationIntervals):
    name = "long_arg_recombination_intervals"

    def plot(self, args):
        df_sites = pd.read_csv("data/long_arg_site_info.csv")
        df_recombs = pd.read_csv("data/long_arg_recombinants.csv")
        df_recombs = df_recombs[df_recombs.max_descendant_samples > 1]
        self.plot_breakpoints(df_recombs, df_sites, unique_only=False)
        prefix = os.path.join("figures", self.name)
        plt.savefig(prefix + f".{args.outtype}", bbox_inches="tight")


######################################
#
# Main
#
######################################


def main():
    figures = list(get_subclasses(Figure))

    name_map = {fig.name: fig for fig in figures if fig.name is not None}

    parser = argparse.ArgumentParser(description="Make the plots for specific figures.")
    parser.add_argument("-v", "--verbosity", action="count", default=0)
    parser.add_argument(
        "-o",
        "--outtype",
        choices=["pdf", "png", "svg"],
        default="pdf",
        help="The format of the output file",
    )
    parser.add_argument(
        "name",
        type=str,
        help="figure name",
        choices=sorted(list(name_map.keys()) + ["all", "Pango_X_graph"]),
    )
    args = parser.parse_args()

    levels = [logging.WARNING, logging.INFO, logging.DEBUG]
    level = levels[min(args.verbosity, len(levels) - 1)]  # cap to last level index
    logging.basicConfig(level=level)

    if args.name == "all":
        for name, fig in name_map.items():
            if fig in figures:
                logging.info(f"plotting {name}")
                fig().plot(args)

    elif args.name == "Pango_X_graph":
        for name, fig in name_map.items():
            if issubclass(fig, Pango_X_graph):
                logging.info(f"plotting {name}")
                fig().plot(args)

    else:
        fig = name_map[args.name]
        logging.info(f"plotting {args.name}")
        fig().plot(args)


if __name__ == "__main__":
    main()
