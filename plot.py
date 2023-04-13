import argparse
import collections
from datetime import datetime, timedelta
import hashlib
import re
import sys
import json
import os
import tempfile
import subprocess
import logging
import gzip
import shutil

import matplotlib as mpl

mpl.use("Agg")
from matplotlib.colors import rgb2hex
from matplotlib.cm import get_cmap
from matplotlib.lines import Line2D
from matplotlib import pyplot as plt
import numpy as np
import tqdm
import tszip
import tskit
import pandas as pd

import tsconvert  # Not on pip. Install with python -m pip install git+http://github.com/tskit-dev/tsconvert
import sc2ts  # install with python -m pip install git+https://github.com/jeromekelleher/sc2ts

import utils

# Redefine the path to your local dendroscope Java app & chromium app here
dendroscope_binary = (
    "/Applications/Dendroscope/Dendroscope.app/Contents/MacOS/JavaApplicationStub"
)
chromium_binary = "/usr/local/bin/chromium"


class FocalTreeTs:
    """Convenience class to access a single focal tree in a tree sequence"""

    def __init__(self, ts, pos, basetime=None):
        self.tree = ts.at(pos, sample_lists=True)
        self.pos = pos
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
        Note that NextClade also produces a tree with  more samples but no branch lengths
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
    wide_fn = "upgma-full-md-30-mm-3-2021-06-30-recinfo-gisaid-il.ts.tsz"
    long_fn = "upgma-mds-1000-md-30-mm-3-2022-06-30-recinfo-gisaid-il.ts.tsz"

    def plot(self, args):
        raise NotImplementedError()


class Cophylogeny(Figure):
    name = None
    pos = 0  # Position along tree seq to plot trees
    sc2ts_filename = None
    nextstrain_ts_fn = "nextstrain_ncov_gisaid_global_all-time_timetree-2023-01-21.nex"

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

    @staticmethod
    def run_nnet_untangle(trees):
        assert len(trees) == 2
        with tempfile.TemporaryDirectory() as tmpdirname:
            newick_path = os.path.join(tmpdirname, "cophylo.nwk")
            command_path = os.path.join(tmpdirname, "commands.txt")
            with open(newick_path, "wt") as file:
                for tree in trees:
                    print(tree.as_newick(), file=file)
            with open(command_path, "wt") as file:
                print(f"open file='{newick_path}';", file=file)
                print("compute tanglegram method=nnet", file=file)
                print(
                    f"save format=newick file='{newick_path}'", file=file
                )  # overwrite
                print("quit;", file=file)
            subprocess.run([dendroscope_binary, "-g", "-c", command_path])
            order = []
            with open(newick_path, "rt") as newicks:
                for line in newicks:
                    # hack: use the order of `nX encoded in the string
                    order.append([int(n[1:]) for n in re.findall(r"n\d+", line)])
        return order

    def __init__(self, args):
        """
        Defines two simplified tree sequences, focussed on a specific tree. These are
        stored in self.sc2ts and self.nxstr
        """
        sc2ts_arg, basetime = utils.load_tsz(self.ts_dir, self.sc2ts_filename)
        fn = os.path.join(self.ts_dir, self.nextstrain_ts_fn)
        try:
            nextstrain = Nextstrain(fn, span=sc2ts_arg.sequence_length)
        except FileNotFoundError:
            logging.info("Attempting to extract gz compressed file")
            with tempfile.TemporaryDirectory() as tmpdir:
                with gzip.open(fn + ".gz") as f_in:
                    fn = os.path.join(tmpdir, self.nextstrain_ts_fn)
                    with open(fn, "wb") as f_out:
                        shutil.copyfileobj(f_in, f_out)
                    nextstrain = Nextstrain(fn, span=sc2ts_arg.sequence_length)

        # Slow step: find the samples in sc2ts_arg.ts also in nextstrain.ts, and subset
        sc2ts_its, nxstr_its = sc2ts.subset_to_intersection(
            sc2ts_arg, nextstrain.ts, filter_sites=False, keep_unary=True
        )

        logging.info(
            f"Num samples in subsetted ARG={sc2ts_its.num_samples} vs "
            f"NextStrain={nxstr_its.num_samples}"
        )

        # Check first set of samples map
        for u, v in zip(sc2ts_its.samples(), nxstr_its.samples()):
            assert (
                sc2ts_its.node(u).metadata["strain"]
                == nxstr_its.node(v).metadata["strain"]
            )

        ## Filter from entire TS:
        # Some of the samples in sc2_its are recombinants: remove these from both trees
        sc2ts_simp_its = sc2ts_its.simplify(
            sc2ts_its.samples()[0 : nxstr_its.num_samples],
            keep_unary=True,
            filter_nodes=False,
        )
        assert sc2ts_simp_its.num_samples == sc2ts_simp_its.num_samples
        for u, v in zip(sc2ts_simp_its.samples(), nxstr_its.samples()):
            assert (
                sc2ts_simp_its.node(u).metadata["strain"]
                == nxstr_its.node(v).metadata["strain"]
            )

        logging.info(
            f"Removed {sc2ts_its.num_samples-sc2ts_simp_its.num_samples} samples "
            "in sc2 not in nextstrain"
        )

        ## Filter from trees
        # Some samples in sc2ts_simp_its are internal. Remove those from both datasets
        keep = np.array([u for u in sc2ts_simp_its.at(self.pos).leaves()])

        # Change the random seed here to change the untangling start point
        # rng = np.random.default_rng(777)
        # keep = rng.shuffle(keep)
        sc2ts_tip = sc2ts_simp_its.simplify(keep)
        assert nxstr_its.num_trees == 1
        nxstr_tip = nxstr_its.simplify(keep)
        logging.info(
            "Removed internal samples in first tree. Trees now have "
            f"{sc2ts_tip.num_samples} leaf samples"
        )

        # Call the java untangling program
        sc2ts_order, nxstr_order = self.run_nnet_untangle(
            [sc2ts_tip.at(self.pos), nxstr_tip.first()]
        )

        # Align the time in the nextstrain tree to the sc2ts tree
        ns_sc2_time_difference = []
        for s1, s2 in zip(sc2ts_tip.samples(), nxstr_tip.samples()):
            n1 = sc2ts_tip.node(s1)
            n2 = nxstr_tip.node(s2)
            assert n1.metadata["strain"] == n2.metadata["strain"]
            ns_sc2_time_difference.append(n1.time - n2.time)
        dt = timedelta(**{nxstr_tip.time_units: np.median(ns_sc2_time_difference)})

        nxstr_order = list(
            reversed(nxstr_order)
        )  # RH tree rotated so reverse the order

        self.sc2ts = FocalTreeTs(sc2ts_tip.simplify(sc2ts_order), self.pos, basetime)
        self.nxstr = FocalTreeTs(
            nxstr_tip.simplify(nxstr_order), self.pos, basetime - dt
        )

        logging.info(
            f"{self.sc2ts.ts.num_trees} trees in the simplified 'backbone' ARG"
        )

    def plot(self, args):
        prefix = os.path.join("figures", self.name)
        strain_id_map = {
            self.sc2ts.strain(n): n
            for n in self.sc2ts.samples
            if self.sc2ts.strain(n) != ""
        }

        # A few color schemes to try
        cmap = get_cmap("tab20b", 50)
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
                self.sc2ts.timediff(isodate): (isodate[:7] if show else "")
                for isodate, show in {
                    "2020-01-01": True,
                    "2020-02-01": False,
                    "2020-03-01": False,
                    "2020-04-01": True,
                    "2020-05-01": False,
                    "2020-06-01": False,
                    "2020-07-01": True,
                    "2020-08-01": False,
                    "2020-09-01": False,
                    "2020-10-01": True,
                    "2020-11-01": False,
                    "2020-12-01": False,
                    "2021-01-01": True,
                    "2021-02-01": False,
                    "2021-03-01": False,
                    "2021-04-01": True,
                    "2021-05-01": False,
                    "2021-06-01": False,
                    "2021-07-01": True,
                }.items()
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
                self.nxstr.timediff(isodate): (isodate[:7] if show else "")
                for isodate, show in {
                    "2020-01-01": True,
                    "2020-02-01": False,
                    "2020-03-01": False,
                    "2020-04-01": True,
                    "2020-05-01": False,
                    "2020-06-01": False,
                    "2020-07-01": True,
                    "2020-08-01": False,
                    "2020-09-01": False,
                    "2020-10-01": True,
                    "2020-11-01": False,
                    "2020-12-01": False,
                    "2021-01-01": True,
                    "2021-02-01": False,
                    "2021-03-01": False,
                    "2021-04-01": True,
                    "2021-05-01": False,
                    "2021-06-01": False,
                    "2021-07-01": True,
                }.items()
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
        sc2ts_str = f"Sc2ts {self.name[-4:]} ARG: "
        if self.sc2ts.pos == 0:
            sc2ts_str += "first tree"
        else:
            sc2ts_str += f"tree @ position {self.sc2ts.pos}"
        w, h = 900, 800
        mar_in = 0.05
        w_in = w / 96 + mar_in * 2
        h_in = h / 96 + mar_in * 2
        svg_string = (
            f'<svg baseProfile="full" width="{w}" height="{h}" version="1.1" id="main"'
            + ' xmlns="http://www.w3.org/2000/svg" '
            + 'xmlns:ev="http://www.w3.org/2001/xml-events" xmlns:xlink="http://www.w3.org/1999/xlink">'
            + f'<defs><style>@page {{margin: {mar_in}in; padding: 0; size: {w_in:.2f}in {h_in:.2f}in;}} '
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
    sc2ts_filename = "upgma-full-md-30-mm-3-2021-06-30-recinfo-gisaid-il.ts.tsz"
    use_colour = "Pango"


class CophylogenyLong(Cophylogeny):
    name = "supp_cophylogeny_long"
    sc2ts_filename = "upgma-mds-1000-md-30-mm-3-2022-06-30-recinfo-gisaid-il.ts.tsz"
    use_colour = "Pango"


class RecombinationNodeMrcas(Figure):
    name = None
    sc2ts_filename = "upgma-mds-1000-md-30-mm-3-2022-06-30-recinfo-gisaid-il.ts.tsz"
    csv_fn = "breakpoints_{}.csv"
    data_dir = "data"

    def __init__(self, args):
        self.ts, self.basetime = utils.load_tsz(self.data_dir, self.sc2ts_filename)

        prefix = utils.snip_tsz_suffix(self.sc2ts_filename)
        df = pd.read_csv(os.path.join(self.data_dir, self.csv_fn.format(prefix)))
        df["tmrca"] = self.ts.nodes_time[df.mrca.values]
        df["tmrca_delta"] = df.tmrca - self.ts.nodes_time[df.node.values]
        logging.info(f"{len(df)} breakpoints | {len(np.unique(df.node))} re nodes read")
        # Remove potential contaminents
        df = df[df.max_descendant_samples > 1]
        # For all plots, omit the breakpoints which are not in the tree seq but listed
        # in the HMM metadata, see https://github.com/jeromekelleher/sc2ts/issues/121
        self.df = df[df.is_arg_hmm_path_length_consistent == True]
        logging.info(
            f"{len(self.df)} breakpoints | {len(np.unique(self.df.node))} "
            "re nodes initially retained"
        )

    @staticmethod
    def _filter(df):
        return df[
            np.logical_and(
                df.fwd_bck_parents_max_mut_dist == 0, df.is_hmm_mutation_consistent
            )
        ]

    @staticmethod
    def add_common_lines(ax, num, ts, common_proportions):
        v_pos = {k: v for v, k in enumerate(common_proportions.keys())}
        for i, (u, (pango, prop)) in enumerate(common_proportions.items()):
            n_children = len(np.unique(ts.edges_child[ts.edges_parent == u]))
            logging.info(
                f"{ordinal(i+1)} most freq. parent MRCA has id {u} (imputed: {pango}) "
                f"@ time={ts.node(u).time}; "
                f"num_children={n_children}"
            )
            # Find all samples of the focal lineage
            t = ts.node(u).time
            ax.axhline(t, ls=":", c="grey", lw=1)
            sep = "\n" if v_pos[u] == 0 else " "
            most = "Most" if v_pos[u] == 0 else ordinal(i + 1) + " most"
            ax.text(
                (t + 5) / 7,  # Hand tweaked to get nice label positions
                ts.node(u).time,
                f"{most} freq. MRCA,{sep}{n_children} children,{sep}{prop * 100:.1f} % of {pango}",
                fontsize=8,
                va="center",
                bbox=dict(facecolor="white", edgecolor="none", pad=0),
            )

    def do_plot(self, main_ax, hist_ax, df, title, label_tweak, xlab=True, ylab=True):
        logging.info(f"Plotting {title}")
        logging.info(f" {len(df)} points, {len(np.unique(df.node))} rec nodes")
        logging.info(
            f" time diff: min={df.tmrca_delta.min()}, max={df.tmrca_delta.max()}"
        )
        dates = [
            datetime(y, m, 1)
            for y in (2020, 2021, 2022)
            for m in range(1, 13, 3)
            if (self.basetime - datetime(y, m, 1)).days > -2
        ]

        for num_parents_2, colour in zip([True, False], ["green", "blue"]):
            df_ = df[df.num_parents == 2] if num_parents_2 else df[df.num_parents > 2]
            main_ax.scatter(
                df_.tmrca_delta / 7,
                df_.tmrca,
                alpha=0.1,
                c=colour,
                label="num_parents=2" if num_parents_2 else "num_parents>2",
            )
        if xlab:
            hist_ax.set_xlabel("Estimated divergence between lineage pairs (weeks)")
        if ylab:
            main_ax.set_ylabel(f"Estimated MRCA date")
        main_ax.set_title(title)
        main_ax.set_yticks(
            ticks=[(self.basetime - d).days for d in dates],
            labels=[str(d)[:7] for d in dates],
        )
        hist_ax.spines["top"].set_visible(False)
        hist_ax.spines["right"].set_visible(False)
        hist_ax.spines["left"].set_visible(False)
        hist_ax.get_yaxis().set_visible(False)
        hist_ax.hist(df.tmrca_delta / 7, bins=60, density=True)

        x = []
        y = []
        for row in df.itertuples():
            if row.causal_lineage.startswith("X"):
                x.append(row.tmrca_delta / 7)
                y.append(row.tmrca)
                main_ax.text(
                    x[-1] + label_tweak[0],
                    y[-1] + label_tweak[1],  # Tweak so it is above the point
                    row.causal_lineage,
                    size=6,
                    ha="center",
                    rotation=70,
                )
        main_ax.scatter(x, y, c="orange", s=8)


class RecombinationNodeMrcas_all(RecombinationNodeMrcas):
    name = "recombination_node_mrcas"
    num_common_lines = 5

    def plot(self, args):
        prefix = os.path.join("figures", self.name)

        fig, axes = plt.subplots(
            2,
            2,
            figsize=(14, 8),
            sharex=True,
            sharey="row",
            gridspec_kw={"height_ratios": [4, 1]},
        )
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
        proportions = descendant_proportion(self.ts, focal_node_map)
        self.add_common_lines(axes[0][0], 5, self.ts, proportions)
        self.do_plot(
            axes[0][0],
            axes[1][0],
            self.df,
            "A. Unfiltered",
            label_tweak=[1, -8],
        )
        self.do_plot(
            axes[0][1],
            axes[1][1],
            self._filter(self.df),
            "B. After filtering",
            label_tweak=[1, -8],
            ylab=False,
        )
        axes[0][1].legend()
        axes[0][1].invert_yaxis()
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
            f"{'|'.join(restrict)} breakpoints in the “long” ARG",
            label_tweak=[1.3, -13],
            **kwargs,
        )

    def plot(self, args):
        prefix = os.path.join("figures", self.name)

        fig, axes = plt.subplots(
            nrows=4,
            ncols=3,
            figsize=(18, 12),
            sharex=True,
            gridspec_kw={"height_ratios": [4, 1, 4, 1]},
        )
        for row_idx in range(0, axes.shape[0], 2):
            for col_idx in range(axes.shape[1]):
                axes[row_idx, col_idx].set_ylim(
                    self.df.tmrca.min(), self.df.tmrca.max()
                )

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
        self.subplot(axes[2][0], axes[3][0], ["Alpha", "Delta"], parent_variants)
        self.subplot(
            axes[2][1], axes[3][1], ["Alpha", "Omicron"], parent_variants, ylab=False
        )
        self.subplot(
            axes[2][2], axes[3][2], ["Delta", "Omicron"], parent_variants, ylab=False
        )
        axes[0][-1].invert_yaxis()
        axes[2][-1].invert_yaxis()

        plt.savefig(prefix + f".{args.outtype}", bbox_inches="tight")


class RecombinationNodeMrcas_filtered_subset(RecombinationNodeMrcas_subset):
    name = "supp_recombination_node_mrcas_filtered"
    # Only used to get information out really.

    def __init__(self, args):
        super().__init__(args)
        self.df = self.filter(self.df)


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
        # Pango X-lineages in red/brown/pink
        'XA': '#fe812e',
        'XB': '#fc31fb',
        'XQ': '#e780a9',
        'XU': '#c35aab',
        'XAA': '#fdacfe',
        'XAB': '#f2b6b9',
        'XAG': '#fc3131',

        # B lineages in blue/green (AY.100 is B.1.617.2.100)
        # Basic in blue
        'B.1': '#212fe1',
        'B.1.177.18': '#93e8ff',
        'B.1.631': '#add9e5',
        'B.1.634': '#2092e1',
        'B.1.627': '#7b96cc',
        'B.1.1': '#aea4ff',

        'B.1.1.7': '#8666e3', # Alpha

        'AY.100': '#c6c076',  # Delta

        # Omicron (B.1.1.529...) in green-ish
        'BA.1': '#74b058',  # Omicron
        'BA.2': '#4de120',  # Omicron
        'BA.2.9': '#21e1ab',


        None: "lightgray",  # Default

        "Unknown (R)": "k",
        "Unknown": "None",
    }

    @staticmethod
    def grow_graph(ts, input_nodes):
        # Ascend up from input nodes
        up_nodes = sc2ts.node_path_to_samples(input_nodes, ts, ignore_initial=True, stop_at_recombination=True)
        # Descend from these
        nodes = sc2ts.node_path_to_samples(up_nodes, ts, rootwards=False, ignore_initial=False)
        # Ascend again, to get parents of downward nonsamples
        up_nodes = sc2ts.node_path_to_samples(nodes, ts, ignore_initial=False)
        nodes = np.append(nodes, up_nodes)
        # Add parents of recombination nodes up to the nearest sample
        re_nodes = nodes[ts.nodes_flags[nodes] & sc2ts.NODE_IS_RECOMBINANT > 0]
        re_parents = np.unique(ts.edges_parent[np.isin(ts.edges_child, re_nodes)])
        re_ancestors = sc2ts.node_path_to_samples(re_parents, ts, ignore_initial=False)
        nodes = np.append(nodes, re_ancestors)
        # Remove duplicates
        _, idx = np.unique(nodes, return_index=True)
        return nodes[np.sort(idx)]

    def __init__(self, args):
        self.ts, self.basetime = utils.load_tsz(self.ts_dir, self.long_fn)
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

    def __init__(self, args):
        super().__init__(args)
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

    @staticmethod
    def define_nodes(ts, imputed_lineage):
        raise NotImplementedError()

    def used_pango_colours(self, used_nodes):
        used_pango = set([
            self.ts.node(n).metadata.get("Imputed_" + self.imputed_lineage, None)
            for n in used_nodes
        ])
        return {k: v for k, v in self.node_colours.items() if k in used_pango}

    @classmethod
    def save_adjusted_positions(cls, pos, fn):
        cls.adjust_positions(pos)
        logging.info("Saving positions")
        with open(fn, "wt") as f:
            # json.dumps doesn't like np.int's
            f.write(json.dumps({int(k): v for k, v in pos.items()}))

    @staticmethod
    def post_process(ax):
        pass
        
    @staticmethod
    def adjust_positions(pos):
        # Override this to adjust the node positions by altering `pos`
        # e.g. by using change_position
        pass

    def plot_subgraph(self, nodes, ax, treeinfo=None, node_positions=None):
        return sc2ts.plot_subgraph(
            nodes,
            self.ts,
            ti=treeinfo,
            mutations_json_filepath=self.mutations_fn,
            ax=ax,
            exterior_edge_len=self.exterior_edge_len,
            ts_id_labels=self.show_ids,
            node_colours=self.node_colours,
            node_metadata_labels=self.node_metadata_label_key,
            sample_metadata_labels=self.sample_metadata_labels,
            node_size=self.node_size,
            node_font_size=self.node_font_size,
            label_replace=self.label_replace,
            edge_font_size=self.edge_font_size,
            colour_metadata_key="Imputed_" + self.imputed_lineage,
            node_positions=node_positions,
        )

    def plot(self, args):
        fig, ax = plt.subplots(1, 1, figsize=self.figsize)
        nodes = self.define_nodes(self.ts, self.imputed_lineage)
        G, pos = self.plot_subgraph(nodes, ax=ax, node_positions=self.node_positions)
        self.post_process(ax)
        # Could use the following for a legend
        used_colours = self.used_pango_colours(list(pos.keys()))

        if self.node_positions is None:
            self.save_adjusted_positions(pos, self.node_positions_fn)
            fn = self.fn_prefix + "_defaultpos"
        else:
            fn = self.fn_prefix
        plt.savefig(fn + f".{args.outtype}", bbox_inches="tight")

    def make_legend_elements(self, used_colours, size):
        legend_elements = []
        for k, v in used_colours.items():
            if k is None:
                k = "Other"
            elif k == "Unknown (R)":
                k = "Recombination node"
            elif k == "Unknown":
                k = "Pango not imputed"
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
                    marker='o',
                    label=k,
                    linestyle='None',
                    markerfacecolor=v,
                    markeredgecolor=stroke,
                    markersize=size,
                )
            )
        return legend_elements


class Pango_XA_nxcld_tight_graph(Pango_X_tight_graph):
    name = "Pango_XA_nxcld_tight_graph"
    imputed_lineage = "Nextclade_pango"
    figsize=(3, 4)

    @staticmethod
    def define_nodes(ts, imputed_lineage):
        XA = [n.id for n in ts.nodes() if n.metadata.get(imputed_lineage) == "XA"]
        return Pango_X_graph.grow_graph(ts, XA)

    @staticmethod
    def adjust_positions(pos):
        dx = pos[154551][0] - pos[147032][0]
        change_pos(pos, 130298, dx=-dx/2)

    @staticmethod
    def post_process(ax):
        # The points are a bit close to the L and R edges
        x_min, x_max = ax.get_xlim()
        ax.set_xlim(x_min - (x_max - x_min) * 0.15, x_max + (x_max - x_min) * 0.15)

class Pango_XAG_nxcld_tight_graph(Pango_X_tight_graph):
    name = "Pango_XAG_nxcld_tight_graph"
    imputed_lineage = "Nextclade_pango"
    figsize=(11, 4)

    @staticmethod
    def define_nodes(ts, imputed_lineage):
        nodes = Pango_X_graph.grow_graph(ts, [712029])  # This gives a nice layout
        XAG = [n.id for n in ts.nodes() if n.metadata.get(imputed_lineage) == "XAG"]
        assert len(set(XAG) - set(nodes)) == 0
        return nodes

    @staticmethod
    def post_process(ax):
        x_min, x_max = ax.get_xlim()
        ax.set_xlim(x_min + (x_max - x_min) * 0.06, x_max - (x_max - x_min) * 0.06)

class Pango_XB_nxcld_tight_graph(Pango_X_tight_graph):
    name = "Pango_XB_nxcld_tight_graph"
    imputed_lineage = "Nextclade_pango"
    figsize=(16, 6)

    @staticmethod
    def define_nodes(ts, imputed_lineage):
        # Order of nodes here matters. This is found by trial-and-error :(
        basic_nodes = [
            285180, # lone AB on the side
            335592, #
            280287,
            206465,
            334261,
            337869,
            345330,
            394058, # most recent nodes, product of the most recent recombination 
        ]
        for n in basic_nodes:
            pango = ts.node(n).metadata.get("Imputed_" + imputed_lineage)
            if pango != "XB":
                logging.info(f"XB plot: input node {n} is {pango} not XB")
        nodes = Pango_X_graph.grow_graph(ts, basic_nodes)
        if logging.getLogger().isEnabledFor(logging.INFO):
            XB = [n.id for n in ts.nodes() if n.metadata.get(imputed_lineage) == "XB"]
            logging.info(
                f"XB nodes not shown (perhaps below '+N' nodes): {set(XB) - set(nodes)}")
        return nodes

    @staticmethod
    def post_process(ax):
        x_min, x_max = ax.get_xlim()
        ax.set_xlim(x_min + (x_max - x_min) * 0.06, x_max - (x_max - x_min) * 0.06)

    @staticmethod
    def adjust_positions(pos):
        dx = pos[341400][0] - pos[350705][0]
        dy = pos[325792][1] - pos[285180][1]

        # All adjustments below found by tedious trial and error

        change_pos(pos, 351074, xy_from=73385)
        change_pos(pos, 73385, xy_from=322395, dx=5 * dx)
        pos[322395], pos[345330] = pos[345330], pos[322395]
        change_pos(pos, 345330, dx=dx)
        change_pos(pos, 394059, x=pos[345330][0] / 2 + pos[73385][0] / 2)
        change_pos(pos, 394058, xy_from=394059, y=pos[394058][1])
        pos[329950], pos[280287] = pos[280287], pos[329950]
        pos[359092], pos[339516] = pos[339516], pos[359092]
        change_pos(pos, 359092, dx=dx)
        change_pos(pos, 339516, dx=dx)

        change_pos(pos, 392495, dx= -3 * dx, xy_from=339516)

        for p in [380107, 335144, 338964]:
            change_pos(pos, p, dx=-dx, dy=-dy)

        change_pos(pos, 423732, dx=-5 * dx)
        change_pos(pos, 423733, xy_from=423732, y=pos[423733][1])
        for p in [352126, 358835, 379420]:
            change_pos(pos, p, dx=4 * dx)

        change_pos(pos, 300560, dx=dx)
        change_pos(pos, 5731, dx=dx/2)

        change_pos(pos, 285181, dx=1.5 * dx)
        change_pos(pos, 325792, dx=3 * dx)
        change_pos(pos, 282861, dx=5 * dx)
        change_pos(pos, 285180, dx=3 * dx)
        change_pos(pos, 282727, dx=9 * dx)
        change_pos(pos, 265975, dx=10 * dx)
        change_pos(pos, 274176, dx=1.5 * dx)
        change_pos(pos, 305005, dx=-dx)
        change_pos(pos, 358998, dx=dx)

        for p in [273897, 206465]:
            change_pos(pos, p, dx= 4 * dx)

        change_pos(pos, 266716, xy_from=325792, dx=-dx/2, y=pos[266716][1])
        change_pos(pos, 206466, xy_from=266716, dx=dx/2, y=pos[206466][1])
        change_pos(pos, 200603, xy_from=206466, dx=2 * dx, y=pos[200603][1])
        change_pos(pos, 12108, xy_from=206466, dx=-2 * dx, y=pos[12108][1])

        change_pos(pos, 13053, dx=dx/2)
        change_pos(pos, 320601, dx=-dx/2)
        change_pos(pos, 309252, dx=-dx)

        change_pos(pos, 289479, dx=0.3 * dx, dy=dy/2)
        change_pos(pos, 3940, dx=-2 * dx, dy=dy/2)
        change_pos(pos, 1165, dx=-0.2 * dx, dy=dy/2)
        change_pos(pos, 179, dx=-dx, dy=dy/2)

        change_pos(pos, 339517, dx=0.5 * dx)
        change_pos(pos, 338965, dx=-0.5 * dx)
        change_pos(pos, 365162, dx=-0.5 * dx)

        change_pos(pos, 328838, dx=dx)

        change_pos(pos, 341400, xy_from=351074, dx=dx/2, y=pos[341400][1])
        change_pos(pos, 350705, xy_from=351074, dx=-dx/2, y=pos[350705][1])

        change_pos(pos, 334261, xy_from=336014, dx=dx/2, y=pos[334261][1])
        change_pos(pos, 335592, xy_from=336014, dx=-dx/2, y=pos[335592][1])

        change_pos(pos, 328838, dx=-dx)

class Pango_XA_XAG_XB_nxcld_tight_graph(Pango_X_tight_graph):
    name = "Pango_XA_XAG_XB_nxcld_tight_graph"
    imputed_lineage = "Nextclade_pango"
    figsize=(16, 10)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.node_positions is None
        self.node_positions = {}
        self.node_positions_fn = {}
        for Xlin, cls in [
            ("XA", Pango_XA_nxcld_tight_graph),
            ("XAG", Pango_XAG_nxcld_tight_graph),
            ("XB", Pango_XB_nxcld_tight_graph),
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
        width_ratios = [1, 6, 0.9]
        height_ratios=[0.7, 1]
        gs = fig.add_gridspec(2, 3, width_ratios=width_ratios, height_ratios=height_ratios, wspace=0.05, hspace=0.05)
        ax2 = fig.add_subplot(gs[0, 1])
        ax1 = fig.add_subplot(gs[0, 0], sharey=ax2)
        ax3 = fig.add_subplot(gs[1, :])
        ax_legend = fig.add_subplot(gs[0, 2])

        treeinfo = sc2ts.TreeInfo(self.ts)

        fn = self.fn_prefix
        all_nodes = set()
        for Xlin, ax, cls in [
            ("XA", ax1, Pango_XA_nxcld_tight_graph),
            ("XAG", ax2, Pango_XAG_nxcld_tight_graph),
            ("XB", ax3, Pango_XB_nxcld_tight_graph),
        ]:
            logging.info(f"Plotting {Xlin}")
            nodes = cls.define_nodes(self.ts, self.imputed_lineage)
            G, pos = self.plot_subgraph(
                nodes, ax, treeinfo=treeinfo, node_positions=self.node_positions[Xlin])
            all_nodes.update(G.nodes)
            cls.post_process(ax)

            if self.node_positions[Xlin] is None:
                cls.save_adjusted_positions(pos, self.node_positions_fn[Xlin])
                fn = fn + f"_no{Xlin}pos"

        sample_nodes = []
        nonsample_nodes = []
        for n in all_nodes:
            if self.ts.node(n).is_sample():
                sample_nodes.append(n)
            else:
                nonsample_nodes.append(n)

        sample_colours = self.used_pango_colours(sample_nodes)
        nonsample_colours = self.used_pango_colours(nonsample_nodes)
        exclusive_nonsample_colours = {
            k: v for k, v in nonsample_colours.items() if k not in sample_colours}
        sample_node_size = np.sqrt(self.node_size)
        nonsample_node_size = sample_node_size / 3  # Matches the reduction in sc2ts.plot_subgraph


        legend1 = ax_legend.legend(
            title=f"Pango ({self.imputed_lineage.split('_')[0]})",
            handles=self.make_legend_elements(sample_colours, size=sample_node_size),
            loc='upper right',
            bbox_to_anchor=(ax_legend.get_xlim()[1]*1.04, ax_legend.get_ylim()[1]*1.02),
            labelspacing=1.05,
            alignment="left",
            borderpad=0.6,
        )
        ax_legend.margins(x=0, y=0)
        ax_legend.axis('off')
        """
        legend2 = ax_legend.legend(
            title=f"Non-samples (small points)",
            handles=self.make_legend_elements(exclusive_nonsample_colours, size=nonsample_node_size),
            loc='center left',
            alignment="left",
            borderpad=0.7,
        )
        """
        ax1.text(0.01 / (width_ratios[0]/sum(width_ratios)), 1 - 0.04 / (height_ratios[0]/sum(height_ratios)), "XA", fontsize=20, transform=ax1.transAxes)
        ax2.text(0.01 / (width_ratios[1]/sum(width_ratios)), 1 - 0.04  / (height_ratios[0]/sum(height_ratios)), "XAG", fontsize=20, transform=ax2.transAxes)
        ax3.text(0.01, 1 - 0.04 / (height_ratios[1]/sum(height_ratios)), "XB", fontsize=20, transform=ax3.transAxes)

        plt.savefig(fn + f".{args.outtype}", bbox_inches="tight")

large_replace = {
    "Unknown (R)": "Rec\nnode",
    "Unknown": "",
    # Make all the Pango lineages bold
    "XAG": r"$\bf XAG$",
    "XAA": r"$\bf XAA$",
    "XAB": r"$\bf XAB$",
    "XB": r"$\bf XB$",
    "BA.2": r"$\bf BA.2$",
    "AY.100": r"$\bf AY.100$",
    r"$\bf BA.2$.9": r"$\bf BA.2.9$",  # hack, because BA.2.9 already replaced above
    "BA.1": r"$\bf BA.1$",
    # TODO - add the rest of the Pango lineages
}

class Pango_XAG_gisaid_large_graph(Pango_XAG_nxcld_tight_graph):
    name = "Pango_XAG_gisaid_large_graph"
    imputed_lineage = "GISAID_lineage"
    figsize=(44, 16)
    node_size = 2000
    show_ids = None  # Only show for sample nodes
    edge_font_size = 6
    node_font_size = 7.5
    mutations_fn = None
    show_metadata=True
    label_replace = large_replace

class Pango_XA_gisaid_large_graph(Pango_XA_nxcld_tight_graph):
    name = "Pango_XA_gisaid_large_graph"
    imputed_lineage = "GISAID_lineage"
    figsize=(12, 16)
    node_size = 2000
    show_ids = None  # Only show for sample nodes
    edge_font_size = 6
    node_font_size = 7.5
    mutations_fn = None
    show_metadata=True
    label_replace = large_replace

class Pango_XB_gisaid_large_graph(Pango_XB_nxcld_tight_graph):
    name = "Pango_XB_gisaid_large_graph"
    imputed_lineage = "GISAID_lineage"
    figsize=(64, 20)
    node_size = 2000
    show_ids = None  # Only show for sample nodes
    edge_font_size = 6
    node_font_size = 7.5
    mutations_fn = None
    show_metadata=True
    label_replace = large_replace


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


def descendant_proportion(ts, focal_nodes_map):
    """
    Take each focal node which maps to an imputed lineage, and work out how many
    samples of that lineage type have the focal node in their ancestry. If
    it maps to None, use the ts to get the imputed lineage of that node
    """
    focal_lineages = {
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
            for k, v in focal_lineages.items():
                if lineage == v:
                    sample_lists[k].append(nd.id)
    for focal_node, samples in sample_lists.items():
        sts, node_map = ts.simplify(samples, map_nodes=True, keep_unary=True)
        ok = np.zeros(sts.num_samples, dtype=bool)
        assert sts.samples().max() == sts.num_samples - 1
        for tree in sts.trees():
            for u in tree.samples(node_map[focal_node]):
                ok[u] = True
        ret[focal_node] = focal_lineages[focal_node], np.sum(ok) / len(ok)
    return ret


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
        "-o", "--outtype",
        choices=['pdf', 'svg'],
        default='pdf',
        help="The format of the output file",
    )
    parser.add_argument(
        "name",
        type=str,
        help="figure name",
        choices=sorted(list(name_map.keys()) + ["all"]),
    )
    args = parser.parse_args()

    levels = [logging.WARNING, logging.INFO, logging.DEBUG]
    level = levels[min(args.verbosity, len(levels) - 1)]  # cap to last level index
    logging.basicConfig(level=level)

    if args.name == "all":
        for name, fig in name_map.items():
            if fig in figures:
                logging.info(f"plotting {name}")
                fig(args).plot(args)
    else:
        fig = name_map[args.name]
        logging.info(f"plotting {args.name}")
        fig(args).plot(args)


if __name__ == "__main__":
    main()
