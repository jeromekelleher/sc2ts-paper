import argparse
import collections
from datetime import datetime, timedelta
import hashlib
import re
import sys
import os
import tempfile
import subprocess
import logging

from IPython.display import SVG, HTML
from matplotlib.colors import rgb2hex
from matplotlib.cm import get_cmap
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
dendroscope_binary = "/Applications/Dendroscope/Dendroscope.app/Contents/MacOS/JavaApplicationStub"
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
        return getattr(self.basetime - datetime.fromisoformat(isodate), self.ts.time_units)

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
            min_edge_length=0.0001 * 1/365,
            span=span,
        )
        # Remove "samples" without names
        keep = [n.id for n in ts.nodes() if n.is_sample() and "strain" in n.metadata]
        self.ts = ts.simplify(keep)

    @staticmethod
    def pango_names(ts):
        # This is relevant to any nextstrain tree seq, not just the stored one
        return {n.metadata.get("comment", {}).get("pango_lineage", "") for n in ts.nodes()}    

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
        return("alpha")
    if pango.startswith("B.1.351"):
        return "beta"
    if pango.startswith("P.1"):
        return "gamma"
    if pango.startswith("AY") or pango == "B.1.617.2":
        return("delta")
    if pango == "B.1.427" or pango == "B.1.429":
        return "epsilon"
    if pango == "B.1.526":
        return "iota"
    if pango == "B.1.617.1":
        return "kappa"
    if pango == "B.1.526":
        return "iota"
    if pango == "B.1.621" or pango == "B.1.621.1":
        return "mu"
    if pango.startswith("P.2"):
        return "zeta"
    if pango == "B.1.1.529" or pango.startswith("BA."):
        return "omicron"
    return("")


class Figure:
    """
    Superclass for creating figures. Each figure is a subclass
    """
    name = None
    ts_dir = "data"
    wide_fn = "upgma-full-md-30-mm-3-2021-06-30-recinfo-il.ts.tsz"
    long_fn = "upgma-mds-1000-md-30-mm-3-2022-06-30-recinfo-il.ts.tsz"

    def plot(self):
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
                print(f"save format=newick file='{newick_path}'", file=file) # overwrite
                print("quit;", file=file)
            subprocess.run([dendroscope_binary, "-g", "-c", command_path])
            order = []
            with open(newick_path, "rt") as newicks:
                for line in newicks:
                    # hack: use the order of `nX encoded in the string
                    order.append([int(n[1:]) for n in re.findall(r'n\d+', line)])
        return order

    def __init__(self, args):
        """
        Defines two simplified tree sequences, focussed on a specific tree. These are
        stored in self.sc2ts and self.nxstr
        """
        sc2ts_arg, basetime = utils.load_tsz(self.ts_dir, self.sc2ts_filename)
        nextstrain = Nextstrain(self.nextstrain_ts_fn, span=sc2ts_arg.sequence_length)
        
        # Slow step: find the samples in sc2ts_arg.ts also in nextstrain.ts, and subset
        sc2ts_its, nxstr_its = sc2ts.subset_to_intersection(
            sc2ts_arg, nextstrain.ts, filter_sites=False, keep_unary=True)
            
        logging.info(
            f"Num samples in subsetted ARG={sc2ts_its.num_samples} vs "
            f"NextStrain={nxstr_its.num_samples}"
        )
        
        # Check first set of samples map
        for u, v in zip(sc2ts_its.samples(), nxstr_its.samples()):
            assert sc2ts_its.node(u).metadata["strain"] == nxstr_its.node(v).metadata["strain"]
        
        ## Filter from entire TS:
        # Some of the samples in sc2_its are recombinants: remove these from both trees
        sc2ts_simp_its = sc2ts_its.simplify(
            sc2ts_its.samples()[0:nxstr_its.num_samples],
            keep_unary=True,
            filter_nodes=False)
        assert sc2ts_simp_its.num_samples == sc2ts_simp_its.num_samples
        for u, v in zip(sc2ts_simp_its.samples(), nxstr_its.samples()):
            assert sc2ts_simp_its.node(u).metadata["strain"] == nxstr_its.node(v).metadata["strain"]
    
        logging.info(
            f"Removed {sc2ts_its.num_samples-sc2ts_simp_its.num_samples} samples "
            "in sc2 not in nextstrain"
        )
    
        ## Filter from trees
        # Some samples in sc2ts_simp_its are internal. Remove those from both datasets
        keep = np.array([u for u in sc2ts_simp_its.at(self.pos).leaves()])
    
        # Change the random seed here to change the untangling start point
        #rng = np.random.default_rng(777)
        #keep = rng.shuffle(keep)
        sc2ts_tip = sc2ts_simp_its.simplify(keep)
        assert nxstr_its.num_trees == 1
        nxstr_tip = nxstr_its.simplify(keep)
        logging.info(
            "Removed internal samples in first tree. Trees now have "
            f"{sc2ts_tip.num_samples} leaf samples"
        )
        
        # Call the java untangling program
        sc2ts_order, nxstr_order = self.run_nnet_untangle(
            [sc2ts_tip.at(self.pos), nxstr_tip.first()])
    
        # Align the time in the nextstrain tree to the sc2ts tree
        ns_sc2_time_difference = []
        for s1, s2 in zip(
            sc2ts_tip.samples(),
            nxstr_tip.samples()
        ):
            n1 = sc2ts_tip.node(s1)
            n2 = nxstr_tip.node(s2)
            assert n1.metadata["strain"] == n2.metadata["strain"]
            ns_sc2_time_difference.append(n1.time - n2.time)
        dt = timedelta(**{nxstr_tip.time_units: np.median(ns_sc2_time_difference)})
    
        nxstr_order = list(reversed(nxstr_order))  # RH tree rotated so reverse the order

        self.sc2ts = FocalTreeTs(
            sc2ts_tip.simplify(sc2ts_order), self.pos, basetime)
        self.nxstr = FocalTreeTs(
            nxstr_tip.simplify(nxstr_order), self.pos, basetime - dt)

        logging.info(f"{self.sc2ts.ts.num_trees} trees in the simplified 'backbone' ARG")

    def plot(self):
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
            "Nextclade": {"md_key": "clade_membership", "scheme": sc2ts.ns_clade_colours},
            "PangoMpl": {"md_key": "pango_lineage", "scheme": {
                    k: rgb2hex(cmap(i)) for i, k in enumerate(pango)}
            },
            "PangoB.1.1": {"md_key": "pango_lineage", "scheme": {
                k: ("#FF0000" if k == ("B.1.1") else "#000000") for i, k in enumerate(pango)}
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
                    legend[clade] = col['scheme'][clade]
                    nxstr_styles.append(
                        f".nxstr .n{n.id} .edge {{stroke: {col['scheme'][clade]}}}")
                    s = self.nxstr.strain(n.id)
                    if s in strain_id_map:
                        sc2ts_styles.append(
                            f".sc2ts .n{strain_id_map[s]} .edge {{stroke: {col['scheme'][clade]}}}")
        
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
        
        shared_split_keys = set(nxstr_hashes.keys()).intersection(set(sc2ts_hashes.keys()))
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
                        pango = node.metadata.get("comment", {}).get("pango_lineage", "")
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
            node_labels=node_labels['sc2ts'],
            root_svg_attributes = {"class": "sc2ts"},
            mutation_labels={},
            omit_sites=True,
            symbol_size=1,
            y_axis=True,
            y_ticks={
                self.sc2ts.timediff(isodate): (isodate[:7] if show else "")
                for isodate, show in {
                    '2020-01-01': True,
                    '2020-02-01': False,
                    '2020-03-01': False,
                    '2020-04-01': True,
                    '2020-05-01': False,
                    '2020-06-01': False,
                    '2020-07-01': True,
                    '2020-08-01': False,
                    '2020-09-01': False,
                    '2020-10-01': True,
                    '2020-11-01': False,
                    '2020-12-01': False,
                    '2021-01-01': True,
                    '2021-02-01': False,
                    '2021-03-01': False,
                    '2021-04-01': True,
                    '2021-05-01': False,
                    '2021-06-01': False,
                    '2021-07-01': True,
                }.items()
            },
            y_label=" ",
        )
        
        svg2 = self.nxstr.tree.draw_svg(
            size=(800, 400),
            canvas_size=(900, 800),  # Allow for time axis at the other side of the tree
            node_labels = node_labels['nxstr'],
            root_svg_attributes = {"class": "nxstr"},
            mutation_labels={},
            omit_sites=True,
            symbol_size=1,
            y_axis=True,
            y_ticks={
                self.nxstr.timediff(isodate): (isodate[:7] if show else "")
                for isodate, show in {
                    '2020-01-01': True,
                    '2020-02-01': False,
                    '2020-03-01': False,
                    '2020-04-01': True,
                    '2020-05-01': False,
                    '2020-06-01': False,
                    '2020-07-01': True,
                    '2020-08-01': False,
                    '2020-09-01': False,
                    '2020-10-01': True,
                    '2020-11-01': False,
                    '2020-12-01': False,
                    '2021-01-01': True,
                    '2021-02-01': False,
                    '2021-03-01': False,
                    '2021-04-01': True,
                    '2021-05-01': False,
                    '2021-06-01': False,
                    '2021-07-01': True,
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
            lft_rel_time = (self.sc2ts.tree.time(lft_node)-min_lft_time) / (self.sc2ts.tree.time(self.sc2ts.tree.root)-min_lft_time)
            rgt_node = names_rgt[nm]
            rgt_rel_time = (self.nxstr.tree.time(rgt_node)-min_rgt_time) / (self.nxstr.tree.time(self.nxstr.tree.root)-min_rgt_time)
            loc[nm]={
                'lft': (370 - lft_rel_time * 340, 763 - lft_node * ((800 - 77) / self.sc2ts.ts.num_samples) - 22),
                'rgt':(430 + rgt_rel_time * 340, rgt_node * ((800 - 77) / self.nxstr.ts.num_samples) + 22)
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
        if self.sc2ts.pos == 0:
            pos_str = "first tree"
        else:
            pos_str = f"tree @ position {self.sc2ts.pos}"
        svg_string = (
            '<svg baseProfile="full" height="800" version="1.1" width="900" id="main"' +
            ' xmlns="http://www.w3.org/2000/svg" ' +
            'xmlns:ev="http://www.w3.org/2001/xml-events" xmlns:xlink="http://www.w3.org/1999/xlink">' +
            f'<defs><style>{"".join(global_styles)}</style></defs>'
            f'<text text-anchor="middle" transform="translate(200, 12)">SC2ts {pos_str}</text>' +
            '<text text-anchor="middle" transform="translate(600, 12)">Nextstrain tree</text>' +
            '<g>' + ''.join([
                f'<line x1="{v["lft"][0]}" y1="{v["lft"][1]}" x2="{v["rgt"][0]}" y2="{v["rgt"][1]}" stroke="#CCCCCC" />'
                for v in loc.values()
                ])+
            '</g>' +
            '<g class="left_tree" transform="translate(0 800) rotate(-90)">' +
            svg1 +
            '</g><g class="right_tree" transform="translate(800 -37) rotate(90)">' +
            svg2 +
            '</g>' + 
            '<g class="legend" transform="translate(800 30)">' +
            f'<text>{self.use_colour} lineage</text>' +
            "".join(f'<line x1="0" y1="{25+i*15}" x2="15" y2="{25+i*15}" stroke-width="2" stroke="{legend[nm]}" /><text font-size="10pt" x="20" y="{30+i*15}">{nm}</text>' for i, nm in enumerate(sorted(legend))) +
            '</g>' + 
            '</svg>'
        )
    
        with open(f"{prefix}.svg", "wt") as file:
            file.write(svg_string)
        subprocess.run([
            chromium_binary,
            "--headless",
            "--disable-gpu",
            "--run-all-compositor-stages-before-draw",
            "--print-to-pdf-no-header",
            f"--print-to-pdf={prefix}.pdf",
            f"{prefix}.svg",
        ])


class CophylogenyWide(Cophylogeny):
    name = "cophylogeny_wide"
    sc2ts_filename = "upgma-full-md-30-mm-3-2021-06-30-recinfo-il.ts.tsz"
    use_colour = "Pango"


class CophylogenyLong(Cophylogeny):
    name = "supp_cophylogeny_long"
    sc2ts_filename = "upgma-mds-1000-md-30-mm-3-2022-06-30-recinfo-il.ts.tsz"
    use_colour = "Pango"


class RecombinationNodeMrcas(Figure):
    name = None
    sc2ts_filename = "upgma-mds-1000-md-30-mm-3-2022-06-30-recinfo-il.ts.tsz"
    csv_fn = "breakpoints_{}.csv"
    data_dir = "data"
    
    
    def __init__(self, args):
        prefix = utils.snip_tsz_suffix(self.sc2ts_filename)
        self.df = pd.read_csv(os.path.join(self.data_dir, self.csv_fn.format(prefix)))
        self.ts, self.basetime = utils.load_tsz(self.data_dir, self.sc2ts_filename)

    @staticmethod
    def add_common_lines(ax, num, ts, common_proportions):
        v_pos = {k: v for v, k in enumerate(common_proportions.keys())}
        for i, (u, (pango, prop)) in enumerate(common_proportions.items()):
            n_children = len(np.unique(ts.edges_child[ts.edges_parent == u]))
            logging.info(
                f"{ordinal(i+1)} most common parent MRCA has id {u} (imputed: {pango}) "
                f"@ time={ts.node(u).time}; "
                f"num_children={n_children}"
            )
            # Find all samples of the focal lineage
            t = ts.node(u).time
            ax.axhline(t, ls=":", c="grey", lw=1)
            sep = "\n" if v_pos[u] == 0 else " "
            ax.text(
                t/1.3 + 210,
                ts.node(u).time,
                f"Node {u},{sep}{n_children} children,{sep}{prop * 100:.1f} % of {pango}",
                fontsize=8,
                va="center",
                bbox=dict(facecolor='white', edgecolor='none', pad=0)
            )

    def do_plot(self, main_ax, hist_ax, df, title, label_tweak, xlab=True, ylab=True):
        dates = [
            datetime(y, m, 1) 
            for y in (2020, 2021, 2022)
            for m in range(1, 13, 3)
            if (self.basetime-datetime(y, m, 1)).days > -2
        ]
        main_ax.scatter(
            df.tmrca_delta,
            df.tmrca,
            alpha=0.1,
            c=np.array(["blue", "green"])[df.hmm_consistent.astype(int)],
        )
        if xlab:
            hist_ax.set_xlabel("Divergence between parents of a recombinant (days)")
        if ylab:
            main_ax.set_ylabel(f"Date of parental MRCA")
        main_ax.set_title(title)
        main_ax.set_yticks(
            ticks=[(self.basetime-d).days for d in dates],
            labels=[str(d)[:7] for d in dates],
        )
        hist_ax.spines['top'].set_visible(False)
        hist_ax.spines['right'].set_visible(False)
        hist_ax.spines['left'].set_visible(False)
        hist_ax.get_yaxis().set_visible(False)
        hist_ax.hist(df.tmrca_delta, bins=60, density=True)

        
        x = []
        y = []
        for row in df.itertuples():
            if row.origin_nextclade_pango.startswith("X"):
                x.append(row.tmrca_delta)
                y.append(row.tmrca)
                main_ax.text(
                    x[-1] + label_tweak[0],
                    y[-1] + label_tweak[1],  # Tweak so it is above the point
                    row.origin_nextclade_pango,
                    size=6,
                    ha='center',
                    rotation=70,
                )
        main_ax.scatter(x, y, c="orange", s=8)
        main_ax.invert_yaxis()


class RecombinationNodeMrcas_all(RecombinationNodeMrcas):
    name = "recombination_node_mrcas"
    num_common_lines = 5
    def plot(self):
        prefix = os.path.join("figures", self.name)

        fig, axes = plt.subplots(
            2, figsize=(10, 8), sharex=True, gridspec_kw={'height_ratios': [4, 1]})

        mrca_counts = collections.Counter(self.df.parents_mrca)
        common_mrcas = mrca_counts.most_common(self.num_common_lines)
        logging.info(
            "Calculating proportions of descendants for "
            f"{['mrca: {} ({} counts)'.format(id, c) for id, c in common_mrcas]}")
        focal_node_map = {c[0]: None for c in common_mrcas}
        focal_node_map[519829] = "BA.1"
        # 5868 is a common MRCA almost exactly contemporary with the most common node
        del focal_node_map[5868]
        proportions = descendant_proportion(self.ts, focal_node_map)
        self.add_common_lines(axes[0], 5, self.ts, proportions)
        self.do_plot(
            axes[0],
            axes[1],
            self.df,
            "Parental lineages of recombination nodes in the “Long” ARG",
            label_tweak=[6, -8],
        )
        plt.savefig(prefix + ".pdf", bbox_inches='tight')

class RecombinationNodeMrcas_subset(RecombinationNodeMrcas):
    name = "supp_recombination_node_mrcas"

    def plot(self):
        prefix = os.path.join("figures", self.name)

        fig, axes = plt.subplots(
            nrows=4,
            ncols=3,
            figsize=(18, 12),
            sharex=True,
            gridspec_kw={'height_ratios': [4, 1, 4, 1]},
        )
        for row_idx in range(0, axes.shape[0], 2):
            for col_idx in range(axes.shape[1]):
                axes[row_idx, col_idx].set_ylim(self.df.tmrca.min(), self.df.tmrca.max())

        parent_variants = [
            {variant_name(row.left_parent_pango), variant_name(row.right_parent_pango)}
            for row in self.df.itertuples()
        ]
        label_tweak=[10, -13]

        #mrca_counts = collections.Counter(self.df.parents_mrca)
        #self.add_common_lines(axes[0], mrca_counts, 5, self.ts)
        self.do_plot(
            axes[0][0],
            axes[1][0],
            self.df[[v=={'alpha', 'alpha'} for v in parent_variants]],
            "alpha + alpha rec nodes in the “Long” ARG",
            label_tweak=label_tweak,
            xlab=False)

        self.do_plot(
            axes[0][1],
            axes[1][1],
            self.df[[v=={'delta', 'delta'} for v in parent_variants]],
            "delta + delta rec nodes in the “Long” ARG",
            label_tweak=label_tweak,
            xlab=False,
            ylab=False,
        )

        self.do_plot(
            axes[0][2],
            axes[1][2],
            self.df[[v=={'omicron', 'omicron'} for v in parent_variants]],
            "omicron + omicron rec nodes in the “Long” ARG",
            label_tweak=label_tweak,
            xlab=False,
            ylab=False,
        )

        self.do_plot(
            axes[2][0],
            axes[3][0],
            self.df[[v=={'alpha', 'delta'} for v in parent_variants]],
            "alpha + delta rec nodes in the “Long” ARG",
            label_tweak=label_tweak,
        )

        self.do_plot(
            axes[2][1],
            axes[3][1],
            self.df[[v=={'alpha', 'omicron'} for v in parent_variants]],
            "alpha + omicron rec nodes in the “Long” ARG",
            label_tweak=label_tweak,
            ylab=False,
        )

        self.do_plot(
            axes[2][2],
            axes[3][2],
            self.df[[v=={'delta', 'omicron'} for v in parent_variants]],
            "delta + omicron rec nodes in the “Long” ARG",
            label_tweak=label_tweak,
            ylab=False,
        )
        plt.savefig(prefix + ".pdf", bbox_inches='tight')


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
    return ["first", "second", "third", "fourth", "fifth", "sixth", "seventh"][n - 1]

def descendant_proportion(ts, focal_nodes_map):
    """
    Take each focal node which maps to an imputed lineage, and work out how many
    samples of that lineage type have the focal node in their ancestry. If
    it maps to None, use the ts to get the imputed lineage of that node
    """
    focal_lineages = {
        u: ts.node(u).metadata['Imputed_lineage'] if lin is None else lin
        for u, lin in focal_nodes_map.items()}
    sample_lists = {u: [] for u in focal_nodes_map}
    # The time consuming step is finding the samples of each lineage type
    # this would probably be quicker is we could be bothered to do it via TreeInfo
    # but we don't have that calculated in the plotting code
    ret = {}
    for nd in tqdm.tqdm(ts.nodes(), desc="Finding sample lists"):
        if nd.is_sample():
            lineage = nd.metadata.get('Nextclade_pango', "")
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
    parser.add_argument('-v', '--verbosity', action='count', default=0) 
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
        for _, fig in name_map.items():
            if fig in figures:
                fig(args).plot()
    else:
        fig = name_map[args.name](args)
        fig.plot()


if __name__ == "__main__":
    main()