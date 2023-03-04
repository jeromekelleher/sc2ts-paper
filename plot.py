from datetime import datetime, timedelta
import hashlib
import re
import sys
from dataclasses import dataclass
import os
import tempfile
import subprocess


from IPython.display import SVG, HTML
from matplotlib.colors import rgb2hex
from matplotlib.cm import get_cmap
import numpy as np
import tszip
import tskit

import tsconvert  # Not on pip. Install with python -m pip install git+http://github.com/tskit-dev/tsconvert
import sc2ts  # install with python -m pip install git+https://github.com/jeromekelleher/sc2ts

# Redefine the path to your local dendroscope Java app & chromium app here
dendroscope_binary = "/Applications/Dendroscope/Dendroscope.app/Contents/MacOS/JavaApplicationStub"
chromium_binary = "/usr/local/bin/chromium"

@dataclass
class ARG:
    """Associate a tree sequence with a ."""
    ts: tskit.TreeSequence
    day_0: datetime

class FocalTreeTs:
    """Convenience class to access a single focal tree in a tree sequence"""
    def __init__(self, ts, pos, day_0=None):
        self.tree = ts.at(pos, sample_lists=True)
        self.pos = pos
        self.day_0 = day_0

    @property
    def ts(self):
        return self.tree.tree_sequence

    @property
    def samples(self):
        return self.tree.tree_sequence.samples()
        
    def timediff(self, isodate):
        return getattr(self.day_0 - datetime.fromisoformat(isodate), self.ts.time_units)

    def strain(self, u):
        return self.tree.tree_sequence.node(u).metadata.get("strain", "")
        
    def hash_samples_under_node(self, u):
        b2b = hashlib.blake2b(
            " ".join(sorted([self.strain(s) for s in self.tree.samples(u)])).encode(),
            digest_size=20,
        )
        return b2b.digest()

        
    
def load_tsz_file(date, filename):
    return ARG(
        tszip.decompress("data/" + filename.format(date)),
        datetime.fromisoformat(date),
    )

def load_nextstrain(filename, span):
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
        sc2ts.extract_newick_from_nextstrain_nexus(filename),
        min_edge_length=0.0001 * 1/365,
        span=span,
    )
    # Remove "samples" without names
    keep = [n.id for n in ts.nodes() if n.is_sample() and "strain" in n.metadata]
    return ts.simplify(keep)

def pango_names(ts):
    return {n.metadata.get("comment", {}).get("pango_lineage", "") for n in ts.nodes()}    


class Cophylogeny:

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

    def __init__(self, sc2ts_arg, nextstrain_ts, pos):
        """
        Return two simplified trees, which use the timescale in arg.ts
        """
        sc2ts_its, nxstr_its = sc2ts.subset_to_intersection(
            sc2ts_arg.ts, nextstrain_ts, filter_sites=False, keep_unary=True)
            
        print(
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
    
        print(
            "Removed",
            sc2ts_its.num_samples-sc2ts_simp_its.num_samples,
            "samples in sc2 not in nextstrain",
        )
    
        ## Filter from trees
        # Some samples in sc2ts_simp_its are internal. Remove those from both datasets
        keep = np.array([u for u in sc2ts_simp_its.at(pos).leaves()])
    
        # Change the random seed here to change the untangling start point
        #rng = np.random.default_rng(777)
        #keep = rng.shuffle(keep)
        sc2ts_tip = sc2ts_simp_its.simplify(keep)
        assert nxstr_its.num_trees == 1
        nxstr_tip = nxstr_its.simplify(keep)
        print(
            "Removed internal samples in first tree. Trees now have",
            sc2ts_tip.num_samples,
            "leaf samples"
        )
        
        # Call the java untangling program
        sc2ts_order, nxstr_order = self.run_nnet_untangle(
            [sc2ts_tip.at(pos), nxstr_tip.first()])
    
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

        self.sc2ts = FocalTreeTs(sc2ts_tip.simplify(sc2ts_order), pos, sc2ts_arg.day_0)
        self.nxstr = FocalTreeTs(nxstr_tip.simplify(nxstr_order), pos, sc2ts_arg.day_0 - dt)

    def plot(self, prefix, use_colour = "Pango"):
        # Slow step: find the samples in arg.ts also in nextstrain_ts, and subset
    
        strain_id_map = {
            self.sc2ts.strain(n): n
            for n in self.sc2ts.samples
            if self.sc2ts.strain(n) != ""
        }
    
        # A few color schemes to try
        cmap = get_cmap("tab20b", 50)
        pango = pango_names(self.nxstr.ts)
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
        col = colours[use_colour]
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
        
        svg_string = (
            '<svg baseProfile="full" height="800" version="1.1" width="900" id="main"' +
            ' xmlns="http://www.w3.org/2000/svg" ' +
            'xmlns:ev="http://www.w3.org/2001/xml-events" xmlns:xlink="http://www.w3.org/1999/xlink">' +
            f'<defs><style>{"".join(global_styles)}</style></defs>'
            f'<text text-anchor="middle" transform="translate(200, 12)">SC2ts tree @ position {self.sc2ts.pos}</text>' +
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
            f'<text>{use_colour} lineage</text>' +
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
        
if __name__ == "__main__":

    wide = load_tsz_file("2021-06-30", "upgma-full-md-30-mm-3-{}-recinfo-il.ts.tsz")
    long = load_tsz_file("2022-06-30", "upgma-mds-1000-md-30-mm-3-{}-recinfo-il.ts.tsz")
    nextstrain_ts = load_nextstrain(
        "data/nextstrain_ncov_gisaid_global_all-time_timetree-2023-01-21.nex",
        span=wide.ts.sequence_length,
    )
    
    # TODO use argparse to fire off different functions to create each plot

    cophylo = Cophylogeny(
        wide, nextstrain_ts, pos=int(np.mean(sc2ts.core.get_gene_coordinates()["S"])))
    cophylo.plot(prefix="figures/cophylogeny") 
