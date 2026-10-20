import argparse
from pathlib import Path

import imgkit  # To convert the HTML table to a PNG. Also needs wkhtmltox to be installed
import numpy as np
import pandas as pd
import sc2ts
import sc2ts.debug as sd
import tskit
import tszip

from io import BytesIO
from PIL import Image
from tqdm import tqdm

data_dir = Path(__file__).resolve().parent.parent / "data"
png_dir = Path(__file__).resolve().parent.parent / "figures/static"


ts = tszip.load(data_dir / "sc2ts_viridian_v2.3.trees.tsz")

def pangoX_RE_node_labels():
    """
    Return a dict mapping RE node IDs to labels for PangoX designations.
    exclude_dups=False will also return pangoXs that have multiple RE nodes
    (currently only a few XMs, not including the main XM, which is joint with XAL
    """
    df = pd.read_csv(data_dir / "pango_x_event_details.csv")
    # Type IV events are not recombinants
    df = df[df["type"].isin([1, 2])]
    return {int(row['node']): row['name'] for _, row in df.iterrows()}

def save_copying_pattern_image(html_str, label, save_dir, zoom=None):
    options = {"format": "png", "quiet": "", "transparent": ""}  # use transparent so we can crop
    if zoom is not None:
        options['zoom'] = zoom
    img_bytes = imgkit.from_string(html_str, output_path=False, options=options)
    img = Image.open(BytesIO(img_bytes))
    img = img.convert('RGBA')
    # Crop whitespace
    bbox = img.getbbox()
    img = img.crop(bbox)
    img.save(save_dir / f"{label.replace('/','+')}.png", "PNG", optimize=True, compress_level=9)

def get_copying_table(ts, node_id, **kwargs):
    html_str = sd.CopyingTable(ts, node_id).html(**kwargs)
    html_str = html_str.replace('transform:', '-webkit-transform:')
    return html_str.replace('writing-mode:', '-webkit-writing-mode:')

def save_copying_table_image(ts, node_id, label, save_dir, zoom=None,**kwargs):
    # NB - do not zoom if no ATCG letters are printed, as boxes will be of varying size
    html_str = get_copying_table(ts, node_id, **kwargs)
    html_str = html_str.replace('<style>', '<style>.copying-table th {text-align: right; font-weight: normal;}')
    save_copying_pattern_image(html_str, label, save_dir, zoom=zoom)

# Graph descendant utility functions for the tree sequence
# copied from https://github.com/hyanwong/tskit_extras
def _graph_descendants(ts, node):
    """
    Return a boolean array marking the descendants of ``node`` anywhere
    in the graph, regardless of whether those descendants have ``node``
    as a genetic ancestor in any local tree.

    Note that this builds arrays of starts and ends of the child edges for each
    node, which can also be done by using tskit.jit.numba.NumbaTreeSequence
    for potentially faster descendant queries.
    """
    if node < 0:
        raise ValueError(
            "Negative node IDs are disallowed; did you pass tskit.NULL?"
        )

    is_descendant = np.zeros(ts.num_nodes, dtype=bool)
    parent = ts.edges_parent
    # Edges for a given parent are contiguous in the edge table, so the child
    # edges of each node are found from the boundaries between runs of parents.
    child_start = np.zeros(ts.num_nodes, dtype=np.intp)
    child_end = np.zeros(ts.num_nodes, dtype=np.intp)
    if ts.num_edges > 0:
        run_starts = np.flatnonzero(np.r_[True, parent[1:] != parent[:-1]])
        run_ends = np.r_[run_starts[1:], ts.num_edges]
        run_parents = parent[run_starts]
        child_start[run_parents] = run_starts
        child_end[run_parents] = run_ends
    child = ts.edges_child
    # Code from https://github.com/tskit-dev/tskit/discussions/2869

    stack = [node]
    while stack:
        u = stack.pop()
        for e in range(child_start[u], child_end[u]):
            v = child[e]
            if not is_descendant[v]:
                is_descendant[v] = True
                stack.append(v)
    return is_descendant

def graph_samples(ts, node):
    """
    Return the sample nodes that are graph descendants of ``node``.

    Graph descendants are reachable by following edges, regardless of
    whether the edges overlap in genomic coordinates. If ``node`` is a
    sample, it is included in the result.

    :param tskit.TreeSequence ts: The tree sequence to analyse.
    :param int node: The node ID.
    :return: A sorted array of sample node IDs.
    :rtype: numpy.ndarray
    """
    is_descendant = _graph_descendants(ts, node)
    if ts.nodes_flags[node] & tskit.NODE_IS_SAMPLE:
        is_descendant[node] = True
    samples = ts.samples()
    return samples[is_descendant[samples]]

def num_graph_samples(ts, node):
    """
    Return the number of sample nodes graph-reachable from ``node``,
    including ``node`` if it is a sample.

    :param tskit.TreeSequence ts: The tree sequence to analyse.
    :param int node: The node ID.
    :return: The number of descendant samples.
    :rtype: int
    """
    return len(graph_samples(ts, node))

def main_pangoX(pango=None):
    ####
    # The pango X versions
    ####
    for u, label in tqdm(pangoX_RE_node_labels().items()):
        if pango is not None and label != pango:
            continue
        edges = np.where(ts.edges_child == u)[0]
        parents = {left: parent for left, parent in zip(ts.edges_left[edges], ts.edges_parent[edges])}
        parents = {parents[left]: 1 for left in sorted(parents)}
        if len(parents) > 4:
            print(f"Omitting {label} (node {u}) with more than 4 parents:")
            continue
        child_pango = ts.node(u).metadata["pango"]
        for suffix, rgt_label in [("", f"&nbsp;{u}"), ("-no_nodeid", None)]:
            save_copying_table_image(
                ts,
                u,
                label + suffix,
                png_dir,
                hide_extra_rows=True,
                show_bases=None,
                child_label=label if child_pango==label else f"{label}&nbsp;({child_pango})",
                parent_labels=[ts.node(p).metadata["pango"] for p in parents],
                child_rgt_label=rgt_label,
                font_family='Verdana',
            )

def main_quadrants():
    ####
    # The examples for bad quadrants
    ####
    rec = pd.read_csv(data_dir / "recombinants.csv", index_col=0)
    for u, lab in tqdm([
        (360603, "RE_node-QCpass-360603"),  # Q1
        (420996, "RE_node-QCfail-420996"),  # Q2
        (1876328, "RE_node-QCfail-1876328"),  # Q3
        (374020, "RE_node-QCfail-374020"),  # Q4
    ]):
        row = rec.loc[u]
        save_copying_table_image(
            ts,
            u,
            lab,
            png_dir,
            hide_extra_rows=False,
            show_bases=True,
            zoom=4,
            child_label=row.recombinant_pango,
            parent_labels=[row.parent_left_pango, row.parent_right_pango],
            child_rgt_label=f"&nbsp;Child&nbsp;node&nbsp;#{u}",
            font_family='Verdana',
        )

def main_all():
    ####
    # The html file (and all recombs)
    ####
    rec = pd.read_csv(data_dir / "recombinants.csv", index_col=0)
    with open(data_dir / "copy_patterns.html", "wt") as f:
        print(
            "<html>",
            "<head><style>",
            "@media print {@page {size: A3 landscape;}}",
            "table tr td, table tr th {page-break-inside: avoid;}",
            ".nobreak {page-break-inside: avoid !important; margin-bottom: 5px}",
            ".fail-lft {background-color: gainsboro; background-image: repeating-linear-gradient(-45deg, transparent, transparent 5px, silver 5px, silver 6px);}",
            ".fail-rgt {background-color: gainsboro; background-image: repeating-linear-gradient(45deg, transparent, transparent 5px, silver 5px, silver 6px);}",
            ".fail-lft.fail-rgt {background-color: gainsboro; background-image: repeating-linear-gradient(-45deg, transparent, transparent 5px, silver 5px, silver 6px), repeating-linear-gradient(45deg, transparent, transparent 5px, silver 5px, silver 6px);}",
            "</style></head>",
            "<body>",
            sep="\n",
            file=f
        )
        first = True
        for u in tqdm(np.where(ts.nodes_flags & sc2ts.NODE_IS_RECOMBINANT != 0)[0]):
            css_classes = ["nobreak"]
            try:
                row = rec.loc[u]
                num_descendant_samples = row.num_descendant_samples
                child_label = row.recombinant_pango
                parent_labels = [row.parent_left_pango, row.parent_right_pango]
                if row.net_min_supporting_loci_lft < 4:
                    css_classes.append("fail-lft")
                if row.net_min_supporting_loci_rgt < 4:
                    css_classes.append("fail-rgt")
            except KeyError:
                # These should all be >2 parent recombinants
                use_edges = np.where(ts.edges_child == u)[0]
                parents = []
                for p in sorted(use_edges, key=lambda e: ts.edges_left[e]):
                    if p not in parents:
                        parents.append(p)
                assert len(parents) > 2
                pango = ts.node(u).metadata['pango']
                if len(parents) > 4:
                    print(f"Omitting {pango} (node {u}) with more than 4 parents:")
                    continue

                num_descendant_samples = num_graph_samples(ts, u)
                child_label = pango
                parent_labels = [ts.node(p).metadata['pango'] for p in parents]
                # By default we don't calculate net_min_supporting_loci for > 2 parents,
                # but the 4 recombinants with > 2 parents have been inspected by eye
                # and have >4 supporting loci for each parent.

            if num_descendant_samples == 1:
                sample_txt = "sample"
            else:
                sample_txt = "samples"
            if first:    
                child_rgt_label = (
                    f'<span style="white-space: pre;"> Copying pattern for recombinant child, '
                    f'node #{u} ({num_descendant_samples} descendant {sample_txt})</span>'
                )
            else:
                child_rgt_label = f"&nbsp;#{u}&nbsp;({num_descendant_samples})"
            t = get_copying_table(
                ts,
                u,
                child_label=child_label,
                parent_labels=parent_labels,
                child_rgt_label=child_rgt_label,
                exclude_stylesheet=not first,
            )
            print(f'<div class="{" ".join(css_classes)}">{t}</div>', file=f)
            first = False
            #save_copying_pattern_image(
            #    t, f"RE_node-QC{s}-{u}",
            #    png_dir, child_label=row.sample_id, hide_extra_rows=False, hide_labels=False,
            #    show_bases=True, zoom=4, font_family='Verdana'
            #)
        print("</body>", "</html>", sep="\n", file=f)

def main():
    argparser = argparse.ArgumentParser(
        description="Save copying pattern images and HTML for recombination nodes."
    )
    argparser.add_argument(
        "--pango",
        help="Only generate the PangoX copying pattern for this designation.",
    )
    args = argparser.parse_args()

    main_pangoX(args.pango)
    if args.pango is None:
        main_all()
        main_quadrants()

if __name__ == "__main__":
    main()
