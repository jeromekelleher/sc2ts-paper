import argparse
from pathlib import Path

import imgkit  # To convert the HTML table to a PNG. Also needs wkhtmltox to be installed
import numpy as np
import pandas as pd
import sc2ts.debug as sd
import tszip

from io import BytesIO
from PIL import Image
from tqdm import tqdm

data_dir = Path(__file__).resolve().parent.parent / "data"
png_dir = Path(__file__).resolve().parent.parent / "figures/static"


ts = tszip.load(data_dir / "sc2ts_viridian_v2.2.trees.tsz")

def pangoX_RE_node_labels():
    """
    Return a dict mapping RE node IDs to labels for PangoX designations.
    exclude_dups=False will also return pangoXs that have multiple RE nodes
    (currently only a few XMs, not including the main XM, which is joint with XAL
    """
    df = pd.read_csv(data_dir / "pango_x_event_details.csv")
    return {int(row['recombinant']): row['name'] for _, row in df.iterrows()}

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
        (366701, "RE_node-QCpass-366701"),  # Q1
        (420981, "RE_node-QCfail-420981"),  # Q2
        (1876313, "RE_node-QCfail-1876313"),  # Q3
        (374005, "RE_node-QCfail-374005"),  # Q4
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
        df = rec.sort_index()
        for i, row in tqdm(enumerate(df.itertuples()), total=len(df)):
            css_classes = ["nobreak"]
            if row.net_min_supporting_loci_lft < 4:
                css_classes.append("fail-lft")
            if row.net_min_supporting_loci_rgt < 4:
                css_classes.append("fail-rgt")
            if i == 0:
                exclude_stylesheet = False
                if row.num_descendant_samples == 1:
                    sample_txt = "sample"
                else:
                    sample_txt = "samples"
                child_rgt_label = (
                    f'<span style="white-space: pre;"> Copying pattern for recombinant child, '
                    f'node #{row.Index} ({row.num_descendant_samples} descendant {sample_txt})</span>'
                )
            else:
                exclude_stylesheet = True
                child_rgt_label = f"&nbsp;#{row.Index}&nbsp;({row.num_descendant_samples})"
            t = get_copying_table(
                ts,
                row.Index,
                child_label=row.recombinant_pango,
                parent_labels=[row.parent_left_pango, row.parent_right_pango],
                child_rgt_label=child_rgt_label,
                exclude_stylesheet=i>0,
            )
            print(f'<div class="{" ".join(css_classes)}">{t}</div>', file=f)
            #save_copying_pattern_image(
            #    t, f"RE_node-QC{s}-{row.recombinant}",
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
        main_quadrants()
        main_all()

if __name__ == "__main__":
    main()