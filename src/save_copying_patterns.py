from pathlib import Path
import numpy as np
import pandas as pd
import imgkit  # To convert the HTML table to a PNG. Also needs wkhtmltox to be installed
import sc2ts.debug as sd
import tszip
from PIL import Image
from io import BytesIO

from tqdm import tqdm

data_dir = Path(__file__).resolve().parent.parent / "data"
png_dir = Path(__file__).resolve().parent.parent / "figures/static"


ts = tszip.load(data_dir / "sc2ts_viridian_v1.2.trees.tsz")

def pangoX_RE_node_labels(exclude_dups=True):
    """
    Return a dict mapping RE node IDs to labels for PangoX designations.
    exclude_dups=False will also return pangoXs that have multiple RE nodes
    (currently only a few XMs, not including the main XM, which is joint with XAL
    """
    # Only label the Pango X origins as listed in Table 1
    # First find the nodes clearly associated with a RE event
    pango_x_events = pd.read_csv(data_dir / "pango_x_events.csv")
    # Get number of descendants for RE node so we can exclude those with >1e5 descendants, i.e. BA.5
    recombinants = pd.read_csv(data_dir / "recombinants.csv").set_index('recombinant')
    closest_recombinant_num_samples = -np.ones(len(pango_x_events), dtype=int)
    use = (pango_x_events.closest_recombinant >= 0).values
    closest_recombinant_num_samples[use] = recombinants.loc[
        pango_x_events.closest_recombinant[use],
        'num_descendant_samples'
    ]
    # Exclude non-recombinants, and nodes where closes RE node is BA.5
    pango_x = pango_x_events[np.logical_and(
        closest_recombinant_num_samples > 0,
        closest_recombinant_num_samples < 1e5
    )]

    def makelabel(arr):
        arr = sorted(arr.values, key=lambda x: (len(x), x))
        if len(arr) == 1:
            return arr[0]
        if len(arr) == 2:
            return arr[0] + "+" + arr[1]
        else:
            return arr[0] + "++"

    px = pango_x.groupby('closest_recombinant')['root_pango'].apply(makelabel)
    if exclude_dups:
        return dict(px.drop_duplicates(keep=False))
    else:
        return dict(px)

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
    img.save(save_dir / f"{label}.png", "PNG", optimize=True, compress_level=9)

def get_copying_table(ts, node_id, **kwargs):
    html_str = sd.CopyingTable(ts, node_id).html(**kwargs)
    html_str = html_str.replace('transform:', '-webkit-transform:')
    return html_str.replace('writing-mode:', '-webkit-writing-mode:')

def save_copying_table_image(ts, node_id, label, save_dir, zoom=None,**kwargs):
    # NB - do not zoom if no ATCG letters are printed, as boxes will be of varying size
    html_str = get_copying_table(ts, node_id, **kwargs)
    html_str = html_str.replace('<style>', '<style>.copying-table th {text-align: right; font-weight: normal;}')
    save_copying_pattern_image(html_str, label, save_dir, zoom=zoom)

rec = pd.read_csv(data_dir / "recombinants.csv").set_index('recombinant')

####
# The pango X versions
####
for u, label in tqdm(pangoX_RE_node_labels().items()):
    row = rec.loc[u]
    for suffix, rgt_label in [("", f"&nbsp;{u}"), ("-no_nodeid", None)]:
        save_copying_table_image(
            ts,
            u,
            label + suffix,
            png_dir,
            hide_extra_rows=True,
            show_bases=None,
            child_label=row.recombinant_pango,
            parent_labels=[row.parent_left_pango, row.parent_right_pango],
            child_rgt_label=rgt_label,
            font_family='Verdana',
    )

####
# The examples for bad quadrants
####
for u, lab in tqdm([
    (427863, "RE_node-QCpass-427863"),  # Q1
    (748991, "RE_node-QCfail-748991"),  # Q2
    (411345, "RE_node-QCfail-411345"),  # Q3
    (663484, "RE_node-QCfail-663484"),  # Q4
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

####
# The html file (and all recombs)
####
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
    for i, row in tqdm(enumerate(df.itertuples())):
        css_classes = ["nobreak"]
        if row.net_min_supporting_loci_lft < 4:
            css_classes.append("fail-lft")
        if row.net_min_supporting_loci_rgt < 4:
            css_classes.append("fail-rgt")
        if i == 0:
            exclude_stylesheet = False
            child_rgt_label = f'<span style="white-space: pre;"> Copying pattern for recombinant child, node #{row.Index}</span>'
        else:
            exclude_stylesheet = True
            child_rgt_label = f"&nbsp;#{row.Index}"
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
