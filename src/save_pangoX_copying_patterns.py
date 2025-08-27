from pathlib import Path
import numpy as np
import pandas as pd
import imgkit  # To convert the HTML table to a PNG. Also needs wkhtmltox to be installed
import sc2ts
import tszip
from PIL import Image
from io import BytesIO

from tqdm import tqdm

data_dir = Path(__file__).resolve().parent.parent / "data"
png_dir = Path(__file__).resolve().parent.parent / "figures/static"


ts = tszip.load(data_dir / "sc2ts_viridian_v1.1.trees.tsz")

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
    # Exclude non-recombinants, and ondes where closes RE node is BA.5
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


for node_id, label in tqdm(pangoX_RE_node_labels().items()):
    img_bytes = imgkit.from_string(
    sc2ts.info.CopyingTable(ts, node_id).html(hide_extra_rows=True, hide_labels=True, show_bases=None),
        False,  # return the bytes, rather than saving to file
        options={"width": 2000, "format": "png", "quiet": ""})
    img = Image.open(BytesIO(img_bytes))
    img.save(png_dir / f"{label}.png", "PNG", optimize=True, compress_level=9)