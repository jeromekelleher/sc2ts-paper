# Make e.g. csv data files used in plotting
import sc2ts
import logging
from datetime import datetime

import utils

if __name__ == "__main__":

    # Make breakpoints files
    for fn in (
        "upgma-mds-1000-md-30-mm-3-2022-06-30-recinfo2-gisaid-il.ts.tsz",
        "upgma-full-md-30-mm-3-2021-06-30-recinfo2-gisaid-il.ts.tsz",
    ):
        prefix = utils.snip_tsz_suffix(fn)
        ts, basetime = utils.load_tsz("data", fn)
        logging.info("Collecting tree information, may take a while")
        treeinfo = sc2ts.TreeInfo(ts)
        df = treeinfo.export_recombinant_breakpoints()
        df[f"parents_dist_{ts.time_units}"] = ts.nodes_time[df["mrca"]] - ts.nodes_time[df["node"]]
        df.to_csv(f"data/breakpoints_{prefix}.csv")
