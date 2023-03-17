# Make e.g. csv data files used in plotting
import sc2ts
import logging
from datetime import datetime

import utils

if __name__ == "__main__":

    # Make a breakpoints file for the long ARG
    fn = "upgma-mds-1000-md-30-mm-3-2022-06-30-recinfo-il.ts.tsz"
    prefix = utils.snip_tsz_suffix(fn)
    long, basetime = utils.load_tsz("data", fn)
    logging.info("Collecting tree information, may take a while")
    long_treeinfo = sc2ts.TreeInfo(long)
    df = long_treeinfo.export_recombinant_breakpoints()
    df.to_csv(f"data/breakpoints_{prefix}.csv")
