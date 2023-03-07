# Make e.g. csv data files used in plotting
import sc2ts
import logging
from datetime import datetime

from utils import load_tsz_file

if __name__ == "__main__":
    long = load_tsz_file("2022-06-30", "upgma-mds-1000-md-30-mm-3-{}-recinfo-il.ts.tsz")
    logging.info("Collecting tree information, may take a while")
    long_treeinfo = sc2ts.TreeInfo(long.ts)
    df = long_treeinfo.export_recombination_node_breakpoints()
    date_str = long.day_0.date().isoformat()
    df.to_csv(f"data/breakpoints_long_{date_str}.csv")
