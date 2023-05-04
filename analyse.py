import argparse
import os.path

import numpy as np
import tszip

import sc2ts

class CompressedSc2tsARG:
    # Keep details of the compressed ARG so we can extract e.g. compressed sizes
    def __init__(self, fn):
        if not fn.endswith(".tsz"):
            raise ValueError("Requires a tszip compressed file ending in .tsz")
        self.fn = fn
        self.ts = tszip.decompress(fn)
        self.treeinfo = None
        
    @property
    def ti(self):
        # Slow to calculate, so cache it
        if self.treeinfo is None:
            self.treeinfo = sc2ts.TreeInfo(self.ts)
        return self.treeinfo

    def compressed_bytes(self):
        return os.path.getsize(self.fn)


def perc(x, tot):
    # return percentage, formatted nicely
    return f"{x/tot * 100:.2g}%"

class Total:
    # Keep track of a total, and print values as a percentage of the total
    def __init__(self, total):
        self._total = total
    def suffix(self, x, suffix=None):
        # print with "M", "k", etc
        if suffix is None:
            return str(x)
        elif suffix == "k":
            return f"{x/1000:.3g}k"
        elif suffix == "M":
            return f"{x/1000_000:.3g}M"
    def total(self, suffix=None):
        return self.suffix(self._total, suffix)
    def x_perc(self, x, suffix=None):
        return f"{self.suffix(x)} {perc(x, self._total)}"

def tot(df, *rownames):
    # return sum of named rows in the df, or just a single row value
    return df.loc[list(rownames)].sum().value

def prt(*args):
    # shortcut for printing separated by a tab
    print(*args, sep="\t")

def print_stats(wide, long, use_treeinfo=True):
    if use_treeinfo:
        summaryW = wide.ti.summary()
        summaryL = long.ti.summary()
    prt("* Results")
    prt("** Inferred ARGs")
    prt("", "", "Wide ARG", "Long ARG")
    w = Total(wide.ts.num_nodes)
    l = Total(long.ts.num_nodes)
    if use_treeinfo:
        prt("Max_sub_delay", tot(summaryW, "max_submission_delay"), "", tot(summaryL, "max_submission_delay"), "")
        prt("Max_samp/day", tot(summaryW, "max_samples_per_day"), "", tot(summaryL, "max_samples_per_day"), "")
        prt()
        
    # NODES
    prt("Nodes", "total", w.total(), "", l.total(), "")
    prt("", "samples", w.x_perc(wide.ts.num_samples), l.x_perc(long.ts.num_samples))
    if use_treeinfo:
        for ts, summary in [(wide.ts, summaryW), (long.ts, summaryL)]:
            # Check on some properties
            assert  ts.num_samples == tot(summary, 'samples')
            assert (
                (ts.nodes_flags & (sc2ts.NODE_IS_MUTATION_OVERLAP + sc2ts.NODE_IS_REVERSION_PUSH) > 0).sum()
                == tot(summary, 'mc_nodes', 'pr_nodes'))
            # Count cases where the same node is a child for multiple edges
            _, count = np.unique(ts.edges_child, return_counts=True)
            assert (ts.nodes_flags & sc2ts.NODE_IS_RECOMBINANT > 0).sum() == (count > 1).sum()
            assert tot(summary, 're_nodes') == (count > 1).sum()
            assert tot(summary, 'recombinants') == (count > 1).sum()
        tree_adjW = tot(summaryW, 'mc_nodes', 'pr_nodes')
        tree_adjL = tot(summaryL, 'mc_nodes', 'pr_nodes')
        re_nodesW = tot(summaryW, 're_nodes')
        re_nodesL = tot(summaryL, 're_nodes')
        prt("", "UPGMA", w.x_perc(wide.ts.num_nodes - wide.ts.num_samples - re_nodesW - tree_adjW), l.x_perc(long.ts.num_nodes - long.ts.num_samples - re_nodesL - tree_adjL))
        prt("", "treeAdj", w.x_perc(tree_adjW),l.x_perc(tree_adjL))
        prt("", "recomb", w.x_perc(re_nodesW), l.x_perc(re_nodesL))

        
    # MUTATIONS    
    prt()
    w = Total(wide.ts.num_mutations)
    l = Total(long.ts.num_mutations)
    prt("Muts", "total", w.total("M"), "", l.total("M"), "")

    
    # RECOMBINATIONS
    prt()
    wide_no_sgltn = sc2ts.detach_singleton_recombinants(wide.ts, filter_nodes=True)
    long_no_sgltn = sc2ts.detach_singleton_recombinants(long.ts, filter_nodes=True)
    w = Total((wide.ts.nodes_flags & sc2ts.NODE_IS_RECOMBINANT > 0).sum())
    l = Total((long.ts.nodes_flags & sc2ts.NODE_IS_RECOMBINANT > 0).sum())
    prt("REnodes", "total", w.total(), "", l.total(), "")
    
    prt("", "NoSgltn",
        w.x_perc((wide_no_sgltn.nodes_flags & sc2ts.NODE_IS_RECOMBINANT > 0).sum()),
        l.x_perc((long_no_sgltn.nodes_flags & sc2ts.NODE_IS_RECOMBINANT > 0).sum()),
    )
    ##prt("", "1 break",
    #prt("", "2 breaks",
    #prt("", "3+ breaks",
    
    
    # SIZES - currently quoted in MB not MiB - I think this is more useful to the layman
    prt("Size (inc metadata)", Total(wide.compressed_bytes()).total("M")+"B", Total(long.compressed_bytes()).total("M")+"B", )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-build-treeinfo", help="Do not build treeinfo", action="store_true")
    args = parser.parse_args()

    wide = CompressedSc2tsARG("data/upgma-full-md-30-mm-3-2021-06-30-recinfo-gisaid-il.ts.tsz")
    print("Read wide ARG")
    long = CompressedSc2tsARG("data/upgma-mds-1000-md-30-mm-3-2022-06-30-recinfo-gisaid-il.ts.tsz")
    print("Read long ARG")
    print("-------------")

    print_stats(wide, long, not args.no_build_treeinfo)  # for testing, allow skip treeinfo

if __name__ == "__main__":
    main()
