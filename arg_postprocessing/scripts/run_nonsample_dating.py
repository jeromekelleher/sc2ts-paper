import argparse
import json
import sys

import numpy as np
import tszip
import tsdate
import tskit

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description=(
            "Assuming a constant mutation rate, allocate plausible times to the "
            "nonsample nodes of a sc2ts tree sequence, using tsdate"
        )
    )
    argparser.add_argument("input_ts", help="Path to input ts or tsz file")
    argparser.add_argument("output_ts", help="Path to the output .trees file")
    argparser.add_argument(
        "--add-tsdate-metadata",
        "-m",
        action="store_true",
        help="Add the standard tsdate metadata to nodes & mutations (increases the filesize)",
    )
    argparser.add_argument(
        "--verbose", "-v", action="store_true", help="Print extra info"
    )
    args = argparser.parse_args()

    ts = tszip.load(args.input_ts)
    # check that the initial Wuhan strain is there
    assert ts.node(1).metadata["strain"].startswith("Wuhan")
    tables = ts.dump_tables()
    tables.provenances.add_row(
        json.dumps({"command": sys.argv[0], "args": sys.argv[1:]}).encode()
    )
    # Reset the times: these will be reinferred by tsdate anyway
    tables.mutations.time = np.full_like(tables.mutations.time, tskit.UNKNOWN_TIME)

    # For dating, we want to treat the Wuhan node as a sample, so it is fixed in time
    node1_flags = ts.node(1).flags
    tables.nodes[1] = tables.nodes[1].replace(flags=node1_flags | tskit.NODE_IS_SAMPLE)
    ts = tables.tree_sequence()

    # assume to first order approximation that the mutation rate is constant for all muts
    edge_times = ts.nodes_time[ts.edges_parent] - ts.nodes_time[ts.edges_child]
    av_mu = ts.num_mutations / ((ts.edges_right - ts.edges_left) * edge_times).sum()

    dated_ts = tsdate.date(
        ts,
        mutation_rate=av_mu,
        rescaling_intervals=0,
        constr_iterations=1000,
        time_units=ts.time_units,
        allow_unary=True,
        progress=args.verbose,
        set_metadata=args.add_tsdate_metadata,
    )
    assert dated_ts.node(1).metadata["strain"].startswith("Wuhan")

    # revert the Wuhan strain to its original flags
    tables = dated_ts.dump_tables()
    tables.nodes[1] = tables.nodes[1].replace(flags=node1_flags)
    tables.dump(args.output_ts)
