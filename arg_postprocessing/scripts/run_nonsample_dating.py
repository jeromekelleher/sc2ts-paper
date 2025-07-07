import argparse
import json
import sys

import numpy as np
import tszip
import tsdate
import tskit
import sc2ts

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description=(
            "Assuming a constant mutation rate, allocate plausible times to the "
            "nonsample nodes of a sc2ts tree sequence, using tsdate"
        )
    )
    argparser.add_argument("input_ts", help="Path to input ts or tsz file")
    argparser.add_argument(
        "output_ts",
        nargs='?',
        default=None,
        help=(
            "Path to the ts file to output, tszip compressed by default (unless name ends in "
            ".ts or .trees). If not given, adds '.dated' to input filename",
        )
    )
    argparser.add_argument(
        "--leave-recombinant-mutations",
        "-r",
        action="store_true",
        help=(
            "If mutations exist on a single edge under a recombination nodes, should we"
            "leave them there, or move them to the appropriate parent of the node (equivalent "
            "to placing the recombination node as recently as possible "
            "(default=False, as moving them provides consistency and more reliable dating)"
        )
    )
    argparser.add_argument(
        "--no-mutation-time",
        "-n",
        action="store_true",
        help="Should we set mutation times to `UNKNOWN_TIME` (can make plotting nicer)")
    argparser.add_argument(
        "--add-tsdate-metadata",
        "-m",
        action="store_true",
        help="Should we add the standard tsdate metadata to nodes & mutations (increases the filesize)")
    argparser.add_argument("--verbose", "-v", action="store_true", help="Print extra info")
    args = argparser.parse_args()

    ts = tszip.load(args.input_ts)
    # check that the initial Wuhan strain is there, and make it a sample if needed
    assert ts.node(1).metadata["strain"].startswith("Wuhan")
    tables = ts.dump_tables()
    arguments = [args.input_ts]
    if args.output_ts is not None:
        arguments.append(args.output_ts)
    if args.leave_recombinant_mutations:
        arguments.append("--leave-recombinant-mutations")
    if args.no_mutation_time:
        arguments.append("--no-mutation-time")
    if args.add_tsdate_metadata:
        arguments.append("--add-tsdate-metadata")
    tables.provenances.add_row(
        json.dumps({"command": sys.argv[0], "args": arguments}).encode()
    )

    if not args.leave_recombinant_mutations:
        re_nodes = np.where(tables.nodes.flags & sc2ts.NODE_IS_RECOMBINANT)[0]
        re_node_desc_edges = np.isin(tables.edges.parent, re_nodes)
        unique_pairs = np.unique([tables.edges.parent[re_node_desc_edges], tables.edges.child[re_node_desc_edges]], axis=1)
        parent_id, children_per_parent = np.unique(unique_pairs[0,:], return_counts=True)
        re_nodes_with_one_child = parent_id[children_per_parent==1]
        assert np.all((tables.nodes.flags[re_nodes_with_one_child] & sc2ts.NODE_IS_RECOMBINANT) != 0)
        mutations_node = tables.mutations.node
        tot_moved = 0
        for re_node in re_nodes_with_one_child:
            child = np.unique(tables.edges.child[tables.edges.parent == re_node])
            assert len(child) == 1
            muts_to_move = np.where(tables.mutations.node == child[0])[0]
            tot_moved += len(muts_to_move)
            mutations_node[muts_to_move] = re_node
        if args.verbose:
            print(
                f"Moving {tot_moved} mutations from below to above",
                len(re_nodes_with_one_child),
                "recombinant nodes with a single child"
            )
        tables.mutations.node = mutations_node
        # Reset the times: these will be reinferred by tsdate anyway
    tables.mutations.time = np.full_like(tables.mutations.time, tskit.UNKNOWN_TIME)

    # For dating, we want to treat the Wuhan node as a sample, so it is fixed in time
    node1_flags = ts.node(1).flags
    if not ts.node(1).is_sample():
        tables.nodes[1] = tables.nodes[1].replace(flags=tskit.NODE_IS_SAMPLE)
    ts = tables.tree_sequence()
    assert ts.node(1).is_sample()


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
        set_metadata=True if args.add_tsdate_metadata else False,
    )
    
    if "strain" not in dated_ts.node(1).metadata:
        raise ValueError("You are using a version of tsdate that overwrites node metadata")

    assert dated_ts.node(1).metadata["strain"].startswith("Wuhan")
        # revert the Wuhan strain to nonsample if needed

    tables = dated_ts.dump_tables()
    if args.no_mutation_time:
        tables.mutations.time = np.full_like(tables.mutations.time, tskit.UNKNOWN_TIME)
    tables.nodes[1] = tables.nodes[1].replace(flags=node1_flags)
    dated_ts = tables.tree_sequence()
    assert dated_ts.node(1).flags == node1_flags


    if args.output_ts is None:
        if args.input_ts.endswith(".trees"):
            args.output_ts = args.input_ts[:-6] + ".dated.trees"
        elif args.input_ts.endswith(".ts"):
            args.output_ts = args.input_ts[:-3] + ".dated.ts"
        elif args.input_ts.endswith(".tsz"):
            args.output_ts = args.input_ts[:-4] + ".dated.tsz"
    if args.output_ts.endswith(".ts") or args.output_ts.endswith(".trees"):
        dated_ts.dump(args.output_ts)
    else:
        tszip.compress(dated_ts, args.output_ts)

