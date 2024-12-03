import argparse
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
    argparser.add_argument(
        "output_ts",
        nargs='?',
        default=None,
        help=(
            "Path to the ts file to output, tszip compressed by default (unless name ends in "
            ".ts or .trees). If not given, adds '.dated' to input filename",
        )
    )
    argparser.add_argument("--verbose", "-v", action="store_true", help="Print extra info")
    args = argparser.parse_args()

    ts = tszip.load(args.input_ts)
    # check that the initial Wuhan strain is there, and make it a sample if needed
    assert ts.node(1).metadata["strain"].startswith("Wuhan")
    if not ts.node(1).is_sample():
        node1_flags = ts.node(1).flags
        tables = ts.dump_tables()
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
        constr_iterations=100,
        time_units=ts.time_units,
        allow_unary=True,
        progress=args.verbose, 
    )
    
    if "strain" not in dated_ts.node(1).metadata:
        raise ValueError("You are using a version of tsdate that overwrites node metadata")

    assert dated_ts.node(1).metadata["strain"].startswith("Wuhan")
    if not ts.node(1).is_sample():
        # revert the Wuhan strain to nonsample if needed
        tables = dated_ts.dump_tables()
        tables.nodes[1] = tables.nodes[1].replace(flags=node1_flags)
        dated_ts = tables.tree_sequence()
        assert not dated_ts.node(1).is_sample()


    if args.output_ts is None:
        if args.input_ts.endswith(".trees"):
            args.output_ts = args.input_ts + ".dated.trees"
        elif args.input_ts.endswith(".ts"):
            args.output_ts = args.input_ts + ".dated.ts"
        elif args.input_ts.endswith(".tsz"):
            args.output_ts = args.input_ts[:-4] + ".dated.tsz"
    if args.output_ts.endswith(".ts") or args.output_ts.endswith(".trees"):
        dated_ts.dump(args.output_ts)
    else:
        tszip.compress(dated_ts, args.output_ts)

