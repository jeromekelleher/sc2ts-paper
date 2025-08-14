import argparse
import collections

import numpy as np
import tszip
import tskit
import sc2ts
from tqdm import tqdm

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description=(
            "Return an updated sc2ts ARG with the breakpoints shifted to account "
            "for deletions. This requires deletions to have been mapped onto the ARG."
        )
    )
    argparser.add_argument("input_ts", help="Path to input ts or tsz file")
    argparser.add_argument(
        "output_ts",
        nargs="?",
        default=None,
        help=(
            "Path to the ts file to output, tszip compressed by default (unless name ends in "
            ".ts or .trees). If not given, adds '.bpshift' to input filename"
        ),
    )
    argparser.add_argument(
        "--verbose", "-v", action="store_true", help="Print extra info"
    )
    args = argparser.parse_args()

    ts = tszip.load(args.input_ts)

    re_nodes = np.where(ts.nodes_flags & sc2ts.NODE_IS_RECOMBINANT)[0]
    recombinant_edges = [np.where(ts.edges_child == c)[0] for c in re_nodes]
    assert all(
        [len(e) == 2 for e in recombinant_edges]
    )  # all recombinants have only 2 edges
    recombinant_edges = np.array(recombinant_edges)
    # Sort the edges for each child so the left one is first
    srt = np.argsort(ts.edges_left[recombinant_edges], axis=1)
    recombinant_edges = np.array(
        [edge_pair[s] for edge_pair, s in zip(recombinant_edges, srt)]
    )
    recombinant_parents = ts.edges_parent[recombinant_edges]
    breaks = ts.edges_right[recombinant_edges[:, 0]]
    child_and_parents = np.column_stack([re_nodes, recombinant_parents])
    unique_child_and_parents = np.unique(child_and_parents)
    if args.verbose:
        print(
            f"Simplifying to {len(unique_child_and_parents)} recombinant parents and children"
        )

    sts, node_map = ts.simplify(
        unique_child_and_parents, keep_unary=True, map_nodes=True
    )
    new_child_and_parents = node_map[np.array(child_and_parents)]
    assert np.all(new_child_and_parents >= 0)
    assert np.all(new_child_and_parents < len(np.unique(new_child_and_parents)))
    assert sts.num_samples == len(np.unique(new_child_and_parents))
    assert sts.num_samples == np.max(new_child_and_parents) + 1

    variable_sites = collections.defaultdict(list)
    variable_site_left_of_bp = {}
    variable_site_right_of_bp = {}

    # For speed, work with a tree sequence that has been simpified down to the parents and children
    # of all recombinants
    SiteInfo = collections.namedtuple(
        "SiteInfo", "pos, child, left_parent, right_parent"
    )
    for v in sts.variants():
        for i, (bp, (child, p_lft, p_rgt)) in enumerate(
            zip(breaks, new_child_and_parents)
        ):
            if v.genotypes[p_lft] != v.genotypes[p_rgt]:
                variable_sites[i].append(
                    SiteInfo(
                        v.site.position,
                        v.alleles[v.genotypes[child]],
                        v.alleles[v.genotypes[p_lft]],
                        v.alleles[v.genotypes[p_rgt]],
                    )
                )
                if v.site.position < bp:
                    variable_site_left_of_bp[i] = v.site.position
                elif v.site.position >= bp and i not in variable_site_right_of_bp:
                    variable_site_right_of_bp[i] = v.site.position
    assert set(variable_site_left_of_bp.keys()) == set(variable_site_right_of_bp.keys())

    # find the cases where the variable site to the left or right of the
    # breakpoint contains a deletion ("-"), either in the child or the parent
    new_breaks = {}
    site_pos_map = {pos: i for i, pos in enumerate(ts.sites_position)}
    keep_mutation = np.ones(ts.num_mutations, dtype=bool)
    for re_id, siteinfo in variable_sites.items():
        for site_idx in range(len(siteinfo)):
            info = siteinfo[site_idx]
            if info.pos == variable_site_left_of_bp[re_id]:
                if info.child == info.right_parent:
                    assert info.child != info.left_parent
                    if info.child == "-" or info.left_parent == "-":
                        i = site_idx + 1
                        while siteinfo[i - 1].child == siteinfo[i - 1].right_parent:
                            i -= 1
                            mutations = ts.site(site_pos_map[siteinfo[i].pos]).mutations
                            for m in mutations:
                                if m.node == re_nodes[re_id]:
                                    if (
                                        m.derived_state != "-"
                                        and ts.mutation(m.parent).derived_state != "-"
                                    ):
                                        if args.verbose:
                                            print(
                                                f"Exceptional case for RE node {re_nodes[re_id]}: site"
                                                f" at {siteinfo[i].pos} is not to or from a deletion"
                                            )
                                    keep_mutation[m.id] = False
                    else:
                        continue
                    assert siteinfo[site_idx + 1].pos == breaks[re_id]
                    new_breaks[re_id] = siteinfo[i].pos
                    if args.verbose:
                        print(
                            re_nodes[re_id],
                            ": deletion breakpoint misplaced rightwards @",
                            breaks[re_id],
                            "should be @",
                            new_breaks[re_id],
                        )
            elif info.pos == variable_site_right_of_bp[re_id]:
                # NB - we rarely (never?) hit this logic is sc2ts ARGs, because the breakpoint is put to the rightmost location
                if info.child == info.left_parent:
                    assert info.child != info.right_parent
                    i = site_idx
                    if info.child == "-" or info.right_parent == "-":
                        while siteinfo[i].child == info.left_parent:
                            i += 1
                            mutations = ts.site(site_pos_map[siteinfo[i].pos]).mutations
                            for m in mutations:
                                if m.node == re_nodes[re_id]:
                                    assert (
                                        m.derived_state == "-"
                                        or ts.mutation(m.parent).derived_state == "-"
                                    ), (
                                        m.derived_state,
                                        ts.mutation(m.parent).derived_state,
                                    )
                                    keep_mutation[m.id] = False
                    else:
                        continue
                    assert siteinfo[site_idx + 1].pos == breaks[re_id]
                    new_breaks[re_id] = siteinfo[i].pos
                    if args.verbose:
                        print(
                            re_nodes[re_id],
                            ": deletion breakpoint misplaced leftwards @",
                            breaks[re_id],
                            "should be @",
                            new_breaks[re_id],
                        )

    tables = ts.dump_tables()
    edges_left = tables.edges.left
    edges_right = tables.edges.right
    for i, new_bp in new_breaks.items():
        bp = breaks[i]
        lft_edge, rgt_edge = recombinant_edges[i]
        assert edges_right[lft_edge] == bp
        edges_right[lft_edge] = new_bp
        assert edges_left[rgt_edge] == bp
        edges_left[rgt_edge] = new_bp

    tables.edges.left = edges_left
    tables.edges.right = edges_right
    tables.mutations.parent = np.full_like(tables.mutations.parent, tskit.NULL)
    tables.mutations.keep_rows(keep_mutation)
    tables.sort()
    tables.build_index()
    tables.compute_mutation_parents()
    # Set times as unknown, in case and mutations have changed edges
    tables.mutations.time = np.full_like(tables.mutations.time, tskit.UNKNOWN_TIME)
    new_ts = tables.tree_sequence()

    for v1, v2 in tqdm(
        zip(
            ts.variants(samples=re_nodes, isolated_as_missing=False),
            new_ts.variants(samples=re_nodes, isolated_as_missing=False),
        ),
        total=ts.num_sites,
        desc="Check unchanged RE seq",
        disable=not args.verbose,
    ):
        assert v1.site.position == v2.site.position
        assert np.all(v1.states() == v2.states())

    if args.verbose:
        print(
            f"Out of {len(re_nodes)} RE nodes, {len(new_breaks)} breakpoints were shifted"
        )
    if args.output_ts is None:
        if args.input_ts.endswith(".trees"):
            args.output_ts = args.input_ts[:-6] + ".bpshift.trees"
        elif args.input_ts.endswith(".ts"):
            args.output_ts = args.input_ts[:-3] + ".bpshift.ts"
        elif args.input_ts.endswith(".tsz"):
            args.output_ts = args.input_ts[:-4] + ".bpshift.tsz"
    if args.output_ts.endswith(".ts") or args.output_ts.endswith(".trees"):
        new_ts.dump(args.output_ts)
    else:
        tszip.compress(new_ts, args.output_ts)
