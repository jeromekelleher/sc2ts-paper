import argparse
import collections
from tqdm import tqdm
from pathlib import Path

import numpy as np
import pandas as pd
import sc2ts
import tszip


def recombinant_supporting_locations(ts, adjacent_distance):
    # return a dictionary mapping each recombination node to
    # the number of "supporting sites" for each edge above that node
    sorted_edges = {}
    nodes_used = []
    run_positions = collections.defaultdict(list)
    for u in np.where(ts.nodes_flags & sc2ts.NODE_IS_RECOMBINANT)[0]:
        edges = np.where(ts.edges_child == u)[0]
        sorted_edges[u] = edges[np.argsort(ts.edges_left[edges])]
        nodes_used.append(u)
        nodes_used += list(ts.edges_parent[edges])
    nodes_used, inverse = np.unique(nodes_used, return_inverse=True)
    node_map = {index: id_ for id_, index in zip(inverse, nodes_used[inverse])}

    last_pos = {u: np.full(len(v), -np.inf) for u, v in sorted_edges.items()}
    last_parent = {u: np.full(len(v), -1) for u, v in sorted_edges.items()}
    supporting_location_count = {u: np.full(len(v), 0) for u, v in sorted_edges.items()}
    edge_index = {u: 0 for u in sorted_edges.keys()}

    sts = ts.simplify(nodes_used)
    for v in tqdm(sts.variants(), total=sts.num_sites, disable=not args.verbose):
        genotypes = v.genotypes
        pos = v.site.position
        for re_node, edge_list in sorted_edges.items():
            parents = ts.edges_parent[edge_list]
            parent_genos = [genotypes[node_map[p]] for p in parents]
            if np.all(np.diff(parent_genos) == 0):
                # irrelevant site, as parents are all identical genotype
                continue
            last_position = last_pos[re_node]
            re_geno = genotypes[node_map[re_node]]
            assert re_geno >= 0

            # pick the correct edge for this RE node, advancing along the sequence if necessary
            edge = ts.edge(edge_list[edge_index[re_node]])
            while edge.right <= pos:
                edge_index[re_node] += 1
                edge = ts.edge(edge_list[edge_index[re_node]])
            idx = edge_index[re_node]
            assert edge.child == re_node
            parent_geno = genotypes[node_map[edge.parent]]
            if pos - last_position[idx] > adjacent_distance:
                if re_geno == parent_geno:
                    supporting_location_count[re_node][idx] += 1
                elif re_geno in parent_genos:
                    supporting_location_count[re_node][idx] -= 1
                else:
                    pass  # this is a de-novo mutation
            else:
                run_positions[re_node].append(int(last_position[idx]))
                if last_parent[re_node][idx] != edge.parent:
                    raise ValueError(
                        f"conflicting parents within {adjacent_distance}bp"
                    )
            last_position[idx] = pos
            last_parent[re_node][idx] = edge.parent
    return supporting_location_count, run_positions


def main(args):
    net_min_supp_loci_cutoff = 4
    ts = tszip.load(args.input_ts)
    df = pd.read_csv(args.input_csv)
    output_csv = args.output_csv

    supporting_loci_count, _ = recombinant_supporting_locations(
        ts, args.adjacent_distance
    )
    arr = []
    for u, support in supporting_loci_count.items():
        if len(support) != 2:
            raise ValueError(
                f"Expected 2 edges above recombinant node {u} but found "
                f" {len(support)}, implying it does not have exactly 2 parents"
            )
        arr.append([u, *support])
    new_cols = np.array(arr).T

    new_cols = np.vstack(
        (
            new_cols,
            [np.minimum(new_cols[1, :], new_cols[2, :]) >= net_min_supp_loci_cutoff],
        )
    )

    new_df = df.set_index("recombinant")
    colnames = {  # column in new_cols => name in new df
        1: ("net_min_supporting_loci_lft", int),
        2: ("net_min_supporting_loci_rgt", int),
        3: (f"net_min_supporting_loci_lft_rgt_ge_{net_min_supp_loci_cutoff}", bool),
    }
    for use, (colname, dtype) in colnames.items():
        new_df.loc[new_cols[0, :], colname] = new_cols[use, :].astype(dtype)
        df[colname] = new_df[colname].values
        df[colname] = df[colname].astype(dtype)
    df.to_csv(output_csv, index=False)
    if args.verbose:
        print(f"Output written to {output_csv}")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description=(
            "Count the number of `loci` supporting each side of a recombinant, where a "
            "locus can span multiple positions, up to `adjacent_distance` apart. "
        )
    )
    argparser.add_argument("input_ts", help="Path to input ts or tsz file")
    argparser.add_argument("input_csv", help="Recombinant csv file in pandas format")
    argparser.add_argument(
        "output_csv",
        nargs="?",
        default=None,
        help=(
            "Path to the csv file to output. "
            "If not given, add '.minl' to input filename prefix."
        ),
    )
    argparser.add_argument(
        "--verbose", "-v", action="store_true", help="Print extra info"
    )
    argparser.add_argument(
        "--adjacent_distance",
        "-d",
        type=int,
        help="The max distance between sites at the same locus",
        default=3,
    )

    args = argparser.parse_args()
    main(args)
