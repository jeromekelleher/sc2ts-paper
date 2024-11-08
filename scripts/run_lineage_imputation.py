import sc2ts
import tszip
import sklearn.tree
import numpy as np
from tqdm.auto import tqdm
import pandas as pd
from collections import defaultdict
import time
import argparse

class InferLineage:
    def __init__(self, num_nodes, true_lineage):
        self.lineages_true = [None] * num_nodes
        self.lineages_pred = [None] * num_nodes
        self.num_nodes = num_nodes
        self.lineages_type = [
            0
        ] * num_nodes  # 0 if can't infer, 1 if inherited, 2 if imputed
        self.num_sample_imputed = 0
        self.num_intern_imputed = (
            1  # This is the root node which I'm taking to be lineage B
        )
        self.lineages_pred[0] = "B"
        self.lineages_type[0] = 1
        self.change = 1
        self.current_node = None
        self.linfound = False
        self.true_lineage = true_lineage
        self.recombinants = None

    def reset(self):
        self.change = 0

    def total_inferred(self, ti):
        return self.num_sample_imputed + self.num_intern_imputed

    def set_node(self, node):
        self.current_node = node
        self.linfound = False

    def add_imputed_values(self, X_index, y):
        for ind, pred in zip(X_index, y):
            self.lineages_pred[ind] = pred
            self.lineages_type[ind] = 2
            if self.lineages_true[ind] is not None:
                self.num_sample_imputed += 1
            else:
                self.num_intern_imputed += 1
            self.change += 1

    def record_recombinants(self, ts, ti):
        for r in tqdm(ti.recombinants, desc="Record recombinants"):
            r_node = ts.node(r)
            if self.true_lineage not in r_node.metadata:
                # Just recording that this is a recombinant lineage for which we don't have a Pango name
                self.lineages_pred[r] = "Unknown"
        self.recombinants = ti.recombinants

    def record_true_lineage(self, node):
        if self.true_lineage in node.metadata and self.lineages_true[node.id] is None:
            self.lineages_true[node.id] = node.metadata[self.true_lineage]

    def inherit_from_node(self, node, is_child=False):
        if self.true_lineage in node.metadata:
            self.lineages_pred[self.current_node.id] = node.metadata[self.true_lineage]
            self.lineages_type[self.current_node.id] = 1
            self.linfound = True
        elif is_child and not (self.lineages_pred[node.id] in [None, "Unknown"]):
            self.lineages_pred[self.current_node.id] = self.lineages_pred[node.id]
            self.lineages_type[self.current_node.id] = 1
            self.linfound = True
        elif not is_child and self.lineages_pred[node.id] is not None:
            self.lineages_pred[self.current_node.id] = self.lineages_pred[node.id]
            self.lineages_type[self.current_node.id] = 1
            self.linfound = True

    def inherit_from_children(self, ts, t, mut_dict):
        if not self.linfound:
            for child_node_ind in t.children(self.current_node.id):
                if child_node_ind not in mut_dict.names:
                    child_node = ts.node(child_node_ind)
                    self.inherit_from_node(child_node, is_child=True)
                    if self.linfound:
                        break

    def inherit_from_parent(self, ts, t, mut_dict):
        if not self.linfound:
            if self.current_node.id not in mut_dict.names:
                parent_node_ind = t.parent(self.current_node.id)
                if parent_node_ind != -1:
                    self.inherit_from_node(ts.node(parent_node_ind), is_child=False)

    def update(self):
        if self.linfound:
            if self.current_node.is_sample():
                self.num_sample_imputed += 1
            else:
                self.num_intern_imputed += 1
            self.change += 1

    def check_node(self, node, ti):
        self.set_node(node)
        if (
            self.current_node.id not in ti.recombinants
            and self.lineages_pred[self.current_node.id] is None
        ):
            return True
        else:
            return False

    def print_info(self, ts, ti, target):
        print("-" * 30)
        print(
            "Sample nodes imputed:",
            self.num_sample_imputed,
            "out of possible",
            ts.num_samples,
        )
        print(
            "Internal nodes imputed:",
            self.num_intern_imputed,
            "out of possible",
            target - target_samples,
        )
        print(
            "Total imputed:",
            self.num_sample_imputed + self.num_intern_imputed,
            "out of possible",
            target,
        )
        print("Number of recombinants (not imputed):", len(ti.recombinants))

        print("-" * 30)
        correct = incorrect = 0
        type1 = type2 = 0
        for lt, lp, ltype in zip(
            self.lineages_true, self.lineages_pred, self.lineages_type
        ):
            if ltype == 1:
                type1 += 1
            elif ltype == 2:
                type2 += 1
            if lt is not None and lp != "Unknown":
                if lt == lp:
                    correct += 1
                else:
                    incorrect += 1
        print(
            "Correctly imputed samples:",
            correct,
            "(",
            round(100 * correct / (correct + incorrect), 3),
            "% )",
        )
        print(
            "Incorrectly imputed samples:",
            incorrect,
            "(",
            round(100 * incorrect / (correct + incorrect), 3),
            "% )",
        )
        print(
            "Imputed using inheritance:",
            type1,
            "(",
            round(100 * type1 / (self.total_inferred(ti)), 3),
            "% )",
            "using characteristic mutations:",
            type2,
            "(",
            round(100 * type2 / (self.total_inferred(ti)), 3),
            "% )",
        )
        print("-" * 30)

    def get_results(self):
        all_lineages = [None] * self.num_nodes
        for i, (lt, lp) in enumerate(zip(self.lineages_true, self.lineages_pred)):
            if lt is not None:
                all_lineages[i] = lt
            elif i in self.recombinants:
                all_lineages[i] = "Unknown (R)"
            else:
                all_lineages[i] = lp
        return all_lineages


def impute_lineages(
    ts,
    ti,
    node_to_mut_dict,
    df,
    ohe_encoder,
    clf_tree,
    true_lineage="Viridian_pangolin",
):
    """
    Impute lineages for all nodes, and save in the node metadata
    """

    tic = time.time()

    inferred_lineages = InferLineage(ts.num_nodes, true_lineage)
    t = ts.first()

    # Assigning "Unknown" as the lineage for recombinant nodes that don't have a Pango designation
    inferred_lineages.record_recombinants(ts, ti)

    for n in tqdm(ts.nodes(), desc="Record true lineages"):
        inferred_lineages.record_true_lineage(n)

    target = ts.num_nodes - len(ti.recombinants)

    with tqdm(total=target - 1, desc="Infer lineages") as pbar:
        while inferred_lineages.total_inferred(ti) < target:
            impute_lineages_inheritance(
                inferred_lineages,
                ts,
                t,
                ti,
                node_to_mut_dict,
                pbar,
            )
            impute_lineages_decisiontree(
                inferred_lineages,
                ts,
                t,
                ti,
                node_to_mut_dict,
                df,
                ohe_encoder,
                clf_tree,
                target,
                pbar,
            )
            # print("Imputed so far:", inferred_lineages.num_sample_imputed + inferred_lineages.num_intern_imputed, "out of possible", target)
    inferred_lineages.print_info(ts, ti, target)

    edited_ts = add_lineages_to_ts(inferred_lineages, ts)
    edited_ts = fix_lineages(inferred_lineages, edited_ts)

    print("Time:", time.time() - tic)

    return edited_ts


def impute_lineages_inheritance(
    inferred_lineages,
    ts,
    t,
    ti,
    node_to_mut_dict,
    pbar,
):
    """
    For each node for which a lineage has not yet been assigned, try and copy the lineage of the parent or
    one of the children (if there are no lineage-defining mutations on the connecting edge).
    This is run iteratively on the nodes until no further assignment is possible.
    """

    # print("Inheriting lineages...", end="")
    # Need to loop through until all known lineages have been copied where possible
    while inferred_lineages.change:
        inferred_lineages.reset()
        for n_ in t.nodes(order="timedesc"):
            n = ts.node(n_)
            if inferred_lineages.check_node(n, ti):
                # Try to inherit lineage from parent or children, if there is at least one edge
                # without a mutation
                inferred_lineages.inherit_from_children(ts, t, node_to_mut_dict)
                inferred_lineages.inherit_from_parent(ts, t, node_to_mut_dict)
                inferred_lineages.update()
        # print(inferred_lineages.change, end="...")
        pbar.update(inferred_lineages.change)
    # print("done")


def impute_lineages_decisiontree(
    inferred_lineages,
    ts,
    t,
    ti,
    node_to_mut_dict,
    df,
    ohe_encoder,
    clf_tree,
    target,
    pbar,
):
    """
    For each node, impute a lineage based on that of the parent node (if known or already imputed) plus
    the lineage-defining mutations on the connecting edge. This uses the decision tree constructed using
    COVIDCG lineage-defining mutations data.
    """

    # Impute lineages for the rest of the nodes where possible (one pass)
    X = pd.DataFrame(
        index=range(target - inferred_lineages.total_inferred(ti)), columns=df.columns
    )
    X_index = np.zeros(target - inferred_lineages.total_inferred(ti), dtype=int)
    ind = 0
    # print("Imputing lineages...", end = "")
    inferred_lineages.reset()
    for n_ in t.nodes(order="timedesc"):
        n = ts.node(n_)
        if inferred_lineages.check_node(n, ti):
            parent_node_ind = t.parent(inferred_lineages.current_node.id)
            if parent_node_ind != -1:
                parent_node_md = ts.node(parent_node_ind).metadata
                if (
                    inferred_lineages.true_lineage in parent_node_md
                    or inferred_lineages.lineages_pred[parent_node_ind] is not None
                ):
                    # Check if we can now copy the parent's lineage
                    if n_ not in node_to_mut_dict.names or (
                        inferred_lineages.true_lineage not in parent_node_md
                        and inferred_lineages.lineages_pred[parent_node_ind]
                        == "Unknown"
                    ):
                        inferred_lineages.inherit_from_node(
                            ts.node(parent_node_ind)
                        )
                        inferred_lineages.update()
                    # If not, then add to dataframe for imputation
                    else:
                        if inferred_lineages.true_lineage in parent_node_md:
                            parent_lineage = parent_node_md[
                                inferred_lineages.true_lineage
                            ]
                        else:
                            parent_lineage = inferred_lineages.lineages_pred[
                                parent_node_ind
                            ]
                        X_index[ind] = n_
                        X.loc[ind] = df.loc[parent_lineage]
                        positions, alts = node_to_mut_dict.get_mutations(n_)
                        X.loc[ind][positions] = alts
                        ind += 1
                        # print(n_)
    if ind > 0:
        X = X.iloc[0:ind]
        X_index = X_index[0:ind]
        y = clf_tree.predict(ohe_encoder.transform(X))
        inferred_lineages.add_imputed_values(X_index, y)
    pbar.update(inferred_lineages.change)


def check_lineages_in_ts(ts, linmuts_dict):
    """
    Error out if any lineage assignments from ts samples not in linmuts_dict
    """
    for node in tqdm(ts.nodes(), desc="Checking lineages in ts"):
        md = node.metadata
        if "Viridian_pangolin" in md:
            if md["Viridian_pangolin"] not in linmuts_dict.names:
                raise ValueError(
                    "Lineage assignment not in lineage-defining mutations list"
                )


def add_lineages_to_ts(il, ts):
    """
    Adds imputed lineages to ts metadata.
    """

    imputed_lineages = il.get_results()
    tables = ts.tables
    new_metadata = []
    for node in ts.nodes():
        md = node.metadata
        md["Imputed_" + il.true_lineage] = imputed_lineages[node.id]
        new_metadata.append(md)
    validated_metadata = [
        tables.nodes.metadata_schema.validate_and_encode_row(row)
        for row in new_metadata
    ]
    tables.nodes.packset_metadata(validated_metadata)
    edited_ts = tables.tree_sequence()
    return edited_ts

def fix_lineages(il, ts):
    """
    Check if lineage of parent and child nodes are the same
    If so ensure assign this as the imputed lineage of the node
    """

    stop = False
    edited_ts = ts
    while not stop:
        differences = defaultdict(list)
        edits = {}
        t = edited_ts.at_index(int(edited_ts.num_trees / 2))
        for n in t.nodes():
            md = edited_ts.node(n).metadata
            if "Collection_date" not in md:
                if md["Imputed_" + il.true_lineage] != "Unknown" and md["Imputed_" + il.true_lineage] != "Unknown (R)":
                    differences_ = []
                    lineages = set()
                    p_diff = ch_diff = False
                    l1 = md["Imputed_" + il.true_lineage]
                    l2 = edited_ts.node(t.parent(n)).metadata["Imputed_" + il.true_lineage]
                    if l1 != l2 and l2 != "Unknown" and l2 != "Unknown (R)":
                        differences_.append(("p", t.parent(n), l2))
                        lineages.add(l2)
                        p_diff = True
                    for i, ch in enumerate(t.children(n)):
                        l = edited_ts.node(ch).metadata["Imputed_" + il.true_lineage]
                        if l != l1 and l != "Unknown" and l != "Unknown (R)":
                            differences_.append(("ch", ch, l))
                            lineages.add(l)
                            ch_diff = True
                    if p_diff and ch_diff:
                        differences[("n", n, l1)] = differences_
                        if len(lineages) == 1:
                            edits[n] = lineages.pop()

        print("Matching parent-child lineages where possible: " + str(len(edits)) + " out of " + str(len(differences)))
        stop = len(edits) == 0

        if not stop:
            tables = edited_ts.tables
            new_metadata = []
            for node in edited_ts.nodes():
                md = node.metadata
                if node.id in edits:
                    md["Imputed_" + il.true_lineage] = edits[node.id]
                new_metadata.append(md)
            validated_metadata = [
                tables.nodes.metadata_schema.validate_and_encode_row(row)
                for row in new_metadata
            ]
            tables.nodes.packset_metadata(validated_metadata)
            edited_ts = tables.tree_sequence()

    return edited_ts


def imputation_setup(filepath, verbose=False):
    """
    Reads in JSON of lineage-defining mutations and constructs decision tree classifier
    JSON can be downloaded from covidcg.org -> 'Compare AA mutations' -> Download -> 'Consensus mutations'
    (setting mutation type to 'NT' and consensus threshold to 0.9)
    """
    linmuts_dict = sc2ts.lineages.read_in_mutations(filepath)
    df, df_ohe, ohe = sc2ts.lineages.read_in_mutations_json(filepath)

    # Get decision tree
    y = df_ohe.index  # lineage labels
    clf = sklearn.tree.DecisionTreeClassifier()
    clf = clf.fit(df_ohe, y)

    if verbose:
        # Check tree works and that lineages-defining mutations are unique for each lineage
        y_pred = clf.predict(df_ohe)
        correct = incorrect = lineage_definition_issue = 0
        for yy, yy_pred in zip(y, y_pred):
            if yy == yy_pred:
                correct += 1
            else:
                incorrect += 1
                if linmuts_dict.get_mutations(yy) == linmuts_dict.get_mutations(
                    yy_pred
                ):
                    lineage_definition_issue += 1
                    print(yy_pred, "same mutations as", yy)
        print(
            "Correct:",
            correct,
            "incorrect:",
            incorrect,
            "of which due to lineage definition ambiguity:",
            lineage_definition_issue,
        )

    return linmuts_dict, df, df_ohe, ohe, clf


def lineage_imputation(filepath, ts, ti, verbose=False):
    """
    Runs lineage imputation on input ts
    """
    linmuts_dict, df, df_ohe, ohe, clf = imputation_setup(filepath, verbose)
    check_lineages_in_ts(ts, linmuts_dict)
    print("Recording relevant mutations for each node...")
    node_to_mut_dict = sc2ts.lineages.get_node_to_mut_dict(ts, ti, linmuts_dict)
    edited_ts = impute_lineages(
        ts, ti, node_to_mut_dict, df, ohe, clf, "Viridian_pangolin"
    )
    return edited_ts


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Run lineage imputation on ts")
    argparser.add_argument(
        "mutations_json_filepath",
        help="Path to JSON file of lineage-defining mutations",
    )
    argparser.add_argument("input_ts", help="Path to input ts or tsz file")
    argparser.add_argument(
        "output_tsz",
        nargs='?',
        default=None,
        help="Path to the compressed tsz file to output. If not given, adds '.il' to input filename",)
    argparser.add_argument("--verbose", "-v", action="store_true", help="Print extra info")
    args = argparser.parse_args()

    ts = tszip.load(args.input_ts)
    ti = sc2ts.info.TreeInfo(ts)

    new_ts = lineage_imputation(
        args.mutations_json_filepath,
        ts, 
        ti,
        verbose=args.verbose,
    )
    if args.output_tsz is None:
        if args.input_ts.endswith(".tsz"):
            args.output_tsz = args.input_ts[:-4] + ".il.tsz"
        elif args.input_ts.endswith(".trees"):
            args.output_tsz = args.input_ts + ".il.trees"
        elif args.input_ts.endswith(".ts"):
            args.output_tsz = args.input_ts + ".il.ts"
    tszip.compress(new_ts, args.output_tsz)

