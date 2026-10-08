import numpy as np
import pandas as pd
import pytest
import tskit
from click.testing import CliRunner
from sc2ts.core import NODE_IS_EXACT_MATCH

from pipeline import (
    classify_samples,
    collapse_unsupported,
    evaluate,
    event_table,
    make_samples_leaves,
    num_recombinant_nodes,
    placement_table,
    prepare_for_comparison,
    score_arg,
    score_recombinants,
)


def to_sc2ts_style(true_ts, truth, recombinant=None, unplaced=(), exact=()):
    """
    Return a copy of the true ARG laid out as sc2ts would infer it: 1-based
    coordinates with sequence length L + 1, times relative to the last sample,
    strain and HMM match metadata on sample nodes, a non-sample reference node
    and the samples in a different order. Samples named in ``recombinant`` (by
    default the sampled recombinants) are given two-parent HMM matches at their
    recombinant ancestor's breakpoints, and the others single-parent matches.
    ``exact`` samples are flagged as exact matches, and ``unplaced`` samples are
    left as non-sample nodes.
    """
    truth = truth.set_index("strain")
    if recombinant is None:
        recombinant = set(truth.index[truth.is_recombinant])
    strains = {
        u: f"seq_{true_ts.individual(true_ts.node(u).individual).metadata['pedigree_id']}"
        for u in true_ts.samples()
    }
    L = int(true_ts.sequence_length)
    min_time = true_ts.nodes_time[true_ts.samples()].min()

    tables = tskit.TableCollection(L + 1)
    tables.nodes.metadata_schema = tskit.MetadataSchema.permissive_json()
    for node in true_ts.nodes():
        metadata = {}
        flags = 0
        if node.id in strains and strains[node.id] not in unplaced:
            strain = strains[node.id]
            flags = tskit.NODE_IS_SAMPLE
            if strain in exact:
                flags |= NODE_IS_EXACT_MATCH
            path = [{"left": 0, "right": L + 1, "parent": 0}]
            sc2ts = {}
            if strain in recombinant:
                # Pick a breakpoint for false positives.
                bp = int(truth.ancestor_breakpoint[strain])
                if bp <= 0:
                    bp = L // 2
                path = [
                    {"left": 0, "right": bp + 1, "parent": 0},
                    {"left": bp + 1, "right": L + 1, "parent": 1},
                ]
                sc2ts["breakpoint_intervals"] = [[bp - 4, bp + 1]]
            sc2ts["hmm_match"] = {"path": path}
            metadata = {"strain": strain, "sc2ts": sc2ts}
        tables.nodes.add_row(
            flags=flags, time=node.time - min_time, metadata=metadata
        )
    tables.nodes.add_row(time=0, metadata={"strain": "Reference"})
    for edge in true_ts.edges():
        left = 0 if edge.left == 0 else edge.left + 1
        tables.edges.add_row(left, edge.right + 1, edge.parent, edge.child)
    for site in true_ts.sites():
        tables.sites.add_row(site.position + 1, site.ancestral_state)
    for mutation in true_ts.mutations():
        tables.mutations.add_row(
            mutation.site, node=mutation.node, derived_state=mutation.derived_state
        )
    tables.sort()
    # Put the samples in the reverse order.
    tables.subset(np.arange(tables.nodes.num_rows)[::-1])
    tables.sort()
    tables.build_index()
    tables.compute_mutation_parents()
    return tables.tree_sequence()


@pytest.fixture(scope="module")
def true_ts(exported):
    return tskit.load(exported / "true.trees")


@pytest.fixture(scope="module")
def truth(exported):
    return pd.read_csv(exported / "truth.csv")


class TestPrepareForComparison:

    def test_identical(self, true_ts, truth):
        inferred = to_sc2ts_style(true_ts, truth)
        assert inferred.samples()[0] != 0
        a, b = prepare_for_comparison(true_ts, inferred)
        assert a.num_samples == b.num_samples == true_ts.num_samples
        assert a.sequence_length == b.sequence_length == true_ts.sequence_length
        np.testing.assert_array_equal(
            a.nodes_time[a.samples()], b.nodes_time[b.samples()]
        )
        result = score_arg(a, b)
        assert result["arf"] == pytest.approx(0)
        assert result["tpr"] == pytest.approx(1)
        assert result["rmse"] == pytest.approx(0)

    def test_unplaced(self, true_ts, truth):
        unplaced = set(truth.strain[1:4])
        inferred = to_sc2ts_style(true_ts, truth, unplaced=unplaced)
        a, b = prepare_for_comparison(true_ts, inferred)
        assert a.num_samples == b.num_samples == true_ts.num_samples - 3
        result = score_arg(a, b)
        assert result["arf"] == pytest.approx(0)

    def test_different_topology(self, true_ts, truth):
        # Swapping the strains of two samples gives a different ARG.
        inferred = to_sc2ts_style(true_ts, truth)
        tables = inferred.dump_tables()
        u, v = inferred.samples()[[1, -2]]
        metadata = [inferred.node(j).metadata for j in range(inferred.num_nodes)]
        metadata[u], metadata[v] = metadata[v], metadata[u]
        tables.nodes.packset_metadata(
            [tables.nodes.metadata_schema.validate_and_encode_row(m) for m in metadata]
        )
        result = score_arg(*prepare_for_comparison(true_ts, tables.tree_sequence()))
        assert result["arf"] > 0
        assert result["tpr"] < 1

    def test_bad_sequence_length(self, true_ts, truth):
        with pytest.raises(ValueError):
            prepare_for_comparison(true_ts, true_ts)


def make_ts(L, times, edges, mutations=(), num_samples=None):
    """
    Return a tree sequence with nodes at the given times, the first
    ``num_samples`` of which (by default those at time 0) are samples, the
    given (left, right, parent, child) edges and a mutation on each of the
    given (position, node) pairs.
    """
    tables = tskit.TableCollection(L)
    for j, time in enumerate(times):
        is_sample = time == 0 if num_samples is None else j < num_samples
        tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE if is_sample else 0, time=time)
    for left, right, parent, child in edges:
        tables.edges.add_row(left, right, parent, child)
    for position, node in mutations:
        site = tables.sites.add_row(position, "A")
        tables.mutations.add_row(site, node=node, derived_state="T")
    tables.sort()
    return tables.tree_sequence()


# Four samples resolved as (((0, 1)4, 2)5, 3)6.
CATERPILLAR = dict(
    L=10,
    times=[0, 0, 0, 0, 1, 2, 3],
    edges=[
        (0, 10, 4, 0), (0, 10, 4, 1), (0, 10, 5, 4), (0, 10, 5, 2),
        (0, 10, 6, 5), (0, 10, 6, 3),
    ],
)


def assert_same_haplotypes(ts1, ts2):
    np.testing.assert_array_equal(ts1.samples(), ts2.samples())
    for v1, v2 in zip(ts1.variants(), ts2.variants(), strict=True):
        assert v1.site.position == v2.site.position
        np.testing.assert_array_equal(v1.states(), v2.states())


class TestCollapseUnsupported:

    def test_unsupported_node_collapsed(self):
        ts = make_ts(**CATERPILLAR, mutations=[(1, 5)])
        collapsed = collapse_unsupported(ts)
        assert collapsed.num_nodes == ts.num_nodes - 1
        tree = collapsed.first()
        assert tree.num_children(tree.parent(0)) == 3
        assert_same_haplotypes(ts, collapsed)

    def test_supported_nodes_kept(self):
        ts = make_ts(**CATERPILLAR, mutations=[(1, 4), (2, 5), (3, 6)])
        collapsed = collapse_unsupported(ts)
        assert collapsed.tables.nodes == ts.tables.nodes
        assert collapsed.tables.edges == ts.tables.edges

    def test_chain_collapsed_to_root(self):
        ts = make_ts(**CATERPILLAR)
        collapsed = collapse_unsupported(ts)
        # The unsupported root is kept, and all samples are its children.
        assert collapsed.num_nodes == 5
        tree = collapsed.first()
        assert tree.num_roots == 1
        assert tree.num_children(tree.root) == 4
        assert collapsed.node(tree.root).time == 3
        assert_same_haplotypes(ts, collapsed)

    def test_sample_kept(self):
        # Sample 2 at time 1 is the ancestor of samples 0 and 1, and has no
        # mutations, but samples are never collapsed.
        ts = make_ts(
            L=10,
            times=[0, 0, 1, 0, 2],
            edges=[(0, 10, 2, 0), (0, 10, 2, 1), (0, 10, 4, 2), (0, 10, 4, 3)],
            mutations=[(1, 4)],
            num_samples=4,
        )
        collapsed = collapse_unsupported(ts)
        assert collapsed.tables.edges == ts.tables.edges

    def test_recombinant_kept(self):
        # Node 3 is (0, 1), with parent 4 on [0, 5) and 5 on [5, 10), each
        # with sample 2 as its other child.
        ts = make_ts(
            L=10,
            times=[0, 0, 0, 1, 2, 2],
            edges=[
                (0, 10, 3, 0), (0, 10, 3, 1),
                (0, 5, 4, 3), (0, 5, 4, 2), (5, 10, 5, 3), (5, 10, 5, 2),
            ],
            mutations=[(1, 4), (6, 5)],
        )
        collapsed = collapse_unsupported(ts)
        assert collapsed.tables.nodes == ts.tables.nodes
        assert collapsed.tables.edges == ts.tables.edges

    def test_partial_parent(self):
        # Node 3 is (0, 1) everywhere, but only has a parent (4) on [0, 5); on
        # [5, 10) it's a root, so 0 and 1 stay attached to it there.
        ts = make_ts(
            L=10,
            times=[0, 0, 0, 1, 2],
            edges=[
                (0, 10, 3, 0), (0, 10, 3, 1), (0, 5, 4, 3), (0, 5, 4, 2),
            ],
        )
        collapsed = collapse_unsupported(ts)
        assert collapsed.num_nodes == ts.num_nodes
        left, right = collapsed.at(0), collapsed.at(5)
        assert left.parent(0) == left.parent(1) == left.parent(2)
        assert collapsed.node(left.parent(0)).time == 2
        assert right.parent(0) == right.parent(1)
        assert collapsed.node(right.parent(0)).time == 1
        assert right.parent(2) == tskit.NULL
        assert_same_haplotypes(ts, collapsed)

    @pytest.mark.parametrize("last_gen_only", [False, True])
    def test_slim(self, true_ts, last_gen_only):
        # Every individual is sampled in the fixture; keeping only the last
        # generation gives some non-sample nodes.
        ts = true_ts
        if last_gen_only:
            samples = ts.samples()
            ts = ts.simplify(samples[ts.nodes_time[samples] == 0])
        collapsed = collapse_unsupported(ts)
        assert_same_haplotypes(ts, collapsed)
        has_mutation = np.zeros(collapsed.num_nodes, dtype=bool)
        has_mutation[collapsed.mutations_node] = True
        for u in np.unique(collapsed.edges_parent):
            parents = set(collapsed.edges_parent[collapsed.edges_child == u])
            assert (
                collapsed.node(u).is_sample() or has_mutation[u] or len(parents) != 1
            )


class TestMakeSamplesLeaves:

    def assert_samples_leaves(self, ts, leaves):
        assert_same_haplotypes(ts, leaves)
        assert not np.any(np.isin(leaves.samples(), leaves.edges_parent))
        np.testing.assert_array_equal(
            leaves.nodes_time[leaves.samples()], ts.nodes_time[ts.samples()]
        )

    def test_no_internal_samples(self):
        ts = make_ts(**CATERPILLAR, mutations=[(1, 4), (2, 0)])
        leaves = make_samples_leaves(ts)
        assert leaves.tables.nodes == ts.tables.nodes
        assert leaves.tables.edges == ts.tables.edges

    def test_sample_ancestor(self):
        # Sample 2 at time 1, with a mutation, is the ancestor of samples 0
        # and 1, and a sibling of sample 3 under root 4.
        ts = make_ts(
            L=10,
            times=[0, 0, 1, 0, 2],
            edges=[(0, 10, 2, 0), (0, 10, 2, 1), (0, 10, 4, 2), (0, 10, 4, 3)],
            mutations=[(1, 2), (2, 0)],
            num_samples=4,
        )
        leaves = make_samples_leaves(ts)
        self.assert_samples_leaves(ts, leaves)
        assert leaves.num_nodes == ts.num_nodes + 1
        tree = leaves.first()
        new = tree.parent(2)
        assert new == ts.num_nodes
        assert leaves.node(new).time == pytest.approx(1)
        assert leaves.node(new).time > 1
        assert set(tree.children(new)) == {0, 1, 2}
        assert tree.parent(new) == 4
        assert leaves.site(0).mutations[0].node == new

    def test_sample_root(self):
        ts = make_ts(
            L=10, times=[0, 0, 1], edges=[(0, 10, 2, 0), (0, 10, 2, 1)],
            num_samples=3,
        )
        leaves = make_samples_leaves(ts)
        self.assert_samples_leaves(ts, leaves)
        tree = leaves.first()
        assert tree.num_roots == 1
        assert set(tree.children(tree.root)) == {0, 1, 2}

    def test_recombinant_sample(self):
        # Sample 3 has parent 4 on [0, 5) and 5 on [5, 10), and is the
        # ancestor of samples 0 and 1.
        ts = make_ts(
            L=10,
            times=[0, 0, 0, 1, 2, 2],
            edges=[
                (0, 10, 3, 0), (0, 10, 3, 1),
                (0, 5, 4, 3), (0, 5, 4, 2), (5, 10, 5, 3), (5, 10, 5, 2),
            ],
            mutations=[(1, 4), (6, 5)],
            num_samples=4,
        )
        leaves = make_samples_leaves(ts)
        self.assert_samples_leaves(ts, leaves)
        new = leaves.first().parent(3)
        assert leaves.at(0).parent(new) == 4
        assert leaves.at(5).parent(new) == 5

    def test_chain_of_samples(self):
        # Sample 2 is the parent of sample 1, which is the parent of sample 0.
        ts = make_ts(
            L=10, times=[0, 1, 2], edges=[(0, 10, 1, 0), (0, 10, 2, 1)],
            mutations=[(1, 1), (2, 0)],
            num_samples=3,
        )
        leaves = make_samples_leaves(ts)
        self.assert_samples_leaves(ts, leaves)
        assert leaves.num_nodes == 5

    def test_slim(self, true_ts):
        self.assert_samples_leaves(true_ts, make_samples_leaves(true_ts))


class TestScoreArg:

    def test_identical(self):
        ts = make_ts(**CATERPILLAR)
        result = score_arg(ts, ts)
        assert result["arf"] == result["arf_internal"] == pytest.approx(0)
        assert result["tpr"] == result["tpr_internal"] == pytest.approx(1)

    def test_no_internal_nodes(self):
        ts = make_ts(L=10, times=[0, 1], edges=[(0, 10, 1, 0)], num_samples=2)
        result = score_arg(ts, ts)
        assert np.isnan(result["arf_internal"])
        assert np.isnan(result["tpr_internal"])

    def test_star_inferred(self):
        # Only the root of the caterpillar is in the star, so a third of the
        # true internal span is matched. Without the unsupported nodes, the
        # true ARG is the star.
        true = make_ts(**CATERPILLAR)
        inferred = make_ts(
            L=10, times=[0, 0, 0, 0, 3], edges=[(0, 10, 4, j) for j in range(4)]
        )
        result = score_arg(true, inferred)
        assert result["arf_internal"] == pytest.approx(0)
        assert result["tpr_internal"] == pytest.approx(1 / 3)
        # Samples: 4 * 10 of the 7 * 10 true span.
        assert result["tpr"] == pytest.approx(5 / 7)
        result = score_arg(
            collapse_unsupported(true), collapse_unsupported(inferred)
        )
        assert result["tpr"] == result["tpr_internal"] == pytest.approx(1)
        assert result["arf"] == result["arf_internal"] == pytest.approx(0)

    def test_sample_identical_to_ancestor(self):
        # In the true ARG, unsampled node 3 carries a mutation and has
        # children sample 0, identical to it, and sample 1, with another
        # mutation. sc2ts can't infer node 3, so puts sample 1 under sample 0.
        # The two can't be told apart from the sequences.
        true = make_ts(
            L=10,
            times=[0, 0, 0, 1, 2],
            edges=[(0, 10, 3, 0), (0, 10, 3, 1), (0, 10, 4, 3), (0, 10, 4, 2)],
            mutations=[(1, 3), (2, 1)],
        )
        inferred = make_ts(
            L=10,
            times=[1, 0, 0, 2],
            edges=[(0, 10, 0, 1), (0, 10, 3, 0), (0, 10, 3, 2)],
            mutations=[(1, 0), (2, 1)],
            num_samples=3,
        )
        assert_same_haplotypes(true, inferred)
        result = score_arg(true, inferred)
        assert result["tpr"] < 1
        assert result["arf"] > 0
        result = score_arg(
            *[collapse_unsupported(make_samples_leaves(ts)) for ts in (true, inferred)]
        )
        assert result["tpr"] == result["tpr_internal"] == pytest.approx(1)
        assert result["arf"] == result["arf_internal"] == pytest.approx(0)


def score(samples):
    return score_recombinants(samples, event_table(samples))


def make_descendants(truth, strains, ancestor=10**6, breakpoint=50):
    """
    Return a copy of the truth table with the given non-recombinant samples
    made clonal descendants of the same unsampled recombinant.
    """
    truth = truth.copy()
    rows = truth.strain.isin(strains)
    assert not truth.is_recombinant[rows].any()
    truth.loc[rows, "recombinant_ancestor"] = ancestor
    truth.loc[rows, "ancestor_breakpoint"] = breakpoint
    truth.loc[rows, "ancestor_detectable"] = True
    truth.loc[rows, "sampled_between"] = np.nan
    return truth


class TestClassifySamples:
    # Every individual is sampled in the fixture, so if everything is placed
    # the expected recombinants are exactly the sampled ones.

    def test_perfect(self, true_ts, truth):
        samples = classify_samples(to_sc2ts_style(true_ts, truth), truth)
        assert samples.strain.tolist() == truth.strain.tolist()
        assert samples.placed.all()
        np.testing.assert_array_equal(samples.expected_recombinant, truth.is_recombinant)
        np.testing.assert_array_equal(
            samples.inferred_recombinant, samples.expected_recombinant
        )
        rec = samples[samples.expected_recombinant]
        np.testing.assert_array_equal(rec.inferred_breakpoint, rec.ancestor_breakpoint)
        assert (rec.breakpoint_interval_left == rec.ancestor_breakpoint - 5).all()
        assert (rec.breakpoint_interval_right == rec.ancestor_breakpoint).all()
        assert (samples[~samples.expected_recombinant].num_inferred_parents == 1).all()

        result = score(samples)
        n = truth.is_recombinant.sum()
        assert result["num_samples"] == result["num_placed"] == len(truth)
        assert result["num_sampled_recombinants"] == n
        assert result["num_expected_recombinants"] == n
        assert result["num_events"] == n
        assert result["num_detectable_events"] == truth.ancestor_detectable[
            truth.is_recombinant].sum()
        assert result["events_detected"] == n
        assert result["true_positives"] == n
        assert result["false_positives"] == 0
        assert result["precision"] == 1
        assert result["recall"] == 1
        assert result["recall_detectable"] == 1
        assert result["mean_abs_breakpoint_error"] == 0
        assert result["breakpoint_in_interval"] == 1

    def test_missed_and_spurious(self, true_ts, truth):
        recombinants = list(truth.strain[truth.is_recombinant])
        spurious = truth.strain[~truth.is_recombinant].iloc[-1]
        recombinant = set(recombinants[1:]) | {spurious}
        inferred = to_sc2ts_style(true_ts, truth, recombinant=recombinant)
        result = score(classify_samples(inferred, truth))
        n = len(recombinants)
        assert result["true_positives"] == n - 1
        assert result["false_positives"] == 1
        assert result["events_detected"] == n - 1
        assert result["precision"] == pytest.approx((n - 1) / n)
        assert result["recall"] == pytest.approx((n - 1) / n)

    def test_descendant_of_unsampled_recombinant(self, true_ts, truth):
        # sc2ts reporting a descendant of an unsampled recombinant as a
        # recombinant is correct.
        strain = truth.strain[~truth.is_recombinant].iloc[-1]
        truth = make_descendants(truth, [strain])
        recombinant = set(truth.strain[truth.is_recombinant]) | {strain}
        inferred = to_sc2ts_style(true_ts, truth, recombinant=recombinant)
        samples = classify_samples(inferred, truth).set_index("strain")
        assert samples.expected_recombinant[strain]
        assert samples.inferred_breakpoint[strain] == 50
        result = score(samples.reset_index())
        n = truth.is_recombinant.sum()
        assert result["num_sampled_recombinants"] == n
        assert result["num_expected_recombinants"] == n + 1
        assert result["true_positives"] == n + 1
        assert result["false_positives"] == 0
        assert result["recall"] == 1
        assert result["mean_abs_breakpoint_error"] == 0

    def test_event_detected_once(self, true_ts, truth):
        # Two descendants of the same unsampled recombinant: once one is
        # inferred to be a recombinant, the other can match it directly.
        strains = list(truth.strain[~truth.is_recombinant].iloc[-2:])
        truth = make_descendants(truth, strains)
        recombinant = set(truth.strain[truth.is_recombinant]) | {strains[0]}
        inferred = to_sc2ts_style(true_ts, truth, recombinant=recombinant)
        result = score(classify_samples(inferred, truth))
        n = truth.is_recombinant.sum()
        assert result["num_expected_recombinants"] == n + 2
        assert result["num_events"] == n + 1
        events = event_table(classify_samples(inferred, truth)).set_index(
            "recombinant_ancestor")
        event = events.loc[10**6]
        assert event.num_samples == 2
        assert not event.sampled
        assert event.detectable
        assert event.detected
        assert event.breakpoint == 50
        assert event.breakpoint_error == 0
        assert event.breakpoint_in_interval
        assert event.interval_width == 5
        assert result["events_detected"] == n + 1
        assert result["recall"] == 1
        assert result["false_positives"] == 0

    def test_event_missed(self, true_ts, truth):
        strains = list(truth.strain[~truth.is_recombinant].iloc[-2:])
        truth = make_descendants(truth, strains)
        inferred = to_sc2ts_style(true_ts, truth)
        result = score(classify_samples(inferred, truth))
        n = truth.is_recombinant.sum()
        assert result["num_events"] == n + 1
        assert result["events_detected"] == n
        assert result["recall"] == pytest.approx(n / (n + 1))
        events = event_table(classify_samples(inferred, truth)).set_index(
            "recombinant_ancestor")
        assert not events.detected[10**6]
        assert np.isnan(events.interval_width[10**6])

    def test_unplaced_recombinant(self, true_ts, truth):
        # Find a recombinant whose clonal child is sampled, with nothing else
        # in between.
        rec = truth[truth.is_recombinant]
        children = truth[truth.sampled_between.isin(rec.strain)]
        child = children.iloc[0]
        parent = child.sampled_between
        inferred = to_sc2ts_style(true_ts, truth, unplaced={parent})
        samples = classify_samples(inferred, truth).set_index("strain")
        assert not samples.placed[parent]
        assert not samples.inferred_recombinant[parent]
        assert samples.num_inferred_parents[parent] == -1
        # With the recombinant missing from the ARG, its child should be
        # inferred to be a recombinant in its place.
        assert samples.expected_recombinant[child.strain]
        result = score(samples.reset_index())
        assert result["num_placed"] == len(truth) - 1
        assert result["num_exact_matches"] == 0
        assert result["num_held_back"] == 1
        assert result["num_expected_recombinants"] == (
            truth.is_recombinant.sum() - 1 + len(children[children.sampled_between == parent])
        )

    def test_exact_match(self, true_ts, truth):
        strain = truth.strain[~truth.is_recombinant].iloc[-1]
        inferred = to_sc2ts_style(true_ts, truth, exact={strain})
        samples = classify_samples(inferred, truth).set_index("strain")
        assert samples.placed[strain]
        assert samples.exact_match[strain]
        assert not samples.inferred_recombinant[strain]
        assert not samples.exact_match.drop(strain).any()
        result = score(samples.reset_index())
        assert result["num_placed"] == len(truth)
        assert result["num_exact_matches"] == 1
        assert result["num_held_back"] == 0

    def test_exact_match_recombinant(self, true_ts, truth):
        # A recombinant identical to a node already in the ARG is added as an
        # exact match, and accounts for its children's recombination.
        rec = truth[truth.is_recombinant]
        children = truth[truth.sampled_between.isin(rec.strain)]
        child = children.iloc[0]
        parent = child.sampled_between
        recombinant = set(rec.strain) - {parent}
        inferred = to_sc2ts_style(
            true_ts, truth, recombinant=recombinant, exact={parent}
        )
        samples = classify_samples(inferred, truth).set_index("strain")
        assert samples.placed[parent]
        assert samples.exact_match[parent]
        assert not samples.expected_recombinant[child.strain]
        # The recombinant itself is expected, but its single-parent exact
        # match means the event is missed.
        assert samples.expected_recombinant[parent]
        assert not samples.inferred_recombinant[parent]
        result = score(samples.reset_index())
        assert result["events_detected"] == result["num_events"] - 1

    def test_no_recombinants(self, true_ts, truth):
        inferred = to_sc2ts_style(true_ts, truth, recombinant=set())
        result = score(classify_samples(inferred, truth))
        assert result["true_positives"] == 0
        assert np.isnan(result["precision"])
        assert result["recall"] == 0
        assert np.isnan(result["mean_abs_breakpoint_error"])


class TestNumRecombinantNodes:

    def test_simple(self):
        tables = tskit.TableCollection(10)
        for _ in range(4):
            tables.nodes.add_row(time=1)
        tables.nodes.add_row(time=0)
        tables.nodes.add_row(time=0)
        # Node 4 has two parents; node 5 has one parent over two edges.
        tables.edges.add_row(0, 5, 0, 4)
        tables.edges.add_row(5, 10, 1, 4)
        tables.edges.add_row(0, 5, 2, 5)
        tables.edges.add_row(5, 10, 2, 5)
        tables.sort()
        assert num_recombinant_nodes(tables.tree_sequence()) == 1


class TestPlacementTable:

    def test_counts(self, true_ts, truth):
        unplaced = set(truth.strain[1:4])
        exact = set(truth.strain[4:6])
        samples = classify_samples(
            to_sc2ts_style(true_ts, truth, unplaced=unplaced, exact=exact), truth
        )
        df = placement_table(samples)
        assert df.gen.tolist() == sorted(truth.gen.unique())
        assert df.num_samples.sum() == len(truth)
        assert df.num_placed.sum() == len(truth) - 3
        assert df.num_exact_matches.sum() == 2
        expected = truth.groupby("gen").size()
        np.testing.assert_array_equal(df.num_samples, expected.values)


class TestCli:

    def test_run(self, exported, true_ts, truth, tmp_path):
        inferred_path = tmp_path / "inferred.ts"
        to_sc2ts_style(true_ts, truth).dump(inferred_path)
        result = CliRunner().invoke(
            evaluate,
            [str(exported / "true.trees"), str(inferred_path),
             str(exported / "truth.csv"), str(tmp_path),
             "--pathogen", "x", "--rep", "0", "--samples-per-day", "1000", "--k", "4"],
        )
        assert result.exit_code == 0, result.output
        df = pd.read_csv(tmp_path / "evaluation.csv")
        assert len(df) == 1
        row = df.iloc[0]
        assert row.pathogen == "x"
        assert row.k == 4
        assert row.arf == pytest.approx(0)
        assert row.tpr_resolved == pytest.approx(1)
        # Every individual is sampled in this simulation.
        assert np.isnan(row.tpr_internal)
        assert row.false_positives == 0
        assert row.median_interval_width == 5
        events = pd.read_csv(tmp_path / "events.csv")
        assert len(events) == row.num_events == truth.is_recombinant.sum()
        placement = pd.read_csv(tmp_path / "placement.csv")
        assert placement.num_samples.sum() == len(truth)
        for df in [events, placement]:
            assert list(df.columns[:4]) == ["pathogen", "rep", "samples_per_day", "k"]
            assert (df.k == 4).all()
        samples = pd.read_csv(tmp_path / "samples.csv")
        assert len(samples) == len(truth)
