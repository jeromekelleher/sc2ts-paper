import numpy as np
import pandas as pd
import pytest
import tskit
from click.testing import CliRunner

from pipeline import (
    classify_samples,
    evaluate,
    event_table,
    num_recombinant_nodes,
    placement_table,
    prepare_for_comparison,
    score_arg,
    score_recombinants,
)


def to_sc2ts_style(true_ts, truth, recombinant=None, unplaced=()):
    """
    Return a copy of the true ARG laid out as sc2ts would infer it: 1-based
    coordinates with sequence length L + 1, times relative to the last sample,
    strain and HMM match metadata on sample nodes, a non-sample reference node
    and the samples in a different order. Samples named in ``recombinant`` (by
    default the sampled recombinants) are given two-parent HMM matches at their
    recombinant ancestor's breakpoints, and the others single-parent matches; ``unplaced`` samples
    are left as non-sample nodes.
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
    tables.sort()
    # Put the samples in the reverse order.
    tables.subset(np.arange(tables.nodes.num_rows)[::-1])
    tables.sort()
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
        assert result["num_expected_recombinants"] == (
            truth.is_recombinant.sum() - 1 + len(children[children.sampled_between == parent])
        )

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
        samples = classify_samples(
            to_sc2ts_style(true_ts, truth, unplaced=unplaced), truth
        )
        df = placement_table(samples)
        assert df.gen.tolist() == sorted(truth.gen.unique())
        assert df.num_samples.sum() == len(truth)
        assert df.num_placed.sum() == len(truth) - 3
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
             "--pathogen", "x", "--rep", "0", "--p", "1.0", "--k", "4"],
        )
        assert result.exit_code == 0, result.output
        df = pd.read_csv(tmp_path / "evaluation.csv")
        assert len(df) == 1
        row = df.iloc[0]
        assert row.pathogen == "x"
        assert row.k == 4
        assert row.arf == pytest.approx(0)
        assert row.false_positives == 0
        assert row.median_interval_width == 5
        events = pd.read_csv(tmp_path / "events.csv")
        assert len(events) == row.num_events == truth.is_recombinant.sum()
        placement = pd.read_csv(tmp_path / "placement.csv")
        assert placement.num_samples.sum() == len(truth)
        for df in [events, placement]:
            assert list(df.columns[:4]) == ["pathogen", "rep", "p", "k"]
            assert (df.k == 4).all()
        samples = pd.read_csv(tmp_path / "samples.csv")
        assert len(samples) == len(truth)
