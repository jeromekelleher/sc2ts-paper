import numpy as np
import pandas as pd
import pytest
import tskit
from click.testing import CliRunner

from scripts.evaluate import (
    classify_samples,
    num_recombinant_nodes,
    prepare_for_comparison,
    run,
    score_arg,
    score_recombinants,
)


def to_sc2ts_style(true_ts, truth, recombinant=None, unplaced=()):
    """
    Return a copy of the true ARG laid out as sc2ts would infer it: 1-based
    coordinates with sequence length L + 1, times relative to the last sample,
    strain and HMM match metadata on sample nodes, a non-sample reference node
    and the samples in a different order. Samples named in ``recombinant`` (by
    default the true recombinants) are given two-parent HMM matches at their
    true breakpoints, and the others single-parent matches; ``unplaced`` samples
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
                bp = int(truth.breakpoint[strain])
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


class TestClassifySamples:

    def test_perfect(self, true_ts, truth):
        samples = classify_samples(to_sc2ts_style(true_ts, truth), truth)
        assert samples.strain.tolist() == truth.strain.tolist()
        assert samples.placed.all()
        np.testing.assert_array_equal(
            samples.inferred_recombinant, samples.is_recombinant
        )
        rec = samples[samples.is_recombinant]
        np.testing.assert_array_equal(rec.inferred_breakpoint, rec.breakpoint)
        assert (rec.breakpoint_interval_left == rec.breakpoint - 5).all()
        assert (rec.breakpoint_interval_right == rec.breakpoint).all()
        assert (samples[~samples.is_recombinant].num_inferred_parents == 1).all()

        score = score_recombinants(samples)
        assert score["num_samples"] == score["num_placed"] == len(truth)
        assert score["num_true_recombinants"] == truth.is_recombinant.sum()
        assert score["true_positives"] == truth.is_recombinant.sum()
        assert score["false_positives"] == 0
        assert score["false_negatives"] == 0
        assert score["precision"] == 1
        assert score["recall"] == 1
        assert score["recall_detectable"] == 1
        assert score["mean_abs_breakpoint_error"] == 0
        assert score["breakpoint_in_interval"] == 1

    def test_missed_and_spurious(self, true_ts, truth):
        true_recombinants = list(truth.strain[truth.is_recombinant])
        missed = true_recombinants[0]
        spurious = truth.strain[~truth.is_recombinant].iloc[-1]
        recombinant = set(true_recombinants[1:]) | {spurious}
        inferred = to_sc2ts_style(true_ts, truth, recombinant=recombinant)
        score = score_recombinants(classify_samples(inferred, truth))
        n = len(true_recombinants)
        assert score["true_positives"] == n - 1
        assert score["false_positives"] == 1
        assert score["false_negatives"] == 1
        assert score["precision"] == pytest.approx((n - 1) / n)
        assert score["recall"] == pytest.approx((n - 1) / n)

    def test_unplaced(self, true_ts, truth):
        unplaced = truth.strain[truth.is_recombinant].iloc[0]
        inferred = to_sc2ts_style(true_ts, truth, unplaced={unplaced})
        samples = classify_samples(inferred, truth)
        row = samples.set_index("strain").loc[unplaced]
        assert not row.placed
        assert not row.inferred_recombinant
        assert row.num_inferred_parents == -1
        score = score_recombinants(samples)
        assert score["num_placed"] == len(truth) - 1
        # Unplaced samples are excluded rather than counted as missed.
        assert score["false_negatives"] == 0
        assert score["num_true_recombinants"] == truth.is_recombinant.sum() - 1

    def test_no_recombinants(self, true_ts, truth):
        inferred = to_sc2ts_style(true_ts, truth, recombinant=set())
        score = score_recombinants(classify_samples(inferred, truth))
        assert score["true_positives"] == 0
        assert np.isnan(score["precision"])
        assert score["recall"] == 0
        assert np.isnan(score["mean_abs_breakpoint_error"])


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


class TestCli:

    def test_run(self, exported, true_ts, truth, tmp_path):
        inferred_path = tmp_path / "inferred.ts"
        to_sc2ts_style(true_ts, truth).dump(inferred_path)
        output = tmp_path / "evaluation.csv"
        output_samples = tmp_path / "samples.csv"
        result = CliRunner().invoke(
            run,
            [str(exported / "true.trees"), str(inferred_path),
             str(exported / "truth.csv"), str(output), str(output_samples),
             "--pathogen", "x", "--rep", "0", "--p", "1.0", "--k", "4"],
        )
        assert result.exit_code == 0, result.output
        df = pd.read_csv(output)
        assert len(df) == 1
        row = df.iloc[0]
        assert row.pathogen == "x"
        assert row.k == 4
        assert row.arf == pytest.approx(0)
        assert row.false_positives == 0
        assert len(pd.read_csv(output_samples)) == len(truth)
