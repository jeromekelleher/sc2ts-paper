import numpy as np
import pandas as pd
import pytest
import tskit
from sc2ts.core import NODE_IS_RECOMBINANT

from pipeline import recombinant_table, relative_segments


def make_true_ts(relative_recombines=True):
    """
    Return a true ARG with a focal sample x (node 0, pedigree ID 10) from a
    later generation than samples R (node 1, ID 11) and y (node 2, ID 12). x
    and R descend from a, and y from b. If ``relative_recombines``, R is a
    recombinant that takes the right half of the genome from b instead, so x's
    nearest earlier relative is R on the left but not on the right.
    """
    tables = tskit.TableCollection(100)
    tables.individuals.metadata_schema = tskit.MetadataSchema.permissive_json()
    for pedigree_id in [10, 11, 12]:
        tables.individuals.add_row(metadata={"pedigree_id": pedigree_id})
    for j, time in enumerate([0, 1, 1]):
        tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=time, individual=j)
    a = tables.nodes.add_row(time=2)
    b = tables.nodes.add_row(time=2)
    root = tables.nodes.add_row(time=3)
    tables.edges.add_row(0, 100, a, 0)
    if relative_recombines:
        tables.edges.add_row(0, 50, a, 1)
        tables.edges.add_row(50, 100, b, 1)
    else:
        tables.edges.add_row(0, 100, a, 1)
    tables.edges.add_row(0, 100, b, 2)
    tables.edges.add_row(0, 100, root, a)
    tables.edges.add_row(0, 100, root, b)
    tables.sort()
    return tables.tree_sequence()


class TestRelativeSegments:

    def test_relative_recombinant(self):
        ts = make_true_ts()
        assert relative_segments(ts, 0, [1, 2]) == [(0, 50, 3), (50, 100, 5)]

    def test_no_recombination(self):
        ts = make_true_ts(relative_recombines=False)
        assert relative_segments(ts, 0, [1, 2]) == [(0, 100, 3)]

    def test_no_earlier_samples(self):
        ts = make_true_ts()
        assert relative_segments(ts, 0, []) == [(0, 100, -1)]


def make_inferred_ts():
    """
    Return an sc2ts-style ARG (nodes only) with two recombinant nodes: 1, in
    group g1, above sample seq_10, and 3, in group g2, above sample seq_11.
    seq_12 has a single parent.
    """
    tables = tskit.TableCollection(101)
    tables.nodes.metadata_schema = tskit.MetadataSchema.permissive_json()
    two_parents = {"path": [{"left": 0, "right": 51, "parent": 0},
                            {"left": 51, "right": 101, "parent": 0}]}
    one_parent = {"path": [{"left": 0, "right": 101, "parent": 0}]}
    tables.nodes.add_row(time=3, metadata={"strain": "Reference"})
    for group, strain, match in [
        ("g1", "seq_10", two_parents), ("g2", "seq_11", two_parents)
    ]:
        tables.nodes.add_row(
            flags=NODE_IS_RECOMBINANT,
            time=1,
            metadata={"sc2ts": {"group_id": group, "date_added": "2026-01-03"}},
        )
        tables.nodes.add_row(
            flags=tskit.NODE_IS_SAMPLE,
            time=0,
            metadata={"strain": strain, "sc2ts": {"group_id": group, "hmm_match": match}},
        )
    tables.nodes.add_row(
        flags=tskit.NODE_IS_SAMPLE,
        time=0,
        metadata={"strain": "seq_12", "sc2ts": {"group_id": "g3", "hmm_match": one_parent}},
    )
    return tables.tree_sequence()


def rematch(node, num_mutations, k1000_muts):
    mutation = {"site_position": 1, "inherited_state": "A", "derived_state": "T"}
    return {
        "recombinant": node,
        "original_match": {
            "path": [{"left": 0, "right": 51, "parent": 0},
                     {"left": 51, "right": 101, "parent": 0}],
            "mutations": [mutation] * num_mutations,
        },
        "no_recomb_match": {
            "path": [{"left": 0, "right": 101, "parent": 0}],
            "mutations": [mutation] * k1000_muts,
        },
    }


@pytest.fixture
def samples():
    # seq_10 is a false positive; seq_11 a true recombinant.
    return pd.DataFrame({
        "strain": ["seq_10", "seq_11", "seq_12"],
        "gen": [2, 1, 1],
        "placed": [True, True, True],
        "expected_recombinant": [False, True, False],
        "recombinant_ancestor": [-1, 11, -1],
        "ancestor_breakpoint": [-1, 50, -1],
        "ancestor_detectable": [False, True, False],
        "breakpoint_interval_left": [40, 45, -1],
        "breakpoint_interval_right": [60, 48, -1],
    })


class TestRecombinantTable:

    def test_table(self, samples):
        df = recombinant_table(
            make_true_ts(), make_inferred_ts(), samples,
            [rematch(1, 2, 5), rematch(3, 4, 12)],
        ).set_index("recombinant")
        assert list(df.index) == [1, 3]
        assert list(df.causal_strains) == ["seq_10", "seq_11"]
        assert list(df.num_mutations) == [2, 4]
        assert list(df.k1000_muts) == [5, 12]
        assert list(df.mutations_averted) == [3, 8]
        assert list(df.true_positive) == [False, True]
        # The false positive's relative R recombines at 50, inside its interval.
        assert df.category[1] == "relative_recombinant"
        assert df.relative_breakpoints[1] == "50"
        assert df.relative_breakpoint_in_interval[1]
        assert pd.isna(df.category[3])
        assert df.recombinant_ancestor[3] == 11
        # The true breakpoint, 50, is outside the interval [45, 48].
        assert not df.breakpoint_in_interval[3]

    def test_unresolved_branch(self, samples):
        df = recombinant_table(
            make_true_ts(relative_recombines=False), make_inferred_ts(), samples,
            [rematch(1, 2, 5), rematch(3, 4, 12)],
        ).set_index("recombinant")
        assert df.category[1] == "unresolved_branch"
        assert df.relative_breakpoints[1] == ""
        assert not df.relative_breakpoint_in_interval[1]

    def test_no_recombinants(self, samples):
        tables = make_inferred_ts().dump_tables()
        tables.nodes.flags = np.where(
            tables.nodes.flags == NODE_IS_RECOMBINANT, 0, tables.nodes.flags
        ).astype(np.uint32)
        df = recombinant_table(make_true_ts(), tables.tree_sequence(), samples, [])
        assert len(df) == 0
        assert "mutations_averted" in df.columns
