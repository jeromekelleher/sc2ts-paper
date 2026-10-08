import pandas as pd
import pytest

from pipeline import (
    recombinant_ancestors,
    required_sequences,
    sample_individuals,
    sample_pedigree_ids,
    truth_table,
)


def make_pedigree(rows):
    return pd.DataFrame(
        rows,
        columns=["gen", "child", "parent1", "parent2", "is_clonal",
                 "copy_parent", "breakpoints", "is_recombinant"],
    )


# 0 is the founder, with clonal children 1 and 2. 3 is a recombinant of 1 and 2
# that differs from both; 4 is a recombinant identical to 1. 5 is a clonal child
# of 3, and 6 a clonal child of 5.
PEDIGREE = make_pedigree([
    [0, 0, -1, -1, 0, -1, ".", 0],
    [1, 1, 0, 0, 1, 0, ".", 0],
    [1, 2, 0, 0, 1, 0, ".", 0],
    [2, 3, 1, 2, 0, 1, "2", 1],
    [2, 4, 1, 2, 0, 1, "1", 1],
    [3, 5, 3, 3, 1, 3, ".", 0],
    [4, 6, 5, 5, 1, 5, ".", 0],
])
SEQUENCES = {
    "seq_0": "AAAA",
    "seq_1": "CAAA",
    "seq_2": "AAAT",
    "seq_3": "CAAT",
    "seq_4": "CAAA",
    "seq_5": "CAAT",
    "seq_6": "CGAT",
}


class TestRecombinantAncestors:

    @pytest.mark.parametrize(
        "ids, ancestors, between",
        [
            # Recombinants are their own ancestor.
            ([0, 3, 4], [-1, 3, 4], [[], [], []]),
            # Non-recombinant lineages have none.
            ([0, 1, 2], [-1, -1, -1], [[], [], []]),
            # Descendants of an unsampled recombinant.
            ([0, 5], [-1, 3], [[], []]),
            ([0, 6], [-1, 3], [[], []]),
            # Sampled individuals in between, including the recombinant.
            ([0, 3, 5, 6], [-1, 3, 3, 3], [[], [], [3], [5, 3]]),
            ([0, 5, 6], [-1, 3, 3], [[], [], [5]]),
        ],
    )
    def test_ancestors(self, ids, ancestors, between):
        assert recombinant_ancestors(PEDIGREE, ids) == (ancestors, between)

    def test_required_sequences(self):
        assert required_sequences(PEDIGREE, [0, 1, 6]) == {1, 2, 3}
        assert required_sequences(PEDIGREE, [0, 1]) == set()


class TestTruthTable:

    def test_samples(self):
        df = truth_table(PEDIGREE, [0, 1, 3, 4, 6], SEQUENCES)
        assert df.strain.tolist() == ["seq_0", "seq_1", "seq_3", "seq_4", "seq_6"]
        assert df.gen.tolist() == [0, 1, 2, 2, 4]
        assert df.is_recombinant.tolist() == [False, False, True, True, False]
        assert df.breakpoint.tolist() == [-1, -1, 2, 1, -1]
        assert df.recombinant_ancestor.tolist() == [-1, -1, 3, 4, 3]
        assert df.ancestor_breakpoint.tolist() == [-1, -1, 2, 1, 2]
        # 4's breakpoint is where its parents don't differ.
        assert df.ancestor_detectable.tolist() == [False, False, True, False, True]
        assert df.sampled_between.tolist() == ["", "", "", "", "seq_3"]

    def test_descendant(self):
        df = truth_table(PEDIGREE, [0, 5, 6], SEQUENCES)
        assert df.recombinant_ancestor.tolist() == [-1, 3, 3]
        assert df.sampled_between.tolist() == ["", "", "seq_5"]


class TestSlim:

    @pytest.mark.parametrize("probability", [0.2, 0.5, 1])
    def test_ancestors(self, slim_ts, slim_prefix, probability):
        pedigree = pd.read_csv(f"{slim_prefix}.slim.pedigree.tsv", sep="\t")
        df = pedigree.set_index("child")
        ids = sample_pedigree_ids(sample_individuals(slim_ts, probability, seed=1))
        sampled = set(ids)
        ancestors, between = recombinant_ancestors(pedigree, ids)
        num_unaccounted = 0
        for child, ancestor, path in zip(ids, ancestors, between):
            if df.is_recombinant[child]:
                assert ancestor == child
                assert path == []
            elif ancestor >= 0:
                # Walking up the clonal line reaches the ancestor, passing
                # through exactly the sampled individuals listed.
                expected = []
                u = df.parent1[child]
                while u != ancestor:
                    assert not df.is_recombinant[u]
                    if u in sampled:
                        expected.append(u)
                    u = df.parent1[u]
                if ancestor in sampled:
                    expected.append(ancestor)
                assert path == expected
                num_unaccounted += len(path) == 0
            else:
                assert path == []
        if probability == 1:
            assert num_unaccounted == 0
        else:
            assert num_unaccounted > 0

    def test_truth_csv(self, exported, slim_prefix):
        truth = pd.read_csv(exported / "truth.csv", keep_default_na=False)
        pedigree = pd.read_csv(f"{slim_prefix}.slim.pedigree.tsv", sep="\t")
        pedigree = pedigree.set_index("child").loc[[int(s[4:]) for s in truth.strain]]
        assert truth.is_recombinant.tolist() == pedigree.is_recombinant.astype(bool).tolist()
        assert truth.is_recombinant.sum() > 0
        rec = truth[truth.is_recombinant]
        assert (rec.recombinant_ancestor == rec.strain.str[4:].astype(int)).all()
        assert (rec.ancestor_breakpoint == rec.breakpoint).all()
        assert (rec.sampled_between == "").all()
        assert 0 < rec.ancestor_detectable.sum() <= len(rec)
        # Everything is sampled, so every other recombinant ancestor is
        # accounted for by a sampled individual.
        other = truth[~truth.is_recombinant & (truth.recombinant_ancestor >= 0)]
        assert len(other) > 0
        assert (other.sampled_between != "").all()
