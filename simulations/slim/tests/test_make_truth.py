import numpy as np
import pandas as pd
import pytest

from pipeline import read_candidates, read_fasta, strain_name, truth_table


def walk_pedigree(pedigree, ids, sampled):
    """
    The reference for what SLiM tracks: for each individual, walk up its clonal
    line in the full pedigree to its nearest recombinant ancestor (itself, if it
    is a recombinant), or to the founder if there is none (ancestor -1),
    collecting the sampled individuals on the way, including the ancestor if it
    is sampled and not the individual itself.
    """
    parent = dict(zip(pedigree.child, pedigree.parent1))
    is_recombinant = dict(zip(pedigree.child, pedigree.is_recombinant.astype(bool)))
    ancestors = []
    between = []
    for child in ids:
        ancestor = -1
        path = []
        u = child
        while True:
            if is_recombinant[u]:
                ancestor = u
                break
            # Non-recombinants have parent1 == parent2.
            u = parent[u]
            if u < 0:
                break
            if u in sampled:
                path.append(u)
        ancestors.append(ancestor)
        between.append(path)
    return ancestors, between


@pytest.fixture(scope="module")
def candidates(slim_prefix):
    return read_candidates(f"{slim_prefix}.slim.samples.tsv")


@pytest.fixture(scope="module")
def pedigree(slim_prefix):
    return pd.read_csv(f"{slim_prefix}.slim.pedigree.tsv", sep="\t")


@pytest.fixture(scope="module")
def recombinants(slim_prefix):
    return pd.read_csv(f"{slim_prefix}.slim.recombinants.tsv", sep="\t")


class TestSlimAncestry:
    # The ancestry SLiM tracks agrees with walking the full pedigree.

    def test_recombinant_ancestor(self, candidates, pedigree):
        ancestors, _ = walk_pedigree(pedigree, candidates.index, set(candidates.index))
        np.testing.assert_array_equal(candidates.recombinant_ancestor, ancestors)
        assert (candidates.recombinant_ancestor >= 0).any()

    def test_prev_candidate(self, candidates, pedigree):
        _, between = walk_pedigree(pedigree, candidates.index, set(candidates.index))
        previous = [path[0] if len(path) > 0 else -1 for path in between]
        np.testing.assert_array_equal(candidates.prev_candidate, previous)
        assert (candidates.prev_candidate >= 0).any()

    def test_is_recombinant(self, candidates, pedigree):
        expected = pedigree.set_index("child").is_recombinant[candidates.index]
        np.testing.assert_array_equal(candidates.is_recombinant, expected)

    def test_recombinants(self, recombinants, pedigree):
        expected = pedigree[pedigree.is_recombinant == 1]
        assert recombinants.id.tolist() == expected.child.tolist()
        assert recombinants.gen.tolist() == expected.gen.tolist()
        assert recombinants.parent1.tolist() == expected.parent1.tolist()
        assert recombinants.parent2.tolist() == expected.parent2.tolist()
        assert (recombinants.parent1 != recombinants.parent2).all()
        assert ((recombinants.breakpoint > 0) & (recombinants.breakpoint < 200)).all()

    def test_detectable(self, recombinants, slim_prefix):
        ids = set(recombinants.id) | set(recombinants.parent1) | set(recombinants.parent2)
        sequences = read_fasta(
            f"{slim_prefix}.slim.all_sequences.fa", [strain_name(u) for u in ids]
        )
        expected = [
            sequences[strain_name(row.id)] != sequences[strain_name(row.parent1)]
            and sequences[strain_name(row.id)] != sequences[strain_name(row.parent2)]
            for row in recombinants.itertuples()
        ]
        assert recombinants.detectable.astype(bool).tolist() == expected
        assert 0 < sum(expected)

    def test_candidate_sequences(self, candidates, slim_prefix):
        names = [strain_name(u) for u in candidates.index]
        assert read_fasta(f"{slim_prefix}.slim.sequences.fa", names) == read_fasta(
            f"{slim_prefix}.slim.all_sequences.fa", names
        )


# Candidates: 0 is the founder. 3 is a recombinant candidate; 5 a candidate
# clonal descendant of 3 and 6 a candidate clonal descendant of 5. 7 is a
# candidate descendant of the recombinant 4, which isn't a candidate, and 8 a
# candidate descendant of the founder with no recombinant ancestor.
CANDIDATES = pd.DataFrame(
    [
        [0, 0, 0, 0, -1, -1],
        [2, 3, 0, 1, 3, -1],
        [3, 5, 0, 0, 3, 3],
        [4, 6, 0, 0, 3, 5],
        [4, 7, 1, 0, 4, -1],
        [4, 8, 2, 0, -1, 0],
    ],
    columns=["gen", "id", "rank", "is_recombinant", "recombinant_ancestor",
             "prev_candidate"],
).set_index("id")
RECOMBINANTS = pd.DataFrame(
    [[2, 3, 1, 2, 120, 1], [2, 4, 1, 2, 60, 0]],
    columns=["gen", "id", "parent1", "parent2", "breakpoint", "detectable"],
)


class TestTruthTable:

    def test_columns(self):
        df = truth_table([0, 3, 5, 6, 7, 8], CANDIDATES, RECOMBINANTS)
        assert df.strain.tolist() == ["seq_0", "seq_3", "seq_5", "seq_6", "seq_7", "seq_8"]
        assert df.gen.tolist() == [0, 2, 3, 4, 4, 4]
        assert df.is_recombinant.tolist() == [False, True, False, False, False, False]
        assert df.breakpoint.tolist() == [-1, 120, -1, -1, -1, -1]
        assert df.recombinant_ancestor.tolist() == [-1, 3, 3, 3, 4, -1]
        assert df.ancestor_breakpoint.tolist() == [-1, 120, 120, 120, 60, -1]
        assert df.ancestor_detectable.tolist() == [False, True, True, True, False, False]
        # 8 has no recombinant ancestor, so nothing between.
        assert df.sampled_between.tolist() == ["", "", "seq_3", "seq_5 seq_3", "", ""]

    def test_subset(self):
        # The chain runs through candidates, sampled or not.
        df = truth_table([6], CANDIDATES, RECOMBINANTS)
        assert df.sampled_between.tolist() == ["seq_5 seq_3"]

    def test_agrees_with_pedigree(self, candidates, pedigree, recombinants):
        ids = list(candidates.index[candidates["rank"] < 4])
        df = truth_table(ids, candidates, recombinants)
        ancestors, between = walk_pedigree(pedigree, ids, set(candidates.index))
        assert df.recombinant_ancestor.tolist() == ancestors
        # Without a recombinant ancestor, nothing needs accounting for.
        assert df.sampled_between.tolist() == [
            " ".join(strain_name(u) for u in path) if a >= 0 else ""
            for a, path in zip(ancestors, between)
        ]
        assert (df.ancestor_breakpoint >= 0).tolist() == [a >= 0 for a in ancestors]

    def test_truth_csv(self, exported):
        # Everything is sampled, so the recombinant ancestor of every
        # non-recombinant is accounted for by a sampled individual in between.
        truth = pd.read_csv(exported / "truth.csv", keep_default_na=False)
        rec = truth[truth.is_recombinant]
        assert len(rec) > 0
        assert (rec.recombinant_ancestor == rec.strain.str[4:].astype(int)).all()
        assert (rec.ancestor_breakpoint == rec.breakpoint).all()
        assert (rec.sampled_between == "").all()
        other = truth[~truth.is_recombinant & (truth.recombinant_ancestor >= 0)]
        assert len(other) > 0
        assert (other.sampled_between != "").all()
