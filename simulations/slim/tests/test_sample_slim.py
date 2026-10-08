import pandas as pd
import pytest
import tskit
from click.testing import CliRunner

from pipeline import find_founder, read_candidates, sample, sample_individuals


def pedigree_ids(ts):
    return [ts.individual(ts.node(u).individual).metadata["pedigree_id"]
            for u in ts.samples()]


@pytest.fixture(scope="module")
def candidates(slim_prefix):
    return read_candidates(f"{slim_prefix}.slim.samples.tsv")


@pytest.fixture(scope="module")
def pedigree(slim_prefix):
    return pd.read_csv(f"{slim_prefix}.slim.pedigree.tsv", sep="\t")


class TestSlimSampling:
    # Checks on how SLiM picks candidate samples.

    def test_candidates_per_generation(self, candidates, pedigree):
        sizes = pedigree.groupby("gen").size()
        counts = candidates.groupby("gen").size()
        assert counts.index.tolist() == sizes.index.tolist()
        assert (counts == sizes.clip(upper=8)).all()
        # The cap binds once the population has grown.
        assert (sizes > 8).any()

    def test_ranks(self, candidates):
        for _, df in candidates.groupby("gen"):
            assert df["rank"].tolist() == list(range(len(df)))

    def test_founder(self, candidates, slim_ts):
        assert candidates.loc[0].gen == 0
        assert candidates.loc[0]["rank"] == 0
        founder = slim_ts.individual(find_founder(slim_ts))
        assert founder.metadata["pedigree_id"] == 0


class TestSample:

    def test_counts(self, slim_ts, candidates):
        ts = sample_individuals(slim_ts, candidates, 3)
        ids = pedigree_ids(ts)
        assert sorted(ids) == sorted(candidates.index[candidates["rank"] < 3])
        counts = candidates.loc[ids].groupby("gen").size()
        available = candidates.groupby("gen").size()
        assert (counts == available.clip(upper=3)).all()

    def test_founder_first(self, slim_ts, candidates):
        ts = sample_individuals(slim_ts, candidates, 3)
        assert pedigree_ids(ts)[0] == 0
        assert ts.samples().tolist() == list(range(ts.num_samples))

    def test_one_node_per_individual(self, slim_ts, candidates):
        ts = sample_individuals(slim_ts, candidates, 8)
        ids = pedigree_ids(ts)
        assert len(ids) == len(set(ids)) == len(candidates)
        # Vacant nodes have no ancestry, so would be isolated in every tree.
        assert all(ts.node(u).metadata["is_vacant"] == [0] for u in ts.samples())
        assert ts.first().num_roots == 1

    def test_nested(self, slim_ts, candidates):
        small = set(pedigree_ids(sample_individuals(slim_ts, candidates, 2)))
        large = set(pedigree_ids(sample_individuals(slim_ts, candidates, 5)))
        assert 1 < len(small) < len(large)
        assert small < large

    def test_bad_samples_per_day(self, slim_ts, candidates):
        with pytest.raises(ValueError):
            sample_individuals(slim_ts, candidates, 0)

    def test_cli(self, slim_prefix, candidates, tmp_path):
        output = tmp_path / "out.trees"
        result = CliRunner().invoke(
            sample,
            [f"{slim_prefix}.slim.ts", f"{slim_prefix}.slim.samples.tsv",
             str(output), "--samples-per-day", "4"],
        )
        assert result.exit_code == 0, result.output
        expected = sample_individuals(tskit.load(f"{slim_prefix}.slim.ts"), candidates, 4)
        assert pedigree_ids(tskit.load(output)) == pedigree_ids(expected)
