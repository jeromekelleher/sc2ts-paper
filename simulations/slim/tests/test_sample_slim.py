import numpy as np
import pytest
import tskit
from click.testing import CliRunner

from scripts.sample_slim import find_founder, run, sample


def pedigree_ids(ts):
    return [ts.individual(ts.node(u).individual).metadata["pedigree_id"]
            for u in ts.samples()]


class TestSample:

    def test_founder(self, slim_ts):
        founder = slim_ts.individual(find_founder(slim_ts))
        assert founder.metadata["pedigree_id"] == 0

    def test_only_founder(self, slim_ts):
        ts = sample(slim_ts, 0, seed=1)
        assert pedigree_ids(ts) == [0]

    def test_all(self, slim_ts):
        ts = sample(slim_ts, 1, seed=1)
        ids = pedigree_ids(ts)
        # One sample node per individual, founder first.
        assert ids[0] == 0
        assert len(ids) == len(set(ids)) == slim_ts.num_individuals
        assert ts.samples().tolist() == list(range(ts.num_samples))

    def test_no_vacant_samples(self, slim_ts):
        ts = sample(slim_ts, 1, seed=1)
        # Vacant nodes have no ancestry, so would be isolated in every tree.
        assert all(ts.node(u).metadata["is_vacant"] == [0] for u in ts.samples())
        tree = ts.first()
        assert tree.num_roots == 1

    def test_nested(self, slim_ts):
        small = set(pedigree_ids(sample(slim_ts, 0.3, seed=2)))
        large = set(pedigree_ids(sample(slim_ts, 0.7, seed=2)))
        assert 1 < len(small) < len(large)
        assert small < large

    def test_seed(self, slim_ts):
        a = pedigree_ids(sample(slim_ts, 0.5, seed=2))
        b = pedigree_ids(sample(slim_ts, 0.5, seed=3))
        assert a != b

    @pytest.mark.parametrize("probability", [-0.1, 1.1])
    def test_bad_probability(self, slim_ts, probability):
        with pytest.raises(ValueError):
            sample(slim_ts, probability, seed=1)

    def test_cli(self, slim_prefix, tmp_path):
        output = tmp_path / "out.trees"
        result = CliRunner().invoke(
            run,
            [f"{slim_prefix}.slim.ts", str(output), "--probability", "0.5",
             "--seed", "2"],
        )
        assert result.exit_code == 0, result.output
        ts = tskit.load(output)
        expected = sample(tskit.load(f"{slim_prefix}.slim.ts"), 0.5, seed=2)
        assert pedigree_ids(ts) == pedigree_ids(expected)
