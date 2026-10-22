import numpy as np
import pandas as pd
import pytest
import tskit
from click.testing import CliRunner

from scripts import node_dating_validation

OTHER_FLAG = 1 << 21


@pytest.fixture
def ts_path(tmp_path):
    """
    A balanced binary tree in which 2/3 of the internal nodes are samples,
    and some samples carry an extra flag.
    """
    ts = tskit.Tree.generate_balanced(64, arity=2).tree_sequence
    tables = ts.dump_tables()
    flags = tables.nodes.flags
    internal = np.where(flags == 0)[0]
    flags[internal[: 2 * len(internal) // 3]] = tskit.NODE_IS_SAMPLE
    flags[::5] |= OTHER_FLAG
    tables.nodes.flags = flags
    path = tmp_path / "in.trees"
    tables.dump(path)
    return path


def run(*args):
    result = CliRunner().invoke(node_dating_validation.cli, [str(a) for a in args])
    assert result.exit_code == 0, result.output
    return result


def test_mask(ts_path, tmp_path):
    out = tmp_path / "masked.trees"
    run("mask", ts_path, out)
    ts = tskit.load(ts_path)
    masked = tskit.load(out)
    changed = np.where(ts.nodes_flags != masked.nodes_flags)[0]
    has_children = np.isin(np.arange(ts.num_nodes), ts.edges_parent)
    is_sample = (ts.nodes_flags & tskit.NODE_IS_SAMPLE) != 0
    num_internal = np.sum(~is_sample & has_children)
    assert len(changed) == num_internal
    # Only the sample bit of samples with children is cleared
    assert np.all(is_sample[changed] & has_children[changed])
    np.testing.assert_array_equal(
        masked.nodes_flags[changed], ts.nodes_flags[changed] - tskit.NODE_IS_SAMPLE
    )
    np.testing.assert_array_equal(ts.nodes_time, masked.nodes_time)


def test_mask_seed(ts_path, tmp_path):
    flags = []
    for name, seed in [("a", 1), ("b", 1), ("c", 2)]:
        out = tmp_path / f"{name}.trees"
        run("mask", ts_path, out, "--seed", seed)
        flags.append(tskit.load(out).nodes_flags)
    np.testing.assert_array_equal(flags[0], flags[1])
    assert not np.array_equal(flags[0], flags[2])


def test_compare(ts_path, tmp_path):
    masked = tmp_path / "masked.trees"
    run("mask", ts_path, masked)
    out = tmp_path / "out.csv"
    run("compare", ts_path, masked, out)
    df = pd.read_csv(out)
    ts = tskit.load(ts_path)
    chosen = np.where(ts.nodes_flags != tskit.load(masked).nodes_flags)[0]
    np.testing.assert_array_equal(df.node_id, chosen)
    np.testing.assert_allclose(df.time, df.redated_time)


def test_compare_reports_moved_samples(ts_path, tmp_path):
    masked = tmp_path / "masked.trees"
    run("mask", ts_path, masked)
    tables = tskit.load(masked).dump_tables()
    time = tables.nodes.time
    time[0] -= 1  # node 0 is a leaf sample
    tables.nodes.time = time
    tables.dump(masked)
    result = run("compare", ts_path, masked, tmp_path / "out.csv")
    assert "1 of" in result.output
    assert "max change 1.00" in result.output
