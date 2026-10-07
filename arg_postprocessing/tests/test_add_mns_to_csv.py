import pandas as pd
import pytest
import tskit
from click.testing import CliRunner

from scripts.add_mns_to_csv import run


SEQUENCE_LENGTH = 29904
BREAKPOINT = 25050
INTERVAL_LEFT = 25000
INTERVAL_RIGHT = 25100
# MNM1 (and its subset MNM1a) lie left of the breakpoint, MNM2 to the right.
LEFT_MNS = {21302: ("C", "T"), 21304: ("C", "A"), 21305: ("G", "A")}
RIGHT_MNS = {28877: ("A", "T"), 28878: ("G", "C")}


def make_recombinant_ts(left_mns_node):
    """
    Return a single recombinant (node 3) copying from parent_left (node 1)
    left of BREAKPOINT and parent_right (node 2) to the right. The left MNS
    mutations are placed above left_mns_node; the right MNS above parent_right.
    """
    tables = tskit.TableCollection(SEQUENCE_LENGTH)
    root = tables.nodes.add_row(time=2)
    parent_left = tables.nodes.add_row(time=1)
    parent_right = tables.nodes.add_row(time=1)
    sample = tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
    tables.edges.add_row(0, SEQUENCE_LENGTH, root, parent_left)
    tables.edges.add_row(0, SEQUENCE_LENGTH, root, parent_right)
    tables.edges.add_row(0, BREAKPOINT, parent_left, sample)
    tables.edges.add_row(BREAKPOINT, SEQUENCE_LENGTH, parent_right, sample)

    nodes = {"root": root, "parent_left": parent_left}
    for position in [INTERVAL_LEFT, INTERVAL_RIGHT]:
        tables.sites.add_row(position, "A")
    for position, (ref, alt) in LEFT_MNS.items():
        site = tables.sites.add_row(position, ref)
        tables.mutations.add_row(site, nodes[left_mns_node], alt)
    for position, (ref, alt) in RIGHT_MNS.items():
        site = tables.sites.add_row(position, ref)
        tables.mutations.add_row(site, parent_right, alt)
    tables.sort()
    return tables.tree_sequence()


def run_script(tmp_path, ts):
    ts_file = tmp_path / "recomb.trees"
    csv_file = tmp_path / "recombinants.csv"
    output = tmp_path / "recombinants_mns.csv"
    ts.dump(ts_file)
    dfr = pd.DataFrame({
        "recombinant": [100],
        "sample": [3],
        "parent_left": [1],
        "parent_right": [2],
        "interval_left": [INTERVAL_LEFT],
        "interval_right": [INTERVAL_RIGHT],
    })
    dfr.to_csv(csv_file, index=False)
    result = CliRunner().invoke(run, [str(ts_file), str(csv_file), str(output)])
    assert result.exit_code == 0, result.output
    df = pd.read_csv(output, keep_default_na=False)
    pd.testing.assert_frame_equal(df[dfr.columns], dfr)
    return df.iloc[0]


class TestAddMnsToCsv:

    def test_mns_both_sides(self, tmp_path):
        row = run_script(tmp_path, make_recombinant_ts("parent_left"))
        assert row.num_mns_left == 2
        assert row.mns_left == "MNM1,MNM1a"
        assert row.num_mns_right == 1
        assert row.mns_right == "MNM2"

    def test_mns_shared_by_both_parents(self, tmp_path):
        # Both parents carry the left MNS, so it doesn't discriminate.
        row = run_script(tmp_path, make_recombinant_ts("root"))
        assert row.num_mns_left == 0
        assert row.mns_left == ""
        assert row.num_mns_right == 1
        assert row.mns_right == "MNM2"
