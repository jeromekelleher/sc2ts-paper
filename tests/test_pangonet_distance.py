import pytest
from pathlib import Path
import sys
sys.path.append("../notebooks/")
import nb_utils


class TestPangonetDistance():
    def start_pangonet(self):
        rebar_dir = Path("../notebooks/dataset/rebar")
        alias_key_file = rebar_dir / "alias_key.json"
        lineage_notes_file = rebar_dir / "lineage_notes.txt"
        return nb_utils.initialise_pangonet(
            alias_key_file=alias_key_file,
            lineage_notes_file=lineage_notes_file,
        )


    @pytest.mark.parametrize(
        "labels, expected",
        [
            # Special case
            (("BM.1", "BM.1"), 0),
            # Parent-child pair
            (("BM.1", "BM.1.1"), 1),
            # Ancestor-descendant pair
            (("BM.1", "BM.1.1.1"), 2),
            # Compressed pair
            (("BM.1", "BJ.1"), 6),
            # Uncompressed pair
            (("B.1.1.529.2.75.3", "B.1.1.529.2.10.1"), 4),
            # Uncompressed/compressed pair
            (("B.1.1.529.2.75.3", "BM.1"), 1),
        ],
    )
    def test_pangonet_distance(self, labels, expected):
        pangonet = self.start_pangonet()
        actual = nb_utils.get_pangonet_distance(
            pangonet=pangonet,
            label_1=labels[0],
            label_2=labels[1],
        )
        assert actual == expected
        # Flip input labels
        actual = nb_utils.get_pangonet_distance(
            pangonet=pangonet,
            label_1=labels[1],
            label_2=labels[0],
        )
        assert actual == expected
