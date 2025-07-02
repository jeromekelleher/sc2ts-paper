import pytest

from pangonet.pangonet import PangoNet
from scripts.add_pangonet_distance_to_csv import get_pangonet_distance


@pytest.fixture(scope="module")
def pango():
    # Note: pangonet downloads these files each time by default which can result
    # in rate limit errors. Uncomment this to get things to work.
    # return PangoNet().build(alias_key="alias_key.json", lineage_notes="lineage_notes.txt")
    return PangoNet().build()


class TestPangonetDistance():

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
    def test_pangonet_distance(self, pango, labels, expected):
        actual = get_pangonet_distance(pango, *labels)
        assert actual == expected
        # Flip input labels
        actual = get_pangonet_distance(pango, *labels)
        assert actual == expected
