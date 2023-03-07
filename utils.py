from dataclasses import dataclass
from datetime import datetime
import os

import tskit
import tszip

@dataclass
class ARG:
    """Associate a tree sequence with a start date."""
    ts: tskit.TreeSequence
    day_0: datetime


def load_tsz_file(date, filename):
    return ARG(
        tszip.decompress(os.path.join("data", filename.format(date))),
        datetime.fromisoformat(date),
    )

