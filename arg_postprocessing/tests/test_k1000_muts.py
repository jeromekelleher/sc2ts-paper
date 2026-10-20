import json

import pandas as pd
from click.testing import CliRunner

from scripts import add_recombinant_k1000_muts_to_csv, extract_k1000_muts


def match(num_segments, num_mutations):
    """
    Return a minimal HmmMatch dict with the given path and mutation counts.
    """
    return {
        "path": [{"left": j, "right": j + 1, "parent": j} for j in range(num_segments)],
        "mutations": [{"site_position": j} for j in range(num_mutations)],
    }


# Node 20 is a two-parent recombinant, node 10 a multi-parent one.
REMATCHES = [
    {
        "recombinant": 20,
        "original_match": match(2, 3),
        "no_recomb_match": match(1, 12),
    },
    {
        "recombinant": 10,
        "original_match": match(4, 5),
        "no_recomb_match": match(1, 30),
    },
]


def run_extract(tmp_path):
    rematches = tmp_path / "recombinants.json"
    rematches.write_text(json.dumps(REMATCHES))
    output = tmp_path / "k1000_muts.json"
    result = CliRunner().invoke(extract_k1000_muts.run, [str(rematches), str(output)])
    assert result.exit_code == 0, result.output
    return output


def test_extract_counts_sorted(tmp_path):
    with open(run_extract(tmp_path)) as f:
        data = json.load(f)
    assert list(data) == ["10", "20"]
    assert data["10"] == {"k4_muts": 5, "k1000_muts": 30}
    assert data["20"] == {"k4_muts": 3, "k1000_muts": 12}


def test_add_to_csv_ignores_multi_parent(tmp_path):
    k1000_muts = run_extract(tmp_path)
    csv = tmp_path / "recombinants.csv"
    pd.DataFrame({"recombinant": [20], "num_mutations": [3]}).to_csv(csv, index=False)
    output = tmp_path / "out.csv"
    result = CliRunner().invoke(
        add_recombinant_k1000_muts_to_csv.run, [str(k1000_muts), str(csv), str(output)]
    )
    assert result.exit_code == 0, result.output
    assert "WARNING" not in result.output
    df = pd.read_csv(output)
    assert list(df.recombinant) == [20]
    assert list(df.k1000_muts) == [12]
