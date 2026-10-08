import pathlib
import shutil
import subprocess

import pytest
import tskit
from click.testing import CliRunner

import pipeline


SLIM_SCRIPT = pathlib.Path(__file__).parent.parent / "santasim_like.slim"


@pytest.fixture(scope="session")
def slim_prefix(tmp_path_factory):
    """
    Run a small SLiM simulation, with plenty of recombination and mutation,
    and return the output prefix.
    """
    if shutil.which("slim") is None:
        pytest.skip("slim is not installed")
    prefix = tmp_path_factory.mktemp("slim") / "sim"
    subprocess.run(
        [
            "slim", "-s", "1",
            "-d", "N=10", "-d", "GENS=15", "-d", "GROWTH_GENS=5",
            "-d", "L=200", "-d", "EFF_RECOMB=0.3", "-d", "MU=0.01",
            "-d", f'OUT_PREFIX="{prefix}"',
            str(SLIM_SCRIPT),
        ],
        check=True,
        capture_output=True,
    )
    return prefix


@pytest.fixture(scope="session")
def slim_ts(slim_prefix):
    return tskit.load(f"{slim_prefix}.slim.ts")


@pytest.fixture(scope="session")
def exported(slim_prefix, tmp_path_factory):
    """
    Sample every individual and export them, returning the directory of
    outputs: true.trees, sequences.fa, metadata.tsv and truth.csv.
    """
    out = tmp_path_factory.mktemp("exported")
    runner = CliRunner()
    result = runner.invoke(
        pipeline.sample,
        [f"{slim_prefix}.slim.ts", str(out / "true.trees"),
         "--probability", "1", "--seed", "1"],
    )
    assert result.exit_code == 0, result.output
    result = runner.invoke(
        pipeline.export_samples,
        [str(out / "true.trees"), f"{slim_prefix}.slim.sequences.fa",
         f"{slim_prefix}.slim.pedigree.tsv", str(out / "sequences.fa"),
         str(out / "metadata.tsv")],
    )
    assert result.exit_code == 0, result.output
    result = runner.invoke(
        pipeline.make_truth,
        [str(out / "true.trees"), f"{slim_prefix}.slim.sequences.fa",
         f"{slim_prefix}.slim.pedigree.tsv", str(out / "truth.csv")],
    )
    assert result.exit_code == 0, result.output
    return out
