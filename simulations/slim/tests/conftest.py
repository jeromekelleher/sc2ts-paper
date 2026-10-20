import pathlib
import shutil
import subprocess

import pytest
import tskit
from click.testing import CliRunner

import pipeline


SLIM_SCRIPT = pathlib.Path(__file__).parent.parent / "santasim_like.slim"


def run_slim(prefix, N, s_max):
    """
    Run a small SLiM simulation, with plenty of recombination and mutation,
    writing the full pedigree and every sequence too.
    """
    if shutil.which("slim") is None:
        pytest.skip("slim is not installed")
    subprocess.run(
        [
            "slim", "-s", "1",
            "-d", f"N={N}", "-d", "GENS=15", "-d", "GROWTH_GENS=5",
            "-d", "L=200", "-d", "EFF_RECOMB=0.3", "-d", "MU=0.01",
            "-d", f"S_MAX={s_max}", "-d", "WRITE_ALL=T",
            "-d", f'OUT_PREFIX="{prefix}"',
            str(SLIM_SCRIPT),
        ],
        check=True,
        capture_output=True,
    )
    return prefix


@pytest.fixture(scope="session")
def slim_prefix(tmp_path_factory):
    """
    A simulation where the number of candidate samples per generation, 8, is
    well below the final population size, 30.
    """
    return run_slim(tmp_path_factory.mktemp("slim") / "sim", N=30, s_max=8)


@pytest.fixture(scope="session")
def slim_ts(slim_prefix):
    return tskit.load(f"{slim_prefix}.slim.ts")


@pytest.fixture(scope="session")
def slim_all_prefix(tmp_path_factory):
    """
    A simulation where every individual is a candidate sample.
    """
    return run_slim(tmp_path_factory.mktemp("slim_all") / "sim", N=10, s_max=1000)


@pytest.fixture(scope="session")
def exported(slim_all_prefix, tmp_path_factory):
    """
    Sample every individual of slim_all_prefix and export them, returning the
    directory of outputs: true.trees, sequences.fa, metadata.tsv and truth.csv.
    """
    prefix = slim_all_prefix
    out = tmp_path_factory.mktemp("exported")
    runner = CliRunner()
    result = runner.invoke(
        pipeline.sample,
        [f"{prefix}.slim.ts", f"{prefix}.slim.samples.tsv", str(out / "true.trees"),
         "--samples-per-day", "1000"],
    )
    assert result.exit_code == 0, result.output
    result = runner.invoke(
        pipeline.export_samples,
        [str(out / "true.trees"), f"{prefix}.slim.sequences.fa",
         f"{prefix}.slim.samples.tsv", str(out / "sequences.fa"),
         str(out / "metadata.tsv")],
    )
    assert result.exit_code == 0, result.output
    result = runner.invoke(
        pipeline.make_truth,
        [str(out / "true.trees"), f"{prefix}.slim.samples.tsv",
         f"{prefix}.slim.recombinants.tsv", str(out / "truth.csv")],
    )
    assert result.exit_code == 0, result.output
    return out
