import json
import tomllib

import pytest
from click.testing import CliRunner

from scripts.write_sc2ts_config import run, toml_value


class TestWriteSc2tsConfig:

    def run_script(self, tmp_path, extend_parameters):
        output = tmp_path / "config.toml"
        result = CliRunner().invoke(
            run,
            [
                str(output),
                "--dataset", "dataset.vcz.zip",
                "--reference", "reference.fa",
                "--run-id", "x_rep0_p0.5_k3",
                "--results-dir", "sc2ts/results",
                "--log-dir", "sc2ts/logs",
                "--matches-dir", "sc2ts/matches",
                "--num-mismatches", "3",
                "--num-threads", "2",
                "--extend-parameters", json.dumps(extend_parameters),
            ],
        )
        assert result.exit_code == 0, result.output
        with open(output, "rb") as f:
            return tomllib.load(f)

    def test_config(self, tmp_path):
        config = self.run_script(
            tmp_path, {"hmm_cost_threshold": 7, "deletions_as_missing": True}
        )
        assert config["dataset"] == "dataset.vcz.zip"
        assert config["reference_fasta"] == "reference.fa"
        assert config["reference_date"] == "2025-12-31"
        assert config["run_id"] == "x_rep0_p0.5_k3"
        assert config["results_dir"] == "sc2ts/results"
        assert config["log_dir"] == "sc2ts/logs"
        assert config["matches_dir"] == "sc2ts/matches"
        assert config["date_field"] == "date"
        assert config["extend_parameters"] == {
            "hmm_cost_threshold": 7,
            "deletions_as_missing": True,
            "num_mismatches": 3,
            "num_threads": 2,
        }

    def test_k_overrides_extend_parameters(self, tmp_path):
        config = self.run_script(tmp_path, {"num_mismatches": 10})
        assert config["extend_parameters"]["num_mismatches"] == 3


class TestTomlValue:

    @pytest.mark.parametrize(
        "value", [True, False, 1, 1.5, 1e-3, "a\"b", [], [1, "x", [2.0]]]
    )
    def test_round_trip(self, value):
        assert tomllib.loads(f"x = {toml_value(value)}")["x"] == value

    def test_unsupported(self):
        with pytest.raises(ValueError):
            toml_value({"a": 1})
