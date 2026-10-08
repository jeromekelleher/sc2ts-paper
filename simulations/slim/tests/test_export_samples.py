import datetime

import pandas as pd
import pytest
import tskit

from scripts.export_samples import START_DATE, read_fasta


def fasta_names(path):
    with open(path) as f:
        return [line[1:].strip() for line in f if line.startswith(">")]


class TestExportSamples:

    def test_names_agree(self, exported):
        ts = tskit.load(exported / "true.trees")
        names = [
            f"seq_{ts.individual(ts.node(u).individual).metadata['pedigree_id']}"
            for u in ts.samples()
        ]
        metadata = pd.read_csv(exported / "metadata.tsv", sep="\t")
        truth = pd.read_csv(exported / "truth.csv")
        assert fasta_names(exported / "sequences.fa") == names
        assert metadata.Run.tolist() == names
        assert truth.strain.tolist() == names

    def test_dates(self, exported):
        metadata = pd.read_csv(exported / "metadata.tsv", sep="\t")
        truth = pd.read_csv(exported / "truth.csv")
        expected = [str(START_DATE + datetime.timedelta(days=g)) for g in truth.gen]
        assert metadata.date.tolist() == expected
        assert metadata.date[0] == str(START_DATE)

    def test_sequences_match_slim(self, exported, slim_prefix):
        names = fasta_names(exported / "sequences.fa")
        assert read_fasta(exported / "sequences.fa", names) == read_fasta(
            f"{slim_prefix}.slim.sequences.fa", names
        )


class TestReadFasta:

    def test_multiline(self, tmp_path):
        path = tmp_path / "x.fa"
        path.write_text(">a\nAC\nGT\n>b\nTT\n>c\nGG\n")
        assert read_fasta(path, ["a", "c"]) == {"a": "ACGT", "c": "GG"}

    def test_missing(self, tmp_path):
        path = tmp_path / "x.fa"
        path.write_text(">a\nAC\n")
        with pytest.raises(ValueError):
            read_fasta(path, ["a", "b"])
