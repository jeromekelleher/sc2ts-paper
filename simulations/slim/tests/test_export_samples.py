import datetime

import pandas as pd
import pytest
import tskit

from scripts.export_samples import START_DATE, read_fasta, truth_table


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

    def test_truth_matches_pedigree(self, exported, slim_prefix):
        truth = pd.read_csv(exported / "truth.csv")
        pedigree = pd.read_csv(f"{slim_prefix}.slim.pedigree.tsv", sep="\t")
        pedigree = pedigree.set_index("child").loc[
            [int(s[4:]) for s in truth.strain]
        ]
        assert truth.is_recombinant.tolist() == pedigree.is_recombinant.astype(bool).tolist()
        assert truth.is_recombinant.sum() > 0
        recombinants = truth[truth.is_recombinant]
        assert (recombinants.breakpoint > 0).all()
        assert (truth[~truth.is_recombinant].breakpoint == -1).all()
        # Not every recombinant is detectable, but with a high mutation rate
        # most should be.
        assert 0 < recombinants.detectable.sum() <= len(recombinants)
        assert not truth[~truth.is_recombinant].detectable.any()

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


class TestTruthTable:

    def make_pedigree(self, rows):
        return pd.DataFrame(
            rows,
            columns=["gen", "child", "parent1", "parent2", "is_clonal",
                     "copy_parent", "breakpoints", "is_recombinant"],
        )

    def test_detectable(self):
        pedigree = self.make_pedigree([
            [0, 0, -1, -1, 0, -1, ".", 0],
            [1, 1, 0, 0, 1, 0, ".", 0],
            [1, 2, 0, 0, 1, 0, ".", 0],
            # Differs from both parents.
            [2, 3, 1, 2, 0, 1, "2", 1],
            # Breakpoint where the parents don't differ, so identical to 1.
            [2, 4, 1, 2, 0, 1, "1", 1],
        ])
        sequences = {
            "seq_0": "AAAA",
            "seq_1": "CAAA",
            "seq_2": "AAAT",
            "seq_3": "CAAT",
            "seq_4": "CAAA",
        }
        df = truth_table(pedigree, sequences)
        assert df.strain.tolist() == [f"seq_{j}" for j in range(5)]
        assert df.is_recombinant.tolist() == [False, False, False, True, True]
        assert df.breakpoint.tolist() == [-1, -1, -1, 2, 1]
        assert df.detectable.tolist() == [False, False, False, True, False]
