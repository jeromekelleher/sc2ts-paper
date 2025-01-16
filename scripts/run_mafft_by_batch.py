"""
This script was used to generate MAFFT alignments on the BMRC.

Note that the gaps at the termini of the aligned sequences are replaced with Ns.
"""


import argparse
import subprocess
from pathlib import Path
import pyfaidx as pf


parser = argparse.ArgumentParser()
parser.add_argument("batch", help="Batch ID", type=str)
args = parser.parse_args()
BATCH_ID = args.batch


if BATCH_ID != "test":
    # Run on cluster.
    BASE_DIR= Path("./")
    MAFFT_BIN = BASE_DIR / "local/bin/mafft"
    REF_FILE = BASE_DIR / "sc2ts/sc2ts/data/reference.fasta"
    DATA_DIR = BASE_DIR / "viridian/v0.4"
    IN_FILE = DATA_DIR / "Viridian_tree_cons_seqs_split" / str(BATCH_ID + ".cons.fa")
    OUT_FILE = DATA_DIR / "Viridian_tree_cons_seqs_mafft" / str(BATCH_ID + ".cons.mafft.aln")
else:
    # Local test run.
    MAFFT_BIN = "mafft"
    REF_FILE = "mafft.test.ref.fa"
    IN_FILE = "mafft.test.cons.fa"
    OUT_FILE = "mafft.test.cons.aln"


ref_dict = pf.Fasta(REF_FILE)
qry_dict = pf.Fasta(IN_FILE)

assert len(ref_dict.keys()) == 1, \
    "More than one entry found in the reference sequence file."
ref_name = list(ref_dict.keys())[0]
ref_seq = str(ref_dict[ref_name])
ref_entry = "\n".join(
    [
        ">" + ref_name,
        ref_seq,
    ]
)


# Taken from
# https://github.com/martinghunt/ushonium/blob/93b0be2301645bfb0b050b9485a4d187cebb7031/ushonium/mafft.py#L10
def mafft_stdout_to_seqs(mafft_stdout, ref_name):
    # The aligned sequences are printed to stdout. We should have the
    # reference genome and the aligned genome in there (do not assume what order
    # they are output). We just want to extract the aligned seqs.
    seqs = mafft_stdout.split(">")
    # seqs looks like: ["", "ref\nACGT...", "to_align\nACGT...", ...]
    assert len(seqs) >= 3
    assert seqs[0] == ""
    ref_seq = None
    aln_seqs = {}
    seqs_to_parse = [x.split("\n", maxsplit=1) for x in seqs[1:]]
    for (name, seq_str) in seqs_to_parse:
        if name == ref_name:
            ref_seq = seq_str.replace("\n", "")
        else:
            assert name not in aln_seqs
            aln_seqs[name] = seq_str.replace("\n", "")
    return ref_seq, aln_seqs


# Taken from
# https://github.com/martinghunt/ushonium/blob/93b0be2301645bfb0b050b9485a4d187cebb7031/ushonium/mafft.py#L78
def replace_start_end_indels_with_N(seq):
    new_seq = list(seq)
    chars = {"n", "N", "-"}
    i = 0
    while i < len(new_seq) and new_seq[i] in chars:
        new_seq[i] = "N"
        i += 1
    i = len(new_seq) - 1
    while i >= 0 and new_seq[i] in chars:
        new_seq[i] = "N"
        i -= 1
    return "".join(new_seq)


# Adapted from
# https://github.com/martinghunt/ushonium/blob/93b0be2301645bfb0b050b9485a4d187cebb7031/ushonium/mafft.py#L11
with open(OUT_FILE, 'w') as f:
    for qry_name in qry_dict.keys():
        qry_seq = str(qry_dict[qry_name])
        qry_entry = "\n".join(
            [
                ">" + qry_name,
                qry_seq,
            ]
        )
        script = "\n".join(
            [
                f'''ref=">{ref_name}\n{ref_seq}"''',
                f'''qry=">{qry_name}\n{qry_seq}"''',
                f'''{MAFFT_BIN} --quiet --keeplength --add <(echo "$qry") <(echo "$ref")''',
            ]
        )
        p = subprocess.run(
            ["bash"],
            input=script,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if p.returncode != 0:
            raise Exception(
                f"Error running mafft. Stdout:\n{p.stdout}\n\nStderr:{p.stderr}"
            )
        _, aln_seqs = mafft_stdout_to_seqs(p.stdout, ref_name)
        aln_qry_seq = aln_seqs[qry_name].upper()
        aln_qry_seq = replace_start_end_indels_with_N(aln_qry_seq)
        aln_qry_entry = "\n".join(
            [
                ">" + qry_name,
                aln_qry_seq,
            ]
        )
        f.write(aln_qry_entry + "\n")
