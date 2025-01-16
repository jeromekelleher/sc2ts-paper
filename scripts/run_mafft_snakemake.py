import click
import subprocess
import pyfaidx as pf


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


def get_ref_seq(file):
    ref_dict = pf.Fasta(file)
    if len(ref_dict.keys()) != 1:
        raise ValueError(
            "More than one entry found in the reference sequence file."
        )
    ref_name = list(ref_dict.keys())[0]
    ref_seq = str(ref_dict[ref_name])
    return (ref_name, ref_seq)


@click.command()
@click.argument(
    'ref_file',
    type=click.Path(exists=True, dir_okay=False, readable=True),
)
@click.argument(
    'qry_file',
    type=click.Path(exists=True, dir_okay=False, readable=True),
)
@click.argument(
    'aln_file',
    type=click.Path(exists=False, dir_okay=False, writable=True),
)
@click.argument(
    'mafft_bin',
    type=click.Path(exists=True, dir_okay=False, executable=True),
)
def run_mafft(ref_file, qry_file, aln_file, *, mafft_bin):
    (ref_name, ref_seq) = get_ref_seq(ref_file)

    qry_dict = pf.Fasta(qry_file)

    # Adapted from
    # https://github.com/martinghunt/ushonium/blob/93b0be2301645bfb0b050b9485a4d187cebb7031/ushonium/mafft.py#L11
    with open(aln_file, 'w') as f:
        for qry_name in qry_dict.keys():
            qry_seq = str(qry_dict[qry_name])
            script = "\n".join(
                [
                    f'''ref=">{ref_name}\n{ref_seq}"''',
                    f'''qry=">{qry_name}\n{qry_seq}"''',
                    f'''{mafft_bin} --quiet --keeplength --add <(echo "$qry") <(echo "$ref")''',
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
            # NOTE: MAFFT aligned sequences are in lowercase letters.
            aln_qry_seq = aln_seqs[qry_name]
            aln_qry_entry = "\n".join([">" + qry_name, aln_qry_seq])
            f.write(aln_qry_entry + "\n")


if __name__ == '__main__':
    run_mafft()
