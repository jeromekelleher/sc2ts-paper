import datetime
import re
import click
import pandas as pd
import pyfaidx as pf


@click.command()
@click.argument("santasim_fasta_file")
@click.argument("output_prefix")
def run(santasim_fasta_file, output_prefix):
    # Arbitrary date for first sample.
    start_date = datetime.date(2019, 12, 31)
    out_fasta_file = output_prefix + ".sim.fasta"
    out_metadata_file = output_prefix + ".sim.metadata.tsv"
    out_recombs_file = output_prefix + ".sim.recombs.tsv"
    seqs = pf.Fasta(santasim_fasta_file)
    metadata = []
    recombs = []
    patt = re.compile(r'_g\d+_')
    with open(out_fasta_file, 'w') as f:
        for entry in seqs.keys():
            s = entry.split(":")
            strain = s[0]
            if len(s) == 1:
                # Non-recombinant, e.g., sample_r1_g1_i1.
                recombs.append(
                    {
                        "strain": strain,
                        "is_recombinant": False,
                        "breakpoints": "n/a",
                    }
                )
            else:
                # Recombinant where additional elements are breakpoint locations,
                # e.g., sample_r1_g1_i1:101:255.
                recombs.append(
                    {
                        "strain": strain,
                        "is_recombinant": True,
                        "breakpoints": ";".join(s[1:]),
                    }
                )
            # Treat one generation as one day.
            m = patt.search(strain)
            num_gens = int(m.group(0)[2:-1])    # e.g., '_g2000_'
            metadata.append(
                {
                    "Run": strain,
                    "date": start_date + datetime.timedelta(days=num_gens),
                }
            )
            f.write(">" + strain + "\n")
            f.write(str(seqs[entry]) + "\n")
    df_metadata = pd.DataFrame(metadata)
    df_metadata.to_csv(out_metadata_file, index=False, sep="\t")
    df_recombs = pd.DataFrame(recombs)
    df_recombs.to_csv(out_recombs_file, index=False, sep="\t")


if __name__ == "__main__":
    run()
