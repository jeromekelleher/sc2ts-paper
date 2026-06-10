import click
import pandas as pd
import bte  # https://jmcbroome.github.io/BTE/build/html/


def export_mutations(mat):
    df_bte = []
    for node in mat.breadth_first_expansion():
        list_muts = node.get_mutation_information()
        if len(list_muts) == 0:
            continue
        nt_mutations = []
        for mut in list_muts:
            # e.g., {'chrom': 'NC_045512', 'position': 28854, 'ref_nuc': 'C', 'par_nuc': 'C', 'mut_nuc': 'T'}
            assert len(mut['par_nuc']) == 1
            assert len(mut['mut_nuc']) == 1
            nt_mutations.append(mut['par_nuc'] + str(mut['position']) + mut['mut_nuc'])
        df_bte.append(
            {
                "node_id": node.id,
                "nt_mutations": ";".join(nt_mutations),
                }
        )
    return pd.DataFrame(df_bte)


@click.command()
@click.argument("in_protobuf_file")
@click.argument("out_mutations_file")
def run(in_protobuf_file, out_mutations_file):
    mat = bte.MATree(in_protobuf_file)
    df_bte = export_mutations(mat)
    df_bte.to_csv(out_mutations_file, sep="\t", index=False)


if __name__ == "__main__":
    run()
