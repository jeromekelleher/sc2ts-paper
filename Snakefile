from pathlib import Path
import glob


BASE_DIR = Path("alignments")
DATA_DIR = BASE_DIR / "data"
MAFFT_DIR = BASE_DIR / "mafft"

MAFFT_BIN = "/opt/homebrew/bin/mafft"
MAFFT_SCRIPT = "./scripts/run_mafft_snakemake.py"


# Figshare
URLS = {
    #'batch1': "https://figshare.com/ndownloader/files/45969777",
    'batch2': "https://figshare.com/ndownloader/files/49692480",
}


def get_output_files(wildcards):
    all_files = [
        f"{DATA_DIR}/reference.fasta",
        f"{DATA_DIR}/run_metadata.v05.tsv.gz",
    ]
    for dir_name, url in URLS.items():
        checkpoints.extract_tar.get(dir_name=dir_name)
        #xz_files = glob.glob(f"{DATA_DIR}/{dir_name}.extracted/*.cons.fa.xz")
        #all_files.extend([f.replace('.xz', '') for f in xz_files])
        aln_files = glob.glob(f"{MAFFT_DIR}/{dir_name}/*.mafft.aln")
        print(aln_files)
        all_files.extend(aln_files)
    return all_files


rule all:
    input:
        get_output_files


"""Fetch reference sequence file from the sc2ts GitHub repository."""
rule download_reference:
    output:
        f"{DATA_DIR}/reference.fasta"
    shell:
        """
        wget --quiet \
            https://raw.githubusercontent.com/jeromekelleher/sc2ts/e9d1fdcc7e7ae2c172da64b47da2eb0373dd4d39/sc2ts/data/reference.fasta \
            -O {output}
        """


"""Fetch sample metadata and consensus sequences (Viridian v04 and v05) from Figshare."""
rule download_viridian_metadata:
    output:
        f"{DATA_DIR}/run_metadata.v05.tsv.gz"
    shell:
        """
        wget --quiet --content-disposition \
            https://figshare.com/ndownloader/files/49694808 \
            -O {output}
        """


rule download_viridian_sequences:
    output:
        DATA_DIR / "{dir_name}.tar"
    params:
        url = lambda wildcards: URLS[wildcards.dir_name]
    shell:
        """
        wget --quiet --content-disposition {params.url} -O {output}
        """


checkpoint extract_tar:
    input:
        DATA_DIR / "{dir_name}.tar"
    output:
        directory(DATA_DIR / "{dir_name}.extracted")
    shell:
        """
        mkdir {output} && tar -xf {input} -C {output} --strip-components 1
        """


rule decompress_files:
    input:
        DATA_DIR / "{dir_name}.extracted" / "{part}.cons.fa.xz"
    output:
        DATA_DIR / "{dir_name}.extracted" / "{part}.cons.fa"
    shell:
        """
        xz --decompress --keep --stdout {input} > {output}
        """


"""Get pairwise MAFFT alignments."""
rule run_mafft:
    input:
        DATA_DIR / "{dir_name}.extracted" / "{part}.cons.fa"
    output:
        MAFFT_DIR / "{dir_name}" / "{part}.mafft.aln"
    shell:
        """
        python ./scripts/run_mafft_snakemake.py \
            alignments/data/reference.fasta \
            {input} \
            {output} \
            /opt/homebrew/bin/mafft
        """
