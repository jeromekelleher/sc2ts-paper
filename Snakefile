from pathlib import Path
import glob


BASE_DIR = Path("alignments")
DATA_DIR = BASE_DIR / "data"

MAFFT_BIN = "/opt/homebrew/bin/mafft"
CHUNK_SIZE = 1000

# Figshare
URLS = {
    #'batch1': "https://figshare.com/ndownloader/files/45969777",
    'batch2': "https://figshare.com/ndownloader/files/49692480",
}

def get_output_files(wildcards):
    all_files = [f"{DATA_DIR}/run_metadata.v05.tsv.gz"]
    
    for dir_name, url in URLS.items():
        fa_files = glob.glob(f"{checkpoints.process_tar.get(dir_name=dir_name).output[0]}/*.fa")
        print(checkpoints.process_tar.get(dir_name=dir_name).output[0], fa_files)
        for fa in fa_files:
            aln_file = fa.replace(".fa", ".aln").replace(".extracted", ".aln")
            all_files.append(aln_file)
    
    print(all_files)
    return all_files

rule all:
    input:
        get_output_files

rule download_reference:
    output:
        f"{DATA_DIR}/reference.fasta"
    shell:
        """
        wget --quiet \
            https://raw.githubusercontent.com/jeromekelleher/sc2ts/e9d1fdcc7e7ae2c172da64b47da2eb0373dd4d39/sc2ts/data/reference.fasta \
            -O {output}
        """

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

checkpoint process_tar:
    input:
        DATA_DIR / "{dir_name}.tar"
    output:
        directory(DATA_DIR / "{dir_name}.extracted")
    shell:
        """
        # Extract tar
        mkdir -p {output}
        tar -xf {input} -C {output} --strip-components 1
        
        # Decompress files
        for f in {output}/*.cons.fa.xz; do
            base=$(basename "$f" .cons.fa.xz)
            xz --decompress --stdout "$f" > "{output}/$base.fa"
        done
        """

# If we pass the unmodified fasta file to mafft, it will attempt a multiple sequence alignment
# of all the sequences. We only want an alignment of each sequence to the reference, so
# mafft needs to be run for each sequence separately.
rule align_sequences:
    input:
        reference = DATA_DIR / "reference.fasta",
        sequences = DATA_DIR / "{dir_name}.extracted" / "{part}.fa"
    output:
        DATA_DIR / "{dir_name}.aln" / "{part}.aln"
    run:
        import os
        from Bio import SeqIO
        import subprocess
        os.makedirs(Path(output[0]).parent / "logs", exist_ok=True)
        ref_seq = str(next(SeqIO.parse(input.reference, "fasta")).seq)
        ref_name = "reference"
        with open(output[0], 'w') as output_file:
            for seq in SeqIO.parse(input.sequences, "fasta"):
                log_file = Path(output[0]).parent / "logs" / f"{seq.id}.log"
                seq_str = f">{seq.id}\n{str(seq.seq)}\n"
                script = "\n".join([
                    f'''ref=">{ref_name}\n{ref_seq}"''',
                    f'''qry=">{seq.id}\n{str(seq.seq)}"''',
                    f'''{MAFFT_BIN} --quiet --keeplength --add <(echo "$qry") <(echo "$ref")'''
                ])
                process = subprocess.run(
                    ["bash"],
                    input=script,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                
                if len(process.stderr) > 0:
                    with open(log_file, 'w') as f:
                        f.write(process.stderr)
                if process.returncode != 0:
                    raise Exception(
                        f"Error running mafft for {seq.id}. Stdout:\n{process.stdout}\n\nStderr:{process.stderr}"
                    )
                aln_seqs = process.stdout.split(">")
                assert len(aln_seqs) == 3
                output_file.write(f">{aln_seqs[2].strip()}\n")