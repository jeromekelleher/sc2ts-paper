### Install Dependencies
micromamba env create -f notebooks/novel-recombinants-analysis.yml

python -m ipykernel install --user --name=novel-recombinants

### Download data.
# - Viridan metadata: `data/run_metadata.v04.tsv.gz`
#   - URL: https://figshare.com/articles/dataset/Supplementary_table_S1/25712982?file=45969195
# - Viridian sequences: `data/Viridian_tree_cons_seqs/<#>.cons.fa.xz`
#   - URL: https://doi.org/10.6084/m9.figshare.25713225
# - Viridian index: `data/Viridian_tree_cons_seqs/index.tsv.xz`

### Download dependencies.
wget -q -O csvtk.tar.gz https://github.com/shenwei356/csvtk/releases/download/v0.32.0/csvtk_linux_386.tar.gz
tar -xf csvtk.tar.gz
mv csvtk bin/
rm -f csvtk.tar.gz
#bin/csvtk --help | head -n 5

wget -q -O seqkit.tar.gz https://github.com/shenwei356/seqkit/releases/download/v2.9.0/seqkit_linux_amd64.tar.gz
tar -xf seqkit.tar.gz
mv seqkit bin/
rm -f seqkit.tar.xz
#bin/seqkit --help | head -n 5

wget -q -O bin/nextclade https://github.com/nextstrain/nextclade/releases/download/3.10.2/nextclade-x86_64-unknown-linux-musl
#bin/nextclade --help | head -n 5

wget -q -O bin/rebar https://github.com/phac-nml/rebar/releases/download/v0.2.1/rebar-x86_64-unknown-linux-musl
#bin/rebar --help | head -n 5

data_dir = "data"
results_dir = "results"

metadata_file = ${data_dir}"/run_metadata.v04.tsv.gz"
sequences = ${data_dir}"/Viridian_tree_cons_seqs"
index_file = ${data_dir}"/Viridian_tree_cons_seqs/index.tsv.xz"
recombinants_file = ${data_dir}"/recombinants.csv"

# Extract Viridian metadata for the novel recombinants.
bin/csvtk cut -f sample_id ${recombinants_file} \
    tail -n+2 \
    bin/csvtk grep -t -f Run -P - ${metadata_file} \
    bin/csvtk merge -t -f Run ${index_file} - | \
    > results/novel_recombinants/metadata.tsv

# Extract Viridian batch numbers for the novel recombinants.
bin/csvtk cut -t -f Batch ${results_dir}"/metadata.tsv" | \
    tail -n+2 | \
    sort -g | \
    uniq > results/novel_recombinants//batches.txt

# Extract Viridian consensus sequences for the novel recombinants.
cat results/novel_recombinants/batches.txt | \
    while read batch; do \
    echo Batch: ${batch} 1>&2; \
    bin/csvtk grep -t -f Batch -p ${batch} results/novel_recombinants/metadata.tsv \
        | bin/csvtk cut -t -f Run \
        | tail -n+2 \
        | bin/seqkit grep -w 0 -f - "data/Viridian_tree_cons_seqs/"${batch}".cons.fa.xz"; \
    done > results/novel_recombinants/sequences.fasta

# Download the sars-cov-2 lineage model.
bin/nextclade dataset get \
    --name sars-cov-2 \
    --tag  2025-01-28--16-39-09Z \
    --output-dir dataset/nextclade

# Align the sequences.
bin/nextclade run \
    --input-dataset dataset/nextclade \
    --jobs 2 \
    --output-tsv results/novel_recombinants/nextclade.tsv \
    --output-fasta results/novel_recombinants/nextclade.fasta \
    results/novel_recombinants/sequences.fasta

# Detect recombination using rebar.
bin/rebar dataset download \
    --name sars-cov-2 \
    --tag 2025-01-28 \
    --verbosity error \
    --output-dir dataset/rebar

bin/rebar dataset download \
    --name sars-cov-2 \
    --tag 2025-01-28 \
    --verbosity error \
    --output-dir dataset/rebar

bin/rebar run \
    --dataset-dir dataset/rebar \
    --threads 2 \
    --alignment results/novel_recombinants/nextclade.fasta \
    --output-dir results/novel_recombinants/rebar

cp results/novel_recombinants/rebar/linelist.tsv data/rebar.tsv

bin/rebar plot \
    --run-dir results/novel_recombinants/rebar \
    --annotations dataset/rebar/annotations.tsv \
    --verbosity error
