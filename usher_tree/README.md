# Making a tskit Usher tree

## Requirements

Needs an installation that includes the following packages

numpy numba tskit matplotlib IPython tszip snakemake pyfaidx tqdm

also install sc2ts and tsdate

The snakefile will attempt to locate the matUtils binary, installed as part of Usher.
If you cannot install Usher (e.g. on OSX ARM), you can use a docker image od Usher instead,
installed under `pathogengenomics/usher:latest`, as described at
https://usher-wiki.readthedocs.io/en/latest/Installation.html#docker

## Running

Run 

    snakemake --cores all all

This uses utilities provided in ../src/mat2tsk.py. Note that the Tsdate step may take some time.
The final result should be a file nameed "all_viridian.XXXXXX.trees.tsz"


