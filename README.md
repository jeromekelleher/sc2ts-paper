# sc2ts-paper

This repository contains all analysis for the preprint
[A pandemic-scale Ancestral Recombination Graph for SARS-CoV-2](https://www.biorxiv.org/content/10.1101/2023.06.08.544212v3.full)

## Links

- Live [in-browser demo](https://tskit.dev/explore/lab/index.html?path=sc2ts.ipynb)
- Data processing example [notebook](notebooks/example_data_processing.ipynb)
- sc2ts [homepage](https://tskit.dev/sc2ts)
- sc2ts package on [PyPi](https://pypi.org/project/sc2ts/)
- sc2ts [documentation](https://tskit.dev/sc2ts/docs/)
- Inferred ARG in [tszip format](https://zenodo.org/records/17558489/files/sc2ts_viridian_v1.2.trees.tsz)
- Viridian dataset in [VCZ format](https://zenodo.org/records/16314739/files/viridian_mafft_2024-10-14_v1.vcz.zip)
- UShER Viridian tree [tszip format](https://zenodo.org/records/17558489/files/usher_viridian_v1.0.trees.tsz)

## Repo layout

- The [latex](latex) directory contains the manuscript
- The [viridian_dataset](viridian_dataset) directory contains the Snakemake pipeline
    for aligning the Viridian consensus sequences and converting to VCF Zarr format.
- The [inference](inference) directory contains the configuration file used to run
    primary inference with sc2ts.
- The [arg_postprocessing](arg_postprocessing) directory contains Snakemake workflow
    for postprocessing the ARG, and running all major analyses in the paper
    (RIPPLES, CovRecomb, etc)
- The [notebooks](notebooks) directory contains the Jupyter notebooks used to
    perform analyses, genererate tables and figures  in the manuscript.
- The [src](src) directory contains a few miscellaneous scripts

