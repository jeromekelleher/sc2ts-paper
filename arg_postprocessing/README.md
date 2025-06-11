# Requirements

The pipeline is designed to work with conda, and should be run using

```
snakemake --cores=all --sdm=conda
```

Some elements of the pipeline depend on Usher, which has not been
packaged for ARM macs. Thus, it is simplest to run the pipeline 
on x86 servers, where possible.

## Usher tree

The final result should be a file nameed "all_viridian.XXXXXX.trees.tsz"


