Place data files in this directory, e.g.

* `find_problematic_v2-2021-11-18.ts.il.tsz`:  This is the ARG with imputed lineages, before filtering out problematic sites. For the moment, please contact the sc2ts developers to obtain the equivalent ARG without imputed lineages. The file can then be created by `python scripts/run_lineage_imputation.py -v data/consensus_mutations.json.bz2 data/find_problematic_v2-2021-11-18.ts.il.tsz`
* `nextstrain_ncov_gisaid_global_all-time_timetree-2023-01-21.nex`: downloaded from https://nextstrain.org/ncov/gisaid/global/all-time on 21 Jan 2023, and stored in this directory as a zipped file.
* `consensus_mutations.json.bz2`: download from http://covidcg.org/ -> 'Compare AA mutations' -> Download -> 'Consensus mutations', setting mutation type to 'NT' and consensus threshold to 0.9. To reduce the size of the repository, we further compress this with bzip2.
