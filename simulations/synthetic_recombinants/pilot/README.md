### Pilot run

A small run kept as a record, and read by
`notebooks/analysis_synthetic_recombinants.ipynb`. It was produced with the
steps in the parent `Snakefile`, but run by hand with a reduced set of strains
so it finished in under an hour.

`truth.csv` holds the 100 recombinants and 338 controls that were generated
with `--num-recombinants 100 --seed 42`. Of those, the 100 strains in
`strains.txt`, being the first 50 recombinants and the first 50 controls, were
matched against `data/v2_2026-09-22_2021-05-15.ts.tsz` for each of k = 3, 4 and
5, giving `hmm_k{3,4,5}.jsonl`. `results.csv` is the summary over the strains
that were matched, so it has 300 rows rather than 3 x 438.

The false positive rate rests on 50 controls here, which only bounds it below
about 6%. Tightening that is the main reason to do the full run.
