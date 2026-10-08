### Description
This Snakemake workflow measures how well sc2ts recovers recombinants and the
overall ARG from simulated pathogen sequences, where the truth is known.

For each pathogen and replicate, `santasim_like.slim` simulates a neutral, haploid
Wright-Fisher population that grows from a single founder, recording the full
pedigree and tree sequence. Individuals are sampled from every generation with each
of the pathogen's sampling probabilities `P`, and sc2ts is run on their sequences
for each value of `k` (`num_mismatches`). The inferred ARG is then scored against:

- the **pedigree**, for recombinant detection in each sample, and
- the **true ARG**, the SLiM tree sequence simplified to the samples, with
  [tscompare](https://github.com/tskit-dev/tscompare).

All the code after the SLiM simulation is in `pipeline.py`, as click subcommands:
`sample`, `export-samples`, `make-truth`, `write-sc2ts-config` and `evaluate`.
Tests are in `tests/`.


### Running
Create the environment and run from `simulations/slim/`:
```
mamba env create -f environment.yml
conda activate sc2ts-slim
snakemake --cores 8
```
sc2ts 1.1 or later is needed for custom reference genomes. This runs every pathogen
in `config.yaml` and copies each one's combined results to `summaries/{pathogen}/`.
To run one pathogen, ask for its summaries, e.g.:
```
snakemake --cores 4 summaries/coronavirus/summary.csv
```

Tests run a small SLiM simulation, so need `slim` on the `PATH`:
```
python -m pytest tests
```
`tests/tiny_config.yaml` runs the whole pipeline for a `tiny` pathogen in under a
minute. Config files are merged, so it adds `tiny` to the pathogens in
`config.yaml`; ask for its results to run it alone:
```
snakemake --cores 4 results/tiny/summary.csv --configfile tests/tiny_config.yaml
```


### Configuration
`config.yaml` lists the pathogens studied under `pathogens`, each with its SLiM
parameters under `slim` (see `santasim_like.slim` for what they mean) and its
sampling probabilities `P` under `p_values`. To add a pathogen, add an entry
there. The rest of the config is shared: the number of `replicates`, the
`k_values`, and the other sc2ts `extend` parameters under `sc2ts`. Replicate `r`
uses SLiM seed `seed + r`, and the same seed for sampling, so for a given
replicate the samples are nested as `P` increases. The founder is always sampled
and is used as the sc2ts reference.

Pathogens so far:
- `coronavirus`: a generic coronavirus, with a 30 kb genome, 0.0008 substitutions per
  site per year and a 5.5 day generation time (about 0.36 mutations per genome per
  generation), growing to 1,000 cases per generation over 20 generations and
  staying there for 80 more. 1% of transmissions are recombinant. Sampled with
  P = 0.02 and 0.1.
- `sars_like`: SARS-CoV-like, with a 30 kb genome, 5e-4 substitutions per site per year
  and one generation per day, growing to 10,000 cases per generation over 6 months and
  simulated for 2 years. Sampled with P = 0.02 and 0.1.
- `flu_like`: pdm2009 H1N1-like, with a 13 kb genome, 2.75e-3 substitutions per site per
  year and one generation per day, with the same demography as `sars_like`. Sampled with
  P = 0.02 and 0.1.

Each generation is one day, starting on 2026-01-01.

#### No time-traveller filtering
On real data, sc2ts holds back any sample whose HMM cost (mismatches plus `k` times
the number of breakpoints) is above `hmm_cost_threshold`, and only adds held-back
samples later if enough of them form a plausible retrospective group, as judged by
`min_group_size`, `min_different_dates`, `retrospective_window`,
`max_pango_lineages`, `min_root_mutations`, `max_recurrent_mutations` and
`max_mutations_per_sample`. This is there to keep out "time travellers", samples
whose recorded date is badly wrong, and samples with many sequencing errors (see
"Filtering time travellers" and "Inserting saltational lineages" in the paper's
methods).

Simulated samples have exact dates and no sequencing errors, so here the filter
only costs us samples: in the coronavirus simulation with the Viridian settings
(threshold 7), 39% of samples at P = 0.02 and 3% at P = 0.1 were held back and
never added to the ARG. So `hmm_cost_threshold` is set high enough (1,000,000) that
every sample is added on its own day. It can't simply be left out, as sc2ts then defaults it to 5.
With nothing held back the retrospective group parameters are never used, so they
are left out and take sc2ts's defaults.


### Steps
| Rule | Command | Output |
|---|---|---|
| `run_slim` | `santasim_like.slim` | `sim.slim.{ts,sequences.fa,pedigree.tsv}` |
| `make_reference` | | `reference.fa`, the founder's sequence |
| `sample` | `pipeline.py sample` | `p{P}/true.trees`, the true ARG of the samples |
| `export_samples` | `pipeline.py export-samples` | `p{P}/{sequences.fa,metadata.tsv}` |
| `make_truth` | `pipeline.py make-truth` | `p{P}/truth.csv` |
| `import_dataset`, `zip_dataset` | `sc2ts import-*` | `p{P}/dataset.vcz.zip` |
| `infer` | `pipeline.py write-sc2ts-config`, `sc2ts infer` | `p{P}/k{k}/inferred.ts`, the final day's ARG, and the match DB |
| `postprocess` | `sc2ts postprocess` | `p{P}/k{k}/inferred_pp.ts`, with exact matches added |
| `evaluate` | `pipeline.py evaluate` | `p{P}/k{k}/{evaluation,events,placement,samples}.csv` |
| `combine` | | `results/{pathogen}/{summary,events,placement}.csv` |
| `copy_summary` | | `summaries/{pathogen}/{summary,events,placement}.csv` |

Everything else is written under `results/{pathogen}/rep{r}/`.

Truth is made separately from the exported sequences so that scoring can be
changed without re-running sc2ts, e.g.:
```
snakemake --cores 4 summaries/coronavirus/summary.csv \
    --rerun-triggers mtime --forcerun make_truth
```


### Truth
SLiM records which individuals were made from two parents, with a single
breakpoint. But a sample can also carry a recombination it inherited: if the
recombinant ancestor isn't in the inferred ARG, the sample is the first place
sc2ts can see it. So `truth.csv` records, for each sample, its nearest
`recombinant_ancestor` on its clonal line (itself, if it is a recombinant), that
ancestor's breakpoint, whether the ancestor is `detectable` (its sequence differs
from both of its parents'), and the sampled individuals in between.

A sample is then *expected* to be inferred to be a recombinant if it has a
recombinant ancestor and none of the sampled individuals in between (including the
ancestor) were placed in the ARG (see below). This depends on what sc2ts did, so it is
decided in `evaluate`.


### Output
`results/{pathogen}/summary.csv` has one row per replicate, `P` and `k`.

During inference sc2ts doesn't add a node for a sample identical to one already in
the ARG (an HMM cost of 0), but only counts it against that node. `sc2ts
postprocess` adds these *exact matches* as sample nodes from the match DB, as for
the published ARG, so the post-processed ARG is the one scored. A sample is
*placed* if it has a node in it; anything else was held back by sc2ts. Recombinant
detection is scored over placed samples, where a sample is inferred to be a
recombinant if its HMM match has more than one parent. Precision is per sample.
Recall and breakpoint accuracy are per recombination event, as several samples can
share an expected recombinant ancestor and once sc2ts has inferred the
recombination in one of them the others correctly match it with a single parent.
Breakpoints are taken from the earliest sample the event was found in.

| Column | |
|---|---|
| `num_samples` | Samples |
| `num_placed`, `num_exact_matches`, `num_held_back` | Samples with a node in the ARG, those of them added as exact matches, and samples without a node |
| `num_sampled_recombinants` | Placed samples that are recombinants themselves |
| `num_expected_recombinants` | Placed samples expected to be inferred as recombinants |
| `num_inferred_recombinants` | Samples with a multi-parent HMM match |
| `true_positives`, `false_positives`, `precision` | Inferred recombinants that are, and aren't, expected |
| `num_events`, `num_detectable_events` | Distinct recombinant ancestors of expected samples |
| `events_detected`, `recall`, `recall_detectable` | Events with at least one expected sample inferred to be a recombinant |
| `mean_abs_breakpoint_error` | For events found with one breakpoint, in bases |
| `breakpoint_in_interval` | Fraction of those whose true breakpoint is in sc2ts's breakpoint interval |
| `median_interval_width` | Of those breakpoint intervals, in bases |
| `num_recombinant_nodes` | Nodes with more than one parent in the inferred ARG |
| `arf`, `tpr`, `rmse` | `tscompare.haplotype_arf(inferred, true)` |

For tscompare, both ARGs are simplified to the placed samples in the same order,
sc2ts's 1-based coordinates are shifted to SLiM's 0-based ones, and the inferred
times are shifted to match the true ones. `arf` is the fraction of the inferred
ARG's span not represented in the true ARG, `tpr` the fraction of the true ARG's
span represented in the inferred one.

`results/{pathogen}/events.csv` has one row per expected recombination event per
run: how many samples carry it, whether the recombinant itself was sampled,
whether it is detectable and detected, and the breakpoint interval width and
error. `results/{pathogen}/placement.csv` has the number of samples, the number
placed, and the number of those added as exact matches, per generation per run.
Both stay small as runs are added, so are what to keep. The per-sample truth and
inferred matches are in each run's `samples.csv`, which is not combined.


### Summaries
`summaries/{pathogen}/{summary,events,placement}.csv` are copies of each pathogen's
combined results, kept in the repository (`results/` is not). They are analysed in
`notebooks/analysis_slim_pathogen_simulations.ipynb`, which reads every pathogen
found there.
