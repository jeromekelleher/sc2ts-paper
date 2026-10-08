### Description
This Snakemake workflow measures how well sc2ts recovers recombinants and the
overall ARG from simulated pathogen sequences, where the truth is known.

For each pathogen parameter set and replicate, `santasim_like.slim` simulates a
neutral, haploid Wright-Fisher population that grows from a single founder,
recording the full pedigree and tree sequence. Individuals are sampled from every
generation with probability `P`, and sc2ts is run on their sequences for each
value of `k` (`num_mismatches`). The inferred ARG is then scored against:

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
sc2ts 1.1 or later is needed for custom reference genomes.

To run one pathogen, ask for its summary, overriding config values as needed.
For example, the `coronavirus` pilot:
```
snakemake --cores 4 results/coronavirus/summary.csv --config 'p_values=[0.02,0.1]'
```

Tests run a small SLiM simulation, so need `slim` on the `PATH`:
```
python -m pytest tests
```
`tests/tiny_config.yaml` runs the whole pipeline for a `tiny` pathogen in under a minute:
```
snakemake --cores 4 results/tiny/summary.csv --configfile tests/tiny_config.yaml
```


### Configuration
`config.yaml` has a named set of SLiM parameters for each pathogen under
`pathogens` (see `santasim_like.slim` for what they mean), the number of
`replicates`, the sampling probabilities `P` (`p_values`), the `k_values`, and the
other sc2ts `extend` parameters under `sc2ts`. Replicate `r` uses SLiM seed `seed + r`, and the
same seed for sampling, so for a given replicate the samples are nested as `P`
increases. The founder is always sampled and is used as the sc2ts reference.

Each generation is one day, starting on 2026-01-01.


### Steps
| Rule | Command | Output |
|---|---|---|
| `run_slim` | `santasim_like.slim` | `sim.slim.{ts,sequences.fa,pedigree.tsv}` |
| `make_reference` | | `reference.fa`, the founder's sequence |
| `sample` | `pipeline.py sample` | `p{P}/true.trees`, the true ARG of the samples |
| `export_samples` | `pipeline.py export-samples` | `p{P}/{sequences.fa,metadata.tsv}` |
| `make_truth` | `pipeline.py make-truth` | `p{P}/truth.csv` |
| `import_dataset`, `zip_dataset` | `sc2ts import-*` | `p{P}/dataset.vcz.zip` |
| `infer` | `pipeline.py write-sc2ts-config`, `sc2ts infer` | `p{P}/k{k}/inferred.ts`, the final day's ARG |
| `evaluate` | `pipeline.py evaluate` | `p{P}/k{k}/{evaluation,events,placement,samples}.csv` |
| `combine` | | `results/{pathogen}/{summary,events,placement}.csv` |

Everything else is written under `results/{pathogen}/rep{r}/`.

Truth is made separately from the exported sequences so that scoring can be
changed without re-running sc2ts, e.g.:
```
snakemake --cores 4 results/coronavirus/summary.csv --config 'p_values=[0.02,0.1]' \
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
ancestor) were placed in the ARG. This depends on what sc2ts placed, so it is
decided in `evaluate`.


### Output
`results/{pathogen}/summary.csv` has one row per replicate, `P` and `k`. Recombinant
detection is scored over the samples sc2ts placed in the ARG, where a sample is
inferred to be a recombinant if its HMM match has more than one parent. Precision
is per sample. Recall and breakpoint accuracy are per recombination event, as several samples
can share an expected recombinant ancestor and once sc2ts has inferred the
recombination in one of them the others correctly match it with a single parent.
Breakpoints are taken from the earliest sample the event was found in.

| Column | |
|---|---|
| `num_samples`, `num_placed` | Samples, and those placed in the inferred ARG |
| `num_sampled_recombinants` | Placed samples that are recombinants themselves |
| `num_expected_recombinants` | Placed samples expected to be inferred as recombinants |
| `num_inferred_recombinants` | Placed samples with a multi-parent HMM match |
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
error. `results/{pathogen}/placement.csv` has the number of samples, and the number
placed, per generation per run. Both stay small as runs are added, so are what to
keep. The per-sample truth and inferred matches are in each run's `samples.csv`,
which is not combined.


### Pilot
`pilot/coronavirus_{summary,events,placement}.csv` are the results of the
`coronavirus` pilot, copied from `results/`. They are analysed in
`notebooks/analysis_slim_coronavirus_pilot.ipynb`.
