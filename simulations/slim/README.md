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

All significant code is in `scripts/`, one click command per step, with tests in
`tests/`.


### Running
Create the environment and run from `simulations/slim/`:
```
mamba env create -f environment.yml
conda activate sc2ts-slim
snakemake --cores 8
```
sc2ts 1.1 or later is needed for custom reference genomes.

Tests run a small SLiM simulation, so need `slim` on the `PATH`:
```
python -m pytest tests
```
`tests/tiny_config.yaml` runs the whole pipeline in a minute or so:
```
snakemake --cores 4 --configfile tests/tiny_config.yaml
```


### Configuration
`config.yaml` has a named set of SLiM parameters for each pathogen under
`pathogens` (see `santasim_like.slim` for what they mean), the number of
`replicates`, the sampling probabilities `P`, the `k_values`, and the other sc2ts
`extend` parameters under `sc2ts`. Replicate `r` uses SLiM seed `seed + r`, and the
same seed for sampling, so for a given replicate the samples are nested as `P`
increases. The founder is always sampled and is used as the sc2ts reference.

Each generation is one day, starting on 2026-01-01.


### Steps
| Rule | Script | Output |
|---|---|---|
| `run_slim` | `santasim_like.slim` | `sim.slim.{ts,sequences.fa,pedigree.tsv}` |
| `make_reference` | | `reference.fa`, the founder's sequence |
| `sample` | `sample_slim.py` | `p{P}/true.trees`, the true ARG of the samples |
| `export_samples` | `export_samples.py` | `p{P}/{sequences.fa,metadata.tsv,truth.csv}` |
| `import_dataset`, `zip_dataset` | `sc2ts import-*` | `p{P}/dataset.vcz.zip` |
| `infer` | `write_sc2ts_config.py`, `sc2ts infer` | `p{P}/k{k}/inferred.ts`, the final day's ARG |
| `evaluate` | `evaluate.py` | `p{P}/k{k}/{evaluation,samples}.csv` |
| `combine` | | `results/summary.csv` |

Everything is written under `results/{pathogen}/rep{r}/`.


### Truth
A sample is a true recombinant if SLiM made it from two parents, with a single
breakpoint. Because the parents may not differ on one side of the breakpoint,
`truth.csv` also records whether it is `detectable`: whether its sequence differs
from both of its parents'.


### Output
`results/summary.csv` has one row per pathogen, replicate, `P` and `k`. Recombinant
detection is scored over the samples sc2ts placed in the ARG, where a sample is
inferred to be a recombinant if its HMM match has more than one parent.

| Column | |
|---|---|
| `num_samples`, `num_placed` | Samples, and those placed in the inferred ARG |
| `num_true_recombinants`, `num_detectable_recombinants` | Among placed samples |
| `num_inferred_recombinants` | Placed samples with a multi-parent HMM match |
| `true_positives`, `false_positives`, `false_negatives`, `precision`, `recall` | Against `is_recombinant` |
| `recall_detectable` | Recall against `detectable` |
| `mean_abs_breakpoint_error` | For true positives with one inferred breakpoint, in bases |
| `breakpoint_in_interval` | Fraction of those whose true breakpoint is in sc2ts's breakpoint interval |
| `num_recombinant_nodes` | Nodes with more than one parent in the inferred ARG |
| `arf`, `tpr`, `rmse` | `tscompare.haplotype_arf(inferred, true)` |

For tscompare, both ARGs are simplified to the placed samples in the same order,
sc2ts's 1-based coordinates are shifted to SLiM's 0-based ones, and the inferred
times are shifted to match the true ones. `arf` is the fraction of the inferred
ARG's span not represented in the true ARG, `tpr` the fraction of the true ARG's
span represented in the inferred one.

`p{P}/k{k}/samples.csv` has the truth and inferred match for every sample.
