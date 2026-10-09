### Description
This Snakemake workflow measures how well sc2ts recovers recombinants and the
overall ARG from simulated pathogen sequences, where the truth is known.

For each pathogen and replicate, `santasim_like.slim` simulates a neutral, haploid
Wright-Fisher population that grows from a single founder. For each of the
pathogen's numbers of samples per day `S`, `S` individuals are sampled from every
generation, and sc2ts is run on their sequences for each value of `k`
(`num_mismatches`). The inferred ARG is then scored against:

- the **recombination history** SLiM records, for recombinant detection in each
  sample, and
- the **true ARG**, the SLiM tree sequence simplified to the samples, with
  [tscompare](https://github.com/tskit-dev/tscompare).

All the code after the SLiM simulation is in `pipeline.py`, as click subcommands:
`sample`, `export-samples`, `make-truth`, `write-sc2ts-config`, `evaluate`,
`rematch-recombinants` and `recombinant-table`.
Tests are in `tests/`.


### Running
Create the environment and run from `simulations/slim/`:
```
mamba env create -f environment.yml
conda activate sc2ts-slim
snakemake --cores 8
```
This runs every pathogen
in `config.yaml` and copies each one's combined results to `summaries/{pathogen}/`.
To run one pathogen, ask for its summaries, e.g.:
```
snakemake --cores 4 summaries/coronavirus_like/summary.csv
```
As `results/` is not kept, Snakemake counts the summaries in the repository as up to
date, and won't remake them after the config changes. Force them to be remade
(along with the runs behind them) with `--forcerun copy_summary`.

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
numbers of samples per day under `samples_per_day`. To add a pathogen, add an
entry there. The rest of the config is shared: the number of `replicates`, the
`k_values`, and the other sc2ts `extend` parameters under `sc2ts`. Replicate `r`
uses SLiM seed `seed + r`. The founder is always sampled and is used as the sc2ts
reference.

Pathogens so far:
- `coronavirus_like`: a generic coronavirus, with a 30 kb genome, 0.0008 substitutions
  per site per year and a 5.5 day generation time (about 0.36 mutations per genome
  per generation), growing to 1,000 cases per generation over 20 generations and
  staying there for 180 more. 0.5% of transmissions are recombinant, several times
  the rate estimated for SARS-CoV-2, to give enough recombination events to measure.
  20 and 100 samples per day.
- `flu_like`: pdm2009 H1N1-like, with a 13 kb genome, 2.75e-3 substitutions per site per
  year and one generation per day, growing to 1,000,000 cases per generation over 50
  days and simulated for 100. 200 and 1,000 samples per day.

A SARS-CoV-like pathogen (5e-4 substitutions per site per year, with the same
demography as `flu_like`) was also tried, but in 100 days builds up too little
diversity (about 4 mutations along a lineage) for sc2ts to find any recombinants.

Each generation is one day, starting on 2026-01-01.

#### Sampling and recombinant ancestry in SLiM
With a million individuals per generation, keeping every individual's sequence and
the full pedigree isn't feasible, so SLiM does the sampling and keeps track of what
the truth needs as it goes (see the comments at the top of `santasim_like.slim`):

- Each generation, SLiM puts the individuals in a random order and keeps the first
  `S_MAX`, the largest of the pathogen's `samples_per_day`, as *candidates*, with
  their rank. Sampling `S` per day takes the candidates of rank below `S`: a uniform
  random sample of each generation (all of it, if smaller), nested as `S`
  increases. Only candidates are remembered in the tree sequence and have their
  sequences written.
- Each individual inherits, along its clonal line, its nearest recombinant ancestor
  and its nearest candidate ancestor since then, which SLiM writes for each
  candidate. Recombinants (rare) are written with their breakpoint and whether they
  differ from both parents.

At N = 1,000,000 with 1,000 samples per day this takes about 2 minutes and 9 GB
per replicate (measured for the SARS-CoV-like pathogen, which has the same
demography as `flu_like`).

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
only costs us samples: in an earlier coronavirus simulation (100 generations, 1% of
transmissions recombinant) with the Viridian settings
(threshold 7) and about 20 and 100 samples per day, 39% and 3% of samples were
held back and never added to the ARG. So `hmm_cost_threshold` is set high enough (1,000,000) that
every sample is added on its own day. It can't simply be left out, as sc2ts then defaults it to 5.
With nothing held back the retrospective group parameters are never used, so they
are left out and take sc2ts's defaults.


### Steps
| Rule | Command | Output |
|---|---|---|
| `run_slim` | `santasim_like.slim` | `sim.slim.{ts,sequences.fa,samples.tsv,recombinants.tsv}` |
| `make_reference` | | `reference.fa`, the founder's sequence |
| `sample` | `pipeline.py sample` | `s{S}/true.trees`, the true ARG of the samples |
| `export_samples` | `pipeline.py export-samples` | `s{S}/{sequences.fa,metadata.tsv}` |
| `make_truth` | `pipeline.py make-truth` | `s{S}/truth.csv` |
| `import_dataset`, `zip_dataset` | `sc2ts import-*` | `s{S}/dataset.vcz.zip` |
| `infer` | `pipeline.py write-sc2ts-config`, `sc2ts infer` | `s{S}/k{k}/inferred.ts`, the final day's ARG, and the match DB |
| `postprocess` | `sc2ts postprocess` | `s{S}/k{k}/inferred_pp.ts`, with exact matches added |
| `evaluate` | `pipeline.py evaluate` | `s{S}/k{k}/{evaluation,events,placement,samples}.csv` |
| `rematch_recombinants` | `pipeline.py rematch-recombinants` | `s{S}/k{k}/recombinants_rematch.json` |
| `recombinant_table` | `pipeline.py recombinant-table` | `s{S}/k{k}/recombinants.csv` |
| `combine` | | `results/{pathogen}/{summary,events,placement,recombinants}.csv` |
| `copy_summary` | | `summaries/{pathogen}/{summary,events,placement,recombinants}.csv` |

Everything else is written under `results/{pathogen}/rep{r}/`.

Truth is made separately from the exported sequences so that scoring can be
changed without re-running sc2ts, e.g.:
```
snakemake --cores 4 summaries/coronavirus_like/summary.csv \
    --rerun-triggers mtime --forcerun make_truth
```


### Truth
Recombinants are individuals made from two parents, with a single breakpoint. But
a sample can also carry a recombination it inherited: if the recombinant ancestor
isn't in the inferred ARG, the sample is the first place sc2ts can see it. So
`truth.csv` records, for each sample, its nearest `recombinant_ancestor` on its
clonal line (itself, if it is a recombinant), that ancestor's breakpoint, whether
the ancestor is `detectable` (its sequence differs from both of its parents'), and
the candidates in between, found by following each candidate's nearest candidate
ancestor.

A sample is then *expected* to be inferred to be a recombinant if it has a
recombinant ancestor and none of the sampled individuals in between (including the
ancestor) were placed in the ARG (see below). This depends on what sc2ts did, so it is
decided in `evaluate`.


### Output
`results/{pathogen}/summary.csv` has one row per replicate, number of samples per
day and `k`.

During inference sc2ts doesn't add a node for a sample identical to one already in
the ARG (an HMM cost of 0), but only counts it against that node. `sc2ts
postprocess` adds these *exact matches* as sample nodes from the match DB, as for
the published ARG, so the post-processed ARG is the one scored. Its other tidy-ups
(pushing up unary recombinant mutations and reversions) are turned off with
`--no-mutation-updates`, so the ARG is otherwise as sc2ts inferred it. A sample is
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
| `arf_internal`, `tpr_internal` | ARF and TPR over non-sample nodes only |
| `*_resolved` | The above, with both ARGs reduced to what sequences can resolve (see below) |

For tscompare, both ARGs are simplified to the placed samples in the same order,
sc2ts's 1-based coordinates are shifted to SLiM's 0-based ones, and the inferred
times are shifted to match the true ones. `arf` is the fraction of the inferred
ARG's span not represented in the true ARG, `tpr` the fraction of the true ARG's
span represented in the inferred one.

Samples make up much of both ARGs, and tscompare matches each sample only to
itself, which is always right for samples without descendants. `arf_internal`
and `tpr_internal` leave samples out: each non-sample node is credited with the
span it shares with its best match in the other ARG, weighted by its span. They
are blank if an ARG has no non-sample nodes.

The sequences can't resolve every part of an ARG, so the `_resolved` columns
compare both ARGs after reducing them to the same canonical form:

1. Every sample is made a leaf. A sample with children is replaced by a new
   node just above it, which takes its parents, mutations and children, with
   the sample as one more child. Whether a sample is the ancestor of others, or
   their sibling under an identical ancestor, can't be told from sequences:
   sc2ts often puts samples under an earlier sample where the true ancestor was
   unsampled. tscompare only matches samples to themselves, so without this
   these would count as errors.
2. A non-sample node with no mutations and a single parent has the same
   haplotype as its parent, so whether its children descend from it or from
   the parent can't be told apart either. Many nodes in the true ARG are like
   this. Their children are attached to their parent (where they have one) and
   they are removed. Recombinant nodes are kept.

After step 1 every sample matches itself exactly, so `arf_resolved` and
`tpr_resolved` are dominated by samples; `arf_internal_resolved` and
`tpr_internal_resolved` are the measures to look at.

`results/{pathogen}/events.csv` has one row per expected recombination event per
run: how many samples carry it, whether the recombinant itself was sampled,
whether it is detectable and detected, and the breakpoint interval width and
error. `results/{pathogen}/placement.csv` has the number of samples, the number
placed, and the number of those added as exact matches, per generation per run.
Both stay small as runs are added, so are what to keep. The per-sample truth and
inferred matches are in each run's `samples.csv`, which is not combined.


#### Recombinants and mutations averted
`results/{pathogen}/recombinants.csv` has one row per recombinant node in the
inferred ARG per run, to compare the evidence for each recombinant with the truth.
As for the published ARG (`arg_postprocessing/scripts/rematch_recombinants.py`),
each recombinant is rematched against sc2ts's ARG for the day before it was added,
using `sc2ts.inference.rematch_recombinant` (as `sc2ts rematch-recombinant` does),
once with recombination and once forced to a single parent. This is run on the
ARG from `sc2ts infer` (`inferred.ts`), as it needs sc2ts's daily ARGs, which are
kept in each run's `sc2ts/results`. `mutations_averted` is `k1000_muts`, the number
of mutations in the single-parent match, less `num_mutations`, the number in the
recombinant match: the measure of support for recombinants used in the paper. A
recombinant whose single-parent match costs the same as its recombinant match
(for one breakpoint, `mutations_averted` equal to `k`) is a tie that sc2ts resolved
in favour of recombination.

The causal samples of a recombinant node are the samples added with it (in the
same group) whose HMM match has more than one parent. The node is a
`true_positive` if any of them is an expected recombinant, and the truth columns
(`recombinant_ancestor`, `ancestor_breakpoint`, `ancestor_detectable` and whether
the true breakpoint is in the inferred interval) are from the earliest such sample.
Being per node, these counts differ slightly from the per-sample precision in
`summary.csv`.

False positives are put into one of two categories, from the true ARG. For the
earliest causal sample, the MRCA with its nearest relatives among the placed
samples of earlier generations is found along the genome:

- `relative_recombinant`: the MRCA changes along the genome. The sample has no
  unrepresented recombination of its own, but a relative is (or descends from)
  a recombinant, and was the first sample carrying the new mutations of one of
  its parental lineages. Those mutations are on the relative's node in the ARG,
  so a later, non-recombinant member of that lineage is matched to the relative
  over the segment it inherited from the lineage, and to another node over the
  rest. The recombination is real, but placed on the wrong lineage, usually with
  the relative's breakpoint. `relative_breakpoints` are the positions where the
  MRCA changes, and `relative_breakpoint_in_interval` whether one of them is in
  the inferred breakpoint interval.
- `unresolved_branch`: the MRCA is the same along the genome. The ARG has no
  node close to the sample's true ancestor, typically because a long branch
  carries both mutations the sample shares and mutations it doesn't, and those
  it doesn't are concentrated on one side of the genome.

The rematches are not combined, as they are larger. Summaries from before this
step was added don't have `recombinants.csv`, and need the runs redone.


### Summaries
`summaries/{pathogen}/{summary,events,placement,recombinants}.csv` are copies of each pathogen's
combined results, kept in the repository (`results/` is not). They are analysed in
`notebooks/analysis_slim_pathogen_simulations.ipynb`, which reads every pathogen
found there.
