### Description
This Snakemake workflow estimates the false positive and detection rates of sc2ts
recombinant matching as a function of the recombination penalty `k`.

The paper chooses `k = 4` on qualitative grounds: `k = 3` gave "many dubious
recombination events", `k = 5` "failed to include the well supported recombination
events documented in" Jackson et al. Those are a false positive claim and a false
negative claim, and this workflow measures both.

Synthetic single-breakpoint recombinants are made by splicing together pairs of real
sequences, and matched against the raw inference ARG using `sc2ts run-hmm`. Real
sequences held out of the parent pool are matched the same way as controls, so a
control reported with more than one parent is a false positive.

All the code is in `simulate.py`, as click subcommands: `make-pool`, `generate` and
`summarise`.


### Date and lineages
Samples are taken from **2021-05-16** in the UK, when Alpha and Delta were
co-circulating at close to equal frequency. Lineages are read from `Viridian_scorpio`:
Alpha is any call containing `B.1.1.7-like`, Delta any call beginning `Delta (`. Among
the UK samples from that day accepted by the published inference, 342 are Alpha and 334
are Delta.

Matching is against the ARG for **2021-05-15**, the day before, so none of the parents
or controls are in it and the synthetic samples are matched as new arrivals, as
`extend` would see them.


### Crosses
Two crosses bracket the range of parent divergence in the real ARG, where
`parent_pangonet_distance` runs from 0 to 19 with a median of 2:

- **Alpha x Alpha** — parents are all plain `B.1.1.7`, so distance 0, matching the 31%
  of real recombinants whose parents share a lineage. Few sites separate the parents,
  so this is the regime where detection is hardest and false positives would arise.
- **Alpha x Delta** — distance 4 or 5, with far more signal.

The pool is split in half within each lineage, one half for parents and one for
controls, so no control is a parent of any recombinant.


### Detectability
A breakpoint that leaves one flank with no sites distinguishing the parents yields a
sequence identical to one parent, which no value of `k` could recover. Rather than draw
over the genome and reject those, breakpoints are drawn directly from the window in
which they are detectable — between the first and last site separating the parents —
which gives the same distribution without the loop. `detectable_window` in the output
records the span that was sampled.


### Parent mutations
The parents of real recombinants are rarely sampled themselves, so the closest sampled
relatives the HMM can copy from differ from them by a few private mutations. To model
this, each parent is given Poisson mutations before splicing, at positions drawn
uniformly over the genome, each to one of the other three bases (missing and ambiguous
positions are left alone). Each parent is mutated independently every time it is used.

The mean is a multiple of the number expected in one transmission generation, with a
substitution rate of 0.0008 per site per year (about 24 per genome per year) and a
generation time of 5.5 days, which gives 0.36 mutations per sequence. There are four
arms:

- **0x** — the sequences as they are, as a control.
- **1x** — as if every case were sequenced, so a parent is one generation from its
  closest sampled relative.
- **5x** — as if one case in five were sequenced, about 1.8 mutations per sequence.
- **10x** — as if one case in ten were sequenced, about 3.6 mutations per sequence.

Controls get the same mutations as parents, so that false positive rates are
comparable across arms. Mutations draw from their own random stream, so the parents,
breakpoints and controls are identical in every arm, and differences between arms are
due to the mutations alone. `num_added_mutations` in the output records how many sites
differ from the unmutated sequence, and `num_added_mutations_arg` how many of those are
sites in the ARG.


### Characterisation
Recombinants are characterised with the quantities the paper reports, reusing the
pipeline's own code where possible:

- **Breakpoint intervals** come from `sc2ts.inference.characterise_recombinants`, the
  function the inference pipeline itself calls. It derives both edges of the interval
  from the matched parents' haplotypes; no reverse HMM pass is involved, and none is
  needed, because the forward pass already lands exactly on the first site supporting
  the right parent.
- **Net supporting loci** follow the paper's QC measure: sites where the inferred
  parents differ, clustered into one locus when within 3 bases, scored +1 where the
  recombinant carries the assigned parent's allele and -1 where it carries the other's.
  The gate is 4 or more on both flanks, which leaves 647 of 1,319 real events. The
  scoring is ported from `arg_postprocessing/scripts/add_recombinant_minlength_to_csv.py`,
  which cannot be imported directly because it works on recombinant nodes already in an
  ARG.
- **Pangonet distance** between the parents uses `get_pangonet_distance` from
  `arg_postprocessing/scripts/add_pangonet_distance_to_csv.py` with the pinned
  pango-designation data in `arg_postprocessing/pangonet_data`.


### Parent identification
For each detected recombinant, the two parents the HMM assigned are compared with the
true parents in two ways, midway along each of the matched segments:

- **Pango lineage.** An assigned parent node takes its own `Viridian_pangolin` if it is
  a sample, and otherwise the most common one among its descendant samples (at most
  1,000). It is compared with the true parent's `Viridian_pangolin`, by exact match and
  by pangonet distance.
- **Steps to the true parent.** The true parents are not in the ARG, and never appear as
  controls, because the pool is split in half. So each one is matched directly, as the
  unmutated sequence in the Viridian dataset, with `run-hmm` at `k = 4` (the inference's
  value) against the same ARG. The node it matches is where it would attach. The
  number of edges between that node and the assigned parent in the local tree gives
  `{side}_parent_steps`. `{side}_parent_relation` records whether the assigned parent
  is the same node, an ancestor or descendant of it, or neither. A different node is
  not necessarily a wrong one: if the segment holds none of the sites separating the
  two, they cannot be told apart. `{side}_parent_diffs` counts the sites within the
  matched segment at which the assigned and true parents differ.


### Output
`results.csv` has one row per (`mutation_multiplier`, strain, `k`); strain names repeat
across arms. Each arm's files carry an `_m{multiplier}` suffix. Columns that also appear in
`data/recombinants.csv` carry the same names: `interval_left`, `interval_right`,
`net_min_supporting_loci_lft`, `net_min_supporting_loci_rgt`,
`net_min_supporting_loci_lft_rgt_ge_4`, `parent_pangonet_distance`. Parent
identification adds `{side}_inferred_pango`, `{side}_pango_correct`,
`{side}_pango_distance`, `{side}_true_node`, `{side}_parent_steps`,
`{side}_parent_relation` and `{side}_parent_diffs` for `side` in `left` and `right`,
filled for detected
recombinants only. The parents' direct matches are in `hmm_parents.jsonl`, for the
strains in `parents.strains.txt`. Analysis is in
`notebooks/analysis_synthetic_recombinants.ipynb`, which also writes the paper's
supplementary figure to `figures/synthetic_recombinants.pdf`.


### Running
```
snakemake --cores 4
```

Matching is the expensive step, at roughly 8 seconds per sample per value of `k` on 4
cores against an ARG of 242,799 samples, repeated for each mutation multiplier, plus one
direct match of the parents. The
committed configuration takes about two and a half hours, k = 5 being the slowest; `config.yaml` notes
the larger values for a full run.
