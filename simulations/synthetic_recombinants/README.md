### Description
This Snakemake workflow estimates the false positive and detection rates of
sc2ts recombinant matching as a function of the number of mismatches, `k`,
under idealised conditions.

Synthetic single-breakpoint Alpha x Delta recombinants are made by splicing
together pairs of real sequences, and matched against the raw inference ARG
using `sc2ts run-hmm`. Real non-recombinant sequences held out of the parent
pool are matched in the same way as controls, so a control reported with more
than one parent is a false positive.


### Date
Samples are taken from **2021-05-16** in the UK, when Alpha and Delta were
co-circulating at close to equal frequency: among the UK samples from that day
accepted by the published inference, 342 are Alpha (`B.1.1.7`, `Q.*`) and 334
are Delta (`B.1.617.2`, `AY.*`). Delta's share of UK samples rose from about 5%
on 19 April 2021 to over 95% by mid-June, crossing 50% on 15-16 May.


### Setup
The pool of real samples is split in half within each lineage: one half
supplies recombinant parents, the other supplies controls. The two are
disjoint, so no control is a parent of any recombinant.

Each recombinant draws one Alpha and one Delta parent, chooses at random which
is on the left, and takes a single breakpoint uniformly over the genome.
Positions below the breakpoint come from the left parent and the rest from the
right parent.

Matching is against the ARG for **2021-05-15**, the day before the samples were
collected, so none of the parents or controls are in it and the synthetic
samples are matched as new arrivals, as `extend` would see them.


### Detectability
A breakpoint drawn uniformly will sometimes land where the two parents have no
distinguishing sites on one side, and such a recombinant cannot be detected for
any `k`. These are kept rather than redrawn, and `num_informative_left` and
`num_informative_right` in the output record how many sites separate the parents
either side of the true breakpoint, so detection can be reported conditional on
being detectable. Only sites present in the ARG are counted, as those are the
only ones the HMM sees.


### Output
`results.csv` has one row per (strain, `k`), giving the number of parents
matched, the HMM cost, the inferred breakpoints, the lineage of each matched
parent, and the simulation truth. Note that `breakpoint_correct` is judged
against the informative sites of the *true* parents, whereas the HMM matches
nodes in the ARG that are relatives of those parents, so it is a strict
criterion; `breakpoint_error` gives the raw distance.


### Running
```
snakemake --cores 4
```

Matching is the expensive step: roughly 7 seconds per sample per value of `k`
on 4 cores, against an ARG of 242,799 samples.
