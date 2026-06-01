### Description
This Snakemake workflow:
1. Generates SANTA-SIM XML config files over a grid of mutation rates, recombination rates, and sampling frequencies.
2. Runs SANTA-SIM and parses output for sc2ts runs.
3. Imports alignments and metadata into sc2ts datasets.
4. Runs sc2ts inference for each replicate and each value of `num_mismatches` (k).
5. Postprocesses the final trees (adding exact matches).
6. Summarises sc2ts recombination detection performance.
7. Visualises performance results.


### Dependencies
This workflow uses a modified version of SANTA-SIM (https://github.com/koadman/santa-sim/). This particular version prints out breakpoint locations for simulated recombinant sequences in the description lines of output FastA files. The config files to run SANTA-SIM, as done in Jaya et al. (2023), are available in https://github.com/fredjaya/rec-bench/tree/master/data/xml.

Java and Ant need to be installed before running this Snakemake pipeline. During testing, SANTA-SIM was built and run using Java version 1.8.0_441 and Ant version 1.10.15.


### Simulation setup
We consider host-to-host transmission, where one transmission event is one generation (in SANTA-SIM, generations are discrete and non-overlapping). We assume neutral fitness and a constant environment (only one epoch). We consider a sampling period of 30 days, collecting genomes at different frequencies: once per 3 days, 5 days, and 7 days. TODO: Appropriate values for these parameters need to be determined.


### Sc2ts inference settings
In the sc2ts config files, exclude_dates and exclude_sites are empty, as there are no problematic collection dates or genomic positions. Missing data are not simulated. Indels are not simulated, but possible using SANTA-SIM. No samples are unconditionally inserted. HMM cost threshold is increased, and the requirements for adding sample groups are relaxed (decreased min_group_size and increased retrospective_window). TODO: Appropriate values for these parameters need to be determined.


### Additional notes
A sc2ts run on a dataset of 1,000 SARS-CoV-2-like samples (10 samples collected per day over 100 days) took ~7 minutes (8 CPUs on a MacBook Pro, M3 Pro).

The workflow crashes if a final tree sequence has no samples. This can happen for simulated datasets which are thinly sampled such that all samples have high HMM costs and the retrospective sample grouping parameters do not allow for rejected samples to be added.
