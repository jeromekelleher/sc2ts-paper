# TODO add this file to the paper repo at some point
#!/bin/bash
set -e
set -u
# set -x

mismatches=4
max_hmm_cost=5
num_threads=40
max_missing_sites=500
min_group_size=10
min_root_mutations=2
max_recurrent_mutations=2
max_mutations_per_sample=5
retrospective_window=7
# max_daily_samples=1000

datadir=data
run_id="maskdel-v1-mm_${mismatches}-f_500-mrm_${min_root_mutations}-"
run_id+="mms_${max_mutations_per_sample}-mrec_${max_recurrent_mutations}-"
run_id+="rw_${retrospective_window}-mgs_${min_group_size}"

resultsdir=results/$run_id
results_prefix=$resultsdir/$run_id-
logfile=logs/$run_id.log

options="--num-threads $num_threads -vv -l $logfile "
options+="--min-group-size $min_group_size "
options+="--min-root-mutations $min_root_mutations "
options+="--max-mutations-per-sample $max_mutations_per_sample "
options+="--max-recurrent-mutations $max_recurrent_mutations "
options+="--deletions-as-missing "
options+="--num-mismatches $mismatches "
options+="--retrospective-window $retrospective_window "
# options+="--max-daily-samples $max_daily_samples "

mkdir -p $resultsdir
mkdir -p logs

alignments=$datadir/Viridian_tree_cons_seqs_imported/aln.db
metadata=$datadir/Viridian_tree_cons_seqs_imported/metadata.db
# matches=$resultsdir/matches.db
# Putting the match_db on HDD seems to slow things down quite a bit.
# Probably because we need to sync it after each day. Putting it on 
# the SSD helps.
matches=/scratch/jk/tmp/matches-$run_id.db

last_ts=$resultsdir/initial.ts
python3 -m sc2ts initialise -v $last_ts $matches \
    --problematic-sites=maskdel_problematic_sites_v1.txt \
    --mask-flanks 

initial_phase=2020-03-01
dates=`python3 -m sc2ts list-dates $metadata --before $initial_phase` 
for date in $dates; do
    out_ts="$results_prefix$date".ts
    python3 -m sc2ts extend $last_ts $date $alignments $metadata \
        $matches $out_ts $options
    last_ts=$out_ts
done

# Start filtering on missign data after initial phase
options+="--max-missing-sites $max_missing_sites "
# Exclude 2020-12-31
dates=`python3 -m sc2ts list-dates $metadata --after $initial_phase | grep -v 2020-12-31`

# NOTE: used to pick inference up from a given point if it needs to be stopped.
# start_date=2021-02-22
# dates=`python3 -m sc2ts list-dates --after=$start_date $metadata | grep -v 2020-12-31`
# last_ts="$results_prefix$start_date".ts

for date in $dates; do
    out_ts="$results_prefix$date".ts
    python3 -m sc2ts extend $last_ts $date $alignments $metadata \
        $matches $out_ts $options
    last_ts=$out_ts
done


