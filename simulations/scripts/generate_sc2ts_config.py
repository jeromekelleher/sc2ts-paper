"""
Generate a sc2ts TOML config for one pair of replicate index and k value.
"""

import click


TEMPLATE = """dataset="{dataset}"

run_id="{run_id}"
results_dir = "results_sc2ts/trees"
log_dir = "results_sc2ts/logs"
matches_dir= "results_sc2ts/matches"
log_level = 2

exclude_dates = []
exclude_sites = []

date_field="date"


[extend_parameters]
num_mismatches={k}
hmm_cost_threshold={hmm_cost_threshold}
max_missing_sites={max_missing_sites}
deletions_as_missing={deletions_as_missing}

# Knobs for tuning retro group insertion
min_group_size={min_group_size}
min_root_mutations={min_root_mutations}
max_recurrent_mutations={max_recurrent_mutations}
max_mutations_per_sample={max_mutations_per_sample}
retrospective_window={retrospective_window}

num_threads={num_threads}
memory_limit={memory_limit}

include_samples=[]


[[override]]
start = "2020-01-01"
stop = "3000-01-01"
parameters.max_missing_sites = 10000
"""


def run_id(sim_id: str, rep: int, k: int) -> str:
    return f"{sim_id}_rep{rep}_k{k}"


@click.command()
@click.argument("output_toml", type=click.Path(dir_okay=False))
@click.option("--sim-id", required=True)
@click.option("--rep", type=int, required=True)
@click.option("--k", type=int, required=True)
@click.option("--num-threads", type=int, required=True)
@click.option("--memory-limit", type=int, required=True)
@click.option("--hmm-cost-threshold", type=int, required=True)
@click.option("--max-missing-sites", type=int, required=True)
@click.option(
    "--deletions-as-missing",
    type=click.Choice(["true", "false"]),
    required=True,
)
@click.option("--min-group-size", type=int, required=True)
@click.option("--min-root-mutations", type=int, required=True)
@click.option("--max-recurrent-mutations", type=int, required=True)
@click.option("--max-mutations-per-sample", type=int, required=True)
@click.option("--retrospective-window", type=int, required=True)
def main(
    output_toml,
    sim_id,
    rep,
    k,
    num_threads,
    memory_limit,
    hmm_cost_threshold,
    max_missing_sites,
    deletions_as_missing,
    min_group_size,
    min_root_mutations,
    max_recurrent_mutations,
    max_mutations_per_sample,
    retrospective_window,
):
    dataset = f"results_santasim/{sim_id}_rep{rep}.aln"
    text = TEMPLATE.format(
        dataset=dataset,
        run_id=run_id(sim_id, rep, k),
        k=k,
        hmm_cost_threshold=hmm_cost_threshold,
        max_missing_sites=max_missing_sites,
        deletions_as_missing=deletions_as_missing,
        min_group_size=min_group_size,
        min_root_mutations=min_root_mutations,
        max_recurrent_mutations=max_recurrent_mutations,
        max_mutations_per_sample=max_mutations_per_sample,
        retrospective_window=retrospective_window,
        num_threads=num_threads,
        memory_limit=memory_limit,
    )
    with open(output_toml, "w") as f:
        f.write(text)


if __name__ == "__main__":
    main()
