"""
Write the TOML config file for an sc2ts inference run.
"""
import json

import click


def toml_value(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, list):
        return "[" + ", ".join(toml_value(v) for v in value) + "]"
    raise ValueError(f"Unsupported value: {value!r}")


def make_config(
    *, dataset, reference, run_id, results_dir, log_dir, matches_dir,
    num_mismatches, num_threads, extend_parameters
):
    top = {
        "dataset": dataset,
        "run_id": run_id,
        "results_dir": results_dir,
        "log_dir": log_dir,
        "matches_dir": matches_dir,
        "log_level": 2,
        "exclude_dates": [],
        "exclude_sites": [],
        "date_field": "date",
        "reference_fasta": reference,
        # The day before the founder (the reference) is sampled, as set by
        # START_DATE in export_samples.py: the reference must predate every sample.
        "reference_date": "2025-12-31",
    }
    extend = {
        **extend_parameters,
        "num_mismatches": num_mismatches,
        "num_threads": num_threads,
    }
    lines = [f"{k} = {toml_value(v)}" for k, v in top.items()]
    lines.append("")
    lines.append("[extend_parameters]")
    lines.extend(f"{k} = {toml_value(v)}" for k, v in extend.items())
    return "\n".join(lines) + "\n"


@click.command()
@click.argument("output", type=click.Path(dir_okay=False))
@click.option("--dataset", required=True)
@click.option("--reference", required=True)
@click.option("--run-id", required=True)
@click.option("--results-dir", required=True)
@click.option("--log-dir", required=True)
@click.option("--matches-dir", required=True)
@click.option("--num-mismatches", type=int, required=True)
@click.option("--num-threads", type=int, required=True)
@click.option(
    "--extend-parameters",
    default="{}",
    help="JSON object of other sc2ts extend parameters",
)
def run(
    output, dataset, reference, run_id, results_dir, log_dir, matches_dir,
    num_mismatches, num_threads, extend_parameters
):
    config = make_config(
        dataset=dataset,
        reference=reference,
        run_id=run_id,
        results_dir=results_dir,
        log_dir=log_dir,
        matches_dir=matches_dir,
        num_mismatches=num_mismatches,
        num_threads=num_threads,
        extend_parameters=json.loads(extend_parameters),
    )
    with open(output, "w") as f:
        f.write(config)


if __name__ == "__main__":
    run()
