import datetime
import dataclasses
import os
import concurrent.futures as cf

import sc2ts
import pandas as pd
import click
import tqdm


@dataclasses.dataclass
class MatchWork:
    ts_path: str
    ds_path: str
    sample_id: str
    num_mismatches: int


def run_match(m):
    runs = sc2ts.run_hmm(
        m.ds_path,
        m.ts_path,
        strains=[m.sample_id],
        num_mismatches=m.num_mismatches,
        mismatch_threshold=1000,
        # direction=direction,
        deletions_as_missing=True,
        num_threads=0,
        show_progress=False,
    )
    return runs[0]


@click.command()
@click.argument("ts_prefix")
@click.argument("ds_path", type=click.Path(dir_okay=False, file_okay=True))
@click.argument("samples_csv_path", type=click.Path(dir_okay=False, file_okay=True))
@click.argument("output_path")
@click.option("--num-mismatches", "-k", type=int, multiple=True)
@click.option("--mismatch-threshold", "-m", type=int, default=100)
@click.option("--workers", "-w", type=int, default=1)
def run(
    ts_prefix,
    ds_path,
    samples_csv_path,
    output_path,
    num_mismatches,
    mismatch_threshold,
    workers,
):

    ds = sc2ts.Dataset(ds_path, date_field="Date_tree")
    recomb_df = pd.read_csv(samples_csv_path).set_index("sample_id")

    work = []
    for s in recomb_df.index:
        try:
            date = datetime.datetime.fromisoformat(ds.metadata[s]["Date_tree"])
        except KeyError:
            print(f"missing {s}")
            continue
        match_date = str(date - datetime.timedelta(days=1)).split()[0]
        ts_path = f"{ts_prefix}{match_date}.ts"
        if not os.path.exists(ts_path):
            raise ValueError(f"Missing path: {ts_path}")
        for k in num_mismatches:
            work.append(MatchWork(ts_path, ds.path, s, k))

    # Clear the file
    with open(output_path, "w") as f:
        pass

    with cf.ProcessPoolExecutor(workers) as executor:
        futures = [executor.submit(run_match, w) for w in work]
        for future in tqdm.tqdm(cf.as_completed(futures), total=len(futures)):
            run = future.result()
            with open(output_path, "a") as f:
                print(run.asjson(), file=f)

    # for w in tqdm.tqdm(work):
    #     run = run_match(w)
    #     with open(output_path, "a") as f:
    #         print(run.asjson(), file=f)


run()
