import collections
import random
import concurrent.futures as cf
import pathlib
import tszip
import click
import sc2ts
import numpy as np
import dataclasses
import tqdm
import subprocess
import json


@dataclasses.dataclass
class Work:
    pattern: str
    node: int
    date: str


def worker(work):
    cmd = (
        f"python -m sc2ts rematch-recombinant {work.node} "
        f"--path-pattern={work.pattern} --date={work.date} -vv"
    )
    out = subprocess.check_output(cmd, shell=True)
    result = json.loads(out.decode())
    return result


def dump(json_data, path):
    with open(path, "w") as f:
        json.dump(json_data, f, indent=4)


@click.command()
@click.argument("ts")
@click.argument("pattern")
@click.argument("output")
def run(ts, pattern, output):
    ts = tszip.load(ts)

    recombinants = np.where(ts.nodes_flags & sc2ts.NODE_IS_RECOMBINANT > 0)[0]

    all_work = []
    for u in recombinants:
        md = ts.node(u).metadata
        date = md["sc2ts"]["date_added"]
        all_work.append(Work(pattern, u, date))

    # The Delta wave was particularly difficult in terms of memory usage,
    # so limiting parallelism based on date. Post-Omicron wasn't quite as bad
    bins = ["2020-01", "2021-09", "2022-01", "2030-01"]
    cores_per_bin = [20, 4, 8]
    assert len(cores_per_bin) == len(bins) - 1

    random.seed(42)

    split_work = {}
    for j in range(len(bins) - 1):
        start = bins[j]
        stop = bins[j + 1]
        cores = cores_per_bin[j]
        work = [w for w in all_work if start <= w.date < stop]
        # Randomise so we're not doing all the later onces (which need more RAM)
        # at the same time
        random.shuffle(work)
        split_work[cores] = work

    json_data = []
    for cores in reversed(cores_per_bin):
        work = split_work[cores]
        print(f"Running {len(work)} on {cores}")
        with cf.ProcessPoolExecutor(cores) as executor:
            futures = [executor.submit(worker, w) for w in work]
            for future in tqdm.tqdm(cf.as_completed(futures), total=len(work)):
                result = future.result()
                json_data.append(result)
                dump(json_data, output)


if __name__ == "__main__":
    run()
