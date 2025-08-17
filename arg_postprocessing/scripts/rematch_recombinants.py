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
    result["recombinant"] = int(work.node)
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

    # First, make sure we've pushed up all unary recombinant nodes so we
    # get accurate information on easy/hard
    ts = sc2ts.push_up_unary_recombinant_mutations(ts)

    recomb_nodes = np.where(ts.nodes_flags & sc2ts.NODE_IS_RECOMBINANT > 0)[0]
    node_mutations = np.bincount(ts.mutations_node)

    all_work = collections.defaultdict(list)

    for u in recomb_nodes:
        md = ts.node(u).metadata
        date = md["sc2ts"]["date_added"]
        muts = node_mutations[u]
        all_work[muts].append(Work(pattern, u, date))

    random.seed(42)
    for k, v in all_work.items():
        # Randomise so we're not doing all the later onces (which need more RAM)
        # at the same time
        random.shuffle(v)
        print(f"muts: {k} has {len(v)}")

    cores_for_muts = {0: 20, 1: 10, 2: 5, 3: 2}
    json_data = []
    for muts, cores in cores_for_muts.items():
        easy_work = all_work[muts]
        print(f"Running {len(easy_work)} for {muts} mutations")
        with cf.ProcessPoolExecutor(cores) as executor:
            futures = [executor.submit(worker, work) for work in easy_work]
            for future in tqdm.tqdm(cf.as_completed(futures), total=len(easy_work)):
                result = future.result()
                json_data.append(result)
                dump(json_data, output)

    hard_work = []
    for k, v in all_work.items():
        if k > max(cores_for_muts.keys()):
            hard_work.extend(v)

    print(f"Running remaining {len(hard_work)} one by one")
    for work in tqdm.tqdm(hard_work):
        result = worker(work)
        json_data.append(result)
        dump(json_data, output)


if __name__ == "__main__":
    run()
