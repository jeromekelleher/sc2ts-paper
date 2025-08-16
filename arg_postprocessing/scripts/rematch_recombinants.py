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
@click.option("--cores", type=int)
def run(ts, pattern, output, cores):
    ts = tszip.load(ts)

    # First, make sure we've pushed up all unary recombinant nodes so we
    # get accurate information on easy/hard
    ts = sc2ts.push_up_unary_recombinant_mutations(ts)

    recomb_nodes = np.where(ts.nodes_flags & sc2ts.NODE_IS_RECOMBINANT > 0)[0]
    node_mutations = np.bincount(ts.mutations_node)

    easy_work = []
    hard_work = []

    for u in recomb_nodes:
        md = ts.node(u).metadata
        date = md["sc2ts"]["date_added"]
        muts = node_mutations[u]
        if muts <= 2:
            easy_work.append(Work(pattern, u, date))
        else:
            hard_work.append(Work(pattern, u, date))

    json_data = []
    with cf.ProcessPoolExecutor(cores) as executor:
        futures = [executor.submit(worker, work) for work in easy_work]
        for future in tqdm.tqdm(cf.as_completed(futures), total=len(easy_work)):
            result = future.result()
            json_data.append(result)
            dump(json_data, output)

    for work in tqdm.tqdm(hard_work):
        result = worker(work)
        json_data.append(result)
        dump(json_data, output)


if __name__ == "__main__":
    run()
