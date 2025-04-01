import datetime
import dataclasses
import concurrent.futures as cf

import sc2ts
import pandas as pd


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


def run():

    ds = sc2ts.Dataset(
        "data/viridian_mafft_2024-10-14_v1.vcz.zip", date_field="Date_tree"
    )
    # recomb_df = pd.read_csv("data/recombinants.csv").set_index("sample_id")
    recomb_df = pd.read_csv("data/samples_pangos_absent_in_arg.csv").set_index("Run")

    ts_prefix = "results/v1-beta1/v1-beta1_"
    work = []
    for s in recomb_df.index:
        try:
            date = datetime.datetime.fromisoformat(ds.metadata[s]["Date_tree"])
        except KeyError:
            print(f"missing {s}")
            continue
        match_date = str(date - datetime.timedelta(days=1)).split()[0]
        ts_path = f"{ts_prefix}{match_date}.ts"
        for num_mismatches in [4, 1000]:
            work.append(MatchWork(ts_path, ds.path, s, num_mismatches))

    # output_path = "results/recombinant_reruns.json"
    output_path = "results/pango_x_not_in_arg.json"
    # Clear the file
    with open(output_path, "w") as f:
        pass
    # for w in work:
    #     # print(work)
    #     print(w)
    #     run = run_match(w)
    #     print(run)
    #     print(run.asjson())

    with cf.ProcessPoolExecutor(2) as executor:
        futures = [executor.submit(run_match, w) for w in work]
        for future in cf.as_completed(futures):
            run = future.result()
            print(run)
            with open(output_path, "a") as f:
                print(run.asjson(), file=f)


if __name__ == "__main__":
    run()
