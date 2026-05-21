import click
import numpy as np
import pandas as pd
import tskit


def parse_detection_results(file):
    ts = tskit.load(file)
    df = []
    if len(ts.samples()) == 0:
        raise ValueError("No sample nodes found in tree sequence.")
    for u in ts.samples():
        node = ts.node(id_=u)
        strain = node.metadata["strain"]
        num_edges = np.sum(ts.edges_child == u)
        is_recombinant = True if num_edges > 1 else False
        df.append(
            {
                "strain": strain,
                "is_recombinant": is_recombinant,
                "node": u,
            }
        )
    return pd.DataFrame(df)


def evaluate_performance(df_truth, df_detected):
    num_true_recombs = np.sum(df_truth["is_recombinant"])
    num_inferred_recombs = np.sum(df_detected["is_recombinant"])
    # FIXME: These counts ignore rejected samples.
    df_merged = df_truth.merge(df_detected, on="strain")
    num_inserted = len(df_merged)
    num_rejected = len(df_truth) - num_inserted
    tp = np.sum(np.logical_and(df_merged["is_recombinant_x"], df_merged["is_recombinant_y"]))
    tn = np.sum(np.logical_and(~df_merged["is_recombinant_x"], ~df_merged["is_recombinant_y"]))
    fp = np.sum(np.logical_and(~df_merged["is_recombinant_x"], df_merged["is_recombinant_y"]))
    fn = np.sum(np.logical_and(df_merged["is_recombinant_x"], ~df_merged["is_recombinant_y"]))
    return {
        "num_true_recombs": int(num_true_recombs),
        "num_inferred_recombs": int(num_inferred_recombs),
        "num_inserted": int(num_inserted),
        "num_rejected": int(num_rejected),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


@click.command()
@click.argument("ts_file")
@click.argument("truth_file")
def run(ts_file, truth_file):
    df_detected = parse_detection_results(ts_file)
    df_truth = pd.read_csv(truth_file, sep="\t")
    assert len(df_truth) >= len(df_detected)
    results = evaluate_performance(df_truth, df_detected)
    print(results)


if __name__ == "__main__":
    run()
