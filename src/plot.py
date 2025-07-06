import click
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tszip

import sc2ts


def main_arg():
    return tszip.load(
        "data/sc2ts_v1_2023-02-21_pp_dated_remapped_bps_pango_mmps.trees.tsz"
    )


def samples_csv():
    return pd.read_csv("arg_postprocessing/sc2ts_v1_2023-02-21_samples.csv")


def resources_csv():
    return pd.read_csv("arg_postprocessing/sc2ts_v1_2023-02-21_resources.csv")


def _wide_plot(*args, height=4, **kwargs):
    return plt.subplots(*args, figsize=(16, height), **kwargs)


def savefig(name):
    plt.savefig(f"figures/{name}.png")
    plt.savefig(f"figures/{name}.pdf")


@click.command()
def samples_per_day():

    start_date = "2020-05-01"
    end_date = "3000-01-01"
    scorpio_fraction = 0.05

    df = samples_csv()
    df = df[(df.date >= start_date) & (df.date < end_date)]

    dfa = df.groupby("date").sum().reset_index().astype({"date": "datetime64[s]"})
    dfa["mean_hmm_cost"] = dfa["total_hmm_cost"] / dfa["total"]

    fig, (ax1, ax2, ax3, ax4) = _wide_plot(4, height=12, sharex=True)
    exact_col = "tab:red"
    in_col = "tab:purple"
    ax1.plot(dfa.date, dfa.inserted, label="In ARG", color=in_col)
    ax1.plot(dfa.date, dfa.total, label="Processed")
    ax1.plot(dfa.date, dfa.exact_matches, label="Exact matches", color=exact_col)

    ax2.plot(
        dfa.date,
        dfa.inserted / dfa.total,
        label="Fraction processed in ARG",
        color=in_col,
    )
    ax2.plot(
        dfa.date,
        dfa.exact_matches / dfa.total,
        label="Fraction processed exact matches",
        color=exact_col,
    )

    ax3.plot(dfa.date, dfa.rejected / dfa.total, label="Fraction excluded")
    ax3_2 = ax3.twinx()
    ax3_2.plot(dfa.date, dfa.mean_hmm_cost, label="mean HMM cost", color="tab:orange")
    ax2.set_ylabel("Fraction of samples")
    ax3.set_ylabel("Fraction of samples")
    ax4.set_xlabel("Date")
    ax3_2.set_ylabel("Mean HMM cost")
    ax1.set_ylabel("Number of samples")
    ax1.legend()
    ax2.legend()
    ax3.legend(loc="upper right")
    ax3_2.legend(loc="upper left")

    for ax in [ax1, ax2, ax3]:
        ax.grid()

    df_scorpio = df.pivot_table(
        columns="scorpio", index="date", values="total", aggfunc="sum", fill_value=0
    ).reset_index()
    # Need force conversion back to datetime here for some reason
    df_scorpio = df_scorpio.astype({"date": "datetime64[s]"}).set_index("date")
    # convert to fractions
    df_scorpio = df_scorpio.divide(df_scorpio.sum(axis="columns"), axis="index")
    # Remove columns that don't have more than the threshold
    keep_cols = []
    for col in df_scorpio:
        if np.any(df_scorpio[col] >= scorpio_fraction):
            keep_cols.append(col)

    df_scorpio = df_scorpio[keep_cols]
    ax4.set_title("Scorpio composition of processed samples")
    ax4.stackplot(
        df_scorpio.index,
        *[df_scorpio[s] for s in df_scorpio],
        labels=[" ".join(s.split("_")) for s in df_scorpio],
    )
    ax4.legend(loc="upper left", ncol=2)

    savefig("samples-per-day")


@click.group()
def cli():
    pass


cli.add_command(samples_per_day)
cli()
