import datetime

import click
import humanize
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
    return pd.read_csv(
        "arg_postprocessing/sc2ts_v1_2023-02-21_resources.csv"
    ).set_index("date")


def _wide_plot(*args, height=4, **kwargs):
    return plt.subplots(*args, figsize=(16, height), **kwargs)


def savefig(name):
    plt.savefig(f"figures/{name}.png")
    plt.savefig(f"figures/{name}.pdf")


@click.command()
def inference_resources():
    fig, ax = _wide_plot(3, height=8, sharex=True)

    start_date = "2020-05-01"
    end_date = "3000-01-01"

    dfs = samples_csv().set_index("date")

    dfa = dfs.groupby("date").sum()
    dfa["mean_hmm_cost"] = dfa["total_hmm_cost"] / dfa["total"]
    df = dfa.join(resources_csv(), how="inner")
    df = df.rename(columns={"inserted": "smaples_in_arg", "total": "samples_processed"})
    df = df[(df.index >= start_date) & (df.index < end_date)]

    df["cpu_time"] = df.user_time + df.sys_time
    x = np.array(df.index, dtype="datetime64[D]")

    total_elapsed = datetime.timedelta(seconds=np.sum(df.elapsed_time))
    total_cpu = datetime.timedelta(seconds=np.sum(df.cpu_time))
    title = (
        f"{humanize.naturaldelta(total_elapsed)} elapsed "
        f"using {humanize.naturaldelta(total_cpu)} of CPU time "
        f"(utilisation = {np.sum(df.cpu_time) / np.sum(df.elapsed_time):.2f})"
    )

    ax[0].set_title(title)
    ax[0].plot(x, df.elapsed_time / 60, label="elapsed time")
    ax[-1].set_xlabel("Date")
    ax_twin = ax[0].twinx()
    ax_twin.plot(x, df.samples_processed, color="tab:red", alpha=0.5, label="samples")
    ax_twin.legend(loc="upper left")
    ax_twin.set_ylabel("Samples processed")
    ax[0].set_ylabel("Elapsed time (mins)")
    ax[0].legend()
    ax_twin.legend()
    ax[1].plot(x, df.elapsed_time / df.samples_processed, label="Mean time per sample")
    ax[1].set_ylabel("Elapsed time per sample (s)")
    ax[1].legend(loc="upper right")

    ax_twin = ax[1].twinx()
    ax_twin.plot(x, df.mean_hmm_cost, color="tab:orange", alpha=0.5, label="HMM cost")
    ax_twin.set_ylabel("HMM cost")
    ax_twin.legend(loc="upper left")
    ax[2].plot(x, df.max_memory / 1024**3)
    ax[2].set_ylabel("Max memory (GiB)")

    for a in ax:
        a.grid()
    savefig("inference-resources")


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
cli.add_command(inference_resources)
cli()
