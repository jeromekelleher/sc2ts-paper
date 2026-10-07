import random
import click
import pyfaidx


def sample_sequences(in_file, out_file, p, seed=None):
    if not 0 <= p <= 1:
        raise ValueError("Probability must be in [0, 1].")

    rng = random.Random(seed)
    selected = 0
    with pyfaidx.Fasta(str(in_file), read_long_names=True) as seqs:
        total = len(seqs.keys())
        with open(out_file, "x") as f:
            for seq in seqs:
                if rng.random() < p:
                    f.write(f">{seq.name}\n{seq[:]}\n")
                    selected += 1

    return selected, total


@click.command()
@click.option(
    "-i",
    "in_file",
    type=click.Path(file_okay=True, dir_okay=False),
    required=True,
    help="Input FastA file",
)
@click.option(
    "-o",
    "out_file",
    type=click.Path(file_okay=True, dir_okay=False),
    required=True,
    help="New output FastA file (must not already exist)",
)
@click.option(
    "-p",
    "prob",
    type=float,
    required=True,
    help="Inclusion probability in [0, 1]",
)
@click.option(
    "--seed",
    type=int,
    help="Random seed",
)
def main(in_file, out_file, prob, seed):
    if not 0 <= prob <= 1:
        raise click.BadParameter("Probability must be in [0, 1].")

    selected, total = sample_sequences(in_file, out_file, prob, seed)

    click.echo(f"Selected {selected} of {total} sequences.")


if __name__ == "__main__":
    main()
