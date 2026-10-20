import click
import json


@click.command()
@click.argument("rematches_json")
@click.argument("output")
def run(rematches_json, output):
    """
    Save the number of mutations in the original (k=4) match and the
    no-recombination (k=1000) match for each recombinant, including
    multi-parent recombinants.
    """
    muts = {}
    with open(rematches_json) as f:
        for r in json.load(f):
            nrm = r["no_recomb_match"]
            assert len(nrm["path"]) == 1
            muts[r["recombinant"]] = {
                "k4_muts": len(r["original_match"]["mutations"]),
                "k1000_muts": len(nrm["mutations"]),
            }

    # Sort by node ID so that the file is stable under git
    muts = dict(sorted(muts.items()))
    with open(output, "w") as f:
        json.dump(muts, f, indent=4)


if __name__ == "__main__":
    run()
