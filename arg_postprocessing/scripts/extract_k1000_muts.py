import click
import json


@click.command()
@click.argument("rematches_json")
@click.argument("output")
def run(rematches_json, output):
    k1000_muts = {}
    with open(rematches_json) as f:
        for r in json.load(f):
            nrm = r["no_recomb_match"]
            assert len(nrm["path"]) == 1
            k1000_muts[r["recombinant"]] = len(nrm["mutations"])

    # Sort by node ID so that the file is stable under git
    k1000_muts = dict(sorted(k1000_muts.items()))
    with open(output, "w") as f:
        json.dump(k1000_muts, f, indent=4)


if __name__ == "__main__":
    run()
