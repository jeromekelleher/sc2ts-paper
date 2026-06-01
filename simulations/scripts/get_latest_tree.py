"""
Print the path to the most recent dated sc2ts tree for a run.
"""

import datetime
from pathlib import Path
import click


def tree_date(path: Path, run_id: str) -> datetime.date:
    prefix = f"{run_id}_"
    if not path.name.startswith(prefix) or not path.name.endswith(".ts"):
        raise ValueError(f"Unexpected ts file: {path.name}")
    return datetime.date.fromisoformat(path.name[len(prefix) : -3])


def find_last_tree(tree_dir: Path, run_id: str) -> Path:
    candidates = [
        p
        for p in tree_dir.glob(f"{run_id}_*.ts")
        if "_init" not in p.name and not p.name.endswith(".pp.ts")
    ]
    if not candidates:
        raise FileNotFoundError(f"No dated trees found in {tree_dir} for run {run_id}")
    return max(candidates, key=lambda p: tree_date(p, run_id))


@click.command()
@click.argument(
    "tree_dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
@click.argument("run_id")
def main(tree_dir, run_id):
    print(find_last_tree(tree_dir, run_id))


if __name__ == "__main__":
    main()
