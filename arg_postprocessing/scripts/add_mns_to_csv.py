from dataclasses import dataclass

import click
import numpy as np
import pandas as pd
import tszip
import tskit


# Table 1 from https://academic.oup.com/mbe/article/42/11/msaf272/8301027
MNS = [
  {
    "id": "MNM1",
    "is_artefact": False,
    "mutations": [
      { "reference": "C", "position": 21302, "derived": "T" },
      { "reference": "C", "position": 21304, "derived": "A" },
      { "reference": "G", "position": 21305, "derived": "A" }
    ]
  },
  {
    "id": "MNM1a",
    "is_artefact": False,
    "mutations": [
      { "reference": "C", "position": 21304, "derived": "A" },
      { "reference": "G", "position": 21305, "derived": "A" }
    ]
  },
  {
    "id": "MNM2",
    "is_artefact": False,
    "mutations": [
      { "reference": "A", "position": 28877, "derived": "T" },
      { "reference": "G", "position": 28878, "derived": "C" }
    ]
  },
  {
    "id": "MNM3",
    "is_artefact": False,
    "mutations": [
      { "reference": "G", "position": 27382, "derived": "C" },
      { "reference": "A", "position": 27383, "derived": "T" },
      { "reference": "T", "position": 27384, "derived": "C" }
    ]
  },
  {
    "id": "MNM4",
    "is_artefact": False,
    "mutations": [
      { "reference": "T", "position": 26491, "derived": "C" },
      { "reference": "A", "position": 26492, "derived": "T" },
      { "reference": "T", "position": 26497, "derived": "C" }
    ]
  },
  {
    "id": "MNM5",
    "is_artefact": False,
    "mutations": [
      { "reference": "G", "position": 27758, "derived": "A" },
      { "reference": "T", "position": 27760, "derived": "A" }
    ]
  },
  {
    "id": "MNM6",
    "is_artefact": False,
    "mutations": [
      { "reference": "C", "position": 25162, "derived": "A" },
      { "reference": "C", "position": 25163, "derived": "A" }
    ]
  },
  {
    "id": "MNM7",
    "is_artefact": False,
    "mutations": [
      { "reference": "T", "position": 27875, "derived": "C" },
      { "reference": "C", "position": 27881, "derived": "T" },
      { "reference": "G", "position": 27882, "derived": "C" },
      { "reference": "C", "position": 27883, "derived": "T" }
    ]
  },
  {
    "id": "MNM7a",
    "is_artefact": False,
    "mutations": [
      { "reference": "C", "position": 27881, "derived": "T" },
      { "reference": "G", "position": 27882, "derived": "C" },
      { "reference": "C", "position": 27883, "derived": "T" }
    ]
  },
  {
    "id": "MNM8",
    "is_artefact": False,
    "mutations": [
      { "reference": "T", "position": 21294, "derived": "A" },
      { "reference": "G", "position": 21295, "derived": "A" },
      { "reference": "G", "position": 21296, "derived": "A" }
    ]
  },
  {
    "id": "mnS1",
    "is_artefact": True,  # Sequencing error due to incomplete read trimming
    "mutations": [
      { "reference": "A", "position": 507, "derived": "T" },
      { "reference": "T", "position": 508, "derived": "C" },
      { "reference": "G", "position": 509, "derived": "A" }
    ]
  },
  {
    "id": "MNM9",
    "is_artefact": False,
    "mutations": [
      { "reference": "A", "position": 27038, "derived": "T" },
      { "reference": "T", "position": 27039, "derived": "A" },
      { "reference": "C", "position": 27040, "derived": "A" }
    ]
  },
  {
    "id": "MNM10",
    "is_artefact": False,
    "mutations": [
      { "reference": "A", "position": 21550, "derived": "C" },
      { "reference": "A", "position": 21551, "derived": "T" }
    ]
  },
  {
    "id": "mnS2",
    "is_artefact": True,  # Sequencing error due to deletion
    "mutations": [
      { "reference": "T", "position": 21994, "derived": "C" },
      { "reference": "T", "position": 21995, "derived": "C" }
    ]
  },
  {
    "id": "MNM11",
    "is_artefact": False,
    "mutations": [
      { "reference": "C", "position": 13423, "derived": "A" },
      { "reference": "C", "position": 13424, "derived": "A" }
    ]
  },
  {
    "id": "MNM12",
    "is_artefact": False,
    "mutations": [
      { "reference": "A", "position": 4576, "derived": "T" },
      { "reference": "T", "position": 4579, "derived": "A" }
    ]
  },
  {
    "id": "mnS3",
    "is_artefact": True,  # Sequencing error due to reference bias
    "mutations": [
      { "reference": "T", "position": 28881, "derived": "A" },
      { "reference": "G", "position": 28882, "derived": "A" },
      { "reference": "G", "position": 28883, "derived": "C" }
    ]
  },
  {
    "id": "MNM13",
    "is_artefact": False,
    "mutations": [
      { "reference": "A", "position": 20284, "derived": "T" },
      { "reference": "T", "position": 20285, "derived": "C" }
    ]
  },
  {
    "id": "mnS4",
    "is_artefact": False,
    "mutations": [
      { "reference": "G", "position": 11083, "derived": "T" },
      { "reference": "C", "position": 21575, "derived": "T" }
    ]
  }
]


# NOTE: This might need to be updated.
EXCLUDED_SITES = {16290, 20056, 22802}


ALLELES = ("A", "C", "G", "T", "-")
NODE_COLUMNS = ["sample", "parent_left", "parent_right"]
# Adding these data to recombinants.csv
ANNOTATION_COLUMNS = ["num_mns_left", "mns_left", "num_mns_right", "mns_right"]


@dataclass(frozen=True)
class MnsGroup:
    name: str
    site_ids: np.ndarray
    genotype_rows: np.ndarray
    derived: np.ndarray


def prepare_mns_groups(site_map):
    """Index complete MNS groups into a shared set of genotype rows."""
    available = [
        group for group in MNS
        if all(mut["position"] in site_map for mut in group["mutations"])
    ]

    site_ids = sorted({
        site_map[mut["position"]]
        for group in available
        for mut in group["mutations"]
    })

    site_rows = {site: row for row, site in enumerate(site_ids)}

    groups = []
    for group in available:
        sites = np.array([
            site_map[mut["position"]] for mut in group["mutations"]
        ], dtype=int)

        groups.append(
            MnsGroup(
                name=group["id"],
                site_ids=sites,
                genotype_rows=np.array([site_rows[site] for site in sites], dtype=int),
                derived=np.array([
                    ALLELES.index(mut["derived"]) for mut in group["mutations"]
                ], dtype=int),
            )
        )

    return groups, site_ids


def cache_genotypes(ts, node_ids, site_ids):
    """Decode MNS sites once and map original node IDs to genotype columns."""
    simplified, node_map = ts.simplify(
        node_ids, filter_sites=False, map_nodes=True
    )

    samples = simplified.samples()
    sample_columns = {node: column for column, node in enumerate(samples)}
    node_columns = {node: sample_columns[node_map[node]] for node in node_ids}
    genotypes = np.empty((len(site_ids), len(samples)), dtype=np.int8)
    variant = tskit.Variant(simplified, samples=samples, alleles=ALLELES)

    for row, site_id in enumerate(site_ids):
        variant.decode(site_id)
        genotypes[row] = variant.genotypes

    return genotypes, node_columns


def find_supporting_mns(trio_genotypes, groups, left_site, right_site):
    left_matches = []
    right_matches = []

    for group in groups:
        recombinant, left_parent, right_parent = trio_genotypes[group.genotype_rows].T
        matches_derived = recombinant == group.derived
        matches_left = recombinant == left_parent
        matches_right = recombinant == right_parent

        if np.all(
            (group.site_ids <= left_site)
            & matches_derived & matches_left & ~matches_right
        ):
            left_matches.append(group.name)

        if np.all(
            (group.site_ids >= right_site)
            & matches_derived & matches_right & ~matches_left
        ):
            right_matches.append(group.name)

    return left_matches, right_matches


def annotate_recombinants(recomb_df, ts):
    annotations = []

    if not recomb_df.empty:
        site_map = {int(position): site for site, position in enumerate(ts.sites_position)}
        groups, site_ids = prepare_mns_groups(site_map)
        node_ids = np.unique(recomb_df[NODE_COLUMNS].to_numpy())
        genotypes, node_columns = cache_genotypes(ts, node_ids, site_ids)

        for row in recomb_df.itertuples(index=False):
            columns = [
                node_columns[node]
                for node in (row.sample, row.parent_left, row.parent_right)
            ]

            # Excluded left endpoints use the adjacent site to the left.
            interval_left = row.interval_left
            if interval_left in EXCLUDED_SITES:
                interval_left -= 1

            left, right = find_supporting_mns(
                genotypes[:, columns], groups,
                site_map[interval_left], site_map[row.interval_right],
            )

            annotations.append((len(left), ",".join(left), len(right), ",".join(right)))

    result = recomb_df.copy()
    annot_df = pd.DataFrame(annotations, columns=ANNOTATION_COLUMNS)

    for column in ANNOTATION_COLUMNS:
        dtype = int if column.startswith("num_") else str
        result[column] = annot_df[column].to_numpy(dtype=dtype)

    return result


def add_mns_to_csv(recomb_file, ts_file, mns_file):
    recomb_df = pd.read_csv(recomb_file)
    ts = tszip.load(ts_file)
    annotate_recombinants(recomb_df, ts).to_csv(mns_file, index=False)


@click.command()
@click.argument(
    "recomb_file",
    type=click.Path(exists=True, dir_okay=False)
)
@click.argument(
    "ts_file",
    type=click.Path(exists=True, dir_okay=False)
)
@click.argument(
    "mns_file",
    type=click.Path(dir_okay=False)
)
def main(recomb_file, ts_file, mns_file):
    add_mns_to_csv(recomb_file, ts_file, mns_file)


if __name__ == "__main__":
    main()
