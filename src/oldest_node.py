"""
Find the oldest node for each clade in a Nextstrain auspice JSON tree, and
write its date and date confidence interval to a TSV file.

data/oldest_clade_nodes.tsv was generated from the global all-time GISAID
build, downloaded on 2026-10-22 (build updated 2025-10-01) from
https://data.nextstrain.org/ncov_gisaid_global_all-time.json

Usage: python oldest_node.py ncov_gisaid_global_all-time.json oldest_clade_nodes.tsv
"""

import csv
import json
import sys


def find_oldest_clades(tree):
    oldest = {}
    stack = [tree]
    while stack:
        node = stack.pop()
        stack.extend(node.get("children", []))
        attrs = node.get("node_attrs", {})
        clade = attrs.get("clade_membership", {}).get("value")
        num_date = attrs.get("num_date")
        if clade is None or num_date is None:
            continue
        date = num_date["value"]
        if clade not in oldest or date < oldest[clade]["date"]:
            lo, hi = num_date.get("confidence", [date, date])
            oldest[clade] = {"node_id": node["name"], "date": date, "ci": (lo, hi)}
    return oldest


if __name__ == "__main__":
    with open(sys.argv[1]) as f:
        tree = json.load(f)["tree"]
    oldest = find_oldest_clades(tree)
    with open(sys.argv[2], "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["clade_membership", "oldest_node_id", "date", "date_CI"])
        for clade in sorted(oldest):
            row = oldest[clade]
            lo, hi = row["ci"]
            writer.writerow([clade, row["node_id"], row["date"], f"{{{lo},{hi}}}"])
