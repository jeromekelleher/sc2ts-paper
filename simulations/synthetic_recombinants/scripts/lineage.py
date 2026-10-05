"""
Alpha/Delta labelling of Pango lineages, shared by the pool and summary steps.
"""

ALPHA_ROOTS = ("B.1.1.7", "Q")
DELTA_ROOTS = ("B.1.617.2", "AY")


def classify(pango):
    for lineage, roots in [("Alpha", ALPHA_ROOTS), ("Delta", DELTA_ROOTS)]:
        for root in roots:
            if pango == root or pango.startswith(root + "."):
                return lineage
    return "Other"
