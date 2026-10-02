#!/usr/bin/env python3
r"""
Generate the LaTeX source for the Pango X lineage events table
(Table~\ref{tab:pango_x_lineages} in main.tex).

The four CSV files in this directory are the interface to the analysis: they
are written by notebooks/tab_pango_x_events.ipynb, one per event type. This
script turns them into a tabularray ``longtabs`` table, which breaks over
pages, written to table.tex and pulled into main.tex with

    \input{pango_x_table/table.tex}

Run it from anywhere; paths are resolved relative to this file:

    python3 pango_x_table/make_table.py

Only the standard library is used, so it runs under any python3 without the
project environment. Row order is taken from the CSV files, so all sorting
decisions stay in the notebook.
"""

import argparse
import ast
import csv
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).parent
NUM_COLS = 7

# Parent lineages arrive from the notebook already wrapped in \textbf{} when
# the parent node is exactly matched by a sampled sequence, so values are
# passed through to LaTeX unescaped. Rather than escape (and mangle that
# markup) we check that nothing else needing escaping has crept in.
BOLD_RE = re.compile(r"^\\textbf\{([^{}\\]*)\}$")
LATEX_SPECIALS = set("#$%&_^~\\{}")

CAPTION = r"""
Events in the ARG associated with Pango X lineages.
Each row corresponds to the origin node for a given Pango lineage (Types I,
III and IV, see text) or associated recombination event (Type II). For origin
nodes that are recombinant (Type I) or near a recombination event (Type II),
the ``averted'' column shows the number of additional mutations required
without recombination, as well as the position of informative sites to the
left and right of the breakpoint. These are not available (NA) for
recombination nodes with more than two parents. Parent lineages shown in
boldface are nodes exactly matched by sampled sequences. The descendants column shows the number
of descendants by Pango lineage. Type II events are named for the Pango X
lineage (or lineages) associated with them. For Pango lineages (Type III)
whose origin node was a descendant of a Type I or II recombination node, the
distance (path length and time in days) from the recombination event is shown,
as well as the number of mutations directly ancestral to the Pango origin
node; lineages with more than one such origin node have one row per origin.
Pango X lineages not associated with a recombination node (Type IV) are shown
with the parent node and number of mutations.
"""

SECTION_TITLES = {
    1: "Type I event: Recombination coinciding with Pango origin node",
    2: "Type II event: Recombination closely associated with Pango lineage(s)",
    3: "Type III events: Pango origin nodes derived from Type I and II events",
    4: "Type IV events: Non recombinants in the ARG",
}


def read_csv(event_type):
    """
    Return (rows, fieldnames) for the CSV holding the given event type.
    """
    path = HERE / f"type{event_type}.csv"
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        return list(reader), list(reader.fieldnames)


def check_safe(value, field):
    r"""
    Pass a CSV value through, raising if it contains LaTeX specials outside a
    recognised \textbf{} wrapper.
    """
    if value == "":
        # Recombinants with more than two parents have no parent lineages
        return "NA"
    match = BOLD_RE.match(value)
    text = match.group(1) if match else value
    bad = sorted(LATEX_SPECIALS & set(text))
    if bad:
        raise ValueError(
            f"{field}={value!r} contains LaTeX special character(s) "
            f"{''.join(bad)!r} that make_table.py does not escape"
        )
    return value


def fmt_int(value):
    """
    Render a numeric CSV field as a whole number. Several columns are floats
    in the CSV (mutations_averted is 67.0, times are full precision); "%.0f"
    matches the float_format used by the notebook's to_latex output.
    """
    if value == "":
        return "NA"
    return f"{float(value):.0f}"


def fmt_descendants(value):
    """
    Render the repr of a {pango: count} dict as, e.g., XCR:84, XBB.1.5.70:1.
    """
    counts = ast.literal_eval(value)
    return ", ".join(f"{check_safe(k, 'descendants')}:{v}" for k, v in counts.items())


def make_row(cells, prefix=""):
    """
    Join cells into a table row, padding out to the full column count. Table
    commands (rules, page break hints) are prefixed to the first cell, which is
    where tabularray looks for them.
    """
    cells = [str(c) for c in cells]
    if len(cells) > NUM_COLS:
        raise ValueError(f"{len(cells)} cells in a {NUM_COLS} column table")
    cells += [""] * (NUM_COLS - len(cells))
    if prefix:
        cells[0] = f"{prefix} {cells[0]}".strip()
    return " & ".join(cells) + r" \\"


def make_banner(title, prefix=""):
    """
    A full-width section heading. tabularray spans take no separators for the
    columns they cover, so this row carries no & at all.
    """
    cell = rf"\SetCell[c={NUM_COLS}]{{l}} \textbf{{{title}}}"
    if prefix:
        cell = f"{prefix} {cell}"
    return cell + r" \\"


def group_header():
    """
    The spanned sub-heading shared by the Type I and Type II sections. The
    columns covered by each span still need their & placeholders.
    """
    return make_row(
        [
            "",
            "",
            r"\SetCell[c=2]{c} Parent lineage",
            "",
            r"\SetCell[c=2]{c} Interval",
            "",
            "",
        ],
        prefix=r"\nopagebreak",
    )


def section_header(event_type, page_break):
    """
    The rule and banner introducing a section. A forced page break replaces the
    midrule: the new page already opens with the repeated toprule.
    """
    if page_break:
        return make_banner(SECTION_TITLES[event_type], prefix=r"\pagebreak")
    rule = r"\toprule" if event_type == 1 else r"\midrule"
    return make_banner(SECTION_TITLES[event_type], prefix=rule)


def type1_section(page_break=False):
    rows, _ = read_csv(1)
    lines = [
        section_header(1, page_break),
        group_header(),
        make_row(
            ["pango", "averted", "left", "right", "left", "right", "descendants"],
            prefix=r"\nopagebreak",
        ),
    ]
    for i, row in enumerate(rows):
        lines.append(
            make_row(
                [
                    check_safe(row["root_pango"], "root_pango"),
                    fmt_int(row["mutations_averted"]),
                    check_safe(row["parent_left_pango"], "parent_left_pango"),
                    check_safe(row["parent_right_pango"], "parent_right_pango"),
                    fmt_int(row["interval_left"]),
                    fmt_int(row["interval_right"]),
                    fmt_descendants(row["descendants"]),
                ],
                prefix=r"\midrule \nopagebreak" if i == 0 else "",
            )
        )
    return lines


def type2_section(page_break=False):
    rows, fieldnames = read_csv(2)
    # The notebook used to emit only a total; prefer the per-lineage breakdown
    # when it is there, and say so when it is not.
    if "descendants" in fieldnames:
        last_name = "descendants"

        def last_cell(row):
            return fmt_descendants(row["descendants"])

    else:
        print(
            "warning: type2.csv has no 'descendants' column, falling back to "
            "num_descendant_samples totals; rerun tab_pango_x_events.ipynb to "
            "get the per-lineage breakdown",
            file=sys.stderr,
        )
        last_name = "samples"

        def last_cell(row):
            return fmt_int(row["num_descendant_samples"])

    lines = [
        section_header(2, page_break),
        group_header(),
        make_row(
            ["name", "averted", "left", "right", "left", "right", last_name],
            prefix=r"\nopagebreak",
        ),
    ]
    for i, row in enumerate(rows):
        lines.append(
            make_row(
                [
                    check_safe(row["alias"], "alias"),
                    fmt_int(row["mutations_averted"]),
                    check_safe(row["parent_left_pango"], "parent_left_pango"),
                    check_safe(row["parent_right_pango"], "parent_right_pango"),
                    fmt_int(row["interval_left"]),
                    fmt_int(row["interval_right"]),
                    last_cell(row),
                ],
                prefix=r"\midrule \nopagebreak" if i == 0 else "",
            )
        )
    return lines


def type3_section(page_break=False):
    rows, _ = read_csv(3)
    lines = [
        section_header(3, page_break),
        make_row(
            ["pango", "event", "path len", "time", "mutations", "", "descendants"],
            prefix=r"\nopagebreak",
        ),
    ]
    for i, row in enumerate(rows):
        lines.append(
            make_row(
                [
                    check_safe(row["root_pango"], "root_pango"),
                    check_safe(row["event_name"], "event_name"),
                    fmt_int(row["closest_recombinant_path_len"]),
                    fmt_int(row["closest_recombinant_time"]),
                    fmt_int(row["root_mutations"]),
                    "",
                    fmt_descendants(row["descendants"]),
                ],
                prefix=r"\midrule \nopagebreak" if i == 0 else "",
            )
        )
    return lines


def type4_section(page_break=False):
    rows, _ = read_csv(4)
    lines = [
        section_header(4, page_break),
        make_row(
            ["pango", "parent", "", "", "mutations", "", "descendants"],
            prefix=r"\nopagebreak",
        ),
    ]
    for i, row in enumerate(rows):
        lines.append(
            make_row(
                [
                    check_safe(row["root_pango"], "root_pango"),
                    check_safe(row["root_parent_pango"], "root_parent_pango"),
                    "",
                    "",
                    fmt_int(row["root_mutations"]),
                    "",
                    fmt_descendants(row["descendants"]),
                ],
                prefix=r"\midrule \nopagebreak" if i == 0 else "",
            )
        )
    return lines


SECTIONS = {
    1: type1_section,
    2: type2_section,
    3: type3_section,
    4: type4_section,
}


def make_table(page_break_before=()):
    caption = " ".join(CAPTION.split())
    lines = [
        "% Generated by make_table.py -- do not edit.",
        "% Regenerate with: python3 pango_x_table/make_table.py",
        "%",
        "% The group is what makes the rows tight: tabularray takes the row",
        "% height from the font in force when the table is built, so setting",
        "% the cell font alone leaves rows spaced for the 12pt body font.",
        r"\begingroup\scriptsize",
        r"\begin{longtabs}[",
        r"    theme   = sc2ts,",
        r"    label   = {tab:pango_x_lineages},",
        r"    entry   = {Events in the ARG associated with Pango X lineages},",
        rf"    caption = {{{caption}}},",
        r"  ]{",
        r"    colspec = {lrrlrlX[l]},",
        r"    width   = \linewidth,",
        r"    hspan   = minimal,",
        r"    colsep  = 4pt,",
        r"  }",
    ]
    for event_type, section in SECTIONS.items():
        lines.extend(section(page_break=event_type in page_break_before))
    lines.append(r"\bottomrule")
    lines.append(r"\end{longtabs}")
    lines.append(r"\endgroup")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--page-break-before",
        type=int,
        choices=[2, 3, 4],
        action="append",
        default=[],
        metavar="TYPE",
        help=(
            "start the given event type's section on a new page; "
            "may be given more than once"
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        type=pathlib.Path,
        default=HERE / "table.tex",
        help="where to write the generated LaTeX (default: %(default)s)",
    )
    args = parser.parse_args()
    args.output.write_text(make_table(set(args.page_break_before)))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
