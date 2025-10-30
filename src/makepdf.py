#!/usr/bin/env python3
import argparse
import subprocess
from pathlib import Path
import os
from tempfile import NamedTemporaryFile

from playwright.sync_api import sync_playwright


def convert_file(infile, outdir):
    infile = Path(infile).resolve(strict=True)
    if outdir is None:
        outdir = infile.parent
    else:
        outdir = Path(os.getcwd()) / outdir
    outfile = outdir / str(infile.with_suffix('').name)

    if infile.suffix == ".svg":
        print(f"Converting {infile} to {outfile}")
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            context = browser.new_context()
            page = context.new_page()
            page.goto("file://" + str(infile))
            page.emulate_media(media="print")
            page.pdf(path=str(outfile) + '.pdf')
            browser.close()

    elif infile.suffix == ".ipynb":
        with open(infile, "r") as input, NamedTemporaryFile(mode="w") as output:
            # Replace the javascript in the notebook code
            # to change mutation symbols and node labels to contain URLs.
            # This is a complete hack that fakes tooltips in a pdf, as URLS appear on hover.
            add_mut_links = False
            for line in input.readlines():
                ## Mutation symbols have a hover that shows the position and state change
                if "var mut_symbol = svg" in line:
                    add_mut_links = True
                if add_mut_links and ".enter()" in line:
                    add_mut_links = False
                    start, end = line.rsplit(r'\n', 1)
                    output.write(start + r'.append(\"a\").attr(\"href\", d => \"mut:\" + d.label)\n' + end)
                #  Line labels where the second line starts with (DRR, (ERR, or (SRR
                #  are turned into a link with the second line as the URL. Line labels where
                #  the first line is   
                #  This is EXTREMELY fragile. 
                elif line.rstrip().endswith(r'const lines = text.split(\"\\n\");\n",'):
                    output.write(r'''       "                let lines = text.split(\"\\n\");\n",
        "                let url = null;\n",
        "                if (lines.length == 2) {\n",
        "                  if (lines[1].startsWith('(DRR') || lines[1].startsWith('(ERR') || lines[1].startsWith('(SRR')) {\n",
        "                    url = 'sample:' + lines[1].replace(/[()]/g, ''); lines = [lines[0]];\n",
        "                  }\n",
        "                  else if (lines[1].startsWith('/')) {\n",
        "                    lines = [lines[1], lines[0]];\n",
        "                  }\n",
        "                }\n",''')
                elif line.rstrip().endswith(r'''d3.select(this).selectAll('tspan')\n",'''):
                    output.write(r'''       "                const container = url ? \n",
        "                    d3.select(this).append(\"a\").attr(\"href\", url) :\n",
        "                    d3.select(this);\n",
        "                container.selectAll('tspan')\n",''')
                else:
                    output.write(line)
            output.flush()
            cmd = [
                "jupyter",
                "nbconvert",
                "--to",
                "webpdf",
                "--no-prompt",
                "--TagRemovePreprocessor.remove_cell_tags='{\"remove_cell\"}'",
                "--PDFExporter.margin_left=0.2cm",
                "--PDFExporter.margin_right=0.2cm",
                "--no-input",
                output.name,
                "--output",
                outfile,
            ]
            subprocess.run(cmd, check=True)
        print(f"PDF saved to {outfile}")
    else:
        print(f"Unsupported file type: {infile.suffix}")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description=(
            "Create a PDF from notebook or an svg using playwright."
            "Notebook files will use the nbconvert webpdf path, "
            "and convert tskit_arg_visualizer instances such that they "
            "have mouseover reveals (this is a hack that may break at any time)."
        )
    )
    argparser.add_argument("infiles", nargs="+", help="Path to input .svg or .ipynb file", type=str)
    argparser.add_argument("--outdir", "-o", default=None, help="Path to output directory", type=str)

    args = argparser.parse_args()
    # find absolute path to the input file
    for infile in args.infiles:
        convert_file(infile, args.outdir)

