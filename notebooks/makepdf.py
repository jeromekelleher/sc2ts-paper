#!/usr/bin/env python3
import argparse
import subprocess
from pathlib import Path
from tempfile import NamedTemporaryFile

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description=(
            "Create a PDF from notebook containing a tskit_arg_visualizer instance, "
            "using webpdf. WARNING: this is a complete hack and may break at any time."
        )
    )
    argparser.add_argument("input_ipynb", help="Path to input ts or tsz file", type=str)
    args = argparser.parse_args()
    # find absolute path to the input file
    infile = Path(args.input_ipynb).resolve(strict=True)
    with open(infile, "r") as input, NamedTemporaryFile(mode="w") as output:
        # Replace the javascript in the notebook code
        # to change mutation symbols and node labels to contain URLs.
        # This is a complete hack that fakes tooltips in a pdf, as URLS appear on hover.
        for line in input.readlines():
            ## Mutation symbols have a hover that shows the position and state change
            if line.rstrip().endswith(r'var mut_symbol_rect = mut_symbol\n",'):
                start, end = line.rsplit(r'\n', 1)
                output.write(start + r'.append(\"a\").attr(\"href\", d => \"mut:\" + d.label)\n' + end)
            #  Line labels where the second line starts with (DRR, (ERR, or (SRR
            #  are turned into a link with the second line as the URL)
            #  This is EXTEREMLY fragile. 
            elif line.rstrip().endswith(r'const lines = text.split(\"\\n\");\n",'):
                output.write(r'''       "                let lines = text.split(\"\\n\");\n",
       "                let url = null;\n",
       "                if ((lines.length == 2) && (lines[1].startsWith('(DRR') || lines[1].startsWith('(ERR') || lines[1].startsWith('(SRR'))) {\n",
       "                        url = 'sample:' + lines[1].replace(/[()]/g, ''); lines = [lines[0]];\n",
       "                    }\n",''')
            elif line.rstrip().endswith(r'''d3.select(this).selectAll('tspan')\n",'''):
                output.write(r'''       "                const container = url ? \n",
       "                    d3.select(this).append(\"a\").attr(\"href\", url) :\n",
       "                    d3.select(this);\n",
       "                container.selectAll('tspan')\n",''')
            else:
                output.write(line)
        output.flush()
        outfile = str(infile.with_suffix(''))
        cmd = [
            "jupyter",
            "nbconvert",
            "--to",
            "webpdf",
            "--no-prompt",
            "--no-input",
            output.name,
            "--output",
            outfile,
        ]
        subprocess.run(cmd, check=True)

    print(f"PDF saved to {outfile}")