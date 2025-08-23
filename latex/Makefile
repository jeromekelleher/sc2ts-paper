DATA=
FIGURES=

all: main.pdf docS1.pdf

main.pdf: main.tex references.bib ${DATA} ${FIGURES}
	pdflatex -shell-escape main.tex
	bibtex main
	pdflatex main.tex
	pdflatex main.tex

docS1.pdf: docS1.tex
	pdflatex -shell-escape docS1.tex

%.pdf : %.svg
	# Needs inkscape >= 1.0
	inkscape --export-type="pdf" $<



.PHONY: spellcheck
spellcheck: aspell.conf
	aspell --conf ./aspell.conf --check main.tex

clean:
	rm -f *.pdf
	rm -f illustrations/*.pdf
	rm -f illustrations/*.svg
	rm -f *.log *.dvi *.aux
	rm -f *.blg *.bbl

mrproper: clean
	rm -f *.ps *.pdf
