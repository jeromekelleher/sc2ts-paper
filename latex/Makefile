DATA=pango_x_table/table.tex ripples_tables/x_lineages.tex copying_patterns/figure.tex
FIGURES=

all: combined.pdf main_text.pdf

combined.pdf: main.tex references.bib ${DATA} ${FIGURES}
	pdflatex -shell-escape main.tex
	bibtex main
	pdflatex main.tex
	pdflatex main.tex
	cp main.pdf combined.pdf

main_text.pdf: main.tex
	cp main.tex split_tmp.tex
	pdflatex -shell-escape "\def\SPLITBUILD{1}\input{split_tmp.tex}"
	bibtex split_tmp
	pdflatex -shell-escape "\def\SPLITBUILD{1}\input{split_tmp.tex}"
	pdflatex -shell-escape "\def\SPLITBUILD{1}\input{split_tmp.tex}"
	qpdf split_tmp.pdf --pages . 1-34 -- main_text.pdf
	qpdf split_tmp.pdf --pages . 35-z -- supplementary.pdf

# The generated tables. The CSVs are written by the notebooks.
pango_x_table/table.tex: pango_x_table/make_table.py $(wildcard pango_x_table/*.csv)
	python3 pango_x_table/make_table.py

# The copying pattern images are written by src/save_copying_patterns.py
copying_patterns/figure.tex: copying_patterns/make_figure.py ../data/pango_x_event_details.csv
	python3 copying_patterns/make_figure.py

# Also writes ripples_tables/event_descendants.tex
ripples_tables/x_lineages.tex: ripples_tables/make_tables.py $(wildcard ripples_tables/*.csv)
	python3 ripples_tables/make_tables.py

# Standalone view of the Pango X event table.
pango_x_table.pdf: pango_x_table.tex pango_x_table/table.tex
	pdflatex pango_x_table.tex
	pdflatex pango_x_table.tex


.PHONY: spellcheck
spellcheck: aspell.conf
	aspell --conf ./aspell.conf --check main.tex

clean:
	rm -f *.pdf
	rm -f illustrations/*.pdf
	rm -f illustrations/*.svg
	rm -f *.log *.dvi *.aux
	rm -f *.blg *.bbl
	rm split_tmp*

mrproper: clean
	rm -f *.ps *.pdf
