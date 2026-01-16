DATA=
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
