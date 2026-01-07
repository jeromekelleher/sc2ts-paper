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
	pdflatex -shell-escape "\def\SPLITBUILD{1}\input{main.tex}"
	bibtex main
	pdflatex -shell-escape "\def\SPLITBUILD{1}\input{main.tex}"
	pdflatex -shell-escape "\def\SPLITBUILD{1}\input{main.tex}"
	qpdf main.pdf --pages . 1-39 -- main_text.pdf


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
