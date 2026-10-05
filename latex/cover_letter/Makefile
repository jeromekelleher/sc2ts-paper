all: cover_letter.pdf response_letter.pdf

cover_letter.pdf: cover_letter.tex
	pdflatex cover_letter.tex

response_letter.pdf: response_letter.tex
	pdflatex response_letter.tex

clean:
	rm -f *.log *.dvi *.aux *.out
	rm -f *.blg *.bbl
	rm -f src/*.mpx *.mpx

mrproper: clean
	rm -f *.ps *.pdf
