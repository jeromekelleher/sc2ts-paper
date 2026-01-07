cover_letter.pdf: cover_letter.tex
	pdflatex cover_letter.tex

clean:
	rm -f *.log *.dvi *.aux
	rm -f *.blg *.bbl
	rm -f src/*.mpx *.mpx

mrproper: clean
	rm -f *.ps *.pdf
