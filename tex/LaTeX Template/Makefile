all: thesis

clean:
ifeq ($(OS),Windows_NT)

	rm *.aux
	rm *.dvi
	rm *.log
	rm *.fls
	rm *.fdb_latexmk
	rm *.synctex.gz
else
	rm -f *.aux *.bbl *.bcf *.blg *.lof *.log *.lol *.lot *.out *.run.xml *.tdo *.tex.blg *.toc
endif

thesis:
	pdflatex thesis.tex
	biber thesis
	pdflatex thesis.tex



