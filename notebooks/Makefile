# Use this Makefile to generate the HTML versions of the notebooks
# and to run the data exports in them.
# This is useful for archival purposes, as the notebooks themselves
# sometimes don't render well on GitHub

# FIXME: the SVG graphics still aren't showing up well in the output
# HTML. Need a better way of systematising this

NOTEBOOKS=s2.1_summary_tables.html\
	  s2.2_cophylo_recombinants.html\
	  s2.3_mutational_spectrum.html\
	  s2.4_jackson_recombinants.html\
	  s2.5_breakpoints.html\
	  s2.7_pango_recombinants.html

all: ${NOTEBOOKS}

%.html : %.ipynb
	jupyter nbconvert $< --to html --execute
