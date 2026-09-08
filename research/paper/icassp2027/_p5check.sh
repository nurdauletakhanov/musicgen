#!/bin/bash
# True page-5 test: force a real PAGE break before the bibliography (\clearpage;
# \newpage only ends a column in two-column mode). ICASSP allows 4 pages of body
# plus a references-only 5th, so the forced-break build must be <= 5 pages.
set -e
sed 's/\\bibliography{references}/\\clearpage\\bibliography{references}/' paper.tex > _p5.tex
latexmk -g -pdf -interaction=nonstopmode -jobname=_p5 _p5.tex > _p5.log 2>&1 || true
P=$(gs -q -dNODISPLAY -dBATCH -dNOPAUSE -dPDFINFO _p5.pdf 2>&1 | grep -oE 'File has [0-9]+' | grep -oE '[0-9]+')
BODY=$((P - 1))
if [ "$BODY" -le 4 ]; then echo "PASS: body = $BODY pages, refs alone on page $P"
else echo "FAIL: body = $BODY pages (limit 4); refs start on page $P"; fi
rm -f _p5.aux _p5.bbl _p5.blg _p5.fdb_latexmk _p5.fls _p5.log _p5.out _p5.tex _p5.pdf
