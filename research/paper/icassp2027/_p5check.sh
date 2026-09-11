#!/bin/bash
# True page-5 test: force a real PAGE break before the bibliography (\clearpage;
# \newpage only ends a column in two-column mode). ICASSP allows 4 pages of body
# plus a references-only 5th, so the forced-break build must be <= 5 pages.
set -e
SRC="${1:-paper.tex}"
# Placeholders left from drafting must never reach a submitted build.
if grep -q "TBD" "$SRC"; then echo "FAIL: $SRC still contains TBD placeholders:"; grep -n "TBD" "$SRC" | head; exit 1; fi
sed 's/\\bibliography{references}/\\clearpage\\bibliography{references}/' "$SRC" > _p5.tex
latexmk -g -pdf -interaction=nonstopmode -jobname=_p5 _p5.tex > _p5.log 2>&1 || true
# A failed compile silently shortens the document, which used to read as a
# PASS. Refuse to report on a build that did not succeed.
ERRS=$(grep -c '^!' _p5.log || true)
if [ "${ERRS:-0}" -gt 0 ]; then
  echo "FAIL: $SRC did not compile ($ERRS LaTeX errors):"
  grep -A2 '^!' _p5.log | head -8
  rm -f _p5.aux _p5.bbl _p5.blg _p5.fdb_latexmk _p5.fls _p5.log _p5.out _p5.tex _p5.pdf
  exit 1
fi
P=$(gs -q -dNODISPLAY -dBATCH -dNOPAUSE -dPDFINFO _p5.pdf 2>&1 | grep -oE 'File has [0-9]+' | grep -oE '[0-9]+')
BODY=$((P - 1))
if [ "$BODY" -le 4 ]; then echo "PASS: $SRC body = $BODY pages, refs alone on page $P"
else echo "FAIL: $SRC body = $BODY pages (limit 4); refs start on page $P"; fi
rm -f _p5.aux _p5.bbl _p5.blg _p5.fdb_latexmk _p5.fls _p5.log _p5.out _p5.tex _p5.pdf
