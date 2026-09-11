#!/bin/bash
# Assemble the arXiv source tarball. arXiv compiles LaTeX itself; it needs the
# .tex, the flattened table/figure inputs, IEEEtran.bst, and the prebuilt .bbl
# (so its pipeline never has to run bibtex against our .bib).
#
#   bash make_arxiv.sh        -> arxiv-upload.tar.gz
set -euo pipefail
cd "$(dirname "$0")"

grep -q "TBD" paper.tex && { echo "ERROR: paper.tex still contains TBD placeholders"; grep -n "TBD" paper.tex | head; exit 1; }
latexmk -g -pdf -interaction=nonstopmode paper.tex > /dev/null   # fresh .bbl
[ -f paper.bbl ] || { echo "ERROR: paper.bbl missing"; exit 1; }

STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"' EXIT
cp paper.tex paper.bbl IEEEtran.bst "$STAGE"/
mkdir -p "$STAGE"/tables "$STAGE"/figures
cp tables/*.tex "$STAGE"/tables/
cp figures/*.pdf "$STAGE"/figures/

# Sanity: the staged sources must compile on their own (as arXiv will).
( cd "$STAGE" && latexmk -pdf -interaction=nonstopmode paper.tex > build.log 2>&1 ) \
  || { echo "ERROR: staged sources do not compile standalone"; exit 1; }
PAGES=$(grep -oE 'Output written on paper.pdf \([0-9]+' "$STAGE"/build.log | grep -oE '[0-9]+$' | tail -1)
echo "staged build OK: $PAGES pages"
# ICASSP: page 5 may hold references only. Force a break before the bibliography;
# if the body already fits in 4 pages the count is unchanged, otherwise it grows.
sed 's/\\bibliography{references}/\\clearpage\\bibliography{references}/' "$STAGE"/paper.tex > "$STAGE"/_p5.tex
( cd "$STAGE" && latexmk -pdf -interaction=nonstopmode -jobname=_p5 _p5.tex > _p5.log 2>&1 ) || true
P5=$(grep -oE 'Output written on _p5.pdf \([0-9]+' "$STAGE"/_p5.log | grep -oE '[0-9]+$' | tail -1)
[ "${P5:-0}" -le 5 ] || { echo "ERROR: body text spills onto page 5 (forced-break build has $P5 pages)"; exit 1; }
echo "page-5 references-only check: PASS"

tar -czf arxiv-upload.tar.gz -C "$STAGE" paper.tex paper.bbl IEEEtran.bst tables figures
echo "wrote $(pwd)/arxiv-upload.tar.gz ($(du -h arxiv-upload.tar.gz | cut -f1))"
echo "upload at https://arxiv.org/submit (category: eess.AS, cross-list cs.SD)"
