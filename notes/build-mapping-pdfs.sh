#!/usr/bin/env bash
# Build the three mapping freeze PDFs into notes/ (plus copy the curriculum PDF).
# Same split as older N2S: simple walkthrough / full notes / CLI code / curriculum.
# Usage: ./build-mapping-pdfs.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

need() { command -v "$1" >/dev/null 2>&1 || { echo "Missing: $1" >&2; exit 1; }; }

ENGINE=""
for eng in latexmk pdflatex lualatex; do
  if command -v "$eng" >/dev/null 2>&1; then ENGINE="$eng"; break; fi
done
[[ -n "$ENGINE" ]] || { echo "Need latexmk or pdflatex" >&2; exit 1; }

compile_tex() {
  local tex="$1"
  if [[ "$ENGINE" == "latexmk" ]]; then
    latexmk -pdf -interaction=nonstopmode -halt-on-error "$tex"
  else
    pdflatex -interaction=nonstopmode "$tex"
    pdflatex -interaction=nonstopmode "$tex"
  fi
}

echo "== mapping-notes-simple.tex =="
compile_tex mapping-notes-simple.tex
echo "== mapping-notes.tex =="
compile_tex mapping-notes.tex

need pandoc
PDF_ENGINE=""
for eng in lualatex xelatex pdflatex; do
  if command -v "$eng" >/dev/null 2>&1; then PDF_ENGINE="$eng"; break; fi
done
[[ -n "$PDF_ENGINE" ]] || { echo "Need a pandoc PDF engine" >&2; exit 1; }

echo "== mapping-code-walkthrough.md =="
CODE_MD="_mapping-code-combined.md"
{
  cat <<'EOF'
---
title: "Dual mapping — code walkthrough"
subtitle: "CPU snippets (same role as N2S-CLI-Walkthrough.pdf)"
author: "cxr-evidence-grounding-lab/notes"
date: "11 September 2026"
documentclass: extarticle
fontsize: 9pt
geometry: margin=0.7in
---

\newpage

EOF
  cat mapping-code-walkthrough.md
} > "$CODE_MD"

PANDOC_ARGS=(
  "$CODE_MD"
  -o mapping-code-walkthrough.pdf
  --pdf-engine="$PDF_ENGINE"
  --toc --toc-depth=2
  -V documentclass=extarticle
  -V fontsize=9pt
  -V geometry:margin=0.7in
  -V colorlinks=true
  -V linkcolor=blue
  -V urlcolor=blue
  --highlight-style=tango
  -f markdown
)

if [[ "$PDF_ENGINE" == "pdflatex" ]]; then
  python3 - <<'PY'
from pathlib import Path
p = Path("_mapping-code-combined.md")
t = p.read_text(encoding="utf-8")
repl = {"→": "->", "←": "<-", "—": "--", "–": "-", "“": '"', "”": '"', "‘": "'", "’": "'", "…": "...", "≠": "!="}
for a, b in repl.items():
    t = t.replace(a, b)
p.write_text(t, encoding="utf-8")
PY
  pandoc "${PANDOC_ARGS[@]}"
else
  pandoc "${PANDOC_ARGS[@]}" -V mainfont="DejaVu Sans" -V monofont="DejaVu Sans Mono"
fi
rm -f "$CODE_MD"

CURR_SRC=""
for p in \
  "$HERE/../curriculum-mapping/CXR-Mapping-Curriculum.pdf" \
  "$HERE/../curriculum-mapping/modules/CXR-Mapping-Curriculum.pdf"
do
  if [[ -f "$p" ]]; then CURR_SRC="$p"; break; fi
done
if [[ -n "$CURR_SRC" ]]; then
  cp -f "$CURR_SRC" "$HERE/CXR-Mapping-Curriculum.pdf"
  echo "copied curriculum PDF from $CURR_SRC"
else
  echo "WARN: curriculum PDF missing; run curriculum-mapping/modules/build-mapping-pdf.sh" >&2
fi

# tidy latex aux
if command -v latexmk >/dev/null 2>&1; then
  latexmk -c mapping-notes-simple.tex mapping-notes.tex >/dev/null 2>&1 || true
fi
rm -f mapping-notes*.aux mapping-notes*.log mapping-notes*.out mapping-notes*.toc \
  mapping-notes-simple.aux mapping-notes-simple.log mapping-notes-simple.out mapping-notes-simple.toc \
  2>/dev/null || true

echo
ls -lh mapping-notes-simple.pdf mapping-notes.pdf mapping-code-walkthrough.pdf CXR-Mapping-Curriculum.pdf
echo "Done. Open notes/MAPPING-PDFS.md for the index."
