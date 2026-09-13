#!/usr/bin/env bash
# Build CXR-Mapping-Curriculum.pdf from markdown modules.
# Usage: ./build-mapping-pdf.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

OUT_PDF="CXR-Mapping-Curriculum.pdf"
COMBINED="_mapping-combined.md"

need() { command -v "$1" >/dev/null 2>&1 || { echo "Missing: $1" >&2; exit 1; }; }
need pandoc

PDF_ENGINE=""
for eng in lualatex xelatex pdflatex; do
  if command -v "$eng" >/dev/null 2>&1; then
    PDF_ENGINE="$eng"
    break
  fi
done
[[ -n "$PDF_ENGINE" ]] || { echo "Need lualatex, xelatex, or pdflatex" >&2; exit 1; }

FILES=(
  00-orientation.md
  01-aim-and-background.md
  02-backend-pipeline.md
  03-three-surfaces.md
  04-filling-extracts.md
  05-nested-cells.md
  06-worked-examples.md
  07-leftovers-and-stop.md
  08-mapping-not-selection.md
  09-replicate-and-ceiling.md
)

for f in "${FILES[@]}"; do
  [[ -f "$f" ]] || { echo "Missing module: $f" >&2; exit 1; }
done

cleanup() {
  rm -f "$COMBINED" \
    _mapping-combined.aux _mapping-combined.log \
    _mapping-combined.out _mapping-combined.toc \
    _mapping-combined.tex _mapping-combined.synctex.gz 2>/dev/null || true
}
trap cleanup EXIT

{
  cat <<'EOF'
---
title: "CXR Dual Translate Mapping Curriculum"
subtitle: "MAP-00…09 — frozen 2026-09-11 pass (mapping, not selection)"
author: "CXR evidence-grounding lab"
date: "Generated from markdown modules"
documentclass: extarticle
fontsize: 9pt
geometry: margin=0.7in
linestretch: 1.08
---

\newpage

# How to use this PDF

**North-star:** By MAP-09 you can explain Dual grounding, what this pass set out
to do, how nested Translate patches work, how to replay them, and what the
evidence does not support.

**Not:** a generic auto-corrector for every LLM failure. **Not** locator-chosen
Encode vs Compute vs Translate. **Not** CXR Foundations or RepEng Ph9–14.

| Role | Where |
|------|--------|
| This PDF | Print / offline |
| Track UI | `http://127.0.0.1:8264/` |
| Map viewer | `http://127.0.0.1:8263/` |
| Science lab | `cxr-evidence-grounding-lab/` |

Order: **MAP-00 → MAP-09**.

\newpage

EOF

  first=1
  for f in "${FILES[@]}"; do
    if [[ "$first" -eq 0 ]]; then
      printf '\n\\newpage\n\n'
    fi
    first=0
    sed -E \
      -e 's/\[\[([^]|]+)\|([^]]+)\]\]/\2/g' \
      -e 's/\[\[([^]]+)\]\]/\1/g' \
      "$f"
    printf '\n'
  done
} > "$COMBINED"

echo "Building $OUT_PDF with pandoc + $PDF_ENGINE ..."

PANDOC_ARGS=(
  "$COMBINED"
  -o "$OUT_PDF"
  --pdf-engine="$PDF_ENGINE"
  --toc
  --toc-depth=2
  -V documentclass=extarticle
  -V fontsize=9pt
  -V geometry:margin=0.7in
  -V linestretch=1.08
  -V colorlinks=true
  -V linkcolor=blue
  -V urlcolor=blue
  --highlight-style=tango
  -f markdown
)

if [[ "$PDF_ENGINE" == "pdflatex" ]]; then
  pandoc "${PANDOC_ARGS[@]}" 2> >(tee /tmp/cxr-mapping-pdf.err >&2) || true
else
  pandoc "${PANDOC_ARGS[@]}" \
    -V mainfont="DejaVu Sans" \
    -V monofont="DejaVu Sans Mono" \
    2> >(tee /tmp/cxr-mapping-pdf.err >&2) || true
fi

if [[ ! -f "$OUT_PDF" ]]; then
  echo "Primary engine failed; retrying with pdflatex + ascii fallbacks..." >&2
  python3 - <<'PY'
from pathlib import Path
p = Path("_mapping-combined.md")
t = p.read_text(encoding="utf-8")
repl = {
    "→": "->", "←": "<-", "↔": "<->", "—": "--", "–": "-",
    "“": '"', "”": '"', "‘": "'", "’": "'", "…": "...",
    "≈": "~=", "×": "x", "§": "Section ", "≠": "!=",
    "∧": "AND", "∨": "OR",
}
for a, b in repl.items():
    t = t.replace(a, b)
p.write_text(t, encoding="utf-8")
PY
  pandoc "$COMBINED" -o "$OUT_PDF" \
    --pdf-engine=pdflatex \
    --toc --toc-depth=2 \
    -V documentclass=extarticle \
    -V fontsize=9pt \
    -V geometry:margin=0.7in \
    -V colorlinks=true \
    --highlight-style=tango \
    -f markdown
fi

cp -f "$OUT_PDF" "$ROOT/../CXR-Mapping-Curriculum.pdf"
NOTES="$ROOT/../../notes"
if [[ -d "$NOTES" ]]; then
  cp -f "$OUT_PDF" "$NOTES/CXR-Mapping-Curriculum.pdf"
fi
ls -lh "$OUT_PDF"
pdfinfo "$OUT_PDF" 2>/dev/null || true
echo "Done: $ROOT/$OUT_PDF"
