# Dual mapping PDFs (2026-09-11 freeze)

These sit next to the older N2S pair `progress-notes.pdf` / `progress-notes-simple.pdf`.
The repo gitignores `*.pdf`, so if the tree hides binaries, open these paths directly.

| PDF | Same role as | What it is |
|-----|----------------|------------|
| **`mapping-notes-simple.pdf`** | `progress-notes-simple.pdf` / newcomer tour | Plain-language walkthrough + GUI |
| **`mapping-code-walkthrough.pdf`** | `N2S-CLI-Walkthrough.pdf` | Runnable snippets (CPU) |
| **`CXR-Mapping-Curriculum.pdf`** | `CXR-RepEng-Curriculum.pdf` | MAP-00…09 beginner track |
| **`mapping-notes.pdf`** | `progress-notes.pdf` | Full freeze record (numbers, cells, claim ceiling) |

Rebuild all four:

```bash
cd ~/staging/cxr-evidence-grounding-lab/notes
./build-mapping-pdfs.sh
```

Live UIs: map viewer http://127.0.0.1:8263/ · curriculum http://127.0.0.1:8264/
