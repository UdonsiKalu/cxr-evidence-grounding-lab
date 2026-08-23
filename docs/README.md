# Public findings demo (GitHub Pages)

**Mode:** artifact replay only — frozen Phase 1–4 JSON.  
**Not included:** live Ollama, your GPU, CXR production.

Visitors see Conditions A–D, loss stages, and multi-model panels without running inference.

## Local preview

```bash
cd /path/to/cxr-evidence-grounding-lab
python3 scripts/prepare-github-pages.py
cd docs
python3 -m http.server 8765
```

Open http://127.0.0.1:8765/

## Enable on GitHub

1. Push this repo (or a public copy of this lab folder).
2. **Settings → Pages → Build and deployment**
   - Source: **Deploy from a branch**
   - Branch: `main` (or `master`)
   - Folder: **`/docs`**
3. After deploy, the demo URL is:

   `https://<user>.github.io/<repo>/`

Re-run `python3 scripts/prepare-github-pages.py` whenever frozen artifacts under `artifacts/` change, then commit `docs/artifacts/` + `docs/manifest.json`.

## What is safe

| Public Pages demo | Local clone (`server.py`) |
|-------------------|---------------------------|
| Loads static JSON only | Can call Ollama on *that* machine |
| No path to your laptop | Your `127.0.0.1:8253` / `:8254` |

Do **not** point a public tunnel at your local Ollama for this demo.
