#!/usr/bin/env python3
"""Dual Translate mapping curriculum UI. Port 8264. Read-only modules."""

from __future__ import annotations

import argparse
import json
import re
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent
STATIC = ROOT / "static"
DATA = ROOT / "data"
MODULES_DIR = ROOT / "modules"
CURRICULUM = ROOT / "CXR-Mapping-Curriculum.md"
PROGRESS_PATH = DATA / "progress.json"
HOST = "127.0.0.1"
PORT = 8264

MODULES = [
    ("00", "00-orientation.md", "MAP-00 · Orientation", True),
    ("01", "01-aim-and-background.md", "MAP-01 · Aim and background", False),
    ("02", "02-backend-pipeline.md", "MAP-02 · Backend pipeline", False),
    ("03", "03-three-surfaces.md", "MAP-03 · Three surfaces", False),
    ("04", "04-filling-extracts.md", "MAP-04 · Filling extracts", False),
    ("05", "05-nested-cells.md", "MAP-05 · Nested cells", False),
    ("06", "06-worked-examples.md", "MAP-06 · Four notes", False),
    ("07", "07-leftovers-and-stop.md", "MAP-07 · Leftovers and stop", False),
    ("08", "08-mapping-not-selection.md", "MAP-08 · Mapping ≠ selection", False),
    ("09", "09-replicate-and-ceiling.md", "MAP-09 · Replicate and ceiling", False),
]


def _json(handler, payload, status=200):
    body = json.dumps(payload, indent=2).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(body)


def _read_json(handler):
    n = int(handler.headers.get("Content-Length") or "0")
    raw = handler.rfile.read(n) if n else b"{}"
    return json.loads(raw.decode("utf-8") or "{}")


def _strip_wiki(text: str) -> str:
    text = re.sub(r"\[\[([^\]|]+)\|([^\]]+)\]\]", r"\2", text)
    return re.sub(r"\[\[([^\]]+)\]\]", r"\1", text)


def _load_progress():
    DATA.mkdir(parents=True, exist_ok=True)
    if not PROGRESS_PATH.exists():
        return {"current_module": "00", "unlocked": [m[0] for m in MODULES], "notes": ""}
    return json.loads(PROGRESS_PATH.read_text(encoding="utf-8"))


def _save_progress(data):
    DATA.mkdir(parents=True, exist_ok=True)
    PROGRESS_PATH.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _module_meta(progress):
    unlocked = set(progress.get("unlocked") or [m[0] for m in MODULES])
    out = []
    for mid, fname, title, start in MODULES:
        out.append(
            {
                "id": mid,
                "file": fname,
                "title": title,
                "start_here": start,
                "exists": (MODULES_DIR / fname).exists(),
                "unlocked": True if not unlocked else mid in unlocked,
            }
        )
    return out


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(STATIC), **kwargs)

    def log_message(self, fmt, *args):
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        if path in {"/", "/index.html"}:
            return super().do_GET()
        if path == "/api/meta":
            progress = _load_progress()
            return _json(
                self,
                {
                    "track": "Dual Translate mapping",
                    "port": PORT,
                    "viewer": "http://127.0.0.1:8263/",
                    "claim": (
                        "V1 demonstrated mapping, not selection. Nested Translate "
                        "wrappers. Not auto-correct."
                    ),
                    "source_dir": str(MODULES_DIR),
                    "curriculum": str(CURRICULUM),
                    "modules": _module_meta(progress),
                    "progress": progress,
                },
            )
        if path == "/api/progress":
            return _json(self, _load_progress())
        if path == "/api/module":
            mid = (parse_qs(parsed.query).get("id") or ["00"])[0]
            match = next((m for m in MODULES if m[0] == mid), None)
            if not match:
                return _json(self, {"ok": False, "error": "unknown"}, 404)
            fpath = MODULES_DIR / match[1]
            if not fpath.exists():
                return _json(self, {"ok": False, "error": "missing"}, 404)
            return _json(
                self,
                {
                    "ok": True,
                    "id": mid,
                    "file": match[1],
                    "title": match[2],
                    "markdown": _strip_wiki(fpath.read_text(encoding="utf-8")),
                    "source_path": str(fpath),
                },
            )
        if path == "/api/curriculum":
            if not CURRICULUM.exists():
                return _json(self, {"ok": False}, 404)
            return _json(
                self,
                {
                    "ok": True,
                    "markdown": _strip_wiki(CURRICULUM.read_text(encoding="utf-8")),
                    "source_path": str(CURRICULUM),
                },
            )
        return super().do_GET()

    def do_POST(self):
        if urlparse(self.path).path != "/api/progress":
            return _json(self, {"ok": False}, 404)
        body = _read_json(self)
        cur = _load_progress()
        if "current_module" in body:
            cur["current_module"] = body["current_module"]
        if "notes" in body:
            cur["notes"] = body["notes"]
        _save_progress(cur)
        return _json(self, {"ok": True, "progress": cur})


def main():
    ap = argparse.ArgumentParser(description="Mapping curriculum :8264")
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--host", default=HOST)
    args = ap.parse_args()
    if not MODULES_DIR.is_dir():
        print("Missing modules/", file=sys.stderr)
        sys.exit(1)
    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"mapping curriculum http://{args.host}:{args.port}/  (viewer :8263)", flush=True)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
