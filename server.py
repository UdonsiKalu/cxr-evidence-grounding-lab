#!/usr/bin/env python3
"""Stdlib HTTP UI for the evidence-grounding lab. Port 8253."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from n2s_lab.experiment import load_cases, run_experiment, run_one  # noqa: E402
from n2s_lab.ground import selftest_ground  # noqa: E402
from n2s_lab.ollama_client import DEFAULT_MODEL, ollama_reachable  # noqa: E402
from n2s_lab.paths import ARTIFACTS_DIR, STATIC_DIR  # noqa: E402
from n2s_lab.predicate import PREDICATE_FORMULA, PREDICATE_ID, PREDICATE_TEXT, selftest  # noqa: E402

HOST = "127.0.0.1"
PORT = 8253


def _json_response(handler: SimpleHTTPRequestHandler, payload: object, status: int = 200) -> None:
    body = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(body)


def _read_json(handler: SimpleHTTPRequestHandler) -> dict:
    length = int(handler.headers.get("Content-Length") or "0")
    raw = handler.rfile.read(length) if length else b"{}"
    if not raw:
        return {}
    return json.loads(raw.decode("utf-8"))


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(STATIC_DIR), **kwargs)

    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path in {"/", "/index.html"}:
            return super().do_GET()
        if path == "/api/meta":
            selftest()
            selftest_ground()
            return _json_response(
                self,
                {
                    "predicate_id": PREDICATE_ID,
                    "predicate_text": PREDICATE_TEXT,
                    "predicate_formula": PREDICATE_FORMULA,
                    "ollama": ollama_reachable(),
                    "model": DEFAULT_MODEL,
                    "cases": load_cases(),
                    "artifacts": {
                        "mock": (ARTIFACTS_DIR / "mock-latest.json").exists(),
                        "live": (ARTIFACTS_DIR / "live-latest.json").exists(),
                    },
                },
            )
        if path in {"/api/results/mock", "/api/results/live"}:
            mode = path.rsplit("/", 1)[-1]
            artifact = ARTIFACTS_DIR / f"{mode}-latest.json"
            if not artifact.exists():
                return _json_response(self, {"error": f"no {mode} artifact yet"}, 404)
            return _json_response(self, json.loads(artifact.read_text(encoding="utf-8")))
        if path.startswith("/api/"):
            return _json_response(self, {"error": "not found"}, 404)
        return super().do_GET()

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        try:
            payload = _read_json(self)
            mode = payload.get("mode") or "mock"
            if mode not in {"mock", "live"}:
                return _json_response(self, {"error": "mode must be mock or live"}, 400)
            if path == "/api/run-one":
                case_id = payload.get("case_id")
                case = next((c for c in load_cases() if c["id"] == case_id), None)
                if not case:
                    return _json_response(self, {"error": f"unknown case {case_id}"}, 404)
                return _json_response(self, run_one(case, mode=mode))
            if path == "/api/run":
                return _json_response(self, run_experiment(mode=mode))
            return _json_response(self, {"error": "not found"}, 404)
        except Exception as exc:  # noqa: BLE001 — lab UI should surface the error
            return _json_response(
                self,
                {"error": str(exc), "trace": traceback.format_exc()},
                500,
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=PORT)
    args = parser.parse_args()
    selftest()
    selftest_ground()
    httpd = ThreadingHTTPServer((HOST, args.port), Handler)
    print(f"N2S lab  http://{HOST}:{args.port}/")
    print(f"Ollama   {'up' if ollama_reachable() else 'down'}  model={DEFAULT_MODEL}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
