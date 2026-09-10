#!/usr/bin/env python3
"""Locator v1 GUI. Port 8260. CPU first-break; gated REVIEW only."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from n2s_lab.n2s_locator import (  # noqa: E402
    apply_gated_fix,
    build_locator_panel,
    selftest,
)
from n2s_lab.paths import STATIC_DIR  # noqa: E402

HOST = "127.0.0.1"
PORT = 8260
PAGE = "locator.html"


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
        parsed = urlparse(self.path)
        path = parsed.path
        try:
            if path in {"/", "/index.html"}:
                self.path = "/" + PAGE
                return super().do_GET()
            if path == "/api/meta":
                panel = build_locator_panel()
                return _json_response(
                    self,
                    {
                        "port": PORT,
                        "protocol": panel.get("protocol"),
                        "quest": panel.get("quest"),
                        "n_cases": panel["n_cases"],
                        "n_located": panel["n_located"],
                        "n_pass": panel["n_pass"],
                        "n_incomplete": panel["n_incomplete"],
                        "claim_hygiene": panel.get("claim_hygiene"),
                        "siblings": {
                            "encode": "http://127.0.0.1:8258/",
                            "compute": "http://127.0.0.1:8259/",
                            "translate": "http://127.0.0.1:8253/",
                        },
                        "cases": [
                            {
                                "id": c["id"],
                                "gold": c["gold"],
                                "first_break": c["first_break"].get("surface"),
                                "first_status": c["first_break"].get("status"),
                                "has_trace": c["has_trace"],
                                "applied": bool(c.get("applied_fix")),
                            }
                            for c in panel["cases"]
                        ],
                    },
                )
            if path == "/api/panel":
                return _json_response(self, build_locator_panel())
            if path == "/api/locate":
                q = parse_qs(parsed.query)
                case_id = (q.get("case_id") or [None])[0]
                panel = build_locator_panel()
                rec = next((c for c in panel["cases"] if c["id"] == case_id), None)
                if rec is None:
                    return _json_response(self, {"error": f"unknown case {case_id}"}, 404)
                return _json_response(self, rec)
            if path.startswith("/api/"):
                return _json_response(self, {"error": "not found"}, 404)
            return super().do_GET()
        except Exception as exc:  # noqa: BLE001
            return _json_response(
                self,
                {"error": str(exc), "trace": traceback.format_exc()},
                500,
            )

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        try:
            body = _read_json(self)
            if path == "/api/fix":
                case_id = str(body.get("case_id") or "").strip()
                if not case_id:
                    return _json_response(self, {"error": "case_id required"}, 400)
                out = apply_gated_fix(case_id)
                status = 200 if out.get("ok") else 400
                return _json_response(self, out, status)
            return _json_response(self, {"error": "not found"}, 404)
        except Exception as exc:  # noqa: BLE001
            return _json_response(
                self,
                {"error": str(exc), "trace": traceback.format_exc()},
                500,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="N2S locator v1 GUI")
    parser.add_argument("--port", type=int, default=PORT)
    args = parser.parse_args()
    page = STATIC_DIR / PAGE
    if not page.is_file():
        raise SystemExit(f"missing {page}")
    selftest()
    httpd = ThreadingHTTPServer((HOST, args.port), Handler)
    print(f"N2S diagnostic routing v1  http://{HOST}:{args.port}/")
    print("  First-break routing (not causal). Response = REVIEW / ABSTAIN contain.")
    print("  Siblings  :8258 encode · :8259 compute · :8253 translate")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
