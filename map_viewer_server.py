#!/usr/bin/env python3
"""V1 Dual Translate map viewer. Port 8263. Read-only; does not expand :8260."""

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

from n2s_lab.n2s_map_viewer import cached_panel, selftest  # noqa: E402
from n2s_lab.paths import STATIC_DIR  # noqa: E402

HOST = "127.0.0.1"
PORT = 8263
PAGE = "map-viewer.html"


def _json_response(handler: SimpleHTTPRequestHandler, payload: object, status: int = 200) -> None:
    body = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(body)


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
                panel = cached_panel()
                return _json_response(
                    self,
                    {
                        "port": PORT,
                        "claim": panel["claim"],
                        "n_cases": panel["n_cases"],
                        "n_miss_none": panel["n_miss_none"],
                        "n_miss_apply_all": panel["n_miss_apply_all"],
                        "cell_help": panel["cell_help"],
                        "not": panel["not"],
                        "cases": [
                            {
                                "id": c["id"],
                                "gold": c["gold"],
                                "role": c["role"],
                                "slice": c["slice"],
                                "selector": c["selector"],
                                "oracle": c["oracle"],
                                "match_none": c["match_none"],
                                "match_selector": c["match_selector"],
                                "match_apply_all": c["match_apply_all"],
                                "leftover_tags": c["leftover_tags"],
                            }
                            for c in panel["cases"]
                        ],
                    },
                )
            if path == "/api/case":
                q = parse_qs(parsed.query)
                case_id = (q.get("id") or [None])[0]
                panel = cached_panel()
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


def main() -> None:
    parser = argparse.ArgumentParser(description="V1 Dual Translate map viewer :8263")
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--host", default=HOST)
    parser.add_argument("--port", type=int, default=PORT)
    args = parser.parse_args()
    if args.selftest:
        selftest()
        cached_panel()
        print("selftest ok")
        return
    cached_panel()
    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"map viewer http://{args.host}:{args.port}/  (read-only; not :8260)", flush=True)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
