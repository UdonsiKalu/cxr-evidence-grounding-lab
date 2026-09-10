#!/usr/bin/env python3
"""Neural-to-neural layer-loss GUI. Port 8259. Frozen replay; optional live GPU."""

from __future__ import annotations

import argparse
import json
import sys
import threading
import traceback
import uuid
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from n2s_lab.n2s_nn_layer_loss import (  # noqa: E402
    analyze_pasted_note,
    build_n2n_lost_review_panel,
    build_panel,
    detect_patch_plan,
    frozen_temporal_scores,
    ground_rules,
    scores_from_live_by_layer,
    selftest,
)
from n2s_lab.paths import ARTIFACTS_DIR, STATIC_DIR  # noqa: E402

HOST = "127.0.0.1"
PORT = 8259
PAGE = "nn-layer-loss.html"

_lock = threading.Lock()
_job: dict = {
    "id": None,
    "state": "idle",
    "step": None,
    "result": None,
    "error": None,
}


PATCH_PANEL_PATH = ARTIFACTS_DIR / "n2s-nn-layer-patch-panel.json"
STACK_PANEL_PATH = ARTIFACTS_DIR / "n2s-nn-layer-patch-l12l16-panel.json"
RESIDUAL_PANEL_PATH = ARTIFACTS_DIR / "n2s-nn-layer-patch-l12l16l20-panel.json"


def _json_response(handler: SimpleHTTPRequestHandler, payload: object, status: int = 200) -> None:
    body = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(body)


def _panel() -> dict:
    return build_panel()


def _read_json(handler: SimpleHTTPRequestHandler) -> dict:
    length = int(handler.headers.get("Content-Length") or "0")
    raw = handler.rfile.read(length) if length else b"{}"
    if not raw:
        return {}
    return json.loads(raw.decode("utf-8"))


def _run_patch(job_id: str, *, stack: bool = False, residual: bool = False) -> None:
    if residual:
        step = "L12+L16+L20 residual d patch (GPU)"
    elif stack:
        step = "L12+L16 d patch (GPU)"
    else:
        step = "first-lost d patch (GPU)"
    with _lock:
        _job["state"] = "running"
        _job["step"] = step
        _job["error"] = None
        _job["result"] = None
    try:
        from n2s_lab.n2s_nn_layer_patch import run_layer_patch_panel, write_readout

        panel = run_layer_patch_panel(unload_after=True, stack=stack, residual=residual)
        if panel.get("ok"):
            write_readout(panel, stack=stack, residual=residual)
        with _lock:
            if _job["id"] == job_id:
                if not panel.get("ok"):
                    _job["state"] = "error"
                    _job["error"] = panel.get("message") or panel.get("error") or str(panel)
                    _job["step"] = "failed"
                else:
                    _job["state"] = "done"
                    _job["step"] = "done"
                    _job["result"] = panel
    except Exception as exc:  # noqa: BLE001
        with _lock:
            if _job["id"] == job_id:
                _job["state"] = "error"
                _job["error"] = f"{exc}\n{traceback.format_exc()}"
                _job["step"] = "failed"


def _run_live(job_id: str, evidence: str, gold: str, case_id: str) -> None:
    with _lock:
        _job["state"] = "running"
        _job["step"] = "loading model / scoring vs frozen d"
        _job["error"] = None
        _job["result"] = None
    try:
        from n2s_lab.n2s_upstream_live import score_note_live

        live = score_note_live(evidence, case_id=case_id, gold=gold, unload_after=True)
        if not live.get("ok"):
            with _lock:
                if _job["id"] == job_id:
                    _job["state"] = "error"
                    _job["error"] = live.get("message") or live.get("error") or str(live)
                    _job["step"] = "failed"
            return
        scores = scores_from_live_by_layer(live.get("by_layer") or {})
        rec = analyze_pasted_note(
            evidence,
            gold=gold,
            scores=scores,
            temporal_scores=frozen_temporal_scores(),
            case_id=case_id,
        )
        rec["live_ok"] = True
        rec["live_best_layer"] = live.get("best_layer")
        with _lock:
            if _job["id"] == job_id:
                _job["state"] = "done"
                _job["step"] = "done"
                _job["result"] = rec
    except Exception as exc:  # noqa: BLE001
        with _lock:
            if _job["id"] == job_id:
                _job["state"] = "error"
                _job["error"] = f"{exc}\n{traceback.format_exc()}"
                _job["step"] = "failed"


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
                panel = _panel()
                return _json_response(
                    self,
                    {
                        "port": PORT,
                        "sibling_n2s": "http://127.0.0.1:8253/",
                        "ground_rules": ground_rules(),
                        "n_cases": panel["n_cases"],
                        "n_with_trace": panel["n_with_trace"],
                        "model_id": panel.get("model_id"),
                        "claim_hygiene": panel.get("claim_hygiene"),
                        "paste": True,
                        "live_needs_gpu": True,
                        "cases": [
                            {
                                "id": c["id"],
                                "category": c["category"],
                                "gold": c["gold"],
                                "side": c["side"],
                                "has_trace": c["has_trace"],
                                "n_lost_steps": c["n_lost_steps"],
                            }
                            for c in panel["cases"]
                        ],
                    },
                )
            if path == "/api/panel":
                return _json_response(self, _panel())
            if path == "/api/analyze":
                q = parse_qs(parsed.query)
                case_id = (q.get("case_id") or [None])[0]
                panel = _panel()
                rec = next((c for c in panel["cases"] if c["id"] == case_id), None)
                if rec is None:
                    return _json_response(self, {"error": f"unknown case {case_id}"}, 404)
                return _json_response(self, rec)
            if path == "/api/job":
                with _lock:
                    return _json_response(self, dict(_job))
            if path == "/api/patch/detect":
                return _json_response(self, detect_patch_plan())
            if path == "/api/patch":
                if PATCH_PANEL_PATH.is_file():
                    return _json_response(
                        self, json.loads(PATCH_PANEL_PATH.read_text(encoding="utf-8"))
                    )
                return _json_response(self, {"ok": False, "error": "no patch panel yet"}, 404)
            if path == "/api/patch-stack":
                if STACK_PANEL_PATH.is_file():
                    return _json_response(
                        self, json.loads(STACK_PANEL_PATH.read_text(encoding="utf-8"))
                    )
                return _json_response(self, {"ok": False, "error": "no L12+L16 panel yet"}, 404)
            if path == "/api/patch-residual":
                if RESIDUAL_PANEL_PATH.is_file():
                    return _json_response(
                        self, json.loads(RESIDUAL_PANEL_PATH.read_text(encoding="utf-8"))
                    )
                return _json_response(
                    self, {"ok": False, "error": "no L12+L16+L20 residual panel yet"}, 404
                )
            if path == "/api/n2n-review":
                return _json_response(self, build_n2n_lost_review_panel())
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
            evidence = str(body.get("evidence") or "").strip()
            gold = str(body.get("gold") or "UNKNOWN")
            case_id = str(body.get("case_id") or "PASTE")
            if path == "/api/preview":
                if not evidence:
                    return _json_response(self, {"error": "paste a note first"}, 400)
                rec = analyze_pasted_note(
                    evidence,
                    gold=gold,
                    scores=None,
                    temporal_scores=frozen_temporal_scores(),
                    case_id=case_id,
                )
                rec["live"] = False
                rec["why"] = "Pasted — not GPU-scored yet. Use Live GPU analyze when ready."
                return _json_response(self, rec)
            if path == "/api/live":
                if not evidence:
                    return _json_response(self, {"error": "paste a note first"}, 400)
                with _lock:
                    if _job["state"] == "running":
                        return _json_response(
                            self,
                            {"error": "a GPU job is already running", "job": dict(_job)},
                            409,
                        )
                    job_id = uuid.uuid4().hex[:10]
                    _job.update(
                        {
                            "id": job_id,
                            "state": "queued",
                            "step": "queued",
                            "result": None,
                            "error": None,
                        }
                    )
                threading.Thread(
                    target=_run_live,
                    args=(job_id, evidence, gold, case_id),
                    daemon=True,
                ).start()
                return _json_response(self, {"ok": True, "job_id": job_id, "state": "queued"})
            if path == "/api/patch":
                with _lock:
                    if _job["state"] == "running":
                        return _json_response(
                            self,
                            {"error": "a GPU job is already running", "job": dict(_job)},
                            409,
                        )
                    job_id = uuid.uuid4().hex[:10]
                    _job.update(
                        {
                            "id": job_id,
                            "state": "queued",
                            "step": "queued first-lost patch",
                            "result": None,
                            "error": None,
                        }
                    )
                threading.Thread(target=_run_patch, args=(job_id,), daemon=True).start()
                return _json_response(self, {"ok": True, "job_id": job_id, "state": "queued"})
            if path == "/api/patch-stack":
                with _lock:
                    if _job["state"] == "running":
                        return _json_response(
                            self,
                            {"error": "a GPU job is already running", "job": dict(_job)},
                            409,
                        )
                    job_id = uuid.uuid4().hex[:10]
                    _job.update(
                        {
                            "id": job_id,
                            "state": "queued",
                            "step": "queued L12+L16 patch",
                            "result": None,
                            "error": None,
                        }
                    )
                threading.Thread(
                    target=_run_patch, args=(job_id,), kwargs={"stack": True}, daemon=True
                ).start()
                return _json_response(self, {"ok": True, "job_id": job_id, "state": "queued"})
            if path == "/api/patch-residual":
                with _lock:
                    if _job["state"] == "running":
                        return _json_response(
                            self,
                            {"error": "a GPU job is already running", "job": dict(_job)},
                            409,
                        )
                    job_id = uuid.uuid4().hex[:10]
                    _job.update(
                        {
                            "id": job_id,
                            "state": "queued",
                            "step": "queued L12+L16+L20 residual patch",
                            "result": None,
                            "error": None,
                        }
                    )
                threading.Thread(
                    target=_run_patch,
                    args=(job_id,),
                    kwargs={"residual": True},
                    daemon=True,
                ).start()
                return _json_response(self, {"ok": True, "job_id": job_id, "state": "queued"})
            return _json_response(self, {"error": "not found"}, 404)
        except Exception as exc:  # noqa: BLE001
            return _json_response(
                self,
                {"error": str(exc), "trace": traceback.format_exc()},
                500,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="N2S neural-to-neural layer-loss GUI")
    parser.add_argument("--port", type=int, default=PORT)
    args = parser.parse_args()
    page = STATIC_DIR / PAGE
    if not page.is_file():
        raise SystemExit(f"missing {page}")
    selftest()
    httpd = ThreadingHTTPServer((HOST, args.port), Handler)
    print(f"N2S neural-to-neural layer-loss  http://{HOST}:{args.port}/")
    print("  Frozen replay of U-A mean_score_d — paste box ready; GPU only on Live / patch")
    print(f"  Sibling N2S lab  http://127.0.0.1:8253/")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
