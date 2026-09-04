from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from typing import Any

DEFAULT_HOST = os.environ.get("N2S_OLLAMA_HOST", "http://127.0.0.1:11434")
DEFAULT_MODEL = os.environ.get("N2S_OLLAMA_MODEL", "llama3:8b-instruct-q4_0")
DEFAULT_TIMEOUT = int(os.environ.get("N2S_OLLAMA_TIMEOUT", "120"))
# Ollama's keep_alive=0 can return before the runner actually leaves /api/ps + VRAM.
UNLOAD_VERIFY_TIMEOUT_S = float(os.environ.get("N2S_OLLAMA_UNLOAD_VERIFY_S", "45"))


def ollama_reachable(host: str = DEFAULT_HOST, timeout: int = 3) -> bool:
    try:
        with urllib.request.urlopen(f"{host}/api/tags", timeout=timeout) as resp:
            return resp.status == 200
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def _chat_raw(
    prompt: str,
    *,
    system: str | None = None,
    model: str = DEFAULT_MODEL,
    host: str = DEFAULT_HOST,
    timeout: int = DEFAULT_TIMEOUT,
    num_predict: int = 400,
    format_json: bool = False,
) -> str:
    """Low-level Ollama /api/chat. format_json=True only forces *some* JSON, not our schema."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    payload: dict[str, Any] = {
        "model": model,
        "stream": False,
        "messages": messages,
        "options": {"temperature": 0, "num_predict": num_predict},
    }
    if format_json:
        payload["format"] = "json"
    req = urllib.request.Request(
        f"{host}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    return (body.get("message") or {}).get("content") or ""


def chat_text(
    prompt: str,
    *,
    system: str | None = None,
    model: str = DEFAULT_MODEL,
    host: str = DEFAULT_HOST,
    timeout: int = DEFAULT_TIMEOUT,
    num_predict: int = 500,
) -> str:
    """Free-text generation — no format:json. Used by optional Phase-1 Condition B/D analysis."""
    return _chat_raw(
        prompt,
        system=system,
        model=model,
        host=host,
        timeout=timeout,
        num_predict=num_predict,
        format_json=False,
    ).strip()


def chat_json(
    prompt: str,
    *,
    system: str | None = None,
    model: str = DEFAULT_MODEL,
    host: str = DEFAULT_HOST,
    timeout: int = DEFAULT_TIMEOUT,
    num_predict: int = 400,
) -> dict[str, Any]:
    """JSON mode via Ollama format:json + prompt schema + parse. Not field-level constrained decoding."""
    content = _chat_raw(
        prompt,
        system=system,
        model=model,
        host=host,
        timeout=timeout,
        num_predict=num_predict,
        format_json=True,
    ) or "{}"
    return parse_json_object(content)


def parse_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        data = json.loads(text[start : end + 1])
        if isinstance(data, dict):
            return data
    raise ValueError(f"Model did not return a JSON object: {text[:240]!r}")


def list_loaded_models(host: str = DEFAULT_HOST, timeout: int = 5) -> list[dict[str, Any]]:
    """Return runners currently in VRAM via GET /api/ps."""
    try:
        with urllib.request.urlopen(f"{host}/api/ps", timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8") or "{}")
        models = data.get("models") or []
        return [m for m in models if isinstance(m, dict)]
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError):
        return []


def model_is_loaded(model: str, host: str = DEFAULT_HOST) -> bool:
    """True if *model* (exact or prefix match) appears in /api/ps."""
    names = [(m.get("name") or m.get("model") or "") for m in list_loaded_models(host=host)]
    return any(n == model or n.startswith(model) or model.startswith(n) for n in names if n)


def _post_json(
    url: str,
    payload: dict[str, Any],
    *,
    timeout: int,
) -> dict[str, Any]:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8") or "{}"
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"raw": raw[:200]}


def _unload_via_generate(model: str, host: str, timeout: int) -> dict[str, Any]:
    return _post_json(
        f"{host}/api/generate",
        {"model": model, "prompt": "", "keep_alive": 0, "stream": False},
        timeout=timeout,
    )


def _unload_via_chat(model: str, host: str, timeout: int) -> dict[str, Any]:
    # Chat is how Track A loads the runner; mirror unload on the same API.
    return _post_json(
        f"{host}/api/chat",
        {
            "model": model,
            "messages": [{"role": "user", "content": ""}],
            "keep_alive": 0,
            "stream": False,
            "options": {"num_predict": 1},
        },
        timeout=timeout,
    )


def _unload_via_cli_stop(model: str, timeout: int = 30) -> dict[str, Any]:
    exe = shutil.which("ollama")
    if not exe:
        return {"ok": False, "skipped": True, "reason": "ollama CLI not on PATH"}
    try:
        proc = subprocess.run(
            [exe, "stop", model],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return {
            "ok": proc.returncode == 0,
            "returncode": proc.returncode,
            "stdout": (proc.stdout or "")[:200],
            "stderr": (proc.stderr or "")[:200],
        }
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"ok": False, "error": str(exc)}


def unload_model(
    model: str = DEFAULT_MODEL,
    *,
    host: str = DEFAULT_HOST,
    timeout: int = 30,
    verify_timeout_s: float | None = None,
    poll_s: float = 0.4,
) -> dict[str, Any]:
    """Drop a loaded Ollama model from VRAM and **verify** it left /api/ps.

    Ollama's keep_alive=0 often returns ``done_reason=unload`` while the runner
    (and ~20 GiB) is still resident for 1–2+ seconds. Callers that check VRAM
    immediately then falsely think unload failed or GPU is busy.

    Strategy (stop when /api/ps no longer lists the model):
      1. POST /api/generate keep_alive=0
      2. POST /api/chat keep_alive=0
      3. ``ollama stop MODEL``
      4. Poll /api/ps until gone or verify_timeout
    """
    verify_s = (
        UNLOAD_VERIFY_TIMEOUT_S if verify_timeout_s is None else float(verify_timeout_s)
    )
    attempts: list[dict[str, Any]] = []
    t0 = time.monotonic()

    if not model_is_loaded(model, host=host):
        return {
            "ok": True,
            "model": model,
            "verified": True,
            "already_unloaded": True,
            "elapsed_s": 0.0,
            "attempts": attempts,
        }

    methods = (
        ("generate_keep_alive_0", lambda: _unload_via_generate(model, host, timeout)),
        ("cli_stop", lambda: _unload_via_cli_stop(model, timeout=timeout)),
        # Chat unload last — only reached if still resident; avoids re-loading after generate cleared.
        ("chat_keep_alive_0", lambda: _unload_via_chat(model, host, timeout)),
    )

    last_error: str | None = None
    for name, fn in methods:
        try:
            body = fn()
            attempts.append({"method": name, "ok": True, "response": body})
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = str(exc)
            attempts.append({"method": name, "ok": False, "error": str(exc)})

        # Poll after each method — generate often needs ~1–2s after HTTP returns.
        deadline = time.monotonic() + max(2.0, verify_s / 3.0)
        while time.monotonic() < deadline:
            if not model_is_loaded(model, host=host):
                return {
                    "ok": True,
                    "model": model,
                    "verified": True,
                    "elapsed_s": round(time.monotonic() - t0, 3),
                    "attempts": attempts,
                    "method_that_cleared": name,
                }
            time.sleep(poll_s)

    # Final wait window (all methods already fired).
    while time.monotonic() - t0 < verify_s:
        if not model_is_loaded(model, host=host):
            return {
                "ok": True,
                "model": model,
                "verified": True,
                "elapsed_s": round(time.monotonic() - t0, 3),
                "attempts": attempts,
                "method_that_cleared": "poll_after_all",
            }
        time.sleep(poll_s)

    still = [
        (m.get("name") or m.get("model"))
        for m in list_loaded_models(host=host)
    ]
    return {
        "ok": False,
        "model": model,
        "verified": False,
        "elapsed_s": round(time.monotonic() - t0, 3),
        "attempts": attempts,
        "still_loaded": still,
        "error": last_error
        or f"model still in /api/ps after {verify_s:.0f}s: {still}",
    }
