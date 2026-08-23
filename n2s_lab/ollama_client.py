from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any

DEFAULT_HOST = os.environ.get("N2S_OLLAMA_HOST", "http://127.0.0.1:11434")
DEFAULT_MODEL = os.environ.get("N2S_OLLAMA_MODEL", "llama3:8b-instruct-q4_0")
DEFAULT_TIMEOUT = int(os.environ.get("N2S_OLLAMA_TIMEOUT", "120"))


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
