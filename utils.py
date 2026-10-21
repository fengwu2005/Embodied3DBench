"""Utility helpers for HTTP calls, retry, and simulation for adapters.

Provides a synchronous http_post helper that will simulate a response when `base_url` is None.
Uses `requests` and a small retry/backoff.
"""
from __future__ import annotations
import time
import logging
from typing import Any, Dict, Optional

import requests

logger = logging.getLogger(__name__)


def retry(func, attempts: int = 3, backoff: float = 0.5):
    last_exc = None
    for i in range(attempts):
        try:
            return func()
        except Exception as e:
            last_exc = e
            logger.debug("Attempt %s failed: %s", i + 1, e)
            time.sleep(backoff * (2 ** i))
    raise last_exc


def http_post(
    url: Optional[str],
    json_payload: Dict[str, Any],
    headers: Optional[Dict[str, str]] = None,
    timeout: int = 30,
    api_key: Optional[str] = None,
) -> Any:
    """Synchronous POST helper.

    If `url` is None, returns a simulated response dict with a `text` field for offline/demo use.
    """
    if url is None:
        # Simulate a provider response for local testing/demo: echo back a short text.
        prompt = json_payload.get("prompt") or json_payload.get("messages")
        if isinstance(prompt, list):
            joined = " ".join([m.get("content", "") if isinstance(m, dict) else str(m) for m in prompt])
        else:
            joined = str(prompt)
        return {"simulated": True, "text": f"[SIMULATED RESPONSE] Echo: {joined[:400]}"}

    def _call():
        hdrs = dict(headers or {})
        if api_key:
            # Common pattern: Authorization: Bearer <key>
            hdrs.setdefault("Authorization", f"Bearer {api_key}")
        resp = requests.post(url, json=json_payload, headers=hdrs, timeout=timeout)
        resp.raise_for_status()
        # Attempt JSON first, but return raw text if not JSON
        try:
            return resp.json()
        except Exception:
            return {"text": resp.text}

    return retry(_call, attempts=3, backoff=0.5)
