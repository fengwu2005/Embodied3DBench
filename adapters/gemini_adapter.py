"""Adapter for Gemini-2.5-pro (synchronous, minimal implementation).
"""
from __future__ import annotations
from typing import Any, Dict, List

from base_api import BaseModel, ChatMessage, ChatResult
from utils import http_post

class GeminiAdapter(BaseModel):
    def build_payload(self, messages: List[ChatMessage], **kwargs) -> Dict[str, Any]:
        # Use a simple prompt concatenation commonly used for some Gemini integrations
        prompt = "\n".join([f"{m.role}: {m.content}" for m in messages])
        return {
            "model": self.model_name,
            "prompt": prompt,
            "max_tokens": self.cfg.max_tokens,
        }

    def send_request(self, payload: Dict[str, Any]) -> Any:
        return http_post(self.cfg.base_url, json_payload=payload, timeout=self.cfg.timeout, api_key=self.cfg.api_key)

    def parse_response(self, resp: Any) -> ChatResult:
        if isinstance(resp, dict):
            if resp.get("simulated"):
                return ChatResult(text=resp.get("text", ""), raw=resp)
            # Google-style Gemini might return 'candidates' or 'outputText'
            text = resp.get("outputText") or resp.get("text")
            if not text and "candidates" in resp:
                c = resp["candidates"][0]
                text = c.get("content") or c.get("text")
            return ChatResult(text=text or "", raw=resp)
        return ChatResult(text=str(resp), raw={"raw": resp})
