"""Adapter for GPT5 (synchronous, minimal implementation).

This adapter implements build_payload/send_request/parse_response using evaluator.utils.http_post.
If `ModelConfig.base_url` is None the adapter will return a simulated response.
"""
from __future__ import annotations
from typing import Any, Dict, List

from base_api import BaseModel, ChatMessage, ChatResult, ModelConfig
from utils import http_post


class GPT5Adapter(BaseModel):
    def build_payload(self, messages: List[ChatMessage], **kwargs) -> Dict[str, Any]:
        # For GPT5 we send messages as a list of role/content dicts; provider may vary in real API
        return {
            "model": self.model_name,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "max_tokens": self.cfg.max_tokens,
            **(kwargs.get("extra") or {}),
        }

    def send_request(self, payload: Dict[str, Any]) -> Any:
        return http_post(self.cfg.base_url, json_payload=payload, timeout=self.cfg.timeout, api_key=self.cfg.api_key)

    def parse_response(self, resp: Any) -> ChatResult:
        # Accept simulated dicts or standard provider formats
        if isinstance(resp, dict):
            if resp.get("simulated"):
                return ChatResult(text=resp.get("text", ""), raw=resp)
            # Common OpenAI-like response
            choices = resp.get("choices")
            if choices and isinstance(choices, list):
                text = choices[0].get("message", {}).get("content") if isinstance(choices[0].get("message"), dict) else choices[0].get("text")
                return ChatResult(text=text or "", raw=resp)
            # Generic fallback
            return ChatResult(text=resp.get("text") or str(resp), raw=resp)
        return ChatResult(text=str(resp), raw={"raw": resp})
