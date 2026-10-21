"""Adapter that uses the OpenAI Python client.

This file is based heavily on the existing `general_adapter.py` implementation but
replaces `send_request` to use the OpenAI Python client API rather than direct
HTTP POST via `requests`.

Class: OpenAIAdaptor
"""
from __future__ import annotations
from typing import Any, Dict, List

from openai import OpenAI
import json
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

from base_api import BaseModel, ChatMessage, ChatResult


class OpenAIAdapter(BaseModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        # keep the same strict system prompt used by other adapters
        self.cfg.extra['system'] = """You are an evaluation model for spatial reasoning QA tasks.
Your job is to produce deterministic, machine-parseable outputs for spatial/function reasoning questions (text + optional images).

RESPONSE FORMAT (strict)

1) Your final output MUST consist of exactly one optional <think> block followed by exactly one triple-backtick fenced JSON block labeled `json`.
     - Allowed structure:
         <think>
         ...one-paragraph deterministic reasoning, include inferred task and function_type...
         </think>

         ```json
         <value>
         ```

     - The fenced JSON block must contain only the JSON data that corresponds to the example `value` field in the task descriptions below (i.e., the consumer expects the JSON inside the fenced block to be directly parseable and correspond to the `value` examples). Do NOT add wrappers, status fields, or extra keys outside that `value` structure.

2) Replace placeholder tokens like "category_function" with a concrete label (e.g., "pan_pour_active"). If you cannot determine a concrete function, use "default".

COORDINATES & VECTORS

- 2D points: [x, y] are image pixel coordinates, origin (0,0) top-left, x right, y down.
- 3D points and 3D vectors: coordinates are in the camera coordinate system.
- Vectors must be normalized to unit length. If you normalize a computed vector, state the original length in <think>.

INFERENCE RULE

- From the human message (and any attached meta/image), infer the task key (one of the METRIC_BY_TASK keys) and whether it is 2D vs 3D and single vs multi. Print the inferred `task` and `function_type` inside <think> with a one-line rationale.

STANDARDIZED TASK VALUE FORMATS (these are the exact JSON shapes you MUST output inside the fenced JSON block)

(omitted here for brevity in the runtime copy; the adapter keeps the same system prompt structure used elsewhere)
"""
        # keep a copy of the base system prompt for per-call augmentation
        # self._system_base = self.cfg.extra['system']
        self._system_base = (
            "You are an evaluation model for spatial reasoning QA tasks.\n"
            "Produce deterministic, machine-parseable outputs for spatial/function reasoning questions.\n\n"
            "RESPONSE CONTRACT:\n"
            "- Your final output must consist of exactly one optional <think></think> block followed by one triple-backtick fenced JSON block labeled `json` without any additional texts or parts. Your should put all your answer except json in the <think></think> block to keep the format clean.\n"
            # "- The fenced JSON block must contain only the `EXAMPLE VALUE` structure (no wrappers).\n"
            # "- The JSON output must be strictly same as the EXAMPLE VALUE's format and have at least one answer in it. This is very important!\n"
            "- Copy the `EXAMPLE VALUE` into the JSON block to keep the format same, then change your answer in it (Ensure at least one meaningful answer).\n"
            # "- At least give one answer in JSON block.\n"
            "- The <think> block must include the inferred `task` and `function_type` and a one-line rationale. You should think step by step and give your thinking process in this <think> block.\n"
            "- We focus on robotics manipulation, you should be careful with your answer.\n"
            "- When you answer 3D questions or obj2cam distance or obj2obj distance, the output unit is `m`.\n"
            "\n"
            "When calling this adapter include the parameter `task_name` so the adapter can append the exact expected JSON `value` example for that task.\n"
        )

        # Load per-task response-format mapping (JSON) if available, same pattern as GeneralAdapter
        from pathlib import Path
        try:
            _TASK_FORMATS_PATH = Path(__file__).parent.parent.joinpath("configs", "task_response_formats.json")
            if _TASK_FORMATS_PATH.exists():
                with open(_TASK_FORMATS_PATH, "r", encoding="utf-8") as _f:
                    self._task_formats = json.load(_f)
            else:
                self._task_formats = {}
        except Exception:
            self._task_formats = {}

    def build_payload(self, messages: List[ChatMessage], **kwargs) -> Dict[str, Any]:
        # Initialize client for this call (kept for compatibility; real request is done in send_request)
        self.client = OpenAI(api_key=self.cfg.api_key, base_url=self.cfg.base_url)
        # Build a per-call system prompt by appending the task-specific value example when task_name provided
        task_name = kwargs.get("task_name")
        if task_name:
            task_entry = self._task_formats.get(task_name, {})
            task_desc = task_entry.get("description", "")
            if task_desc and "(Note: example value below is wrapped in a ```json fenced code block)" not in task_desc:
                task_desc = task_desc + " (Note: example value below is wrapped in a ```json fenced code block.)"
            task_example = task_entry.get("value_example")
            try:
                example_text = json.dumps(task_example, ensure_ascii=False, indent=2)
            except Exception:
                example_text = str(task_example)
            example_text_fenced = "```json\n" + example_text + "\n```"
            system_full = self._system_base + f"\nREQUIRED VALUE FORMAT FOR TASK {task_name}:\n{task_desc}\nEXAMPLE VALUE:\n" + example_text_fenced
        else:
            system_full = self._system_base

        logger.info("Built system prompt for task '%s': %s\n\n\n", task_name, system_full)

        out_messages: List[Dict[str, Any]] = []
        # If caller did not include a system message, inject one with the composed system_full
        if not any(getattr(m, "role", None) == "system" for m in messages):
            out_messages.append({"role": "system", "content": system_full})
        for m in messages:
            md: Dict[str, Any] = {"role": m.role}

            # If the message content is already a list/structure, pass it through
            if isinstance(m.content, (list, tuple)):
                md["content"] = list(m.content)
            else:
                # Build a content array with a text element and optionally an image element
                content_items: List[Dict[str, Any]] = [{"type": "text", "text": str(m.content)}]

                meta = getattr(m, "meta", None)
                if isinstance(meta, dict):
                    # Support multiple images attached as a list under 'images'
                    imgs = meta.get("images")
                    if isinstance(imgs, (list, tuple)) and len(imgs) > 0:
                        for img_entry in imgs:
                            if not isinstance(img_entry, dict):
                                continue
                            img_b64 = img_entry.get("image_b64")
                            img_mime = img_entry.get("image_mime")
                            if img_b64:
                                mime = img_mime or "image/jpeg"
                                data_uri = f"data:{mime};base64,{img_b64}"
                                content_items.append({"type": "image_url", "image_url": {"url": data_uri}})
                    else:
                        # Backwards compatible single-image keys
                        img_b64 = meta.get("image_b64")
                        img_mime = meta.get("image_mime")
                        if img_b64:
                            mime = img_mime or "image/jpeg"
                            data_uri = f"data:{mime};base64,{img_b64}"
                            content_items.append({"type": "image_url", "image_url": {"url": data_uri}})

                md["content"] = content_items

            out_messages.append(md)

        return {
            "model": self.model_name,
            "messages": out_messages,
            "max_tokens": self.cfg.max_tokens,
        }

    def send_request(self, payload: Dict[str, Any]) -> Any:
        """Send a request using the OpenAI Python client and return a plain JSON-serializable dict.

        This implementation prefers the OpenAI Python client instead of raw HTTP. It attempts to
        convert the client response to a dict (via .to_dict() if present) and falls back to string
        decoding. On error it returns a dict with an `error` key and logs the exception.
        """
        # Create client (use cfg values if present)
        try:
            self.client = OpenAI(
                api_key=self.cfg.api_key,
                base_url=self.cfg.base_url,
                max_retries=0,
            )
        except Exception:
            # defensive: still try to proceed without explicit client construction
            self.client = None

        body = dict(payload)
        body.setdefault("stream", False)

        try:
            # Use the chat completions create method on the OpenAI client.
            # The OpenAI Python client used in this repo exposes `chat.completions.create`.
            if self.client is not None and hasattr(self.client, "chat"):
                resp = self.client.chat.completions.create(**body)
            else:
                # if client is not available for some reason, raise to go to exception handler
                raise RuntimeError("OpenAI client not available")

            # Convert to plain python structures
            if hasattr(resp, "to_dict"):
                return resp.to_dict()

            # Some client versions stringify responses; try json.loads fallback
            try:
                return json.loads(str(resp))
            except Exception:
                return {"raw": str(resp)}

        except Exception as e:
            logging.exception("OpenAI API call failed: %s", e)
            return {"error": str(e)}

    def parse_response(self, resp: Any) -> ChatResult:
        print("AAAAAA",resp)
        if isinstance(resp, dict):
            if resp.get("simulated"):
                return ChatResult(text=resp.get("text", ""), raw=resp)
            # Try common fields
            if "output" in resp:
                return ChatResult(text=resp.get("output"), raw=resp)
            if "choices" in resp:
                c = resp["choices"][0]
                text = c.get("message", {}).get("content") if isinstance(c.get("message"), dict) else c.get("text")
                if isinstance(text, dict):
                    text = text.get("text") or str(text)
                return ChatResult(text=text or "", raw=resp)
            return ChatResult(text=resp.get("text") or str(resp), raw=resp)
        return ChatResult(text=str(resp), raw={"raw": resp})
