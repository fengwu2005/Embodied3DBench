"""Adapter for GPT-4o (synchronous, minimal implementation).

This adapter formats messages so that when a ChatMessage contains image data in
its .meta (image_b64, image_mime, image_filename), the outgoing payload uses a
content array where one element is text and another is an image_url entry whose
url is a data URI: data:<mime>;base64,<b64> — this matches the example usage of
the OpenAI client.
"""
from __future__ import annotations
from typing import Any, Dict, List

from openai import OpenAI
import requests
import logging
import json as _json
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

from base_api import BaseModel, ChatMessage, ChatResult
import uuid
import time

class GeneralAdapter(BaseModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        from pathlib import Path
        import json as _json

        # Load per-task response-format mapping (JSON) if available
        _TASK_FORMATS_PATH = Path(__file__).parent.parent.joinpath("configs", "task_response_formats.json")
        try:
            if _TASK_FORMATS_PATH.exists():
                with open(_TASK_FORMATS_PATH, "r", encoding="utf-8") as _f:
                    self._task_formats = _json.load(_f)
            else:
                self._task_formats = {}
        except Exception:
            self._task_formats = {}

        # Minimal base system prompt. Per-task expected 'value' shapes will be
        # appended at call time when build_payload receives a task_name kwarg.
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

    def build_payload(self, messages: List[ChatMessage], **kwargs) -> Dict[str, Any]:
        # Initialize client for this call
        self.client = OpenAI(api_key=self.cfg.api_key, base_url=self.cfg.base_url)
        # Build a per-call system prompt by appending the task-specific value example
        task_name = kwargs.get("task_name")
        if task_name:
            task_entry = self._task_formats.get(task_name, {})
            # Append an explicit note that the example will be wrapped in a ```json fenced block
            task_desc = task_entry.get("description", "")
            if task_desc and "(Note: example value below is wrapped in a ```json fenced code block)" not in task_desc:
                task_desc = task_desc + " (Note: example value below is wrapped in a ```json fenced code block.)"
            task_example = task_entry.get("value_example")
            try:
                example_text = _json.dumps(task_example, ensure_ascii=False, indent=2)
            except Exception:
                example_text = str(task_example)
            # Wrap the example in a labeled fenced JSON block so the model sees an explicit code block
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

        # logger.info(str(out_messages))

        return {
            "model": self.model_name,
            "messages": out_messages,
            "max_tokens": self.cfg.max_tokens,
        }

    def send_request(self, payload: Dict[str, Any]) -> Any:
        # Use HTTP POST to the configured base_url (or default JDCloud API)
        url = getattr(self.cfg, "base_url", None) or "http://ai-api.jdcloud.com/v1/chat/completions"
        ouid = str(uuid.uuid4())
        print(ouid)
        headers = {
            "Content-Type": "application/json",
            "Trace-Id": ouid
        }
        # print(headers["Trace-Id"])
        if getattr(self.cfg, "api_key", None):
            headers["Authorization"] = f"Bearer {self.cfg.api_key}"

        # Attach some default fields expected by the new endpoint if not present
        body = dict(payload)
        # defaults (caller can include overrides in payload if desired)
        # body.setdefault("thinking_budget", 1024)
        # body.setdefault("thinking", True)
        body.setdefault("stream", False)

        try:
            if payload["model"] in [] or 1:
                api_key = self.cfg.api_key
                client = OpenAI(
                    api_key=api_key,
                    base_url=url,
                    default_headers={"requestId": ouid},
                    max_retries=0,
                )
                # time.sleep(1)
                resp=client.chat.completions.create(
                    stream=False,
                    model=payload["model"],
                    messages=payload["messages"],
                    # max_tokens=payload["max_tokens"],
                )
                # print("FLLLLLLLLLLLLLLLLLLLLLLLLLLL")
                # time.sleep(1)
                # print("AAAAAAAAAAAAAAAAAAAAAAAAAAA")
                return resp
            resp = requests.post(url, headers=headers, json=body, timeout=60)
            resp.raise_for_status()
            j = resp.json()
            # Return full json so parse_response can handle it
            return j
        except Exception as e:
            logging.exception("HTTP request to model API failed: %s", e)
            # return a dict similar to the simulated response shape to keep parse_response tolerant
            return {"error": str(e)}

    def parse_response(self, resp: Any) -> ChatResult:
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
        try:
            # print("OOOOOK")
            resp_dict = resp.to_dict()
            # print(resp_dict)
            text = resp_dict["choices"][0]["message"]["content"]
            if isinstance(text, list):
                text = text[0]
            if isinstance(text, dict) and "text" in text:
                text = text["text"]
            return ChatResult(text=text, raw=resp_dict)
        except:
            return ChatResult(text=str(resp), raw={"raw": resp})
