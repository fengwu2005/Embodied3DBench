from __future__ import annotations

import json as _json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from PIL import Image

from base_api import BaseModel, ChatMessage, ChatResult

from configs.get_system_prompt import get_system_prompt
try:
    from esi_vlm.model import ProQwen
except ImportError:
    ProQwen = None

from adapters.tools.get_chat_template import get_chat_template

logger = logging.getLogger(__name__)


class Qwen_depth_Adapter(BaseModel):
    """Local (offline) adapter for ProQwen (Qwen2.5-VL + depth fusion).

    Contract with the evaluator:
    - Synchronous only.
    - Inputs: a list of ChatMessage, optionally with image path under message.meta['image_path'].
    - Output: ChatResult.text is the decoded generation (expected to contain a ```json fenced block).
    """

    def __init__(self, cfg):
        super().__init__(cfg)

        if ProQwen is None:
            raise ImportError("Install the sibling ESI-VLA repository to use qwen-depth-encoder.")

        extra = cfg.extra or {}
        ckpt_type = cfg.extra.get("ckpt_type", None)
        self.ckpt_type = ckpt_type
        qwen_model_path = extra.get("qwen_model_path")
        model_path = extra.get("model_path")
        if not qwen_model_path:
            # raise ValueError("Missing cfg.extra['model_path'] for ProQwenAdapter")
            pass
        if not os.path.exists(qwen_model_path):
            raise ValueError(f"Model path does not exist: {qwen_model_path}")

        depth_backbone_path = extra.get("depth_backbone_path", "./depth-anything-init/da3_backbone.pth")
        if depth_backbone_path and not os.path.exists(depth_backbone_path):
            raise ValueError(f"Depth backbone path does not exist: {depth_backbone_path}")

        # Build ProQwen (it internally loads Qwen2.5-VL + processor)
        self.model_path = qwen_model_path
        print("enable_fusion: ", extra.get("enable_fusion", True))
        print("change_chat_template: ", extra.get("change_chat_template", False))
        self.proqwen = ProQwen(
            qwen_model_path=qwen_model_path,
            da3_offline_bundle_path=depth_backbone_path,
            enable_fusion=extra.get("enable_fusion", True),
            change_chat_template=extra.get("change_chat_template", False),
            vlm_device_map=None,
            debug_model=False,
        )
        # Keep a shortcut for evaluator-style string/template formatting.
        self.processor = self.proqwen.vlm_processor
        
        if self.ckpt_type == 'starvla':
            self.processor.tokenizer.chat_template = (
                "{% for message in messages %}"
                "{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n'}}"
                "{% endfor %}"
                "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
            )
            self._im_end_id = self.processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
    
        self.device = next(self.proqwen.vlm_model.parameters()).device
        
        if model_path and os.path.exists(model_path) and ckpt_type == 'starvla':
            print("loading starvla checkpoint")
            state = torch.load(model_path, map_location="cpu")
            # simplest: only load VLM part from StarVLA checkpoint
            # ckpt key example: qwen_vl_interface.model.model.language_model... / qwen_vl_interface.model.model.visual...
            prefix = "qwen_vl_interface.model."
            if isinstance(state, dict):
                filtered = {
                    k[len(prefix):]: v
                    for k, v in state.items()
                    if isinstance(k, str) and k.startswith(prefix)
                }
                if len(filtered) > 0:
                    state = filtered
            msg = self.proqwen.load_state_dict(state, strict=False)
            print(f"ckpt load done. missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)}")
        else:
            if os.path.exists(model_path):
                print("loading checkpoint")
                state = torch.load(model_path, map_location="cpu")
                self.proqwen.load_state_dict(state, strict=True)
            else:
                print("check checkpoint existance")

        # IMPORTANT: when called by eval_all / subprocess with per-task CUDA_VISIBLE_DEVICES,
        # make sure the model tensors end up on *this* process' current GPU.
        # Under CUDA_VISIBLE_DEVICES="X", torch.cuda.current_device() is usually 0.
        if torch.cuda.is_available():
            device = torch.device("cuda", torch.cuda.current_device())
            self.proqwen = self.proqwen.to(device)

        
        # if not cfg.extra.get("enable_thinking", True):
        #     print("change the chat_template")
        #     self.processor.tokenizer.chat_template = get_chat_template()

        # self._task_formats = self._load_task_formats()
        # self._system_base = (
        #     "You are an evaluation model for spatial reasoning QA tasks.\n"
        #     "Produce deterministic, machine-parseable outputs for spatial/function reasoning questions.\n\n"
        #     "RESPONSE CONTRACT:\n"
        #     "- Your final output must consist of exactly one optional <think></think> block followed by one triple-backtick fenced JSON block labeled `json` without any additional texts or parts.\n"
        #     "- Put all non-JSON text inside <think></think> to keep the format clean.\n"
        #     "- Copy the EXAMPLE VALUE into the JSON block to keep the format, then edit values to be your answer (ensure at least one meaningful answer).\n"
        #     "- The <think> block must include the inferred `task` and `function_type` and a one-line rationale.\n"
        #     "- When you answer 3D questions or distance, the output unit is m.\n\n"
        #     "When calling this adapter include the parameter `task_name` so the adapter can append the exact expected JSON `value` example for that task.\n"
        # )

    # def _load_task_formats(self) -> Dict[str, Any]:
    #     # Keep same behavior as other adapters: ../configs/task_response_formats.json
    #     task_formats_path = Path(__file__).parent.parent.joinpath("configs", "task_response_formats.json")
    #     try:
    #         if task_formats_path.exists():
    #             with open(task_formats_path, "r", encoding="utf-8") as f:
    #                 return _json.load(f)
    #     except Exception:
    #         logger.exception("Failed to load task_response_formats.json")
    #     return {}

    # def _build_system_prompt(self, task_name: Optional[str]) -> str:
    #     if not task_name:
    #         return self._system_base

    #     task_entry = self._task_formats.get(task_name, {}) if isinstance(self._task_formats, dict) else {}
    #     task_desc = task_entry.get("description", "")
    #     task_example = task_entry.get("value_example")
    #     try:
    #         example_text = _json.dumps(task_example, ensure_ascii=False, indent=2)
    #     except Exception:
    #         example_text = str(task_example)

    #     example_text_fenced = "```json\n" + example_text + "\n```"
    #     return (
    #         self._system_base
    #         + f"\nREQUIRED VALUE FORMAT FOR TASK {task_name}:\n{task_desc}\nEXAMPLE VALUE:\n"
    #         + example_text_fenced
    #     )

    def _find_first_image_path(self, messages: List[ChatMessage]) -> Optional[str]:
        for m in messages:
            meta = getattr(m, "meta", None)
            if isinstance(meta, dict):
                p = meta.get("image_path")
                if p:
                    return p
                imgs = meta.get("images")
                if isinstance(imgs, list) and imgs:
                    first = imgs[0]
                    if isinstance(first, dict) and first.get("image_path"):
                        return first["image_path"]
        return None

    def build_payload(self, messages: List[ChatMessage], **kwargs) -> Dict[str, Any]:
        task_name = kwargs.get("task_name")
        system_full = get_system_prompt(task_name)

        out_messages: List[Dict[str, Any]] = []
        if not any(getattr(m, "role", None) == "system" for m in messages):
            out_messages.append(
                {
                    "role": "system",
                    "content": [{"type": "text", "text": system_full}],
                }
            )

        # image_path = self._find_first_image_path(messages)
        # if image_path is not None and not os.path.exists(image_path):
        #     raise ValueError(f"image_path does not exist: {image_path}")
        raw_images = []
        for m in messages:
            content_items: List[Dict[str, Any]] = [{"type": "text", "text": str(m.content)}]
            # Attach a single image only once (common eval setting)
            meta = getattr(m, "meta", None)
            if isinstance(meta, dict):
                # Support multiple images attached as a list under 'images'
                imgs = meta.get("images")
                # if image_url==None:
                #     content_items.append({"image": meta.get("image_path", None)})
                #     image_url = meta.get("image_path", None)
                if isinstance(imgs, (list, tuple)) and len(imgs) > 0:
                    for img_entry in imgs:
                        if not isinstance(img_entry, dict):
                            continue
                        img_url = img_entry.get("resolved_path")
                        if img_url:
                            content_items.append({"type": "image", "image": img_url})
                            raw_images.append(Image.open(img_url).convert("RGB"))
                # if image_path is not None and m.role == "user":
                #     content_items.append({"type": "image", "image": image_path})
            out_messages.append({"role": m.role, "content": content_items})

        return {
            "messages": out_messages,
            "image_path": self._find_first_image_path(messages),
            "raw_images": [raw_images] if raw_images else None,
            "max_tokens": int(kwargs.get("max_tokens") or self.cfg.max_tokens or 1024),
            "do_sample": bool(kwargs.get("do_sample", False)),
            "temperature": float(kwargs.get("temperature", 1.0)),
            "top_p": float(kwargs.get("top_p", 0.9)),
            "top_k": int(kwargs.get("top_k", 0)),
        }

    @torch.inference_mode()
    def send_request(self, payload: Dict[str, Any]) -> Any:
        # ProQwen.generate expects the structured messages list with {role, content:[...]}.
        # Images are loaded by the transformers processor directly from message content items.
        out_text_list = self.proqwen.generate(
            messages=payload["messages"],
            max_new_tokens=int(payload.get("max_tokens", 1024)),
            raw_images=payload.get("raw_images", None),
            do_sample=bool(payload.get("do_sample", False)),
            **(
                {
                    "temperature": float(payload.get("temperature", 1.0)),
                    "top_p": float(payload.get("top_p", 0.9)),
                    "top_k": int(payload.get("top_k", 0)),
                }
                if bool(payload.get("do_sample", False))
                else {}
            ),
            **(
                {
        			"eos_token_id": [
                        self.processor.tokenizer.eos_token_id,
                        self.processor.tokenizer.convert_tokens_to_ids("<|im_end|>"),  # ← 加这行
                    ],
                    "repetition_penalty": 1.05,  # 保险起见
                }
                if self.ckpt_type == 'starvla'
                else {}
            )
        )
        return out_text_list[0] if out_text_list else ""

    def parse_response(self, resp: Any) -> ChatResult:
        return ChatResult(text=str(resp), raw={"text": str(resp)})
