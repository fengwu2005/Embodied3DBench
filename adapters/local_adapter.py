from __future__ import annotations
import torch
import os
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from PIL import Image, ImageDraw, ImageFont
from PIL import ImageColor
from typing import Any, Dict, List

from openai import OpenAI
import requests
import logging
import json as _json
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

from base_api import BaseModel, ChatMessage, ChatResult

class LocalAdapter(BaseModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        # Load weights from the path provided in the model configuration.
        print(cfg)
        model_path = cfg.extra["model_path"]
        if not os.path.exists(model_path):
            raise ValueError(f"Model path does not exist: {model_path}")
        self.model_path = model_path
        self.cfg = cfg
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")
        self.processor = AutoProcessor.from_pretrained(model_path)
        
        from pathlib import Path
        import json as _json
        
        _TASK_FORMATS_PATH = Path(__file__).parent.parent.joinpath("configs", "task_response_formats.json")
        try:
            if _TASK_FORMATS_PATH.exists():
                with open(_TASK_FORMATS_PATH, "r", encoding="utf-8") as _f:
                    self._task_formats = _json.load(_f)
            else:
                self._task_formats = {}
        except Exception:
            self._task_formats = {}
        
        self._system_base = (
            "You are an evaluation model for spatial reasoning QA tasks.\n"
            "Produce deterministic, machine-parseable outputs for spatial/function reasoning questions.\n\n"
            "RESPONSE CONTRACT:\n"
            "- Your final output must consist of exactly one optional <think></think> block followed by one triple-backtick fenced JSON block labeled `json` without any additional texts or parts. Your should put all your answer except json in the <think></think> block to keep the format clean.\n"
            # "- The fenced JSON block must contain only the `EXAMPLE VALUE` structure (no wrappers).\n"
            # "- The JSON output must be strictly same as the EXAMPLE VALUE's format and have at least one answer in it. This is very important!\n"
            "- Copy the `EXAMPLE VALUE` into the JSON block to keep the format same, then change your answer in it (Ensure at least one meaningful answer).\n"
            # "- At least give one answer in JSON block.\n"
            "- The <think> block must include the inferred `task` and `function_type` and a one-line rationale. Don't give so much think process.\n"
            "- When you answer 3D questions or distance, the output unit is m.\n"
            "\n"
            "When calling this adapter include the parameter `task_name` so the adapter can append the exact expected JSON `value` example for that task.\n"
        )

    def inference(self, messages, max_new_tokens=4096):
        # print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAA",image_url)
        # _image = Image.open(image_url)
        # _text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # print("input:\n", _text)
        # inputs = self.processor(text=[_text], images=[_image], padding=True, return_tensors="pt").to('cuda')
        print("messages:", messages)
        inputs = self.processor.apply_chat_template(
        	messages,
        	add_generation_prompt=True,
        	tokenize=True,
        	return_dict=True,
        	return_tensors="pt",
        ).to('cuda')

        output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
        output_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        print("output:\n", output_text[0])

        _input_height = inputs['image_grid_thw'][0][1] * 14
        _input_width = inputs['image_grid_thw'][0][2] * 14

        return output_text[0], _input_height, _input_width

    def build_payload(self, messages: List[ChatMessage], **kwargs) -> Dict[str, Any]:
        # print("CCCCCCCCCCCCCCCCCCC",messages)
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
            out_messages.append({"role": "system", "content": [{"type": "text", "text": system_full}]})
            
        image_url = None
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
                    else:
                        # Backwards compatible single-image keys
                        img_url = meta.get("resolved_path")
                        if img_url:
                            content_items.append({"type": "image", "image": img_url})
                        
                    # if isinstance(imgs, (list, tuple)) and len(imgs) > 0:
                    #     for img_entry in imgs:
                    #         if not isinstance(img_entry, dict):
                    #             continue
                    #         img_b64 = img_entry.get("image_b64")
                    #         img_mime = img_entry.get("image_mime")
                    #         if img_b64:
                    #             mime = img_mime or "image/jpeg"
                    #             data_uri = f"data:{mime};base64,{img_b64}"
                    #             content_items.append({"type": "image_url", "image_url": {"url": data_uri}})
                    # else:
                    #     # Backwards compatible single-image keys
                    #     img_b64 = meta.get("image_b64")
                    #     img_mime = meta.get("image_mime")
                    #     if img_b64:
                    #         mime = img_mime or "image/jpeg"
                    #         data_uri = f"data:{mime};base64,{img_b64}"
                    #         content_items.append({"type": "image_url", "image_url": {"url": data_uri}})

                md["content"] = content_items

            out_messages.append(md)

        # logger.info(str(out_messages))
       
        return {
            "model": self.model_name,
            "messages": out_messages,
            "image_url": image_url,
            "max_tokens": self.cfg.max_tokens,
        }

    def send_request(self, payload: Dict[str, Any]) -> Any:
        # Call the local model's inference function
        # resp = self.model.chat.completions.create(**payload)
        resp = self.inference(payload["messages"], max_new_tokens=payload.get("max_tokens", 4096))
        return resp[0]

    def parse_response(self, resp: Any) -> ChatResult:
        # return ChatResult(text=resp.choices[0].message.content, raw=resp)
        return ChatResult(text=str(resp), raw=resp)
        
