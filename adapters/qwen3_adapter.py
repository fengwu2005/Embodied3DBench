from __future__ import annotations
import torch
import os
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image, ImageDraw, ImageFont
from PIL import ImageColor
from typing import Any, Dict, List

from openai import OpenAI
import requests
import logging
from pathlib import Path
import json as _json
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

from base_api import BaseModel, ChatMessage, ChatResult

from configs.get_system_prompt import get_system_prompt
from adapters.tools.get_chat_template import get_chat_template

class Qwen3_Adapter(BaseModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        # Load weights from the path provided in the model configuration.
        print(cfg)
        model_path = cfg.extra["model_path"]
        ckpt_type = cfg.extra.get("ckpt_type", None)
        ckpt_path = cfg.extra.get("ckpt_path", None)
        if not os.path.exists(model_path):
            raise ValueError(f"Model path does not exist: {model_path}")
        self.model_path = model_path
        self.cfg = cfg
        self.model = AutoModelForImageTextToText.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto")
        self.processor = AutoProcessor.from_pretrained(model_path)

        # if not cfg.extra.get("enable_thinking", True):
        #     self.processor.tokenizer.chat_template = get_chat_template()
        
    def inference(self, messages, image_url, max_new_tokens=4096):
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
        _input_width = inputs['image_grid_thw'][0][2] * 1

        return output_text[0], _input_height, _input_width

    def build_payload(self, messages: List[ChatMessage], **kwargs) -> Dict[str, Any]:
        # print("CCCCCCCCCCCCCCCCCCC",messages)
        task_name = kwargs.get("task_name")
        system_full = get_system_prompt(task_name)

        print("Built system prompt for task '%s': %s\n\n\n", task_name, system_full)

        out_messages: List[Dict[str, Any]] = []
        # If caller did not include a system message, inject one with the composed system_full
        if not any(getattr(m, "role", None) == "system" for m in messages):
            # out_messages.append({"role": "system", "content": system_full})
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
        resp = self.inference(payload["messages"], payload["image_url"], max_new_tokens=payload.get("max_tokens", 4096))
        return resp[0]

    def parse_response(self, resp: Any) -> ChatResult:
        # return ChatResult(text=resp.choices[0].message.content, raw=resp)
        return ChatResult(text=str(resp), raw=resp)
        
