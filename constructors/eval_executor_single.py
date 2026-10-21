"""Evaluation executor: load sampled QA set, run model, and save results.

This script will:
- instantiate QALoader with a provided QA root
- ensure an eval mapping exists for the requested task (build if missing)
- load the QA file paths and contents
- load model config(s) from a YAML file under `evaluator/configs` (or provided path)
- instantiate the GPT-4o adapter and run the chat model on each QA
- collect results as {"qa_path", "ground_truth", "model_answer"} and save to output JSON

Usage (example):
    python -m evaluator.constructors.eval_executor --root D:/data/qa_root --task task_0 \
        --model-config evaluator/configs/model_config.yaml --model-name gpt4o --out results_task0.json --n 50
"""

from __future__ import annotations

import argparse
import json
import base64
import mimetypes
import re
import ast
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple
from tqdm import tqdm
import os

from adapters.gemini_adapter import GeminiAdapter
from registry import SimpleRegistry, load_configs_from_yaml
from adapters.general_adapter import GeneralAdapter
from adapters.openai_adapter import OpenAIAdapter
from metrics.metrics import compute_metric
from base_api import ChatMessage
from constructors.qa_loader import QALoader
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def extract_q_and_gt(qa_obj: Any) -> Tuple[str, Optional[str]]:
    """Try common keys to extract question/prompt and ground-truth answer.

    Returns (question_text, ground_truth_or_None).
    """
    # Common simple forms
    if isinstance(qa_obj, str):
        return qa_obj, None
    if not isinstance(qa_obj, dict):
        return str(qa_obj), None

    # Newer dataset format: {'id':..., 'image':..., 'conversations':[{'value': question}, {'value': answer}]}
    try:
        conv = qa_obj.get("conversations") if isinstance(qa_obj, dict) else None
        if isinstance(conv, list) and len(conv) >= 2:
            # Try to extract from conversations[0]['value'] and conversations[1]['value']
            q_val = conv[0]
            a_val = conv[1]
            q_text = None
            a_text = None
            if isinstance(q_val, dict):
                # common key 'value'
                q_text = q_val.get("value") or q_val.get("text") or q_val.get("content")
            elif isinstance(q_val, str):
                q_text = q_val

            if isinstance(a_val, dict):
                a_text = a_val.get("value") or a_val.get("text") or a_val.get("content")
            elif isinstance(a_val, str):
                a_text = a_val

            if q_text is not None:
                return (str(q_text), str(a_text) if a_text is not None else None)
    except Exception:
        # fall through to legacy heuristics
        logger.exception("Error while extracting conversations field from QA object")

    return "", None


def run_eval(
    qa_root: str,
    task_name: str,
    model_config_path: Optional[str],
    model_name: Optional[str],
    loader_out: str,
    out_path: str,
    num_samples: int = 20,
    seed: Optional[int] = None,
):
    # Prepare loader
    loader = QALoader(qa_root, out=loader_out)

    # Ensure mapping exists
    mapping_path = str(Path(loader.out) / "eval_sets" / f"{task_name}_eval.json")
    if not Path(mapping_path).exists():
        logger.info("Mapping not found at %s, building with n=%d seed=%s", mapping_path, num_samples, seed)
        loader.build_eval_set(task_name, num_samples=num_samples, out_path=mapping_path, seed=seed)
    print('build done')
    mapping = loader.load_eval_mapping(mapping_path)
    qa_paths = mapping.get(task_name, [])
    if not qa_paths:
        logger.warning("No QA paths for task %s (mapping: %s)", task_name, mapping_path)

    # Load QA contents
    qa_objects: List[Dict[str, Any]] = []
    for idx, p in enumerate(qa_paths):
        # print(p)
        # break
        if idx >= num_samples:
            break
        try:
            with open(p, "r", encoding="utf-8") as f:
                qa = json.load(f)
            qa_objects.append({"path": p, "obj": qa})
        except Exception:
            logger.exception("Failed to load QA file %s; skipping", p)

    # Load model configs
    cfg_map: Dict[str, Any] = {}
    if model_config_path:
        cfg_map = load_configs_from_yaml(model_config_path) or {}

    # Choose model config: model_name if provided, else first entry
    cfg = None
    if model_name and model_name in cfg_map:
        cfg = cfg_map[model_name]
    else:
        # pick first config if available
        cfg = next(iter(cfg_map.values()), None)

    if cfg is None:
        # Fall back to a minimal simulated config for gpt4o
        logger.info("No model config found; using a minimal simulated gpt4o config")
        from evaluator.base_api import ModelConfig

        cfg = ModelConfig(model_name=model_name or "gpt4o", provider=model_name or "gpt4o", base_url=None, api_key=None, max_tokens=512)

    # Register adapter and create model
    registry = SimpleRegistry()
    registry.register("gpt-4o", OpenAIAdapter)
    registry.register("Gemini-2.5-pro", GeneralAdapter)
    registry.register("gemini-2.5-flash", GeneralAdapter)
    registry.register("gpt-5", GeneralAdapter)
    registry.register("Claude-sonnet-4", GeneralAdapter)
    registry.register("Qwen3-235B-A22B", GeneralAdapter)
    registry.register("Qwen2.5-VL-72B", GeneralAdapter)
    registry.register("Qwen2.5-VL-7B", GeneralAdapter)
    registry.register("internvl3-5-8b", GeneralAdapter)
    registry.register("doubao-1-5-thinking-vision-pro-250428", GeneralAdapter)
    
    model = registry.create(cfg)

    results: List[Dict[str, Any]] = []

    for item in tqdm(qa_objects, desc="Evaluating QA items"):
        time.sleep(1)
        path = item["path"]
        qa_obj = item["obj"]
        # print(path, qa_obj)
        q_text, gt = extract_q_and_gt(qa_obj)

        # Build messages (system prompt will be injected by adapter based on task_name)
        messages = []

        # If QA references an image, try to read and attach it as base64 in meta
        img_meta: Dict[str, Any] = {}
        try:
            img_field = None
            if isinstance(qa_obj, dict):
                # support multiple possible keys
                for k in ("image", "image_filename", "image_path"):
                    if k in qa_obj:
                        img_field = qa_obj.get(k)
                        break

            images_meta_list: List[Dict[str, Any]] = []

            def _resolve_single_image(entry) -> Optional[Dict[str, Any]]:
                """Resolve a single image entry (string path) and return its meta dict or None."""
                if entry is None:
                    return None
                
                # If entry is a dict, try to extract a path-like field
                if isinstance(entry, dict):
                    for kk in ("image", "image_filename", "image_path", "path"):
                        if kk in entry:
                            entry = entry.get(kk)
                            break
                entry = os.path.expandvars(str(entry))

                # At this point entry should be a path-like string
                candidates = []
                try:
                    candidates.append(Path(entry))
                except Exception:
                    pass
                try:
                    candidates.append(Path(path).parent.joinpath(entry))
                except Exception:
                    pass
                try:
                    candidates.append(Path(loader.root).joinpath(entry))
                except Exception:
                    pass

                img_path = None
                for c in candidates:
                    if c and c.exists() and c.is_file():
                        img_path = c
                        break

                if img_path is None:
                    # If nothing resolved, still include a record with the referenced string
                    return {"referenced": entry}

                try:
                    with open(img_path, "rb") as imf:
                        b = imf.read()
                    b64 = base64.b64encode(b).decode("utf-8")
                    mime, _ = mimetypes.guess_type(str(img_path))
                    if not mime:
                        mime = "image/jpeg"
                    return {"image_b64": b64, "image_mime": mime, "image_filename": img_path.name, "resolved_path": str(img_path)}
                except Exception:
                    logger.exception("Failed to read/encode image for QA %s (entry=%s)", path, entry)
                    return {"referenced": entry}

            if img_field:
                # print(img_field)
                # If field is a list/tuple of images, handle each; otherwise treat as single
                if isinstance(img_field, (list, tuple)):
                    for ent in img_field:
                        meta = _resolve_single_image(ent)
                        if meta:
                            images_meta_list.append(meta)
                else:
                    meta = _resolve_single_image(img_field)
                    if meta:
                        images_meta_list.append(meta)

                if images_meta_list:
                    # Provide backward-compatible top-level keys for the first image
                    first = images_meta_list[0]
                    img_meta = {"images": images_meta_list}
                    # also include legacy single-image keys if present
                    if "image_b64" in first:
                        img_meta.update({"image_b64": first.get("image_b64"), "image_mime": first.get("image_mime"), "image_filename": first.get("image_filename")})
                else:
                    logger.warning("Image file referenced but not found or unreadable: %s", img_field)
        except Exception:
            logger.exception("Failed to process image field for QA %s", path)

        # attach image meta into the user message if present
        if img_meta:
            messages.append(ChatMessage(role="user", content=str(q_text), meta=img_meta))
        else:
            messages.append(ChatMessage(role="user", content=str(q_text)))

        try:
            logger.info(q_text)
            # Pass task_name so adapter can append the task-specific response format
            resp = model.chat(messages, task_name=task_name)
            # print(resp)
            ans = resp.text if resp is not None else ""
        except Exception:
            logger.exception("Model failed on QA %s", path)
            ans = ""

        # print(resp)
        # print(ans)
        logger.info(ans)

        # Try to extract a JSON code block from the model answer (```json ... ```),
        # or fallback to the first JSON-like object/array found in the text.
        json_answer = None
        try:
            content = None
            m = re.search(r"```json\s*(.*?)```", ans, re.IGNORECASE | re.DOTALL)
            if not m:
                # fallback to any triple-backtick block
                m = re.search(r"```\s*(\{.*?\}|\[.*?\])\s*```", ans, re.DOTALL)
            if m:
                content = m.group(1).strip()
            else:
                # fallback to first curly-brace or bracket JSON-like substring
                m2 = re.search(r"(\{[\s\S]*?\}|\[[\s\S]*?\])", ans, re.DOTALL)
                if m2:
                    content = m2.group(1).strip()

            if content:
                try:
                    json_answer = json.loads(content)
                except Exception as e:
                    # try parsing as Python literal (handles single quotes)
                    try:
                        json_answer = ast.literal_eval(content)
                    except Exception:
                        # Try a tolerant fix: replace single quotes with double quotes and attempt again
                        try:
                            fixed = content.replace('\'', '"')
                            json_answer = json.loads(fixed)
                        except Exception:
                            # store raw content and parse error message for diagnostics
                            json_answer = {"_raw": content, "_parse_error": str(e)}
        except Exception:
            logger.exception("Failed to extract json codeblock from model answer for QA %s", path)

        # Normalize ground-truth: QA files sometimes store Python-style reprs (single quotes).
        gt_parsed = gt
        try:
            if isinstance(gt, str):
                s = gt.strip()
                # If it looks like a JSON/object but uses single quotes, prefer ast.literal_eval
                if (s.startswith('{') or s.startswith('[')) and ("'" in s and '"' not in s):
                    try:
                        gt_parsed = ast.literal_eval(s)
                    except Exception:
                        try:
                            gt_parsed = json.loads(s.replace("'", '"'))
                        except Exception:
                            gt_parsed = gt
                else:
                    try:
                        gt_parsed = json.loads(s)
                    except Exception:
                        try:
                            gt_parsed = ast.literal_eval(s)
                        except Exception:
                            try:
                                gt_parsed = json.loads(s.replace("'", '"'))
                            except Exception:
                                gt_parsed = gt
            else:
                gt_parsed = gt
        except Exception:
            gt_parsed = gt

        img_paths = [o.get('resolved_path') for o in images_meta_list if isinstance(o, dict) and o.get('resolved_path')]

        # compute metric for this task
        metric_obj = compute_metric(task_name, gt_parsed, json_answer)
        # print(metric_obj)

        results.append({
            "model_name": model_name,
            "qa_path": path,
            "img_paths": img_paths,
            "ground_truth": gt_parsed,
            "model_answer": ans,
            "json_answer": json_answer,
            "metric": metric_obj,
        })

    # Save results list to out_path
    try:
        out_path = os.path.join(out_path, f'results_{task_name}_{model_name}_{time.time()}.json')
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info("Saved %d evaluation results to %s", len(results), out_path)
    except Exception:
        logger.exception("Failed to save results to %s", out_path)

    return results


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, help="QA root directory")
    p.add_argument("--task", required=True, help="Task name under root to evaluate")
    p.add_argument("--model-config", default="evaluator/configs/model_config.yaml", help="Path to model config YAML")
    p.add_argument("--model-name", default="gpt4o", help="Key of model config to use (if present in YAML)")
    p.add_argument("--loader-out", required=True, help="Path to store sampled eval sets")
    p.add_argument("--out", required=True, help="Path to write evaluation results JSON")
    p.add_argument("--n", type=int, default=20, help="Number of samples to pick if mapping missing")
    p.add_argument("--seed", type=int, default=None, help="Random seed for sampling")
    return p.parse_args()


def main():
    args = parse_args()
    run_eval(args.root, args.task, args.model_config, args.model_name, args.loader_out, args.out, num_samples=args.n, seed=args.seed)


if __name__ == "__main__":
    main()
