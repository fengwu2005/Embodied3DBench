#!/usr/bin/env python3
"""Summarize metrics from a result JSON file by first converting each metric to a score, then averaging.

Input JSON is a top-level list:
[
  {
    "model_name": "gpt4o",
    "qa_path": "...",
    "ground_truth": [...],
    "model_answer": "...",
    "json_answer": [...],
    "metric": {"metric_name": "l2", "metric_value": 123.4}
  },
  ...
]

This script:
- infers the QA task name from the input filename (e.g., results_<qa>_<model>_<ts>.json)
- maps the QA to a scoring function
- extracts a raw metric value per item (scalar or from a dict under well-known keys)
- converts each raw metric value to a score
- returns the number of scored items and the mean score
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any, List, Optional, Dict, Tuple
import math

# -------------------------
# Scoring functions (same logic as score.py)
# -------------------------
def power_score(x: float, a: float) -> float:
    return 100.0 * (x ** a)

def exp_score(x: float, a: float) -> float:
    return 100.0 * math.exp(-a * x)

PI = math.pi

# qa -> { "type": <metric_type>, "func": callable(raw_value)->score }
QA_SCORE_MAP: Dict[str, Dict[str, Any]] = {
    "2d_oriented_bbox_qa_detect_category": {"type": "iou", "func": lambda x: power_score(x, 0.5)},
    "2d_oriented_bbox_qa_single_part_with_desc": {"type": "iou", "func": lambda x: power_score(x, 0.5)},
    "2d_oriented_bbox_qa_single_with_desc": {"type": "iou", "func": lambda x: power_score(x, 0.5)},
    "3d_bbox_qa_single_part_with_desc": {"type": "iou", "func": lambda x: power_score(x, 1.0/3)},
    "correspondence_pairs_customized_2d": {"type": "l2", "func": lambda x: exp_score(x, 0.005)},
    "correspondence_pairs_customized_3d": {"type": "l2", "func": lambda x: exp_score(x, 2.6)},
    "correspondence_pairs_single_object_2d": {"type": "l2", "func": lambda x: exp_score(x, 0.005)},
    "correspondence_pairs_single_object_3d": {"type": "l2", "func": lambda x: exp_score(x, 2.6)},
    "grasp_point_2d": {"type": "l2", "func": lambda x: exp_score(x, 0.005)},
    "grasp_point_3d": {"type": "l2", "func": lambda x: exp_score(x, 2.6)},
    "obj_bbox_qa_detect_category_3d": {"type": "iou", "func": lambda x: power_score(x, 1.0/3)},
    "obj_bbox_qa_single_with_desc_3d": {"type": "iou", "func": lambda x: power_score(x, 1.0/3)},
    "object_point_qa_detect_category_2d": {"type": "l2", "func": lambda x: exp_score(x, 0.005)},
    "object_point_qa_detect_category_3d": {"type": "l2", "func": lambda x: exp_score(x, 2.6)},
    "object_point_qa_single_with_desc_2d": {"type": "l2", "func": lambda x: exp_score(x, 0.005)},
    "object_point_qa_single_with_desc_3d": {"type": "l2", "func": lambda x: exp_score(x, 2.6)},
    "single_func_point_2d": {"type": "l2", "func": lambda x: exp_score(x, 0.005)},
    "single_func_point_3d": {"type": "l2", "func": lambda x: exp_score(x, 2.6)},
    "single_func_vector_2d": {"type": "cosine", "func": lambda x: power_score(1 - (math.acos(max(0.0, min(1.0, x))) / PI * 2.0), 1.0)},
    "single_func_vector_3d": {"type": "cosine", "func": lambda x: power_score(1 - (math.acos(max(0.0, min(1.0, x))) / PI * 2.0), 1.0)},
    "single_part_point_2d": {"type": "l2", "func": lambda x: exp_score(x, 0.005)},
    "single_part_point_3d": {"type": "l2", "func": lambda x: exp_score(x, 2.6)},
    "spatial_rel_qa_obj2cam_distance": {"type": "abs", "func": lambda x: exp_score(x, 2.6)},
    "spatial_rel_qa_obj2obj_choice": {"type": "accuracy", "func": lambda x: 100.0 * x},
    "spatial_rel_qa_obj2obj_distance": {"type": "abs", "func": lambda x: exp_score(x, 2.6)},
    "traj_qa_prediction_2d": {"type": "hausdorff", "func": lambda x: exp_score(x, 0.005)},
    "traj_qa_prediction_3d": {"type": "hausdorff", "func": lambda x: exp_score(x, 2.6)},
}

# Preferred key order for extracting raw metric values from dict metrics
METRIC_KEYS_BY_TYPE: Dict[str, List[str]] = {
    "iou": ["iou", "IoU", "bbox_iou", "oriented_bbox_iou"],
    "l2": ["l2", "mean_l2", "l2_xy", "l2_xyz", "distance", "dist"],
    "abs": ["abs", "abs_diff", "mae", "l1", "distance", "dist"],
    "accuracy": ["accuracy", "acc"],
    "hausdorff": ["Hausdorff_Distance", "hausdorff", "HD", "hd"],
    "cosine": ["cosine", "cos", "cos_sim", "similarity", "sim"],  # if not found, try angle -> cosine
}

def load_results(path: str) -> List[Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Expected top-level JSON array (list) in the result file")
    return data

def _parse_metric_field(item: Any) -> Any:
    """Return metric value or dict as stored under item['metric'] or item['metric']['metric_value'].
    Also parses JSON-encoded strings and numeric strings.
    """
    if not isinstance(item, dict):
        return None
    metric = item.get("metric")
    if metric is None:
        return None

    mv = metric.get("metric_value") if isinstance(metric, dict) and "metric_value" in metric else metric

    # decode JSON/numeric strings
    if isinstance(mv, str):
        s = mv.strip()
        try:
            if "." in s or "e" in s.lower():
                return float(s)
            return int(s)
        except Exception:
            pass
        try:
            return json.loads(s)
        except Exception:
            return None

    if isinstance(mv, (int, float, dict, list)):
        return mv
    return None

def _detect_qa_from_basename(basename: str) -> Optional[str]:
    """Find the QA task by searching for the longest matching QA key substring in the filename."""
    b = (basename or "").lower()
    best = None
    for qa in QA_SCORE_MAP.keys():
        if qa.lower() in b:
            if best is None or len(qa) > len(best):
                best = qa
    return best

def _extract_raw_from_value(v: Any, metric_type: str) -> Optional[float]:
    """Extract the raw numeric value from v according to the metric_type."""
    if v is None:
        return None

    # direct scalar
    if isinstance(v, (int, float)):
        x = float(v)
        if math.isnan(x):
            return None
        return x

    # dict with known keys
    if isinstance(v, dict):
        keys = METRIC_KEYS_BY_TYPE.get(metric_type, [])
        for k in keys:
            if k in v and isinstance(v[k], (int, float)):
                x = float(v[k])
                if math.isnan(x):
                    return None
                return x

        # Special fallback: cosine from angle_diff if available
        if metric_type == "cosine":
            # angle could be in radians or degrees
            angle_keys = ["angle_diff", "angle", "theta", "angle_deg", "angle_rad"]
            for ak in angle_keys:
                val = v.get(ak)
                if isinstance(val, (int, float)):
                    a = float(val)
                    if math.isnan(a):
                        continue
                    # heuristic: if > pi, assume degrees
                    if abs(a) > (PI + 1e-6):
                        a = math.radians(a)
                    # convert angle to cosine similarity
                    return math.cos(a)
        return None

    # unsupported type
    return None

def summarize(path: str) -> Tuple[int, float]:
    data = load_results(path)
    basename = os.path.basename(path)
    qa = _detect_qa_from_basename(basename)
    if qa is None:
        # No QA detected; cannot score properly
        # Fallback: average any numeric scalar metrics directly mapped to exp with a light decay
        scores: List[float] = []
        for item in data:
            v = _parse_metric_field(item)
            if isinstance(v, (int, float)) and not math.isnan(float(v)):
                # neutral fallback scoring
                scores.append(exp_score(float(v), 0.001))
        return (len(scores), 0, float(sum(scores) / len(scores))) if scores else (0, 0, 0.0)

    spec = QA_SCORE_MAP[qa]
    metric_type = spec["type"]
    score_func = spec["func"]

    scores: List[float] = []
    skipped = 0

    for item in data:
        v = _parse_metric_field(item)
        x = _extract_raw_from_value(v, metric_type)
        if x is None:
            skipped += 1
            continue

        # sanitize per type
        if metric_type == "iou":
            # clamp IoU to [0, 1]
            x = max(0.0, min(1.0, x))
        elif metric_type == "cosine":
            # clamp cosine to [-1, 1]
            x = max(-1.0, min(1.0, x))

        try:
            s = float(score_func(x))
            if math.isnan(s):
                skipped += 1
                continue
            scores.append(s)
        except Exception:
            skipped += 1

    if not scores:
        print("No valid metric values found (all None or missing)")
        return 0, 0.0

    mean_score = float(sum(scores) / len(scores))
    count_with_metric = len(scores)
    return count_with_metric, skipped, mean_score

def summarize_with_raw(path: str) -> Tuple[int, float]:
    data = load_results(path)
    basename = os.path.basename(path)
    qa = _detect_qa_from_basename(basename)
    if qa is None:
        # No QA detected; cannot score properly
        # Fallback: average any numeric scalar metrics directly mapped to exp with a light decay
        scores: List[float] = []
        for item in data:
            v = _parse_metric_field(item)
            if isinstance(v, (int, float)) and not math.isnan(float(v)):
                # neutral fallback scoring
                scores.append(exp_score(float(v), 0.001))
        return (len(scores), float(sum(scores) / len(scores))) if scores else (0, 0.0)

    spec = QA_SCORE_MAP[qa]
    metric_type = spec["type"]
    score_func = spec["func"]

    scores: List[float] = []
    raw_dic: dict[str, float] = {}
    skipped = 0

    for item in data:
        v = _parse_metric_field(item)
        x = _extract_raw_from_value(v, metric_type)
        if x is None:
            skipped += 1
            continue

        raw_dic[item["qa_path"]] = x

        # sanitize per type
        if metric_type == "iou":
            # clamp IoU to [0, 1]
            x = max(0.0, min(1.0, x))
        elif metric_type == "cosine":
            # clamp cosine to [-1, 1]
            x = max(-1.0, min(1.0, x))

        try:
            s = float(score_func(x))
            if math.isnan(s):
                skipped += 1
                continue
            scores.append(s)
        except Exception:
            skipped += 1

    if not scores:
        print("No valid metric values found (all None or missing)")
        return 0, 0.0, [], {}

    mean_score = float(sum(scores) / len(scores))
    count_with_metric = len(scores)

    # print(raw_dic)

    return count_with_metric, mean_score, scores, raw_dic


def main(argv: List[str]) -> int:
    if len(argv) < 2:
        print("Usage: summarize.py <result_json_path>")
        return 2
    path = argv[1]
    summarize(path)
    return 0


if __name__ == "__main__":
    rc = main(sys.argv)
    sys.exit(rc)
