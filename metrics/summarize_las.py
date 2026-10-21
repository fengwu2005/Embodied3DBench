#!/usr/bin/env python3
"""Summarize metrics from a result JSON file.

The input JSON is expected to be a top-level list of objects similar to:

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

This script extracts item['metric']['metric_value'] for each item, skips None values,
and prints a JSON summary with the mean value.
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any, List, Optional, Dict


# Preprocess mapping: ability_name (substring) -> preprocess function
# If the input file basename contains the ability_name substring, the corresponding
# preprocess function will be applied to metric values before aggregation.
def _identity(x: float) -> float:
    return x

PREPROCESS_BY_ABILITY = {
    # if filename contains 'vector' we take absolute values before summing
    "vector": abs,
    # other abilities default to no-op (identity). list here for clarity.
    "point": _identity,
    "correspondence": _identity,
}


def _select_preprocessor_from_basename(basename: str):
    b = (basename or "").lower()
    for ability, func in PREPROCESS_BY_ABILITY.items():
        if ability in b:
            return func
    return _identity


def load_results(path: str) -> List[Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Expected top-level JSON array (list) in the result file")
    return data


def extract_metric_value(item: Any) -> Optional[float]:
    """Extract metric_value from an item.

    Handles these common shapes:
    - item['metric']['metric_value'] as an int/float -> return numeric
    - item['metric']['metric_value'] as a dict -> return dict (per-key metrics)
    - item['metric']['metric_value'] as a JSON-encoded string -> parse and return
    - item['metric'] itself may be the numeric/dict value (fallback)
    """
    try:
        if not isinstance(item, dict):
            return None
        metric = item.get("metric")
        # If metric is None, maybe the top-level is the value already
        if metric is None:
            return None

        # If metric is a dict with metric_value key, prefer it
        if isinstance(metric, dict) and "metric_value" in metric:
            mv = metric.get("metric_value")
        else:
            # otherwise assume metric itself may be the value
            mv = metric

        # If value is a JSON string, try to parse
        if isinstance(mv, str):
            s = mv.strip()
            # try numeric parse first
            try:
                if "." in s or "e" in s.lower():
                    return float(s)
                return int(s)
            except Exception:
                pass
            try:
                parsed = json.loads(s)
                return parsed
            except Exception:
                # not JSON, return the raw string (unsupported)
                return None

        # For numeric or dict/list, return as-is
        if isinstance(mv, (int, float, dict, list)):
            if isinstance(mv, dict) and "Hausdorff_Distance" in mv:
                mv = mv.get("Hausdorff_Distance")
            return mv

        return None
    except Exception:
        return None


def summarize(path: str) -> int:
    data = load_results(path)
    preproc = _select_preprocessor_from_basename(os.path.basename(path))
    numeric_values: List[float] = []
    skipped = 0
    # accumulator for dict-style metric values
    dict_acc: Dict[str, List[float]] = {}
    # counts per metric key when metric_value is a dict
    dict_counts: Dict[str, int] = {}
    dict_items_count = 0

    for idx, item in enumerate(data):
        v = extract_metric_value(item)
        if v is None:
            skipped += 1
            continue

        # skip NaN
        if isinstance(v, float) and (v != v):
            skipped += 1
            continue

        # Numeric scalar metric
        if isinstance(v, (int, float)):
            try:
                v2 = preproc(float(v))
            except Exception:
                v2 = float(v)
            numeric_values.append(v2)
            continue

        # Dict-style metric, e.g., {"l2_xy": ..., "iou": ..., "angle_diff": ...}
        if isinstance(v, dict):
            dict_items_count += 1
            # iterate known keys and collect numeric values
            for k, val in v.items():
                # if nested per_object list, try to handle first-level dict only
                if isinstance(val, (int, float)):
                    try:
                        v2 = preproc(float(val))
                    except Exception:
                        v2 = float(val)
                    dict_acc.setdefault(k, []).append(v2)
                    dict_counts[k] = dict_counts.get(k, 0) + 1
                # ignore None or non-numeric entries
            continue

        # otherwise unsupported type: skip
        skipped += 1
    
    total = len(data)
    mean_scalar: Optional[float] = None
    if numeric_values:
        mean_scalar = float(sum(numeric_values) / len(numeric_values))

    # compute means for dict_acc
    dict_means: Dict[str, Optional[float]] = {}
    for k, lst in dict_acc.items():
        dict_means[k] = (float(sum(lst) / len(lst)) if lst else None)

    count_with_metric = len(numeric_values) + dict_items_count

    out = {
        "input_file": os.path.abspath(path),
        "total_items": total,
        "count_with_metric": count_with_metric,
        "count_skipped": skipped,
        "mean_metric_value": (mean_scalar if dict_means == {} else dict_means),
        "per_key_counts": (dict_counts if dict_means != {} else {}),
    }

    # print machine-readable JSON summary
    # print(json.dumps(out, indent=2, ensure_ascii=False))

    # print a concise human summary
    if not numeric_values and not dict_means:
        print("No valid metric values found (all None or missing)")
        return 0, 0

    # if dict_means:
    #     print("Per-key mean metric values:")
    #     for k, mv in dict_means.items():
    #         cnt = dict_counts.get(k, 0)
    #         print(f"  {k}: {mv} (n={cnt})")

    # if mean_scalar is not None:
    #     print(f"Mean scalar metric_value over {len(numeric_values)} entries: {mean_scalar}")

    return count_with_metric, mean_scalar


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
