#!/usr/bin/env python3
"""Inspect a .pt/.pth checkpoint and print what keys/parameter names it contains.

Usage:
  python inspect_pt.py /path/to/checkpoint.pt
"""

import sys
import torch


def _is_state_dict(obj):
    return isinstance(obj, dict) and all(isinstance(k, str) for k in obj.keys())


def _pick_state_dict(ckpt: dict):
    """Best-effort: pick the tensor dict that looks like a model state_dict."""
    if not isinstance(ckpt, dict):
        return None, None

    # Common wrappers
    for k in [
        "state_dict",
        "model",
        "model_state_dict",
        "net",
        "module",
        "ema",
        "weights",
    ]:
        v = ckpt.get(k)
        if _is_state_dict(v):
            return k, v

    # If it itself looks like a state_dict
    if _is_state_dict(ckpt):
        return "<root>", ckpt

    return None, None


def main():
    if len(sys.argv) != 2:
        print("Usage: python inspect_pt.py /path/to/checkpoint.pt")
        return 2

    path = sys.argv[1]
    ckpt = torch.load(path, map_location="cpu")

    print(f"Loaded: {path}")
    print(f"Type: {type(ckpt)}")

    if isinstance(ckpt, dict):
        print("\nTop-level keys:")
        for k in sorted(ckpt.keys()):
            v = ckpt[k]
            t = type(v)
            shape = getattr(v, "shape", None)
            if shape is not None:
                print(f"  - {k}: {t} shape={tuple(shape)}")
            else:
                print(f"  - {k}: {t}")
    else:
        print("\nNot a dict checkpoint (no top-level keys to print).")

    wrap_key, sd = (None, None)
    if isinstance(ckpt, dict):
        wrap_key, sd = _pick_state_dict(ckpt)

    if sd is None and _is_state_dict(ckpt):
        wrap_key, sd = "<root>", ckpt

    if sd is None:
        print("\nNo state_dict-like mapping found.")
        return 0

    keys = list(sd.keys())
    print(f"\nState-dict candidate: {wrap_key}")
    print(f"Num params/buffers: {len(keys)}")

    # Print a small sample + simple prefix histogram
    print("\nFirst 50 parameter names:")
    for k in keys[:50]:
        print(f"  {k}")

    prefixes = {}
    for k in keys:
        p = k.split(".", 1)[0]
        prefixes[p] = prefixes.get(p, 0) + 1

    print("\nTop-level name prefixes (count):")
    for p, c in sorted(prefixes.items(), key=lambda x: (-x[1], x[0]))[:50]:
        print(f"  {p}: {c}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
