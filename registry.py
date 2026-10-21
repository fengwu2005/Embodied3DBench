"""Model registry and config loader utilities.

Provides a SimpleRegistry thin wrapper that works with ModelConfig and adapter classes.
Also provides helpers to load model configs from YAML/JSON if desired.
"""
from __future__ import annotations
import json
import logging
import os
from typing import Dict, Any, Optional

import yaml

from base_api import ModelConfig

logger = logging.getLogger(__name__)


def _expand_env(value):
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, dict):
        return {key: _expand_env(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_env(item) for item in value]
    return value


class SimpleRegistry:
    """Separate registry module: thin wrapper around adapter mapping.

    This duplicates the minimal registry in base_api for convenience/import clarity.
    """

    def __init__(self):
        self._mapping: Dict[str, type] = {}

    def register(self, name: str, cls: type):
        self._mapping[name] = cls

    def create(self, cfg: ModelConfig):
        cls = self._mapping.get(cfg.model_name) or self._mapping.get(cfg.provider)
        if not cls:
            raise ValueError(f"No model registered for {cfg.model_name}/{cfg.provider}")
        return cls(cfg)


def load_configs_from_yaml(path: str) -> Dict[str, ModelConfig]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    out: Dict[str, ModelConfig] = {}
    for name, cfg in (data or {}).items():
        expanded = _expand_env(cfg)
        out[name] = ModelConfig(**expanded)
    return out


def load_configs_from_json(path: str) -> Dict[str, ModelConfig]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    out: Dict[str, ModelConfig] = {}
    for name, cfg in (data or {}).items():
        out[name] = ModelConfig(**cfg)
    return out
