"""Core synchronous model evaluation API.

This module defines ModelConfig, ChatMessage, ChatResult and BaseModel (synchronous-only).
Adapters for concrete models should subclass BaseModel and implement:
  - build_payload
  - send_request
  - parse_response

The design keeps network calls synchronous (blocking) as requested.
"""
from __future__ import annotations
import abc
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
	model_name: str
	provider: str
	base_url: Optional[str] = None
	api_key: Optional[str] = None
	timeout: int = 30
	max_tokens: Optional[int] = None
	supports_images: bool = False
	model: Dict[str, Any] = field(default_factory=dict)
	extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ChatMessage:
	role: str  # 'system' | 'user' | 'assistant'
	content: str
	meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ChatResult:
	text: str
	tokens: Optional[int] = None
	finish_reason: Optional[str] = None
	raw: Optional[Dict[str, Any]] = None


class BaseModel(abc.ABC):
	"""
	Synchronous abstract base class for model adapters.

	Subclasses must implement payload construction, sending the request, and parsing.
	All operations are synchronous and blocking (no async/streaming methods).
	"""

	def __init__(self, cfg: ModelConfig):
		self.cfg = cfg

	@property
	def model_name(self) -> str:
		return self.cfg.model_name

	@property
	def provider(self) -> str:
		return self.cfg.provider

	@abc.abstractmethod
	def build_payload(self, messages: List[ChatMessage], **kwargs) -> Dict[str, Any]:
		"""Convert unified messages into provider-specific payload."""

	@abc.abstractmethod
	def send_request(self, payload: Dict[str, Any]) -> Any:
		"""Synchronously send the request to the model provider and return raw response."""

	@abc.abstractmethod
	def parse_response(self, resp: Any) -> ChatResult:
		"""Parse provider response into ChatResult."""

	def chat(self, messages: List[ChatMessage], **kwargs) -> ChatResult:
		"""High-level synchronous API: build payload -> send -> parse -> return ChatResult."""
		payload = self.build_payload(messages, **kwargs)
		resp = self.send_request(payload)
		return self.parse_response(resp)

	def health_check(self) -> bool:
		try:
			r = self.chat([ChatMessage(role="system", content="ping")], **{"timeout": 5})
			return bool(r and r.text is not None)
		except Exception:
			logger.exception("health_check failed for %s", self.model_name)
			return False

	def estimate_tokens(self, text: str) -> int:
		"""A simple token estimator; subclasses can override with tokenizer-specific logic."""
		return max(1, len(text) // 4)


# Registry Protocol for type hints
class ModelFactoryProtocol(Protocol):
	def register(self, name: str, cls: type[BaseModel]) -> None: ...
	def create(self, cfg: ModelConfig) -> BaseModel: ...


class SimpleRegistry:
	"""Very small registry for adapters. Register adapter classes by name/provider.

	Lookup order when creating: model_name -> provider.
	"""

	def __init__(self):
		self._mapping: Dict[str, type[BaseModel]] = {}

	def register(self, name: str, cls: type[BaseModel]):
		self._mapping[name] = cls

	def create(self, cfg: ModelConfig) -> BaseModel:
		cls = self._mapping.get(cfg.model_name) or self._mapping.get(cfg.provider)
		if not cls:
			raise ValueError(f"No model registered for {cfg.model_name}/{cfg.provider}")
		return cls(cfg)


__all__ = ["ModelConfig", "ChatMessage", "ChatResult", "BaseModel", "SimpleRegistry"]
