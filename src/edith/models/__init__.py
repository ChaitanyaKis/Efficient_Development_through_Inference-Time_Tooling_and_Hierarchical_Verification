"""Model layer: the replaceable seam between Edith and any local LLM runtime."""

from .base import ModelProvider, extract_json_object, render_schema_instruction
from .ollama import OllamaProvider, normalize_model_name
from .registry import available_providers, build_provider, register_provider
from .retry import backoff_delays, with_retry

__all__ = [
    "ModelProvider",
    "OllamaProvider",
    "available_providers",
    "backoff_delays",
    "build_provider",
    "extract_json_object",
    "normalize_model_name",
    "register_provider",
    "render_schema_instruction",
    "with_retry",
]
