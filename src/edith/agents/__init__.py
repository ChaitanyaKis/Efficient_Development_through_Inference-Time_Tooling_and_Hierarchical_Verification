"""Agent layer: the common contract every specialized agent implements."""

from .base import Agent
from .echo import EchoAgent, EchoInput, EchoOutput
from .registry import AgentRegistry, build_default_registry

__all__ = [
    "Agent",
    "AgentRegistry",
    "EchoAgent",
    "EchoInput",
    "EchoOutput",
    "build_default_registry",
]
