"""Agent layer: the common contract every specialized agent implements."""

from .base import Agent
from .coder import CoderInput, CoderOutput, CodingAgent
from .critic import CriticAgent, CriticInput, CriticOutput, Finding, adjudicate
from .debugger import DebuggerInput, DebuggerOutput, DebuggingAgent
from .echo import EchoAgent, EchoInput, EchoOutput
from .planner import PlannerAgent, PlannerInput, PlannerOutput, plan_to_tasks
from .registry import AgentRegistry, build_default_registry

__all__ = [
    "Agent",
    "AgentRegistry",
    "CoderInput",
    "CoderOutput",
    "CodingAgent",
    "CriticAgent",
    "CriticInput",
    "CriticOutput",
    "DebuggerInput",
    "DebuggerOutput",
    "DebuggingAgent",
    "EchoAgent",
    "EchoInput",
    "EchoOutput",
    "Finding",
    "PlannerAgent",
    "PlannerInput",
    "PlannerOutput",
    "adjudicate",
    "build_default_registry",
    "plan_to_tasks",
]
