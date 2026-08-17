"""Tool layer: the controlled gateway through which agents touch the outside world.

Agents never receive unrestricted filesystem or shell access. They receive a
:class:`~edith.tools.gateway.ToolGateway` bound to their own
:class:`~edith.schemas.agent.AgentPermissions`, and every operation flows through it.
"""

from .base import Tool, ToolContext
from .gateway import ToolGateway
from .paths import PathPolicy
from .permissions import UNRESTRICTED, PermissionEngine
from .registry import ToolRegistry, build_default_registry
from .schemas import AccessMode, ToolCall, ToolResult, ToolSpec
from .workspace import Workspace

__all__ = [
    "UNRESTRICTED",
    "AccessMode",
    "PathPolicy",
    "PermissionEngine",
    "Tool",
    "ToolCall",
    "ToolContext",
    "ToolGateway",
    "ToolRegistry",
    "ToolResult",
    "ToolSpec",
    "Workspace",
    "build_default_registry",
]
