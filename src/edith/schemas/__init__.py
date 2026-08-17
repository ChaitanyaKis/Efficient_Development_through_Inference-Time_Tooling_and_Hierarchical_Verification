"""Domain schemas. All cross-boundary data in Edith is a Pydantic model defined here."""

from .agent import (
    AgentHealth,
    AgentIdentity,
    AgentPermissions,
    AgentRequest,
    AgentResponse,
    AgentStatus,
    Capability,
    TaskRef,
)
from .common import EdithModel, Severity, Timestamped, Verdict, new_id, utc_now
from .model import (
    GenerationOptions,
    GenerationResult,
    HealthState,
    Message,
    ProviderHealth,
    Role,
    StructuredMode,
    TokenUsage,
)

__all__ = [
    "AgentHealth",
    "AgentIdentity",
    "AgentPermissions",
    "AgentRequest",
    "AgentResponse",
    "AgentStatus",
    "Capability",
    "EdithModel",
    "GenerationOptions",
    "GenerationResult",
    "HealthState",
    "Message",
    "ProviderHealth",
    "Role",
    "Severity",
    "StructuredMode",
    "TaskRef",
    "Timestamped",
    "TokenUsage",
    "Verdict",
    "new_id",
    "utc_now",
]
