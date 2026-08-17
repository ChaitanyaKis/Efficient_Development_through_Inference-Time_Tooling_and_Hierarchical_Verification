"""The product-development layer: PRD, UX specification, and architecture.

CLAUDE.md's architectural rule is that agents communicate through structured artifacts and
project state rather than by talking to each other. M4 is where that becomes the primary
mechanism: a Product Manager writes a PRD, a UX agent reads it and writes a specification, an
Architect reads both and writes a system design and an implementation plan.

Nothing here executes anything. M4 produces the plan; M2's loop is what runs it.

Importing this package registers every document schema against its artifact kind, so
``Artifact.document()`` can resolve a body without a hand-maintained mapping drifting out of
date.
"""

from .artifacts import (
    Artifact,
    ArtifactDocument,
    ArtifactKind,
    ArtifactRef,
    ArtifactStatus,
    ValidationIssue,
    ValidationOutcome,
    ValidationState,
    build_artifact,
    can_transition,
    element_id,
    is_element_id,
    register_document,
)
from .prd import (
    AcceptanceCriterion,
    OpenQuestion,
    Persona,
    PRDDocument,
    Priority,
    Requirement,
    RequirementKind,
    Risk,
    SuccessMetric,
    UserStory,
)
from .properties import (
    CONTRADICTORY_PAIRS,
    ProductProperty,
    conflicts_with,
    expand,
    find_conflicts,
    hints_in,
)

__all__ = [
    "CONTRADICTORY_PAIRS",
    "AcceptanceCriterion",
    "Artifact",
    "ArtifactDocument",
    "ArtifactKind",
    "ArtifactRef",
    "ArtifactStatus",
    "OpenQuestion",
    "PRDDocument",
    "Persona",
    "Priority",
    "ProductProperty",
    "Requirement",
    "RequirementKind",
    "Risk",
    "SuccessMetric",
    "UserStory",
    "ValidationIssue",
    "ValidationOutcome",
    "ValidationState",
    "build_artifact",
    "can_transition",
    "conflicts_with",
    "element_id",
    "expand",
    "find_conflicts",
    "hints_in",
    "is_element_id",
    "register_document",
]
