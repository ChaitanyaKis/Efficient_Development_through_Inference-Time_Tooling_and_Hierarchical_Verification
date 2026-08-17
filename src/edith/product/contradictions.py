"""Deterministic contradiction detection across product artifacts.

M4.8's examples are the specification for this module:

    PRD: "Must work offline."          Architecture: "Requires cloud-only service."
    PRD: "Authentication required."    Architecture: "No authentication."
    UX:  "Mobile responsive."          Architecture: "Desktop-only fixed layout."

All three are found here without a model being asked anything, because the artifacts declare
:class:`~edith.product.properties.ProductProperty` values from a closed vocabulary and a
contradiction is a set intersection against a table of incompatible pairs.

There is a second, weaker layer. A document that *talks* about offline behaviour in prose but
declares no property is not contradicting anything — it is under-specified, and the lexical
pass raises that as an advisory hint. Hints never block. The distinction matters: a blocking
finding must be one a human would agree with immediately, and a keyword match is not that.

The third layer is structural rather than lexical: an architecture whose API endpoints all
require authentication while the PRD demands anonymous access is a contradiction visible in
the *fields*, not in any declared property. Those checks are written out explicitly below,
because each one is a specific claim about how two documents can disagree.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from edith.observability.logging import get_logger

from .architecture import SystemArchitectureDocument
from .prd import PRDDocument
from .properties import ProductProperty, expand, find_conflicts, hints_in
from .ux import UXSpecDocument

logger = get_logger(__name__)


class ContradictionSeverity(StrEnum):
    """How much confidence the finding carries.

    ``BLOCKING`` findings come from structural comparison and are certain. ``ADVISORY``
    findings come from prose and are suggestions that a human or a reviewing agent should
    look. Keeping them in one type with a severity, rather than two types, means a caller
    cannot accidentally treat a hint as a fact by importing the wrong thing.
    """

    BLOCKING = "BLOCKING"
    ADVISORY = "ADVISORY"


@dataclass(frozen=True)
class Contradiction:
    """Two artifacts making incompatible claims."""

    code: str
    #: Which artifacts disagree, e.g. ``("PRD", "SYSTEM_ARCHITECTURE")``.
    between: tuple[str, str]
    detail: str
    severity: ContradictionSeverity = ContradictionSeverity.BLOCKING
    #: Element ids implicated, when the disagreement is attributable to specific ones.
    elements: tuple[str, ...] = ()

    @property
    def blocking(self) -> bool:
        """Whether this must prevent approval."""
        return self.severity is ContradictionSeverity.BLOCKING

    def render(self) -> str:
        """A single readable line."""
        left, right = self.between
        marker = "" if self.blocking else " (advisory)"
        return f"[{self.code}] {left} vs {right}{marker}: {self.detail}"


PROPERTY_CONFLICT = "PROPERTY_CONFLICT"
UNDECLARED_PROPERTY_HINT = "UNDECLARED_PROPERTY_HINT"
AUTHENTICATION_MISMATCH = "AUTHENTICATION_MISMATCH"
OFFLINE_VS_EXTERNAL_DEPENDENCY = "OFFLINE_VS_EXTERNAL_DEPENDENCY"
SENSITIVE_DATA_UNTHREATENED = "SENSITIVE_DATA_WITHOUT_THREAT_MODEL"
UI_REQUIRED_BUT_HEADLESS = "UI_REQUIRED_BUT_ARCHITECTURE_HEADLESS"


def _properties_of(document: PRDDocument | UXSpecDocument | SystemArchitectureDocument
                   ) -> frozenset[ProductProperty]:
    """The properties a document declares, whatever its type."""
    if isinstance(document, PRDDocument):
        return document.declared_properties
    return document.properties


def _label(document: object) -> str:
    """The artifact kind name for a document, for reporting."""
    kind = getattr(type(document), "kind", None)
    return str(kind) if kind is not None else type(document).__name__


def compare_properties(
    left: PRDDocument | UXSpecDocument | SystemArchitectureDocument,
    right: PRDDocument | UXSpecDocument | SystemArchitectureDocument,
) -> list[Contradiction]:
    """Structural contradictions between two documents' declared properties.

    This is the layer that answers M4.8's three examples, and it is exact: both sides
    declared a property from a closed vocabulary, and the pair is in the incompatible table.
    """
    findings: list[Contradiction] = []
    for first, second in find_conflicts(_properties_of(left), _properties_of(right)):
        findings.append(
            Contradiction(
                code=PROPERTY_CONFLICT,
                between=(_label(left), _label(right)),
                detail=(
                    f"{_label(left)} requires {first.value} while {_label(right)} "
                    f"declares {second.value}; these cannot both hold"
                ),
            )
        )
    return findings


def prose_hints(
    document: PRDDocument | UXSpecDocument | SystemArchitectureDocument,
) -> list[Contradiction]:
    """Properties the prose discusses but the document never declared.

    Always advisory. A sentence containing "offline" might be asserting offline support or
    ruling it out, and a keyword cannot tell the difference — which is exactly why this layer
    is not allowed to block anything.
    """
    if isinstance(document, PRDDocument):
        text = " ".join(
            (
                document.problem,
                *document.goals,
                *document.constraints,
                *(requirement.statement for requirement in document.requirements),
            )
        )
    elif isinstance(document, UXSpecDocument):
        text = " ".join((document.overview, *document.interaction_patterns))
    else:
        text = " ".join((document.overview, *document.constraints_considered))

    declared = expand(_properties_of(document))
    hinted = hints_in(text)
    undeclared = sorted(hinted - declared, key=lambda item: item.value)

    return [
        Contradiction(
            code=UNDECLARED_PROPERTY_HINT,
            between=(_label(document), "declared properties"),
            detail=(
                f"the text discusses {prop.value} but the document does not declare it, so "
                f"no contradiction check can be run against it"
            ),
            severity=ContradictionSeverity.ADVISORY,
        )
        for prop in undeclared
    ]


def check_prd_against_architecture(
    prd: PRDDocument, architecture: SystemArchitectureDocument
) -> list[Contradiction]:
    """Every way a PRD and an architecture can disagree."""
    findings = compare_properties(prd, architecture)
    required = expand(prd.declared_properties)
    provided = expand(architecture.properties)

    # Authentication, checked against the endpoints rather than only the declared property.
    # An architecture can declare AUTHENTICATION_REQUIRED and still expose every endpoint
    # anonymously; the endpoints are the ground truth.
    if ProductProperty.AUTHENTICATION_REQUIRED in required and architecture.endpoints:
        unauthenticated = [
            endpoint.endpoint_id
            for endpoint in architecture.endpoints
            if not endpoint.requires_authentication
        ]
        if len(unauthenticated) == len(architecture.endpoints):
            findings.append(
                Contradiction(
                    code=AUTHENTICATION_MISMATCH,
                    between=("PRD", "SYSTEM_ARCHITECTURE"),
                    detail=(
                        "the PRD requires authentication but no endpoint in the "
                        "architecture requires it"
                    ),
                    elements=tuple(unauthenticated),
                )
            )

    # Offline capability, checked against the component graph. A product that must work
    # offline cannot depend on a component it does not own.
    if ProductProperty.OFFLINE_CAPABLE in required:
        external = [
            component.component_id
            for component in architecture.components
            if component.kind.value == "EXTERNAL"
        ]
        if external:
            findings.append(
                Contradiction(
                    code=OFFLINE_VS_EXTERNAL_DEPENDENCY,
                    between=("PRD", "SYSTEM_ARCHITECTURE"),
                    detail=(
                        f"the PRD requires offline capability but the architecture depends "
                        f"on external component(s) {', '.join(external)}"
                    ),
                    elements=tuple(external),
                )
            )

    # Sensitive data with no threat model is not a contradiction so much as an omission, but
    # it is one worth blocking: nobody decides later to threat-model something retroactively.
    handles_sensitive = ProductProperty.SENSITIVE_DATA in required or any(
        entity.sensitive for entity in architecture.entities
    )
    if handles_sensitive and not architecture.threats:
        findings.append(
            Contradiction(
                code=SENSITIVE_DATA_UNTHREATENED,
                between=("PRD", "SYSTEM_ARCHITECTURE"),
                detail=(
                    "the product handles sensitive data but the architecture records no "
                    "threats; an empty threat model is a claim that nothing can go wrong"
                ),
            )
        )

    # A product needing an interface cannot be built headless.
    ui_required = bool(
        {ProductProperty.MOBILE_RESPONSIVE, ProductProperty.ACCESSIBLE} & required
    )
    if ui_required and ProductProperty.HEADLESS in provided:
        findings.append(
            Contradiction(
                code=UI_REQUIRED_BUT_HEADLESS,
                between=("PRD", "SYSTEM_ARCHITECTURE"),
                detail=(
                    "the PRD requires a user interface but the architecture declares the "
                    "product headless"
                ),
            )
        )
    return findings


def check_prd_against_ux(prd: PRDDocument, ux: UXSpecDocument) -> list[Contradiction]:
    """Contradictions between requirements and the interface designed for them."""
    return compare_properties(prd, ux)


def check_ux_against_architecture(
    ux: UXSpecDocument, architecture: SystemArchitectureDocument
) -> list[Contradiction]:
    """Contradictions between the interface and the system meant to serve it.

    M4.8's third example: a responsive specification against a desktop-only architecture.
    """
    findings = compare_properties(ux, architecture)

    if ux.screens and ProductProperty.HEADLESS in expand(architecture.properties):
        findings.append(
            Contradiction(
                code=UI_REQUIRED_BUT_HEADLESS,
                between=("UX_SPEC", "SYSTEM_ARCHITECTURE"),
                detail=(
                    f"the UX specification defines {len(ux.screens)} screen(s) but the "
                    f"architecture declares the product headless"
                ),
            )
        )
    return findings


def check_all(
    *,
    prd: PRDDocument | None = None,
    ux: UXSpecDocument | None = None,
    architecture: SystemArchitectureDocument | None = None,
    include_hints: bool = True,
) -> tuple[Contradiction, ...]:
    """Run every applicable contradiction check across the artifacts provided.

    Missing documents are skipped rather than treated as empty: a project with no
    architecture yet has nothing to contradict, which is different from having an
    architecture that agrees with everything.
    """
    findings: list[Contradiction] = []

    if prd is not None and architecture is not None:
        findings.extend(check_prd_against_architecture(prd, architecture))
    if prd is not None and ux is not None:
        findings.extend(check_prd_against_ux(prd, ux))
    if ux is not None and architecture is not None:
        findings.extend(check_ux_against_architecture(ux, architecture))

    if include_hints:
        for document in (prd, ux, architecture):
            if document is not None:
                findings.extend(prose_hints(document))

    blocking = sum(1 for finding in findings if finding.blocking)
    logger.info(
        "product.contradictions",
        total=len(findings),
        blocking=blocking,
        advisory=len(findings) - blocking,
    )
    # Blocking first, then by code, so a report leads with what actually stops approval.
    return tuple(
        sorted(findings, key=lambda item: (not item.blocking, item.code, item.detail))
    )


def blocking_contradictions(
    findings: tuple[Contradiction, ...],
) -> tuple[Contradiction, ...]:
    """Only the findings that must prevent approval."""
    return tuple(finding for finding in findings if finding.blocking)
