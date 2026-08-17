"""The M4.1 experiment: does decomposition extract more reliable work from a fixed model?

    Hypothesis: decomposing a large structured-generation task into smaller validated
    inference steps improves the reliability of a fixed 3B local coding model.

    Independent variable: generation strategy (monolithic vs decomposed).
    Dependent variables: validity, completeness, requirement coverage, runtime, model
    calls, context cost.

The measurement that matters is **not** "did the model return JSON". M4.1 item 11 is
explicit: a successful trial needs valid structure, valid references, requirement coverage,
no blocking contradictions, correct authority, every required stage, and a persisted
artifact. :class:`TrialOutcome` records each of those separately, so a run that produced
well-formed nonsense is distinguishable from one that produced a usable specification.

Trials are independent and nothing is discarded. A model this small has real variance, and
the comparison is only meaningful across several runs of each arm.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from edith.agents.architect import (
    ArchitectOutput,
    draft_to_architecture,
    draft_to_plan,
)
from edith.agents.ux_designer import (
    SYSTEM_PROMPT as UX_MONOLITHIC_SYSTEM,
)
from edith.agents.ux_designer import (
    USER_TEMPLATE as UX_MONOLITHIC_USER,
)
from edith.agents.ux_designer import (
    UXDesignerOutput,
    draft_to_ux_spec,
)
from edith.models.base import ModelProvider
from edith.observability.logging import get_logger
from edith.product.architect_pipeline import run_architect_pipeline
from edith.product.artifacts import (
    Artifact,
    ArtifactKind,
    ArtifactStatus,
    build_artifact,
)
from edith.product.contradictions import check_all
from edith.product.prd import PRDDocument
from edith.product.stages import StageLedger, run_stage
from edith.product.store import ProductStore
from edith.product.ux import UXSpecDocument
from edith.product.ux_pipeline import run_ux_pipeline
from edith.product.validation import validate_against_dependencies
from edith.schemas.model import Message, Role

logger = get_logger(__name__)


@dataclass
class TrialOutcome:
    """One trial of one arm, measured against every quality gate.

    ``schema_valid`` and ``successful`` are deliberately separate. The first says the model
    produced parseable, schema-conforming output; the second says the result is a usable
    artifact. M4 produced runs that would have passed the first and failed the second, which
    is exactly why both are recorded.
    """

    arm: str
    trial: int
    #: Every stage produced output its schema accepted.
    schema_valid: bool = False
    #: A document was assembled at all.
    assembled: bool = False
    #: Every attempted stage succeeded; nothing is missing.
    complete: bool = False
    #: The artifact passed validation with no blocking issues.
    artifact_valid: bool = False
    #: No blocking contradiction against the PRD.
    no_blocking_contradictions: bool = False
    #: Fraction of MUST/SHOULD requirements the artifact addresses.
    requirement_coverage: float = 0.0
    #: The artifact reached durable storage.
    persisted: bool = False
    #: Authority is system-controlled: a fresh artifact is a DRAFT recommendation.
    authority_correct: bool = False

    model_calls: int = 0
    attempts: int = 0
    duration_seconds: float = 0.0
    input_chars: int = 0
    output_chars: int = 0
    largest_schema_bytes: int = 0
    largest_input_chars: int = 0
    stages_valid: int = 0
    stages_failed: int = 0
    failures: tuple[str, ...] = ()
    detail: str = ""

    @property
    def successful(self) -> bool:
        """Whether this trial produced a usable artifact by every gate M4.1 lists."""
        return (
            self.assembled
            and self.complete
            and self.artifact_valid
            and self.no_blocking_contradictions
            and self.persisted
            and self.authority_correct
        )

    def as_dict(self) -> dict[str, Any]:
        """Machine-readable summary."""
        return {
            "arm": self.arm,
            "trial": self.trial,
            "successful": self.successful,
            "schema_valid": self.schema_valid,
            "assembled": self.assembled,
            "complete": self.complete,
            "artifact_valid": self.artifact_valid,
            "no_blocking_contradictions": self.no_blocking_contradictions,
            "requirement_coverage": round(self.requirement_coverage, 3),
            "persisted": self.persisted,
            "authority_correct": self.authority_correct,
            "model_calls": self.model_calls,
            "attempts": self.attempts,
            "duration_seconds": round(self.duration_seconds, 2),
            "input_chars": self.input_chars,
            "output_chars": self.output_chars,
            "largest_schema_bytes": self.largest_schema_bytes,
            "largest_input_chars": self.largest_input_chars,
            "stages_valid": self.stages_valid,
            "stages_failed": self.stages_failed,
            "failures": list(self.failures),
            "detail": self.detail[:300],
        }


@dataclass
class ArmSummary:
    """Aggregated results for one arm. Nothing is discarded."""

    arm: str
    trials: list[TrialOutcome] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.trials)

    def _rate(self, predicate: Callable[[TrialOutcome], bool]) -> float:
        return (
            sum(1 for trial in self.trials if predicate(trial)) / self.count
            if self.count
            else 0.0
        )

    def _mean(self, attribute: str) -> float:
        values = [getattr(trial, attribute) for trial in self.trials]
        return sum(values) / len(values) if values else 0.0

    def as_dict(self) -> dict[str, Any]:
        """Machine-readable summary, including every trial."""
        return {
            "arm": self.arm,
            "trials": self.count,
            "successful": sum(1 for trial in self.trials if trial.successful),
            "success_rate": round(self._rate(lambda t: t.successful), 3),
            "schema_valid": sum(1 for trial in self.trials if trial.schema_valid),
            "assembled": sum(1 for trial in self.trials if trial.assembled),
            "complete": sum(1 for trial in self.trials if trial.complete),
            "artifact_valid": sum(1 for trial in self.trials if trial.artifact_valid),
            "no_blocking_contradictions": sum(
                1 for trial in self.trials if trial.no_blocking_contradictions
            ),
            "mean_requirement_coverage": round(self._mean("requirement_coverage"), 3),
            "mean_model_calls": round(self._mean("model_calls"), 1),
            "mean_duration_seconds": round(self._mean("duration_seconds"), 1),
            "mean_input_chars": round(self._mean("input_chars")),
            "mean_output_chars": round(self._mean("output_chars")),
            "largest_schema_bytes": max(
                (trial.largest_schema_bytes for trial in self.trials), default=0
            ),
            "largest_input_chars": max(
                (trial.largest_input_chars for trial in self.trials), default=0
            ),
            "failures": _count(
                failure for trial in self.trials for failure in trial.failures
            ),
            "per_trial": [trial.as_dict() for trial in self.trials],
        }


def _count(values: Any) -> dict[str, int]:
    """Frequency table, for failure classifications."""
    counts: dict[str, int] = {}
    for value in values:
        counts[str(value)] = counts.get(str(value), 0) + 1
    return counts


def _coverage(prd: PRDDocument, covered: frozenset[str]) -> float:
    """Fraction of MUST/SHOULD requirements a document addresses."""
    required = {
        item.requirement_id
        for item in prd.requirements
        if item.priority.value in {"MUST", "SHOULD"}
    }
    if not required:
        return 1.0
    return len(required & covered) / len(required)


def _evaluate_ux(
    outcome: TrialOutcome,
    document: UXSpecDocument,
    prd: PRDDocument,
    prd_artifact: Artifact,
    store: ProductStore,
    project_id: str,
) -> None:
    """Apply every quality gate to an assembled UX specification."""
    outcome.assembled = True
    outcome.requirement_coverage = _coverage(prd, document.covered_requirements)

    artifact = build_artifact(
        kind=ArtifactKind.UX_SPEC,
        project_id=project_id,
        title=f"{document.product_name} UX specification",
        author="ux_designer",
        document=document,
        depends_on=(prd_artifact.ref,),
    )
    validation = validate_against_dependencies(artifact, (prd_artifact,))
    artifact = artifact.model_copy(update={"validation": validation})
    outcome.artifact_valid = validation.valid

    findings = check_all(prd=prd, ux=document, include_hints=False)
    outcome.no_blocking_contradictions = not any(item.blocking for item in findings)

    store.save(artifact)
    stored = store.get(artifact.artifact_id)
    outcome.persisted = stored is not None
    # Authority is system-controlled: a freshly generated artifact is a draft
    # recommendation, never an approved requirement, whatever the model said.
    outcome.authority_correct = (
        stored is not None
        and stored.status is ArtifactStatus.DRAFT
        and stored.authority.value == "AGENT_RECOMMENDATION"
    )


def run_ux_monolithic_trial(
    provider: ModelProvider,
    prd: PRDDocument,
    prd_artifact: Artifact,
    store: ProductStore,
    *,
    project_id: str,
    trial: int,
    constraints: str = "",
) -> TrialOutcome:
    """One trial of the M4 monolithic UX path: a single large structured call."""
    outcome = TrialOutcome(arm="ux_monolithic", trial=trial)
    started = time.monotonic()

    user = UX_MONOLITHIC_USER.format(
        prd=prd.render(), design_system="(none exists yet)", research="(none)"
    )
    result = run_stage(
        "ux.monolithic",
        UXDesignerOutput,
        lambda: provider.structured_generate(
            [
                Message(role=Role.SYSTEM, content=UX_MONOLITHIC_SYSTEM),
                Message(role=Role.USER, content=user),
            ],
            UXDesignerOutput,
            max_repair_attempts=2,
        ),
        prompt_chars=len(UX_MONOLITHIC_SYSTEM) + len(user),
        elements_of=lambda output: len(output.flows) + len(output.screens),
    )

    measurement = result.measurement
    if measurement is not None:
        outcome.model_calls = measurement.model_calls
        outcome.attempts = measurement.attempts
        outcome.input_chars = measurement.input_chars
        outcome.output_chars = measurement.output_chars
        outcome.largest_schema_bytes = measurement.schema_bytes
        outcome.largest_input_chars = measurement.input_chars

    outcome.schema_valid = result.ok
    outcome.stages_valid = 1 if result.ok else 0
    outcome.stages_failed = 0 if result.ok else 1
    if not result.ok:
        outcome.failures = (str(result.failure),)
        outcome.detail = result.detail
        outcome.duration_seconds = time.monotonic() - started
        return outcome

    outcome.complete = True
    try:
        document = draft_to_ux_spec(
            UXDesignerOutput.model_validate(result.output.model_dump()),  # type: ignore[union-attr]
            prd=prd,
            product_name=prd.product_name,
        )
    except ValueError as exc:
        outcome.detail = f"assembly failed: {exc}"
        outcome.failures = ("ARTIFACT_VALIDATION_FAILURE",)
        outcome.duration_seconds = time.monotonic() - started
        return outcome

    _evaluate_ux(outcome, document, prd, prd_artifact, store, project_id)
    outcome.duration_seconds = time.monotonic() - started
    return outcome


def run_ux_decomposed_trial(
    provider: ModelProvider,
    prd: PRDDocument,
    prd_artifact: Artifact,
    store: ProductStore,
    *,
    project_id: str,
    trial: int,
    constraints: str = "",
) -> TrialOutcome:
    """One trial of the decomposed UX path: four small stages."""
    outcome = TrialOutcome(arm="ux_decomposed", trial=trial)
    started = time.monotonic()

    result = run_ux_pipeline(provider, prd, constraints=constraints)
    _apply_ledger(outcome, result.ledger)

    if result.document is None:
        outcome.detail = result.assembly_error or "no document was assembled"
        outcome.duration_seconds = time.monotonic() - started
        return outcome

    outcome.complete = result.complete
    _evaluate_ux(outcome, result.document, prd, prd_artifact, store, project_id)
    outcome.duration_seconds = time.monotonic() - started
    return outcome


def run_architect_trial(
    provider: ModelProvider,
    prd: PRDDocument,
    prd_artifact: Artifact,
    store: ProductStore,
    *,
    project_id: str,
    trial: int,
    ux: UXSpecDocument | None = None,
    constraints: str = "",
    arm: str = "architect_decomposed",
) -> TrialOutcome:
    """One trial of the decomposed architecture path."""
    outcome = TrialOutcome(arm=arm, trial=trial)
    started = time.monotonic()

    result = run_architect_pipeline(provider, prd, ux=ux, constraints=constraints)
    _apply_ledger(outcome, result.ledger)

    if result.architecture is None:
        outcome.detail = result.assembly_error or "no architecture was assembled"
        outcome.duration_seconds = time.monotonic() - started
        return outcome

    outcome.assembled = True
    outcome.complete = result.complete
    outcome.requirement_coverage = _coverage(
        prd, result.architecture.covered_requirements
    )

    artifact = build_artifact(
        kind=ArtifactKind.SYSTEM_ARCHITECTURE,
        project_id=project_id,
        title=f"{result.architecture.product_name} architecture",
        author="architect",
        document=result.architecture,
        depends_on=(prd_artifact.ref,),
    )
    validation = validate_against_dependencies(artifact, (prd_artifact,))
    artifact = artifact.model_copy(update={"validation": validation})
    outcome.artifact_valid = validation.valid

    findings = check_all(prd=prd, ux=ux, architecture=result.architecture, include_hints=False)
    outcome.no_blocking_contradictions = not any(item.blocking for item in findings)

    store.save(artifact)
    stored = store.get(artifact.artifact_id)
    outcome.persisted = stored is not None
    outcome.authority_correct = (
        stored is not None
        and stored.status is ArtifactStatus.DRAFT
        and stored.authority.value == "AGENT_RECOMMENDATION"
    )

    if result.plan is not None:
        plan_artifact = build_artifact(
            kind=ArtifactKind.IMPLEMENTATION_PLAN,
            project_id=project_id,
            title=f"{result.plan.product_name} implementation plan",
            author="architect",
            document=result.plan,
            depends_on=(artifact.ref, prd_artifact.ref),
        )
        plan_validation = validate_against_dependencies(
            plan_artifact, (artifact, prd_artifact)
        )
        plan_artifact = plan_artifact.model_copy(update={"validation": plan_validation})
        store.save(plan_artifact)
        outcome.artifact_valid = outcome.artifact_valid and plan_validation.valid
    else:
        outcome.complete = False
        outcome.detail = "no implementation plan was produced"

    outcome.duration_seconds = time.monotonic() - started
    return outcome


def run_architect_monolithic_trial(
    provider: ModelProvider,
    prd: PRDDocument,
    prd_artifact: Artifact,
    store: ProductStore,
    *,
    project_id: str,
    trial: int,
    ux: UXSpecDocument | None = None,
    constraints: str = "",
) -> TrialOutcome:
    """One trial of the M4 monolithic architecture path, for comparison."""
    from edith.agents.architect import (  # noqa: PLC0415 - avoids a prompt-name clash
        SYSTEM_PROMPT,
        USER_TEMPLATE,
    )

    outcome = TrialOutcome(arm="architect_monolithic", trial=trial)
    started = time.monotonic()

    user = USER_TEMPLATE.format(
        prd=prd.render(),
        ux=ux.render() if ux else "(no UX specification)",
        constraints=constraints or "(none stated)",
        research="(none)",
    )
    result = run_stage(
        "arch.monolithic",
        ArchitectOutput,
        lambda: provider.structured_generate(
            [
                Message(role=Role.SYSTEM, content=SYSTEM_PROMPT),
                Message(role=Role.USER, content=user),
            ],
            ArchitectOutput,
            max_repair_attempts=2,
        ),
        prompt_chars=len(SYSTEM_PROMPT) + len(user),
        elements_of=lambda output: len(output.components) + len(output.tasks),
    )

    measurement = result.measurement
    if measurement is not None:
        outcome.model_calls = measurement.model_calls
        outcome.attempts = measurement.attempts
        outcome.input_chars = measurement.input_chars
        outcome.output_chars = measurement.output_chars
        outcome.largest_schema_bytes = measurement.schema_bytes
        outcome.largest_input_chars = measurement.input_chars

    outcome.schema_valid = result.ok
    outcome.stages_valid = 1 if result.ok else 0
    outcome.stages_failed = 0 if result.ok else 1
    if not result.ok or not isinstance(result.output, ArchitectOutput):
        outcome.failures = (str(result.failure),)
        outcome.detail = result.detail
        outcome.duration_seconds = time.monotonic() - started
        return outcome

    outcome.complete = True
    try:
        architecture = draft_to_architecture(
            result.output, prd=prd, product_name=prd.product_name
        )
        plan = draft_to_plan(
            result.output, architecture, prd=prd, product_name=prd.product_name
        )
    except ValueError as exc:
        outcome.detail = f"assembly failed: {exc}"
        outcome.failures = ("ARTIFACT_VALIDATION_FAILURE",)
        outcome.duration_seconds = time.monotonic() - started
        return outcome

    outcome.assembled = True
    outcome.requirement_coverage = _coverage(prd, architecture.covered_requirements)

    artifact = build_artifact(
        kind=ArtifactKind.SYSTEM_ARCHITECTURE,
        project_id=project_id,
        title=f"{architecture.product_name} architecture",
        author="architect",
        document=architecture,
        depends_on=(prd_artifact.ref,),
    )
    validation = validate_against_dependencies(artifact, (prd_artifact,))
    artifact = artifact.model_copy(update={"validation": validation})
    outcome.artifact_valid = validation.valid

    findings = check_all(prd=prd, ux=ux, architecture=architecture, include_hints=False)
    outcome.no_blocking_contradictions = not any(item.blocking for item in findings)

    store.save(artifact)
    stored = store.get(artifact.artifact_id)
    outcome.persisted = stored is not None
    outcome.authority_correct = (
        stored is not None
        and stored.status is ArtifactStatus.DRAFT
        and stored.authority.value == "AGENT_RECOMMENDATION"
    )
    _ = plan
    outcome.duration_seconds = time.monotonic() - started
    return outcome


def _apply_ledger(outcome: TrialOutcome, ledger: StageLedger) -> None:
    """Copy a ledger's measurements onto a trial outcome."""
    totals = ledger.totals()
    outcome.model_calls = int(totals["model_calls"])
    outcome.attempts = int(totals["attempts"])
    outcome.input_chars = int(totals["input_chars"])
    outcome.output_chars = int(totals["output_chars"])
    outcome.largest_schema_bytes = int(totals["largest_schema_bytes"])
    outcome.largest_input_chars = int(totals["largest_input_chars"])
    outcome.stages_valid = int(totals["stages_valid"])
    outcome.stages_failed = int(totals["stages_failed"])
    outcome.schema_valid = outcome.stages_failed == 0
    outcome.failures = tuple(str(failure) for failure in ledger.failures)
