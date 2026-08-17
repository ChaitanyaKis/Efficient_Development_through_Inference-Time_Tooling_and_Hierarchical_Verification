"""The product pipeline, and the API a future UI will drive it through.

M4.15 asks for clean interfaces a UI can use to select an agent, run it, inspect artifacts,
and see project state -- without the UI knowing anything about orchestration. This module is
that boundary: :class:`ProductService` is the whole surface, and everything it returns is a
serialisable schema object rather than a live handle into the engine.

The pipeline it exposes is the M4 workflow:

    idea -> PRD -> UX specification -> architecture -> implementation plan

Each stage validates before it stores, and refuses to build on an artifact that did not
validate. A UI can therefore drive the pipeline one stage at a time, show the findings, let a
human approve, and continue -- which is the interaction the milestone describes, without any
of it being special-cased for the UI.

Instrumentation is built in rather than bolted on. M4.11 warns against guessing context sizes
for product artifacts, which are much larger than coding tasks, so every stage records what
it sent, what came back, and how long it took. The numbers are evidence for setting budgets
later, not budgets set now.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from edith.agents.architect import (
    ArchitectAgent,
    ArchitectInput,
    ArchitectOutput,
    draft_to_architecture,
    draft_to_plan,
)
from edith.agents.product_manager import (
    ProductManagerAgent,
    ProductManagerInput,
    ProductManagerOutput,
    draft_to_prd,
)
from edith.agents.ux_designer import (
    UXDesignerAgent,
    UXDesignerInput,
    UXDesignerOutput,
    draft_to_ux_spec,
)
from edith.config.schema import EdithConfig
from edith.errors import AgentExecutionError, EdithError, FailureCategory
from edith.models.base import ModelProvider
from edith.observability.logging import bind_context, clear_context, get_logger
from edith.schemas.agent import AgentRequest, TaskRef
from edith.schemas.common import Verdict

from .architect_pipeline import run_architect_pipeline
from .architecture import ImplementationPlanDocument, SystemArchitectureDocument
from .artifacts import (
    Artifact,
    ArtifactKind,
    ArtifactStatus,
    ValidationIssue,
    ValidationOutcome,
    ValidationState,
    build_artifact,
)
from .completion import complete_architecture_coverage, complete_ux_coverage
from .contradictions import Contradiction, check_all
from .coverage import (
    DEFAULT_THRESHOLD,
    CoverageMatrix,
    CoverageThreshold,
    analyse_coverage,
)
from .prd import PRDDocument
from .review import (
    ReviewDocument,
    overall_verdict,
    review_plan,
    review_prd_for_feasibility,
    review_prd_for_ux_coverage,
    review_requirement_quality,
)
from .store import ProductStore, approve
from .ux import UXSpecDocument
from .ux_pipeline import run_ux_pipeline
from .validation import validate_against_dependencies, validate_artifact

logger = get_logger(__name__)


def _apply_coverage(
    validation: ValidationOutcome,
    matrix: CoverageMatrix,
    kind: ArtifactKind,
    threshold: CoverageThreshold,
) -> ValidationOutcome:
    """Fold coverage gaps into an artifact's validation outcome.

    This is what makes M4.2's approval gate real. A critical requirement the artifact does
    not address becomes a *blocking* validation issue, so the existing approval path refuses
    it by the same mechanism that refuses a dangling reference -- no separate gate to
    remember, and no way to approve an artifact that does not do what was asked.

    Non-critical gaps are recorded as advisory issues: visible in every report, but they do
    not stop a project shipping something that deliberately omits a "could-have".
    """
    issues = [
        ValidationIssue(
            code=gap.code,
            message=gap.render(),
            element_id=gap.requirement_id,
            blocking=gap.blocking,
        )
        for gap in matrix.gaps_for(kind)
    ]
    if not issues:
        return validation

    combined = [*validation.issues, *issues]
    blocking = any(issue.blocking for issue in combined)
    return validation.model_copy(
        update={
            "issues": combined,
            "state": ValidationState.INVALID if blocking else validation.state,
        }
    )


@dataclass
class StageMetrics:
    """What one pipeline stage cost.

    M4.11 says not to guess context sizes for product artifacts. These are the measurements
    that would justify a budget, recorded from the first run rather than added once something
    goes wrong.
    """

    stage: str
    #: Characters sent to the model, across every message.
    input_chars: int = 0
    #: Characters of structured output returned.
    output_chars: int = 0
    #: Characters of the resulting artifact document, which is what downstream stages pay for.
    artifact_chars: int = 0
    model_calls: int = 0
    attempts: int = 0
    duration_seconds: float = 0.0
    #: Elements produced: requirements, flows, components, tasks.
    elements: int = 0

    def as_dict(self) -> dict[str, Any]:
        """Machine-readable summary."""
        return {
            "stage": self.stage,
            "input_chars": self.input_chars,
            "output_chars": self.output_chars,
            "artifact_chars": self.artifact_chars,
            "model_calls": self.model_calls,
            "attempts": self.attempts,
            "duration_seconds": round(self.duration_seconds, 2),
            "elements": self.elements,
        }


@dataclass
class StageResult:
    """The outcome of one pipeline stage.

    Always returned, never raised past, so a UI can render a failure the same way it renders
    a success. A stage that could throw would force every caller to reimplement error
    presentation.
    """

    stage: str
    ok: bool
    artifact: Artifact | None = None
    error: str = ""
    failure_category: FailureCategory | None = None
    contradictions: tuple[Contradiction, ...] = ()
    reviews: tuple[ReviewDocument, ...] = ()
    metrics: StageMetrics | None = None

    @property
    def blocked(self) -> bool:
        """Whether a blocking problem stops the pipeline continuing from here."""
        if not self.ok:
            return True
        if any(finding.blocking for finding in self.contradictions):
            return True
        return any(review.blockers for review in self.reviews)

    def summary(self) -> str:
        """A one-line description for a human."""
        if not self.ok:
            return f"{self.stage}: FAILED - {self.error}"
        artifact = self.artifact
        state = artifact.validation.summary() if artifact else "no artifact"
        blockers = sum(len(review.blockers) for review in self.reviews)
        blocking = sum(1 for item in self.contradictions if item.blocking)
        extra = []
        if blockers:
            extra.append(f"{blockers} review blocker(s)")
        if blocking:
            extra.append(f"{blocking} contradiction(s)")
        tail = f" [{'; '.join(extra)}]" if extra else ""
        return f"{self.stage}: {state}{tail}"


class ProductService:
    """Drive the product-development pipeline over one project.

    The single interface a UI, the CLI, or a future orchestrator uses. It owns no state
    beyond its collaborators: artifacts live in the store, and every method is safe to call
    on a project someone else has been working on.
    """

    def __init__(
        self,
        config: EdithConfig,
        store: ProductStore,
        *,
        provider: ModelProvider | None = None,
        coverage_threshold: CoverageThreshold | None = None,
        targeted_completion: bool = False,
    ) -> None:
        """
        Args:
            config: Resolved configuration, for agent defaults.
            store: Where artifacts are persisted.
            provider: Model provider. Injected so tests and a UI can supply their own; a
                stage that needs one and finds none fails cleanly rather than constructing
                a runtime behind the caller's back.
        """
        self.config = config
        self.store = store
        self._provider = provider
        #: The coverage policy the approval gate enforces. Explicit and overridable, so a
        #: project with different standards states them rather than editing the engine.
        self.coverage_threshold = coverage_threshold or DEFAULT_THRESHOLD
        #: Whether to run a targeted completion pass over coverage gaps. Measured in
        #: experiment 0002; off by default until that measurement says otherwise.
        self.targeted_completion = targeted_completion

    # -- Agent surface (M4.15: what a UI selects and talks to) ------------------------

    def available_agents(self) -> tuple[dict[str, Any], ...]:
        """The product agents a UI may offer, with their declared permissions.

        Returned as plain data so a UI never imports an agent class. Permissions are included
        because a user choosing an agent should be able to see what it is allowed to touch.
        """
        return tuple(
            {
                "name": agent.identity.name,
                "description": agent.identity.description,
                "capabilities": sorted(str(item) for item in agent.identity.capabilities),
                "produces": kind.value,
                "read_paths": list(agent.identity.permissions.allowed_read_paths),
                "write_paths": list(agent.identity.permissions.allowed_write_paths),
                "tools": sorted(agent.identity.permissions.allowed_tools),
            }
            for agent, kind in (
                (ProductManagerAgent, ArtifactKind.PRD),
                (UXDesignerAgent, ArtifactKind.UX_SPEC),
                (ArchitectAgent, ArtifactKind.SYSTEM_ARCHITECTURE),
            )
        )

    # -- Pipeline stages ------------------------------------------------------------

    def create_prd(
        self,
        project_id: str,
        idea: str,
        *,
        constraints: str = "",
        research: str = "",
        prior_knowledge: str = "",
    ) -> StageResult:
        """Run the Product Manager and store a validated PRD draft."""
        bind_context(project_id=project_id)
        try:
            agent = ProductManagerAgent(
                provider=self._provider, settings=self.config.agents.for_agent("product_manager")
            )
            payload = ProductManagerInput(
                idea=idea,
                constraints=constraints,
                research=research,
                prior_knowledge=prior_knowledge,
            )
            response, metrics = self._invoke(agent, payload, "prd", project_id)
            if not response.ok:
                return StageResult(
                    stage="prd",
                    ok=False,
                    error=response.error or "the product manager failed",
                    failure_category=response.failure_category,
                    metrics=metrics,
                )

            draft = ProductManagerOutput.model_validate(response.output)
            document = draft_to_prd(draft, source=f"project:{project_id}")
            metrics.elements = len(document.requirements)
            metrics.artifact_chars = len(document.render())

            artifact = build_artifact(
                kind=ArtifactKind.PRD,
                project_id=project_id,
                title=f"{document.product_name} PRD",
                author=agent.name,
                document=document,
                source_references=(f"idea:{idea[:200]}",),
            )
            artifact = artifact.model_copy(
                update={"validation": validate_artifact(artifact)}
            )
            self.store.save(artifact)

            reviews = (review_requirement_quality(document),)
            return StageResult(
                stage="prd",
                ok=True,
                artifact=artifact,
                reviews=reviews,
                metrics=metrics,
            )
        except (EdithError, ValueError) as exc:
            return self._failed("prd", exc)
        finally:
            clear_context()

    def create_ux_spec(
        self,
        project_id: str,
        *,
        research: str = "",
        design_system: str = "",
        prior_knowledge: str = "",
        decomposed: bool = True,
    ) -> StageResult:
        """Run the UX agent against the project's current PRD.

        ``decomposed`` defaults to true because experiment 0001 measured the monolithic path
        at 0/5 and the decomposed path at 5/5 on the target model. The monolithic path is
        kept reachable so the comparison stays reproducible, not because it is a supported
        way to run the pipeline.
        """
        bind_context(project_id=project_id)
        try:
            prd_artifact = self._require(project_id, ArtifactKind.PRD)
            prd = PRDDocument.model_validate(prd_artifact.body)

            if decomposed:
                return self._create_ux_decomposed(
                    project_id, prd, prd_artifact, constraints=design_system
                )

            agent = UXDesignerAgent(
                provider=self._provider, settings=self.config.agents.for_agent("ux_designer")
            )
            payload = UXDesignerInput(
                prd=prd.render(),
                design_system=design_system,
                research=research,
                prior_knowledge=prior_knowledge,
            )
            response, metrics = self._invoke(agent, payload, "ux", project_id)
            if not response.ok:
                return StageResult(
                    stage="ux",
                    ok=False,
                    error=response.error or "the ux agent failed",
                    failure_category=response.failure_category,
                    metrics=metrics,
                )

            draft = UXDesignerOutput.model_validate(response.output)
            document = draft_to_ux_spec(draft, prd=prd, product_name=prd.product_name)
            metrics.elements = len(document.flows) + len(document.screens)
            metrics.artifact_chars = len(document.render())

            artifact = build_artifact(
                kind=ArtifactKind.UX_SPEC,
                project_id=project_id,
                title=f"{document.product_name} UX specification",
                author=agent.name,
                document=document,
                depends_on=(prd_artifact.ref,),
            )
            artifact = artifact.model_copy(
                update={
                    "validation": validate_against_dependencies(artifact, (prd_artifact,))
                }
            )
            self.store.save(artifact)

            return StageResult(
                stage="ux",
                ok=True,
                artifact=artifact,
                contradictions=check_all(prd=prd, ux=document, include_hints=False),
                reviews=(review_prd_for_ux_coverage(prd, document),),
                metrics=metrics,
            )
        except (EdithError, ValueError) as exc:
            return self._failed("ux", exc)
        finally:
            clear_context()

    def _create_ux_decomposed(
        self,
        project_id: str,
        prd: PRDDocument,
        prd_artifact: Artifact,
        *,
        constraints: str,
    ) -> StageResult:
        """Generate a UX specification through the decomposed stage pipeline.

        A partial run still stores its artifact. The document is validated and the ledger's
        failures are attached, so an operator can see what is missing and re-run one stage
        rather than the whole specification -- but the artifact is invalid, and the M4
        approval gate refuses to approve it.
        """
        provider = self._provider
        if provider is None:
            return StageResult(
                stage="ux",
                ok=False,
                error="no model provider was supplied",
                failure_category=FailureCategory.CONFIGURATION_ERROR,
            )

        outcome = run_ux_pipeline(provider, prd, constraints=constraints)
        totals = outcome.ledger.totals()
        metrics = StageMetrics(
            stage="ux",
            input_chars=int(totals["input_chars"]),
            output_chars=int(totals["output_chars"]),
            model_calls=int(totals["model_calls"]),
            attempts=int(totals["attempts"]),
            duration_seconds=float(totals["duration_seconds"]),
        )

        if outcome.document is None:
            return StageResult(
                stage="ux",
                ok=False,
                error=(
                    outcome.assembly_error
                    or f"stages failed: {', '.join(outcome.ledger.failed_stages)}"
                ),
                failure_category=FailureCategory.VALIDATION_FAILURE,
                metrics=metrics,
            )

        document = outcome.document

        if self.targeted_completion and self._provider is not None:
            completion = complete_ux_coverage(
                self._provider, prd, document, threshold=self.coverage_threshold
            )
            if completion.ux is not None:
                document = completion.ux
            totals = completion.ledger.totals()
            metrics.model_calls += int(totals["model_calls"])
            metrics.input_chars += int(totals["input_chars"])
            metrics.output_chars += int(totals["output_chars"])
            metrics.duration_seconds += float(totals["duration_seconds"])

        metrics.elements = len(document.flows) + len(document.screens)
        metrics.artifact_chars = len(document.render())

        artifact = build_artifact(
            kind=ArtifactKind.UX_SPEC,
            project_id=project_id,
            title=f"{document.product_name} UX specification",
            author="ux_designer",
            document=document,
            depends_on=(prd_artifact.ref,),
        )
        validation = validate_against_dependencies(artifact, (prd_artifact,))
        if not outcome.ledger.complete:
            # An incomplete run must never look like a complete one. Recording the missing
            # stages as blocking issues is what stops the approval gate accepting it.
            validation = validation.model_copy(
                update={
                    "state": ValidationState.INVALID,
                    "issues": [
                        *validation.issues,
                        *(
                            ValidationIssue(
                                code="STAGE_INCOMPLETE",
                                message=f"generation stage {stage} did not succeed",
                            )
                            for stage in outcome.ledger.failed_stages
                        ),
                    ],
                }
            )
        validation = _apply_coverage(
            validation,
            analyse_coverage(prd, ux=document),
            ArtifactKind.UX_SPEC,
            self.coverage_threshold,
        )
        artifact = artifact.model_copy(update={"validation": validation})
        self.store.save(artifact)

        return StageResult(
            stage="ux",
            ok=True,
            artifact=artifact,
            contradictions=check_all(prd=prd, ux=document, include_hints=False),
            reviews=(review_prd_for_ux_coverage(prd, document),),
            metrics=metrics,
        )

    def create_architecture(
        self,
        project_id: str,
        *,
        constraints: str = "",
        research: str = "",
        prior_knowledge: str = "",
        decomposed: bool = True,
    ) -> StageResult:
        """Run the Architect against the PRD and UX specification.

        Produces two artifacts: the architecture and the implementation plan. They are stored
        separately because they are revised at different rates -- a plan is re-cut often, an
        architecture rarely -- and versioning them together would make every plan change look
        like an architecture change.
        """
        bind_context(project_id=project_id)
        try:
            prd_artifact = self._require(project_id, ArtifactKind.PRD)
            prd = PRDDocument.model_validate(prd_artifact.body)
            ux_artifact = self.store.latest(project_id, ArtifactKind.UX_SPEC)
            ux = UXSpecDocument.model_validate(ux_artifact.body) if ux_artifact else None

            if decomposed:
                return self._create_architecture_decomposed(
                    project_id, prd, prd_artifact, ux, ux_artifact, constraints=constraints
                )

            agent = ArchitectAgent(
                provider=self._provider, settings=self.config.agents.for_agent("architect")
            )
            payload = ArchitectInput(
                prd=prd.render(),
                ux_spec=ux.render() if ux else "",
                constraints=constraints,
                research=research,
                prior_knowledge=prior_knowledge,
            )
            response, metrics = self._invoke(agent, payload, "architecture", project_id)
            if not response.ok:
                return StageResult(
                    stage="architecture",
                    ok=False,
                    error=response.error or "the architect failed",
                    failure_category=response.failure_category,
                    metrics=metrics,
                )

            draft = ArchitectOutput.model_validate(response.output)
            document = draft_to_architecture(
                draft, prd=prd, product_name=prd.product_name
            )
            plan = draft_to_plan(
                draft, document, prd=prd, product_name=prd.product_name
            )
            metrics.elements = len(document.components) + len(plan.tasks)
            metrics.artifact_chars = len(document.render()) + len(plan.render())

            dependencies = (prd_artifact,) + ((ux_artifact,) if ux_artifact else ())
            architecture_artifact = build_artifact(
                kind=ArtifactKind.SYSTEM_ARCHITECTURE,
                project_id=project_id,
                title=f"{document.product_name} architecture",
                author=agent.name,
                document=document,
                depends_on=tuple(item.ref for item in dependencies),
            )
            architecture_artifact = architecture_artifact.model_copy(
                update={
                    "validation": validate_against_dependencies(
                        architecture_artifact, dependencies
                    )
                }
            )
            self.store.save(architecture_artifact)

            plan_artifact = build_artifact(
                kind=ArtifactKind.IMPLEMENTATION_PLAN,
                project_id=project_id,
                title=f"{plan.product_name} implementation plan",
                author=agent.name,
                document=plan,
                depends_on=(architecture_artifact.ref, prd_artifact.ref),
            )
            plan_artifact = plan_artifact.model_copy(
                update={
                    "validation": validate_against_dependencies(
                        plan_artifact, (architecture_artifact, prd_artifact)
                    )
                }
            )
            self.store.save(plan_artifact)

            return StageResult(
                stage="architecture",
                ok=True,
                artifact=architecture_artifact,
                contradictions=check_all(
                    prd=prd, ux=ux, architecture=document, include_hints=False
                ),
                reviews=(
                    review_prd_for_feasibility(prd, document),
                    review_plan(plan, document, prd=prd),
                ),
                metrics=metrics,
            )
        except (EdithError, ValueError) as exc:
            return self._failed("architecture", exc)
        finally:
            clear_context()

    def _create_architecture_decomposed(
        self,
        project_id: str,
        prd: PRDDocument,
        prd_artifact: Artifact,
        ux: UXSpecDocument | None,
        ux_artifact: Artifact | None,
        *,
        constraints: str,
    ) -> StageResult:
        """Generate architecture and plan through the decomposed stage pipeline."""
        provider = self._provider
        if provider is None:
            return StageResult(
                stage="architecture",
                ok=False,
                error="no model provider was supplied",
                failure_category=FailureCategory.CONFIGURATION_ERROR,
            )

        outcome = run_architect_pipeline(provider, prd, ux=ux, constraints=constraints)
        totals = outcome.ledger.totals()
        metrics = StageMetrics(
            stage="architecture",
            input_chars=int(totals["input_chars"]),
            output_chars=int(totals["output_chars"]),
            model_calls=int(totals["model_calls"]),
            attempts=int(totals["attempts"]),
            duration_seconds=float(totals["duration_seconds"]),
        )

        if outcome.architecture is None:
            return StageResult(
                stage="architecture",
                ok=False,
                error=(
                    outcome.assembly_error
                    or f"stages failed: {', '.join(outcome.ledger.failed_stages)}"
                ),
                failure_category=FailureCategory.VALIDATION_FAILURE,
                metrics=metrics,
            )

        architecture = outcome.architecture

        if self.targeted_completion and self._provider is not None:
            completion = complete_architecture_coverage(
                self._provider,
                prd,
                architecture,
                ux=ux,
                threshold=self.coverage_threshold,
            )
            if completion.architecture is not None:
                architecture = completion.architecture
            totals = completion.ledger.totals()
            metrics.model_calls += int(totals["model_calls"])
            metrics.input_chars += int(totals["input_chars"])
            metrics.output_chars += int(totals["output_chars"])
            metrics.duration_seconds += float(totals["duration_seconds"])

        metrics.elements = len(architecture.components)
        metrics.artifact_chars = len(architecture.render())

        dependencies = (prd_artifact,) + ((ux_artifact,) if ux_artifact else ())
        artifact = build_artifact(
            kind=ArtifactKind.SYSTEM_ARCHITECTURE,
            project_id=project_id,
            title=f"{architecture.product_name} architecture",
            author="architect",
            document=architecture,
            depends_on=tuple(item.ref for item in dependencies),
        )
        validation = validate_against_dependencies(artifact, dependencies)
        if not outcome.ledger.complete:
            validation = validation.model_copy(
                update={
                    "state": ValidationState.INVALID,
                    "issues": [
                        *validation.issues,
                        *(
                            ValidationIssue(
                                code="STAGE_INCOMPLETE",
                                message=f"generation stage {stage} did not succeed",
                            )
                            for stage in outcome.ledger.failed_stages
                        ),
                    ],
                }
            )
        validation = _apply_coverage(
            validation,
            analyse_coverage(prd, ux=ux, architecture=architecture),
            ArtifactKind.SYSTEM_ARCHITECTURE,
            self.coverage_threshold,
        )
        artifact = artifact.model_copy(update={"validation": validation})
        self.store.save(artifact)

        reviews = [review_prd_for_feasibility(prd, architecture)]
        if outcome.plan is not None:
            metrics.elements += len(outcome.plan.tasks)
            plan_artifact = build_artifact(
                kind=ArtifactKind.IMPLEMENTATION_PLAN,
                project_id=project_id,
                title=f"{outcome.plan.product_name} implementation plan",
                author="architect",
                document=outcome.plan,
                depends_on=(artifact.ref, prd_artifact.ref),
            )
            plan_artifact = plan_artifact.model_copy(
                update={
                    "validation": validate_against_dependencies(
                        plan_artifact, (artifact, prd_artifact)
                    )
                }
            )
            self.store.save(plan_artifact)
            reviews.append(review_plan(outcome.plan, architecture, prd=prd))

        return StageResult(
            stage="architecture",
            ok=True,
            artifact=artifact,
            contradictions=check_all(
                prd=prd, ux=ux, architecture=architecture, include_hints=False
            ),
            reviews=tuple(reviews),
            metrics=metrics,
        )

    # -- Inspection (M4.14 and M4.15) -------------------------------------------------

    def artifacts(self, project_id: str) -> tuple[Artifact, ...]:
        """Every current artifact for a project."""
        return self.store.current(project_id)

    def artifact(
        self, project_id: str, kind: ArtifactKind, *, approved_only: bool = False
    ) -> Artifact | None:
        """The latest artifact of a kind, optionally restricted to approved ones."""
        status = ArtifactStatus.APPROVED if approved_only else None
        return self.store.latest(project_id, kind, status=status)

    def history(self, artifact_id: str) -> tuple[Artifact, ...]:
        """Every version of an artifact, newest first."""
        return self.store.history(artifact_id)

    def approve_artifact(self, artifact_id: str, version: int | None = None) -> Artifact:
        """Approve an artifact, refusing anything that did not validate.

        Raises:
            ArtifactConflictError: The artifact is unknown, invalid, or not approvable.
        """
        artifact = self.store.get(artifact_id, version)
        if artifact is None:
            raise AgentExecutionError(
                f"no artifact {artifact_id!r} exists",
                category=FailureCategory.CONFIGURATION_ERROR,
            )
        return approve(self.store, artifact)

    def contradictions(self, project_id: str) -> tuple[Contradiction, ...]:
        """Every contradiction across the project's current artifacts."""
        return check_all(**self._documents(project_id))

    def review_project(self, project_id: str) -> tuple[ReviewDocument, ...]:
        """Run every applicable cross-agent review over the project's artifacts."""
        documents = self._documents(project_id)
        prd = documents.get("prd")
        ux = documents.get("ux")
        architecture = documents.get("architecture")
        if prd is None:
            return ()

        reviews = [review_requirement_quality(prd)]
        if ux is not None:
            reviews.append(review_prd_for_ux_coverage(prd, ux))
        if architecture is not None:
            reviews.append(review_prd_for_feasibility(prd, architecture))
            plan_artifact = self.store.latest(project_id, ArtifactKind.IMPLEMENTATION_PLAN)
            if plan_artifact is not None:
                plan = ImplementationPlanDocument.model_validate(plan_artifact.body)
                reviews.append(review_plan(plan, architecture, prd=prd))
        return tuple(reviews)

    def status(self, project_id: str) -> dict[str, Any]:
        """A serialisable snapshot of where the project stands.

        The one call a UI needs to render a project overview: which artifacts exist, whether
        they validated, what contradicts what, and whether anything blocks progress.
        """
        artifacts = self.artifacts(project_id)
        reviews = self.review_project(project_id)
        found = self.contradictions(project_id)
        return {
            "project_id": project_id,
            "artifacts": [
                {
                    "artifact_id": item.artifact_id,
                    "kind": item.kind.value,
                    "version": item.version,
                    "title": item.title,
                    "status": item.status.value,
                    "author": item.author,
                    "validation": item.validation.state.value,
                    "issues": len(item.validation.issues),
                }
                for item in artifacts
            ],
            "contradictions": [item.render() for item in found],
            "blocking_contradictions": sum(1 for item in found if item.blocking),
            "reviews": [
                {
                    "reviewer": review.reviewer,
                    "verdict": review.verdict.value,
                    "findings": len(review.findings),
                    "blockers": len(review.blockers),
                }
                for review in reviews
            ],
            "verdict": overall_verdict(reviews).value if reviews else Verdict.PASS.value,
        }

    # -- Internals -------------------------------------------------------------------

    def _documents(self, project_id: str) -> dict[str, Any]:
        """Load the project's current documents, skipping any that are absent."""
        result: dict[str, Any] = {}
        prd_artifact = self.store.latest(project_id, ArtifactKind.PRD)
        if prd_artifact is not None:
            result["prd"] = PRDDocument.model_validate(prd_artifact.body)
        ux_artifact = self.store.latest(project_id, ArtifactKind.UX_SPEC)
        if ux_artifact is not None:
            result["ux"] = UXSpecDocument.model_validate(ux_artifact.body)
        architecture_artifact = self.store.latest(
            project_id, ArtifactKind.SYSTEM_ARCHITECTURE
        )
        if architecture_artifact is not None:
            result["architecture"] = SystemArchitectureDocument.model_validate(
                architecture_artifact.body
            )
        return result

    def _require(self, project_id: str, kind: ArtifactKind) -> Artifact:
        """Fetch an artifact a stage depends on, refusing to build on an invalid one."""
        artifact = self.store.latest(project_id, kind)
        if artifact is None:
            raise AgentExecutionError(
                f"project {project_id!r} has no {kind.value} yet; produce one first",
                category=FailureCategory.REQUIREMENT_FAILURE,
            )
        if artifact.validation.state.value == "INVALID":
            raise AgentExecutionError(
                f"the project's {kind.value} did not validate "
                f"({artifact.validation.summary()}); fix it before building on it",
                category=FailureCategory.REQUIREMENT_FAILURE,
            )
        return artifact

    def _invoke(
        self, agent: Any, payload: Any, stage: str, project_id: str
    ) -> tuple[Any, StageMetrics]:
        """Run an agent and record what the call cost."""
        metrics = StageMetrics(stage=stage)
        metrics.input_chars = sum(
            len(str(value)) for value in payload.model_dump().values()
        )
        request = AgentRequest(
            payload=payload.model_dump(),
            task=TaskRef(project_id=project_id, title=stage),
        )
        response = agent.execute(request)
        metrics.attempts = response.attempts
        metrics.model_calls = response.attempts
        metrics.duration_seconds = response.duration_seconds
        metrics.output_chars = len(str(response.output))
        logger.info("product.stage", **metrics.as_dict())
        return (response, metrics)

    @staticmethod
    def _failed(stage: str, exc: Exception) -> StageResult:
        """Build a failed stage result from an exception."""
        category = (
            exc.category if isinstance(exc, EdithError) else FailureCategory.VALIDATION_FAILURE
        )
        message = exc.message if isinstance(exc, EdithError) else str(exc)
        logger.warning("product.stage_failed", stage=stage, error=message)
        return StageResult(
            stage=stage, ok=False, error=message, failure_category=category
        )


@dataclass
class PipelineResult:
    """The result of running the whole pipeline."""

    project_id: str
    stages: list[StageResult] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """Whether every stage completed."""
        return bool(self.stages) and all(stage.ok for stage in self.stages)

    @property
    def blocked(self) -> bool:
        """Whether any stage produced a blocking finding."""
        return any(stage.blocked for stage in self.stages)

    def summary(self) -> str:
        """A short report over every stage."""
        return "\n".join(stage.summary() for stage in self.stages)

    def total_metrics(self) -> dict[str, Any]:
        """Aggregated instrumentation across the pipeline."""
        metrics = [stage.metrics for stage in self.stages if stage.metrics]
        return {
            "input_chars": sum(item.input_chars for item in metrics),
            "output_chars": sum(item.output_chars for item in metrics),
            "artifact_chars": sum(item.artifact_chars for item in metrics),
            "model_calls": sum(item.model_calls for item in metrics),
            "duration_seconds": round(
                sum(item.duration_seconds for item in metrics), 2
            ),
            "stages": [item.as_dict() for item in metrics],
        }


def run_pipeline(
    service: ProductService,
    project_id: str,
    idea: str,
    *,
    constraints: str = "",
    research: str = "",
    stop_on_block: bool = True,
    decomposed: bool = True,
) -> PipelineResult:
    """Run idea -> PRD -> UX -> architecture, stopping when a stage blocks.

    ``stop_on_block`` defaults to true because building an architecture on a PRD that
    contradicts itself produces a design nobody should read. A caller that wants to see the
    whole pipeline's output despite findings can turn it off deliberately.
    """
    result = PipelineResult(project_id=project_id)

    prd_stage = service.create_prd(
        project_id, idea, constraints=constraints, research=research
    )
    result.stages.append(prd_stage)
    if not prd_stage.ok or (stop_on_block and prd_stage.blocked):
        return result

    ux_stage = service.create_ux_spec(
        project_id, research=research, decomposed=decomposed
    )
    result.stages.append(ux_stage)
    if not ux_stage.ok or (stop_on_block and ux_stage.blocked):
        return result

    architecture_stage = service.create_architecture(
        project_id, constraints=constraints, research=research, decomposed=decomposed
    )
    result.stages.append(architecture_stage)
    return result
