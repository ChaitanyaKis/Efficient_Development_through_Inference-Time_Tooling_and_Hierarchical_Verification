"""The Master Orchestrator.

Drives the loop: plan -> context -> code -> verify -> judge -> (debug -> retry) -> report.

It orchestrates and nothing else. It does not read files, write files, run commands, or
decide whether code is correct -- those belong to the tools, the verification runner, and
the Critic respectively. What it owns is *sequencing, persistence, and policy*: which agent
runs next, what gets recorded, and when to stop.

Three properties are structural:

- **Bounded.** Every loop has an explicit ceiling: attempts per task, repairs per task, and
  a total agent-run budget as a backstop.
- **Persistent.** State is written before and after every step, so a kill at any point
  leaves a resumable execution.
- **Scoped.** Each agent receives a gateway narrowed to its task's declared scope, so the
  orchestration layer cannot widen what M1 permits.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from edith.agents.coder import CoderInput, CoderOutput, CodingAgent
from edith.agents.critic import CriticAgent, CriticInput, CriticOutput, adjudicate
from edith.agents.debugger import DebuggerInput, DebuggerOutput, DebuggingAgent
from edith.agents.fanout import (
    FanOutAgent,
    PlanFanOutError,
    fan_out,
    signatures_in,
)
from edith.agents.planner import PlannerAgent, PlannerOutput, plan_to_tasks
from edith.config.schema import EdithConfig
from edith.context.engine import ContextBundle, ContextEngine
from edith.environment.python_env import local_module_names
from edith.errors import EdithError, FailureCategory
from edith.integrity import IntegrityReport, tests_exercise_changes
from edith.memory.governor import (
    ExecutionMemoryBudget,
    GovernorSettings,
    GrantOutcome,
    InjectionRecord,
    MemoryBudgetLimits,
    MemoryGovernor,
    MemoryGrant,
)
from edith.memory.retrieval import MemoryRetriever
from edith.memory.store import MemoryStore
from edith.memory.strategy import MemoryStrategy, RetrievalPoint
from edith.models.base import ModelProvider
from edith.models.registry import build_provider
from edith.observability.logging import bind_context, clear_context, get_logger
from edith.planning.dag import PlanValidationError, TaskGraph
from edith.planning.task import Task, TaskStatus
from edith.policy import FailureAction, decide, is_security_failure
from edith.schemas.agent import AgentPermissions, AgentRequest, AgentResponse, TaskRef
from edith.schemas.common import Verdict
from edith.state.schema import (
    AgentRun,
    Execution,
    FailureRecord,
    Project,
    ProjectState,
    ToolExecution,
    VerificationRecord,
)
from edith.state.store import StateStore
from edith.tools.gateway import ToolGateway
from edith.tools.paths import PathPolicy
from edith.tools.registry import ToolRegistry, build_default_registry
from edith.tools.schemas import ToolCall, ToolResult
from edith.verification.integrity_check import IntegrityChecker
from edith.verification.runner import VerificationReport, VerificationRunner
from edith.workspaces import ProjectWorkspace

logger = get_logger(__name__)

#: The principal verification runs as.
#:
#: Deliberately *not* the coding agent's permissions. The coder may write but must not be
#: able to execute arbitrary programs; the verifier may execute the configured checks but
#: must not be able to edit the code it is checking. Giving one principal both would let a
#: single compromised step both change the code and control what "passing" means.
VERIFIER_PERMISSIONS = AgentPermissions(
    allowed_tools=frozenset(
        {"shell.run", "filesystem.read", "git.status", "git.diff", "git.show"}
    ),
    allowed_read_paths=("**",),
)

#: The principal the integrity check runs as: read-only, with history access.
#:
#: Separate from both the coder and the verifier. The check exists to detect an agent
#: rewriting the definition of correctness, so it must not share a principal with anything
#: that can write, and it needs ``git.show`` to read what the tests used to be.
INTEGRITY_PERMISSIONS = AgentPermissions(
    allowed_tools=frozenset(
        {"filesystem.read", "git.status", "git.diff", "git.show", "git.log"}
    ),
    allowed_read_paths=("**",),
)


def _point_or_default(value: str) -> RetrievalPoint:
    """Read a persisted retrieval point, tolerating one written by an older build."""
    try:
        return RetrievalPoint(value)
    except ValueError:
        return RetrievalPoint.CODER_INITIAL


#: Ceilings high enough to never bind, for the unbudgeted experiment arm. Not "no limits":
#: the same governor code runs, so the arms differ by one number rather than by a branch.
_UNBOUNDED_LIMITS = MemoryBudgetLimits(
    max_total_chars=10**9,
    max_retrievals=10**6,
    max_total_memories=10**6,
    max_chars_per_retrieval=100_000,
    max_memories_per_retrieval=1000,
)


@dataclass
class ExecutionResult:
    """The outcome of one orchestrated run."""

    execution_id: str
    state: ProjectState
    verdict: Verdict
    summary: str
    tasks_total: int = 0
    tasks_succeeded: int = 0
    tasks_failed: int = 0
    agent_runs: int = 0
    model_calls: int = 0
    #: Characters of retrieved memory injected across the run, for the memory experiment.
    memory_chars: int = 0
    memories_used: int = 0
    #: Retrievals that produced an injection, so cost per repair cycle is measurable.
    memory_retrievals: int = 0
    #: Requests refused because the execution's budget was spent. The fail-closed count.
    budget_exhaustions: int = 0
    repairs_attempted: int = 0
    changed_files: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0
    failure_category: FailureCategory | None = None

    @property
    def succeeded(self) -> bool:
        """Whether the execution reached RELEASE."""
        return self.state is ProjectState.RELEASE and self.verdict is Verdict.PASS


@dataclass
class TaskOutcome:
    """The result of driving one task to a terminal state."""

    verdict: Verdict
    reason: str
    changed_files: list[str] = field(default_factory=list)
    repairs: int = 0
    failure_category: FailureCategory | None = None


class Orchestrator:
    """Runs the autonomous loop for one project workspace."""

    def __init__(
        self,
        config: EdithConfig,
        store: StateStore,
        workspace: ProjectWorkspace,
        *,
        provider: ModelProvider | None = None,
        tool_registry: ToolRegistry | None = None,
        memory: MemoryStore | None = None,
    ) -> None:
        """
        Args:
            config: Resolved configuration. Its tool layer is re-rooted at ``workspace``.
            store: Durable state.
            workspace: Where work happens. Never the Edith repository.
            provider: Model provider; built from config when omitted. Injected in tests.
            tool_registry: Tool registry; the M1 default when omitted.
            memory: Memory store. ``None`` disables retrieval entirely, which is the
                control arm of the memory experiment rather than a separate code path.
        """
        self.config = workspace.config_for(config)
        self.store = store
        self.workspace = workspace
        self.settings = config.orchestration
        self._provider = provider or build_provider(config)
        self._registry = tool_registry or build_default_registry()
        self._policy = PathPolicy.create(workspace.root, self.config.tools.paths)
        self._runs = 0
        self._model_calls = 0
        self._run_repairs = 0
        self._baseline_ref: str | None = None
        self._memory = memory
        self._retriever = MemoryRetriever(memory) if memory is not None else None
        self._memory_chars = 0
        self._memories_used = 0
        self._memory_retrievals = 0
        self._budget_exhaustions = 0
        # Built per execution in run(), because the budget is an execution's allowance and
        # an orchestrator instance may drive more than one.
        self._governor: MemoryGovernor | None = None
        self._execution_id = ""

    # -- Gateway construction -------------------------------------------------------

    def _gateway(self, permissions: AgentPermissions, agent: str) -> ToolGateway:
        """Build a gateway scoped to ``permissions`` for one agent."""
        return ToolGateway(
            self.config,
            permissions,
            registry=self._registry,
            agent=agent,
            policy=self._policy,
        )

    @staticmethod
    def _work_remains(graph: TaskGraph, current: Task) -> bool:
        """Whether any task other than ``current`` has yet to reach a terminal state."""
        return any(
            task.task_id != current.task_id and not task.status.terminal
            for task in graph.tasks()
        )

    @staticmethod
    def _scoped_permissions(base: AgentPermissions, task: Task) -> AgentPermissions:
        """Narrow an agent's declared permissions to a single task's scope.

        Intersection, never union: the result can only ever be *narrower* than what the
        agent's identity already allows. A task cannot grant its agent new powers.
        """
        return base.model_copy(
            update={
                "allowed_read_paths": task.scope.read_paths or base.allowed_read_paths,
                "allowed_write_paths": task.scope.write_paths,
            }
        )

    # -- Persistence helpers --------------------------------------------------------

    def _record_run(
        self,
        execution: Execution,
        agent: str,
        response: AgentResponse,
        task_id: str | None,
        attempt: int,
    ) -> AgentRun:
        """Persist one agent invocation and its output artifact."""
        self._runs += 1
        output_ref = None
        if response.output:
            output_ref = self.store.artifacts.put_json(response.output)
        run = AgentRun(
            execution_id=execution.execution_id,
            task_id=task_id,
            agent=agent,
            attempt=attempt,
            status=str(response.status),
            model=response.model,
            duration_seconds=response.duration_seconds,
            output_ref=output_ref,
            error=response.error,
            failure_category=response.failure_category,
        )
        return self.store.save_agent_run(run)

    def _record_tool_result(
        self, execution: Execution, run_id: str | None, result: ToolResult
    ) -> None:
        """Persist one tool call the orchestrator issued directly."""
        self.store.save_tool_execution(
            ToolExecution(
                execution_id=execution.execution_id,
                run_id=run_id,
                tool=result.tool,
                ok=result.ok,
                duration_seconds=result.duration_seconds,
                error=result.error,
                failure_category=result.failure_category,
                detail_ref=self.store.artifacts.put_json(result.model_dump(mode="json")),
            )
        )

    def _record_verification(
        self, execution: Execution, task: Task, report: VerificationReport
    ) -> None:
        """Persist every verification outcome with its full output as an artifact."""
        for outcome in report.outcomes:
            output_ref = self.store.artifacts.put(
                f"$ {outcome.command}\n\n--- stdout ---\n{outcome.stdout}\n"
                f"--- stderr ---\n{outcome.stderr}"
            )
            self.store.save_verification(
                VerificationRecord(
                    execution_id=execution.execution_id,
                    task_id=task.task_id,
                    kind=outcome.kind,
                    command=outcome.command,
                    exit_code=outcome.exit_code,
                    passed=outcome.passed,
                    duration_seconds=outcome.duration_seconds,
                    tests_passed=outcome.tests_passed,
                    tests_failed=outcome.tests_failed,
                    output_ref=output_ref,
                )
            )

    def _record_failure(
        self,
        execution: Execution,
        task: Task | None,
        category: FailureCategory,
        action: str,
        message: str,
        attempt: int,
    ) -> None:
        """Persist a classified failure."""
        self.store.save_failure(
            FailureRecord(
                execution_id=execution.execution_id,
                task_id=task.task_id if task else None,
                category=category,
                action=action,
                message=message[:2000],
                attempt=attempt,
            )
        )

    # -- Phases ---------------------------------------------------------------------

    def _plan(self, execution: Execution) -> list[Task]:
        """Invoke the Planner and translate its output into validated tasks.

        Raises:
            PlanValidationError: The plan was malformed or could not be validated.
        """
        self.store.record_transition(execution, ProjectState.PLANNING, "invoking planner")

        permissions = PlannerAgent.identity.permissions
        gateway = self._gateway(permissions, "planner")
        engine = ContextEngine(gateway, self.settings.context)
        bundle = engine.build(execution.request, task_summary="planning")

        agent = PlannerAgent(provider=self._provider, tools=gateway)
        self._model_calls += 1
        response = agent.execute(
            AgentRequest(
                payload={"request": execution.request, "context": bundle.render()},
                task=TaskRef(project_id=execution.project_id, title="plan"),
            )
        )
        self._record_run(execution, "planner", response, None, 1)

        if not response.ok:
            raise PlanValidationError(
                f"planner failed: {response.error}",
                details={"category": str(response.failure_category)},
            )

        plan = PlannerOutput.model_validate(response.output)

        # A plan carrying several functions in one step does not converge -- measured at
        # seventeen repairs and no delivery. When the planner produced one, fan the request
        # out into single-function steps instead. One extra model call, then pure formatting.
        if _needs_fanout(plan, execution.request, self.workspace.root):
            plan = self._fan_out(execution, plan)

        tasks = plan_to_tasks(plan, max_attempts=self.settings.max_task_attempts)
        if not tasks:
            raise PlanValidationError("planner produced no executable steps")

        # Constructing the graph is the validation: cycles and dangling edges are rejected
        # here, before anything is persisted or executed.
        TaskGraph(tasks)
        self.store.save_tasks(execution.execution_id, tasks)
        logger.info("plan.accepted", execution_id=execution.execution_id, tasks=len(tasks))
        return tasks

    def _fan_out(self, execution: Execution, plan: PlannerOutput) -> PlannerOutput:
        """Re-plan a multi-function request as one task per function.

        Phase A is the only model call; everything after it is deterministic formatting. A
        fan-out failure is not fatal on its own -- the original plan is still a plan, and
        failing the run over a decomposition that could not be produced would be worse than
        attempting the plan the planner gave us.
        """
        permissions = FanOutAgent.identity.permissions
        agent = FanOutAgent(
            provider=self._provider, tools=self._gateway(permissions, "fanout_planner")
        )
        self._model_calls += 1
        try:
            fanned = fan_out(agent, execution.request, goal=plan.goal)
        except PlanFanOutError as exc:
            logger.warning(
                "fanout.declined",
                execution_id=execution.execution_id,
                reason=exc.message,
            )
            return plan
        logger.info(
            "fanout.applied",
            execution_id=execution.execution_id,
            before=len(plan.steps),
            after=len(fanned.steps),
        )
        return fanned

    def _build_governor(self, execution: Execution) -> MemoryGovernor | None:
        """Create this execution's governor, resuming its budget if the run was interrupted.

        The allowance is charged against the *execution*, so a restart must not hand a
        resumed run a fresh full budget — that would turn "crash and retry" into an
        unlimited memory supply, which is precisely the unbounded growth M3.2 exists to
        stop. Prior consumption is read back from durable state.
        """
        settings = self.settings.memory
        if not settings.enabled or self._retriever is None:
            return None

        spent = self.store.memory_consumption(execution.execution_id)
        budget_config = settings.budget
        limits = MemoryBudgetLimits(
            max_total_chars=budget_config.max_total_chars,
            max_retrievals=budget_config.max_retrievals,
            max_total_memories=budget_config.max_total_memories,
            max_chars_per_retrieval=budget_config.max_chars_per_retrieval,
            max_memories_per_retrieval=budget_config.max_memories_per_retrieval,
        )
        if not budget_config.enabled:
            # The "without budget" arm of the M3.2 experiment. Ceilings are raised out of
            # reach rather than removed, so the same code path runs in every arm and the
            # comparison measures the budget rather than two different implementations.
            limits = _UNBOUNDED_LIMITS

        budget = ExecutionMemoryBudget(
            execution.execution_id,
            limits,
            consumed_chars=spent.chars,
            retrieval_count=spent.retrievals,
            injected=tuple(
                InjectionRecord(
                    memory_id=memory_id,
                    title=title,
                    point=_point_or_default(point),
                    agent=agent_name,
                    score=score,
                    reason="injected before this run was interrupted",
                    chars=0,
                )
                for memory_id, title, point, agent_name, score in spent.injected
            ),
        )
        if spent.retrievals:
            logger.info(
                "memory.budget_resumed",
                execution_id=execution.execution_id,
                **budget.snapshot(),
            )
        return MemoryGovernor(
            self._retriever,
            budget,
            GovernorSettings(
                strategy=MemoryStrategy(settings.strategy),
                project_id=self.workspace.project_id,
                min_confidence=settings.min_confidence,
                include_global=settings.include_global,
            ),
        )

    def _recall(
        self,
        task: Task,
        point: RetrievalPoint,
        *,
        error_text: str = "",
        agent: str = "coder",
    ) -> str:
        """Ask the governor for prior knowledge, and take whatever it grants.

        This method holds no policy of its own. Where memory applies, how relevant it must
        be, whether the execution can still afford it, and whether it has already been sent
        are all the governor's decisions — which is what makes the governor impossible to
        bypass, since this is the only path from the loop to a memory.

        Returns the empty string for every refusal, so an exhausted budget is indistinguishable
        from no memory at the call site. Memory is an optimisation; the loop continues without it.
        """
        if self._governor is None:
            return ""

        # An observed error is a far better retrieval key than a task title, so when one
        # exists it drives the query and the title becomes supporting context.
        query = f"{task.title} {task.description}"
        if error_text:
            query = f"{error_text[:1500]} {query}"

        grant = self._governor.request(
            execution_id=self._execution_id,
            query=query,
            purpose=point,
            error_text=error_text[:1500],
            paths=tuple(
                path for path in task.inputs.get("files", "").split(",") if path.strip()
            ),
            agent=agent,
        )
        self._record_grant(task, point, grant, agent)
        return grant.text

    def _record_grant(
        self, task: Task, point: RetrievalPoint, grant: MemoryGrant, agent: str
    ) -> None:
        """Account for one grant, whatever its outcome.

        Refusals are recorded too. "The budget was exhausted four times" is exactly the
        measurement M3.2 exists to produce, and it is invisible if only grants are counted.
        """
        if grant.outcome is GrantOutcome.BUDGET_EXHAUSTED:
            self._budget_exhaustions += 1
        if not grant.granted:
            return
        self._memory_chars += grant.chars
        self._memories_used += len(grant.memory_ids)
        self._memory_retrievals += 1
        governor = self._governor
        titles = (
            [
                record.title
                for record in governor.budget.injections
                if record.memory_id in set(grant.memory_ids)
            ]
            if governor is not None
            else []
        )
        self.store.record_memory_injection(
            execution_id=self._execution_id,
            task_id=task.task_id,
            agent_name=agent,
            point=str(point),
            memory_ids=grant.memory_ids,
            scores=grant.scores,
            titles=titles,
            chars=grant.chars,
            reason=grant.reason,
            remaining_chars=grant.remaining_chars,
        )

    def _build_context(self, gateway: ToolGateway, task: Task) -> ContextBundle:
        """Retrieve the smallest useful context for a task."""
        hints = tuple(
            path for path in task.inputs.get("files", "").split(",") if path.strip()
        )
        engine = ContextEngine(gateway, self.settings.context)
        bundle = engine.build(
            f"{task.title}\n{task.description}",
            hint_paths=hints,
            task_summary=task.title,
        )
        if bundle.degraded:
            # Surfaced, not swallowed. An agent asked to edit code it was never shown will
            # invent something plausible, and the resulting failure looks like a reasoning
            # problem rather than the retrieval problem it actually is.
            logger.warning(
                "context.degraded_for_task",
                task_id=task.task_id,
                reason=bundle.degraded_reason,
            )
        return bundle

    def _implement(
        self,
        execution: Execution,
        task: Task,
        bundle: ContextBundle,
        gateway: ToolGateway,
        *,
        evidence: str = "",
        guidance: str = "",
    ) -> tuple[AgentResponse, CoderOutput | None]:
        """Run the Coding Agent for one attempt."""
        agent = CodingAgent(provider=self._provider, tools=gateway)
        self._model_calls += 1
        response = agent.execute(
            AgentRequest(
                payload=CoderInput(
                    title=task.title,
                    description=task.description,
                    context=bundle.render(),
                    acceptance_criteria=list(task.acceptance_criteria),
                    failure_evidence=evidence,
                    repair_guidance=guidance,
                    prior_knowledge=self._recall(
                        task,
                        RetrievalPoint.CODER_REPAIR
                        if evidence
                        else RetrievalPoint.CODER_INITIAL,
                        error_text=evidence,
                    ),
                ).model_dump(),
                task=TaskRef(
                    task_id=task.task_id,
                    project_id=execution.project_id,
                    title=task.title,
                ),
            )
        )
        self._record_run(execution, "coder", response, task.task_id, task.attempts)
        output = CoderOutput.model_validate(response.output) if response.ok else None
        return response, output

    def _capture_baseline(self) -> str | None:
        """Record the commit representing the workspace before any agent touched it.

        Everything the integrity check does is relative to this ref. Without it there is no
        way to tell "the tests always said that" from "the agent just changed what they
        say", so a workspace with no baseline is reported as unchecked rather than clean.
        """
        gateway = self._gateway(INTEGRITY_PERMISSIONS, "integrity")
        if not gateway.can_use("git.log"):
            return None
        result = gateway.execute(
            ToolCall(tool="git.log", arguments={"max_entries": 1})
        )
        if not result.ok:
            return None
        commits = result.output.get("commits", [])
        return str(commits[0]["sha"]) if commits else None

    def _check_integrity(self, justification: str = "") -> IntegrityReport:
        """Compare the test suite against the baseline.

        Runs as its own read-only principal, so the check on whether the definition of
        correctness was altered shares no authority with anything that can alter it.
        """
        if self._baseline_ref is None:
            return IntegrityReport(baseline_unavailable=True)
        gateway = self._gateway(INTEGRITY_PERMISSIONS, "integrity")
        checker = IntegrityChecker(gateway, self._baseline_ref)
        report = checker.check(justification=justification)
        if report.tampered:
            logger.warning(
                "integrity.tampering_detected",
                findings=len(report.blocking_findings),
                files=report.test_files_changed,
            )
        return report

    def _verify(self, execution: Execution, task: Task) -> VerificationReport:
        """Execute the task's verification requirements and persist the evidence.

        Runs under :data:`VERIFIER_PERMISSIONS`, not the coder's, so the principal that
        decides whether the work passed cannot be the one that wrote it.
        """
        self.store.record_transition(
            execution, ProjectState.VERIFICATION, f"verifying {task.task_id}"
        )
        gateway = self._gateway(VERIFIER_PERMISSIONS, "verifier")
        runner = VerificationRunner(
            gateway,
            self.settings.profile(),
            # Lets the classifier tell "the project's own module is broken" apart from
            # "a third-party package is missing".
            local_modules=frozenset(local_module_names(self.workspace.root)),
        )
        requirements = tuple(
            (requirement.kind, requirement.selector) for requirement in task.verification
        )
        report = runner.run_all(requirements or (("tests", None),))
        self._record_verification(execution, task, report)
        return report

    def _judge(
        self,
        execution: Execution,
        task: Task,
        coder_output: CoderOutput | None,
        report: VerificationReport,
        gateway: ToolGateway,
        integrity: IntegrityReport | None = None,
    ) -> CriticOutput | None:
        """Run the Critic against the real diff, real evidence, and the integrity report."""
        self.store.record_transition(
            execution, ProjectState.REVIEW, f"judging {task.task_id}"
        )
        diff = self._current_diff(execution, gateway)
        agent = CriticAgent(provider=self._provider, tools=gateway)
        self._model_calls += 1
        response = agent.execute(
            AgentRequest(
                payload=CriticInput(
                    title=task.title,
                    description=task.description,
                    acceptance_criteria=list(task.acceptance_criteria),
                    changed_files=coder_output.changed_files if coder_output else [],
                    diff=diff[:20_000],
                    evidence=report.evidence(),
                    integrity=integrity.summary()[:4000] if integrity else "",
                ).model_dump(),
                task=TaskRef(task_id=task.task_id, project_id=execution.project_id),
            )
        )
        self._record_run(execution, "critic", response, task.task_id, task.attempts)
        if not response.ok:
            # A missing verdict is survivable: adjudication falls back to the evidence.
            logger.warning("critic.unavailable", task_id=task.task_id, error=response.error)
            return None
        return CriticOutput.model_validate(response.output)

    def _diagnose(
        self,
        execution: Execution,
        task: Task,
        report: VerificationReport,
        bundle: ContextBundle,
        coder_output: CoderOutput | None,
        gateway: ToolGateway,
    ) -> DebuggerOutput | None:
        """Run the Debugging Agent against the real failure output."""
        self.store.record_transition(
            execution, ProjectState.REPAIR, f"diagnosing {task.task_id}"
        )
        agent = DebuggingAgent(provider=self._provider, tools=gateway)
        self._model_calls += 1
        response = agent.execute(
            AgentRequest(
                payload=DebuggerInput(
                    title=task.title,
                    description=task.description,
                    evidence=report.evidence() or "verification produced no output",
                    context=bundle.render(),
                    changed_files=coder_output.changed_files if coder_output else [],
                    attempt=task.attempts,
                    prior_knowledge=self._recall(
                        task,
                        RetrievalPoint.DEBUGGER,
                        error_text=report.evidence(),
                        agent="debugger",
                    ),
                ).model_dump(),
                task=TaskRef(task_id=task.task_id, project_id=execution.project_id),
            )
        )
        self._record_run(execution, "debugger", response, task.task_id, task.attempts)
        if not response.ok:
            logger.warning("debugger.unavailable", task_id=task.task_id, error=response.error)
            return None
        return DebuggerOutput.model_validate(response.output)

    def _current_diff(self, execution: Execution, gateway: ToolGateway) -> str:
        """Ask git for the working-tree diff, tolerating a non-repository workspace."""
        if not gateway.can_use("git.diff"):
            return ""
        result = gateway.execute(ToolCall(tool="git.diff"))
        self._record_tool_result(execution, None, result)
        return str(result.output.get("diff", "")) if result.ok else ""

    # -- Task loop ------------------------------------------------------------------

    def _final_gate(
        self, execution: Execution, tasks: list[Task], changed: list[str]
    ) -> tuple[Verdict, str, int]:
        """Verify the finished plan as a whole, repairing if it still fails.

        Task-level verification runs the project's whole suite, so a task that correctly
        fixes one of several defects still sees red and cannot, on its own, prove the plan
        succeeded. The authoritative judgement therefore happens here, once, after every
        task has had its turn -- and if it fails, the repair loop runs against the *plan's*
        goal rather than any single task's.
        """
        anchor = tasks[-1]
        report = self._verify(execution, anchor)
        integrity = self._check_integrity()
        if integrity.tampered:
            # A green suite achieved by weakening tests is worse than a red one.
            blocking = integrity.blocking_findings[0]
            self._record_failure(
                execution, anchor, FailureCategory.REQUIREMENT_FAILURE, "ESCALATE",
                f"test integrity violated: {blocking.detail}", 1,
            )
            return (Verdict.FAIL, f"test integrity violated: {blocking.detail}", 0)
        if report.passed:
            # The final gate *is* the review, so move through REVIEW rather than jumping
            # from VERIFICATION straight to RELEASE.
            self.store.record_transition(
                execution, ProjectState.REVIEW, "final verification passed"
            )
            return (Verdict.PASS, "final verification passed", 0)

        repairs = 0
        gateway = self._gateway(
            self._scoped_permissions(CodingAgent.identity.permissions, anchor), "coder"
        )
        while repairs < self.settings.max_repair_attempts:
            if self._runs >= self.settings.max_total_agent_runs:
                return (Verdict.FAIL, "total agent run budget exhausted", repairs)

            bundle = self._build_context(gateway, anchor)
            diagnosis = self._diagnose(execution, anchor, report, bundle, None, gateway)
            repairs += 1

            self.store.record_transition(
                execution, ProjectState.IMPLEMENTATION, f"final repair {repairs}"
            )
            _, coder_output = self._implement(
                execution,
                anchor,
                bundle,
                gateway,
                evidence=report.evidence(),
                guidance=diagnosis.as_guidance() if diagnosis else "",
            )
            if coder_output:
                changed[:] = sorted(set(changed) | set(coder_output.changed_files))

            report = self._verify(execution, anchor)
            if self._check_integrity().tampered:
                return (Verdict.FAIL, "test integrity violated during repair", repairs)
            if report.passed:
                self.store.record_transition(
                    execution, ProjectState.REVIEW, f"passed after repair {repairs}"
                )
                return (Verdict.PASS, f"passed after {repairs} repair attempt(s)", repairs)

            self._record_failure(
                execution,
                anchor,
                report.failure_category or FailureCategory.TEST_FAILURE,
                "REPAIR",
                f"final verification still failing after repair {repairs}",
                repairs,
            )

        return (Verdict.FAIL, "final verification still failing after repairs", repairs)

    def _run_task(self, execution: Execution, graph: TaskGraph, task: Task) -> TaskOutcome:
        """Drive one task to a terminal verdict, within its attempt budget."""
        bind_context(task_id=task.task_id, execution_id=execution.execution_id)
        permissions = self._scoped_permissions(CodingAgent.identity.permissions, task)
        gateway = self._gateway(permissions, "coder")

        evidence = ""
        guidance = ""
        repairs = 0
        changed: list[str] = []
        last_reason = "task did not complete"
        last_category: FailureCategory | None = None

        try:
            while task.attempts < task.max_attempts:
                task.attempts += 1
                self.store.record_transition(
                    execution,
                    ProjectState.IMPLEMENTATION,
                    f"attempt {task.attempts} for {task.task_id}",
                )
                bundle = self._build_context(gateway, task)

                response, coder_output = self._implement(
                    execution, task, bundle, gateway, evidence=evidence, guidance=guidance
                )
                self.store.save_task(execution.execution_id, task)

                if not response.ok:
                    last_category = response.failure_category
                    last_reason = response.error or "the coding agent failed"
                    action = decide(
                        last_category,
                        attempts=task.attempts,
                        max_attempts=task.max_attempts,
                    )
                    self._record_failure(
                        execution, task, last_category or FailureCategory.UNKNOWN,
                        str(action), last_reason, task.attempts,
                    )
                    if action is FailureAction.ABORT:
                        return TaskOutcome(
                            Verdict.BLOCKED, last_reason, changed, repairs, last_category
                        )
                    if action is FailureAction.ESCALATE:
                        break
                    continue

                if coder_output:
                    changed = sorted(set(changed) | set(coder_output.changed_files))
                    # A refused write is a security event, not a coding mistake.
                    if coder_output.rejected_files:
                        self._record_failure(
                            execution, task, FailureCategory.SECURITY_FAILURE, "REPORTED",
                            f"writes refused: {coder_output.rejected_files}", task.attempts,
                        )

                report = self._verify(execution, task)
                integrity = self._check_integrity()
                critic = self._judge(
                    execution, task, coder_output, report, gateway, integrity
                )
                verdict, reason = adjudicate(
                    critic,
                    report,
                    changes_made=bool(coder_output and coder_output.made_changes),
                    integrity=integrity,
                )
                logger.info(
                    "task.adjudicated",
                    task_id=task.task_id,
                    verdict=str(verdict),
                    reason=reason,
                    attempt=task.attempts,
                )

                if verdict is Verdict.PASS:
                    # Green tests are evidence only if they ran against the change. A suite
                    # that passes without importing the changed module proves the suite
                    # works, not the code -- the vacuous verification M5 found in its own
                    # benchmark, reaching adjudication by a third route.
                    unexercised = tests_exercise_changes(self.workspace.root, tuple(changed))
                    if unexercised is None:
                        return TaskOutcome(Verdict.PASS, reason, changed, repairs)
                    verdict, reason = Verdict.FAIL, unexercised
                    report = report.model_copy(
                        update={"failure_category": FailureCategory.TEST_FAILURE}
                    )
                    logger.info(
                        "task.unexercised", task_id=task.task_id, reason=unexercised
                    )

                last_reason = reason
                last_category = report.failure_category or FailureCategory.REQUIREMENT_FAILURE

                if is_security_failure(last_category):
                    self._record_failure(
                        execution, task, last_category, "BLOCKED", reason, task.attempts
                    )
                    return TaskOutcome(
                        Verdict.BLOCKED, reason, changed, repairs, last_category
                    )

                action = decide(
                    last_category, attempts=task.attempts, max_attempts=task.max_attempts
                )
                # Recorded after the policy has spoken, and with what it actually decided.
                # This previously wrote "REPAIR" unconditionally, so an escalated environment
                # failure was filed as though the coder had been asked to fix it -- the exact
                # misattribution M5.2 exists to prevent, in the record rather than the policy.
                self._record_failure(
                    execution, task, last_category, str(action.value), reason, task.attempts
                )
                if action in {FailureAction.ESCALATE, FailureAction.ABORT}:
                    break

                # Repair path: diagnose, then feed the diagnosis into the next attempt.
                if (
                    action is FailureAction.REPAIR
                    and repairs < self.settings.max_repair_attempts
                    and self._run_repairs < self.settings.max_total_repairs
                ):
                    diagnosis = self._diagnose(
                        execution, task, report, bundle, coder_output, gateway
                    )
                    repairs += 1
                    self._run_repairs += 1
                    guidance = diagnosis.as_guidance() if diagnosis else ""
                else:
                    guidance = ""
                evidence = _combined_evidence(report, coder_output)

                if self._runs >= self.settings.max_total_agent_runs:
                    last_reason = "total agent run budget exhausted"
                    break

            return TaskOutcome(Verdict.FAIL, last_reason, changed, repairs, last_category)
        finally:
            clear_context()

    # -- Entry point ----------------------------------------------------------------

    def run(self, execution: Execution) -> ExecutionResult:
        """Execute a request end to end.

        Never raises for an execution-level failure: the result carries the verdict and the
        state store carries the evidence.
        """
        started = time.monotonic()
        bind_context(execution_id=execution.execution_id, project_id=execution.project_id)
        self._execution_id = execution.execution_id
        self._governor = self._build_governor(execution)
        self._baseline_ref = self._capture_baseline()
        logger.info(
            "execution.start",
            request=execution.request[:200],
            baseline=self._baseline_ref or "(none)",
        )

        try:
            tasks = self._plan(execution)
        except (PlanValidationError, EdithError) as exc:
            self._record_failure(
                execution, None, exc.category, "ESCALATE", exc.message, 1
            )
            self.store.record_transition(execution, ProjectState.FAILED, exc.message[:200])
            clear_context()
            return ExecutionResult(
                execution_id=execution.execution_id,
                state=ProjectState.FAILED,
                verdict=Verdict.BLOCKED,
                summary=f"planning failed: {exc.message}",
                duration_seconds=time.monotonic() - started,
                failure_category=exc.category,
            )

        graph = TaskGraph(tasks)
        graph.refresh()
        changed: list[str] = []
        repairs = 0
        deferred: list[str] = []
        last_failure: str | None = None
        aborted_reason: str | None = None
        abort_category: FailureCategory | None = None

        while (task := graph.next_task()) is not None:
            if self._runs >= self.settings.max_total_agent_runs:
                aborted_reason = "total agent run budget exhausted"
                break

            task.transition_to(TaskStatus.RUNNING)
            self.store.save_task(execution.execution_id, task)

            outcome = self._run_task(execution, graph, task)
            changed = sorted(set(changed) | set(outcome.changed_files))
            repairs += outcome.repairs

            if outcome.verdict is Verdict.PASS:
                graph.mark_succeeded(task.task_id)
            elif outcome.verdict is Verdict.BLOCKED:
                graph.mark_failed(task.task_id, outcome.reason, outcome.failure_category)
                aborted_reason = outcome.reason
                abort_category = outcome.failure_category
            elif outcome.changed_files and self._work_remains(graph, task):
                # The task made its change but the suite is still red -- expected when a
                # later task fixes the rest. Blocking here would strand the remaining work
                # and guarantee failure. The final gate is what actually decides.
                logger.info(
                    "task.deferred",
                    task_id=task.task_id,
                    reason="changed files but the suite is not green yet",
                )
                graph.mark_succeeded(task.task_id)
                deferred.append(task.task_id)
            else:
                graph.mark_failed(task.task_id, outcome.reason, outcome.failure_category)
                last_failure = outcome.reason
            self.store.save_tasks(execution.execution_id, list(graph.tasks()))

            if aborted_reason:
                break

        summary = graph.summary()

        # The authoritative gate: verify the finished plan as a whole, repairing if needed.
        # Skipped only when the run was aborted or nothing was produced to verify.
        verdict = Verdict.FAIL
        # Carry the specific reason up: "one or more tasks failed" tells an operator
        # nothing, and a test-integrity violation in particular must be impossible to miss.
        text = (
            aborted_reason or last_failure or "one or more tasks failed verification"
        )
        if not aborted_reason and graph.succeeded():
            verdict, text, final_repairs = self._final_gate(execution, tasks, changed)
            repairs += final_repairs

        # Checked once more over everything the execution changed. The per-task check cannot
        # see this case: one task writes the module before any test exists, the next writes a
        # test that changes no implementation, and each is individually unobjectionable while
        # the pair leaves the code untested. Only the union shows it.
        if verdict is Verdict.PASS:
            unexercised = tests_exercise_changes(self.workspace.root, tuple(changed))
            if unexercised is not None:
                logger.info(
                    "execution.unexercised",
                    execution_id=execution.execution_id,
                    reason=unexercised,
                )
                verdict, text = Verdict.FAIL, unexercised

        if verdict is Verdict.PASS:
            self.store.record_transition(
                execution, ProjectState.RELEASE, "final verification passed"
            )
            text = (
                f"completed {len(tasks)} task(s); changed {len(changed)} file(s); {text}"
            )
        else:
            self.store.record_transition(execution, ProjectState.FAILED, text[:200])
            if aborted_reason:
                verdict = Verdict.BLOCKED

        execution.result_summary = text
        self.store.save_execution(execution)
        clear_context()

        result = ExecutionResult(
            execution_id=execution.execution_id,
            state=execution.state,
            verdict=verdict,
            summary=text,
            tasks_total=len(tasks),
            tasks_succeeded=summary.get("SUCCEEDED", 0),
            tasks_failed=summary.get("FAILED", 0) + summary.get("BLOCKED", 0),
            agent_runs=self._runs,
            model_calls=self._model_calls,
            memory_chars=self._memory_chars,
            memories_used=self._memories_used,
            memory_retrievals=self._memory_retrievals,
            budget_exhaustions=self._budget_exhaustions,
            repairs_attempted=repairs,
            changed_files=changed,
            duration_seconds=time.monotonic() - started,
            failure_category=abort_category,
        )
        logger.info(
            "execution.finished",
            verdict=str(verdict),
            tasks=len(tasks),
            agent_runs=self._runs,
            memory_chars=self._memory_chars,
            memory_retrievals=self._memory_retrievals,
            budget_exhaustions=self._budget_exhaustions,
            duration_seconds=round(result.duration_seconds, 2),
        )
        return result

    def close(self) -> None:
        """Release the model provider."""
        self._provider.close()


def _combined_evidence(report: VerificationReport, coder: CoderOutput | None) -> str:
    """Merge verification output with anything the coder's own edits were rejected for.

    Without this, an attempt whose edits were all refused -- by the scope policy or the
    syntax gate -- would hand the next attempt only stale test output, and the model would
    have no idea why nothing changed.
    """
    parts = [report.evidence()]
    if coder and coder.remaining_concerns:
        listed = "\n".join(f"- {item}" for item in coder.remaining_concerns)
        parts.append(f"The previous attempt's edits were REJECTED:\n{listed}")
    return "\n\n".join(part for part in parts if part.strip())


def create_execution(
    store: StateStore, workspace: ProjectWorkspace, request: str
) -> tuple[Project, Execution]:
    """Create or reuse the project row and start a new execution."""
    project = store.get_project(workspace.project_id) or store.save_project(
        Project(
            project_id=workspace.project_id,
            name=workspace.name,
            workspace_root=str(workspace.root),
        )
    )
    execution = store.save_execution(
        Execution(project_id=project.project_id, request=request, branch=workspace.branch)
    )
    return project, execution


def resume_graph(store: StateStore, execution_id: str) -> TaskGraph | None:
    """Rebuild the task graph for an execution from persisted state.

    The restart path: everything needed to continue is in the database.
    """
    tasks = store.load_tasks(execution_id)
    if not tasks:
        return None
    graph = TaskGraph(tasks)
    graph.refresh()
    return graph


__all__ = [
    "ExecutionResult",
    "Orchestrator",
    "TaskOutcome",
    "create_execution",
    "resume_graph",
]


def _needs_fanout(plan: PlannerOutput, request: str, workspace_root: Path) -> bool:
    """Whether this request should be re-planned as one task per function.

    The trigger reads the *request*, not only the plan, and that correction came from a real
    run. The premise was that a multi-function request produced one multi-function step; on a
    four-function calculator the planner instead produced five steps that all wrote to a
    single shared ``calculator.py`` with no tests, and two of them failed fighting over the
    same file. Counting signatures per step saw nothing wrong with that.

    Two conditions, and the second matters as much as the first:

    A request naming two or more functions in call form is a fan-out candidate, however the
    planner chose to slice it.

    A plan touching files that already exist is not. Fan-out imposes a greenfield layout --
    one module per function under ``src/backend/`` -- which is right for new work and wrong
    for a repair to an existing project, where the files and their arrangement are already
    decided. An existing project is left to the ordinary planner.
    """
    if any((workspace_root / name).exists() for step in plan.steps for name in step.files):
        return False
    if len(signatures_in(request)) >= 2:
        return True
    return any(len(signatures_in(step.description)) > 1 for step in plan.steps)
