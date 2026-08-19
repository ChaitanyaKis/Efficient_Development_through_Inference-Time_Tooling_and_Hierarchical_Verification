"""M12: a refused edit is repairable; an absent one is not.

M11 measured what conflating those costs. SEM-003 failed 5/5 with an empty implementation and
zero repair attempts, because the model chose ``replace_function`` for a file that did not
exist. The resolver caught it and said exactly how to fix it -- and the executor threw that
away, classifying "no files changed" as FAILED, which the repair loop does not reconsider.

The distinction restored here is narrow and deliberate:

    edits produced, all refused, reasons stated  ->  REJECTED, repairable
    no edits produced at all                     ->  FAILED, terminal

The second half matters as much as the first. M5.2 established that spending budget on a
failure the agent cannot fix both wastes the attempt and misattributes the fault; showing an
agent its own absence is exactly that. This is the mirror case, not a reversal of it.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from edith.agents.coder import SYSTEM_PROMPT
from edith.config.loader import load_config
from edith.config.schema import ModelParams
from edith.engineering.executor import (
    REPAIRABLE_FAILURES,
    EngineeringExecutor,
    TaskOutcome,
)
from edith.errors import FailureCategory
from edith.workspaces import ProjectWorkspace

from .fakes import FakeProvider


def executor_for(root: Path, responses: list[str]) -> EngineeringExecutor:
    base = load_config(None)
    config = base.model_copy(
        update={"tools": base.tools.model_copy(update={"workspace_root": root})}
    )
    return EngineeringExecutor(
        config,
        ProjectWorkspace(project_id="p", name="n", root=root),
        provider=FakeProvider(ModelParams(model_name="t"), responses),
        isolate=False,
    )


class TestThePromptCannotBeReadAsForbiddingFileCreation:
    """The wording M11 traced the failure to, asserted so it cannot regress."""

    def test_replace_file_is_named_the_only_way_to_create_a_file(self) -> None:
        assert "ONLY mode that can create a file" in SYSTEM_PROMPT

    def test_the_new_file_rule_is_stated_imperatively(self) -> None:
        assert "you MUST use replace_file" in SYSTEM_PROMPT

    def test_the_old_unconditional_avoidance_is_gone(self) -> None:
        """"Avoid this when append or replace_function would work" outranked the instruction
        above it, and the model obeyed the last thing it read."""
        assert "Avoid this when append or replace_function would work" not in SYSTEM_PROMPT

    def test_narrow_modes_are_still_preferred_for_existing_files(self) -> None:
        """The fix must not undo the reason the guidance existed: whole-file rewrites are
        the least reliable option for a small model."""
        assert "already exists, prefer append or replace_function" in SYSTEM_PROMPT

    def test_the_rejection_reason_the_model_will_see_names_the_remedy(self) -> None:
        from edith.agents.coder import CodingAgent, EditMode, FileEdit

        agent = CodingAgent.__new__(CodingAgent)
        edit = FileEdit(
            path="src/new.py", mode=EditMode.REPLACE_FUNCTION, content="def f():\n    return 1\n"
        )
        _, error = agent._resolve_content(edit, "")
        assert error is not None
        assert "replace_file" in error


class TestARefusedEditIsRepairable:
    def test_a_rejected_edit_becomes_rejected_not_failed(self, tmp_path: Path) -> None:
        """The exact M11 shape: mode cannot apply, so nothing is written."""
        (tmp_path / "src").mkdir()
        executor = executor_for(tmp_path, [])
        execution = _no_change_execution(
            executor, rejected=("src/new.py",), concerns=("src/new.py: use replace_file",)
        )
        assert execution.outcome is TaskOutcome.REJECTED
        assert execution.failure_category is FailureCategory.CODE_FAILURE

    def test_a_rejected_edit_enters_the_existing_repair_budget(self) -> None:
        """No second policy: it routes through M5.2's set."""
        assert FailureCategory.CODE_FAILURE in REPAIRABLE_FAILURES

    def test_the_repair_evidence_carries_the_reason_not_just_the_filename(
        self, tmp_path: Path
    ) -> None:
        """M2.1: an agent told only that it failed rewrites the same thing."""
        executor = executor_for(tmp_path, [])
        execution = _no_change_execution(
            executor,
            rejected=("src/new.py",),
            concerns=(
                "src/new.py: mode is replace_function but src/new.py does not exist yet; "
                "use replace_file to create it",
            ),
        )
        assert "use replace_file" in execution.detail

    def test_producing_nothing_at_all_stays_terminal(self, tmp_path: Path) -> None:
        """Showing an agent its own absence tells it nothing -- M5.2's rule, preserved."""
        executor = executor_for(tmp_path, [])
        execution = _no_change_execution(executor, rejected=(), concerns=())
        assert execution.outcome is TaskOutcome.FAILED
        assert execution.failure_category is None

    def test_a_policy_denial_stays_terminal(self, tmp_path: Path) -> None:
        """The guarantee this change nearly broke.

        An out-of-scope write is not the agent's to fix. Retrying it is how a permission
        refusal becomes a retry loop, so a denial keeps M5.2's treatment: classified
        SECURITY_FAILURE, which the repair policy already excludes.
        """
        executor = executor_for(tmp_path, [])
        execution = _no_change_execution(
            executor,
            rejected=("src/frontend/app.py",),
            denied=("src/frontend/app.py",),
            concerns=("src/frontend/app.py: outside the agent's write scope",),
        )
        assert execution.outcome is TaskOutcome.FAILED
        assert execution.failure_category is FailureCategory.SECURITY_FAILURE
        assert execution.failure_category not in REPAIRABLE_FAILURES

    def test_a_mixed_refusal_containing_a_denial_stays_terminal(self, tmp_path: Path) -> None:
        """One denial is enough: the set is not partly repairable."""
        executor = executor_for(tmp_path, [])
        execution = _no_change_execution(
            executor,
            rejected=("a.py", "b.py"),
            denied=("b.py",),
            concerns=("a.py: use replace_file", "b.py: outside scope"),
        )
        assert execution.outcome is TaskOutcome.FAILED

    def test_the_rejected_files_are_recorded(self, tmp_path: Path) -> None:
        executor = executor_for(tmp_path, [])
        execution = _no_change_execution(
            executor, rejected=("a.py", "b.py"), concerns=("a.py: nope",)
        )
        assert execution.rejected_files == ("a.py", "b.py")


def _no_change_execution(
    executor: EngineeringExecutor,
    *,
    rejected: tuple[str, ...],
    concerns: tuple[str, ...],
    denied: tuple[str, ...] = (),
):
    """Drive ``_attempt_task``'s no-change branch with a stubbed agent response.

    Exercises the real branch rather than reimplementing its logic, so the test fails if the
    classification moves.
    """
    from edith.engineering.ownership import EngineeringRole

    class _Response:
        ok = True
        error = None
        failure_category = None
        attempts = 1
        output: ClassVar[dict[str, object]] = {
            "changed_files": [],
            "rejected_files": list(rejected),
            "denied_files": list(denied),
            "remaining_concerns": list(concerns),
            "diff": "",
        }

    from edith.engineering.agents import agent_for as real_agent_for

    real = real_agent_for(EngineeringRole.BACKEND)

    class _Agent:
        """Stands in for the coding agent, keeping the real identity so the executor's
        permission and scope handling runs unchanged."""

        identity = real.identity

        def __init__(self, **kwargs: object) -> None:
            pass

        def execute(self, request: object) -> object:
            return _Response()

    from edith.product.architecture import PlannedTask

    task = PlannedTask.model_validate(
        {
            "task_id": "TASK-001",
            "title": "make a file",
            "description": "make a file",
            "agent": "backend",
            "paths": ["src/new.py"],
            "verification": [],
            "acceptance_criteria": ["it exists"],
            "depends_on": [],
        }
    )
    from edith.engineering.ownership import Assignment

    assignment = Assignment(
        task=task, role=EngineeringRole.BACKEND, write_paths=("src/**",)
    )
    # Patch the agent factory the executor imported, so the real _attempt_task branch runs.
    import edith.engineering.executor as module

    original = module.agent_for
    module.agent_for = lambda role: _Agent  # type: ignore[assignment,return-value]
    try:
        return executor._attempt_task(
            assignment, prd=None, ux=None, architecture_text="", verify=False, evidence=""
        )
    finally:
        module.agent_for = original  # type: ignore[assignment]
