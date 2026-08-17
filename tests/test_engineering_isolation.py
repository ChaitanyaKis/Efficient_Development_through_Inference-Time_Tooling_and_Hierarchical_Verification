"""M5.1: workspace isolation, worktree lifecycle, merge safety, and fair-comparison setup.

M5 ran every task in one shared workspace and compared specialised agents against a generic
one that had weaker infrastructure. These tests pin down both fixes: the lifecycle that keeps
a rejected task's work out of the main tree, and the structural guarantee that the generic
arm runs through identical machinery.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from edith.engineering.agents import agent_for
from edith.engineering.executor import QualityState, TaskOutcome
from edith.engineering.isolation import (
    MergeDecision,
    TaskWorkspace,
    WorkspaceError,
    WorkspaceLedger,
    WorkspaceState,
    may_merge,
)
from edith.engineering.ownership import EngineeringRole, scope_for


def workspace(
    task_id: str = "TASK-001",
    *,
    path: Path | None = None,
    base: str = "abc123",
    state: WorkspaceState = WorkspaceState.CREATED,
) -> TaskWorkspace:
    return TaskWorkspace(
        workspace_id=f"task-{task_id.lower()}",
        task_id=task_id,
        execution_id="exec_1",
        path=path or Path("workspaces") / task_id,
        base_revision=base,
        state=state,
    )


class TestWorkspaceIdentity:
    """M5.1 item 1: a workspace knows what it is for and what it came from."""

    def test_a_workspace_records_everything_a_merge_needs(self) -> None:
        item = workspace()
        assert item.workspace_id
        assert item.task_id == "TASK-001"
        assert item.execution_id == "exec_1"
        assert item.base_revision
        assert item.path

    def test_ownership_is_explicit(self) -> None:
        item = workspace("TASK-001")
        assert item.owns("TASK-001")
        assert not item.owns("TASK-002")


class TestLifecycle:
    def test_the_happy_path_runs_create_execute_verify_merge(self) -> None:
        item = workspace()
        item.transition(WorkspaceState.EXECUTING)
        item.transition(WorkspaceState.VERIFIED)
        item.transition(WorkspaceState.MERGED)
        assert item.state.terminal
        assert not item.active

    def test_a_rejected_workspace_stays_available_for_repair(self) -> None:
        """M5.1: a failed verification leaves the workspace for a bounded repair."""
        item = workspace()
        item.transition(WorkspaceState.EXECUTING)
        item.transition(WorkspaceState.REJECTED)
        assert item.active
        item.transition(WorkspaceState.EXECUTING)
        assert item.state is WorkspaceState.EXECUTING

    def test_a_rejected_workspace_cannot_be_merged_directly(self) -> None:
        item = workspace(state=WorkspaceState.REJECTED)
        with pytest.raises(WorkspaceError, match="illegal workspace transition"):
            item.transition(WorkspaceState.MERGED)

    def test_a_resolved_workspace_is_final(self) -> None:
        item = workspace(state=WorkspaceState.MERGED)
        with pytest.raises(WorkspaceError):
            item.transition(WorkspaceState.EXECUTING)

    def test_a_created_workspace_can_be_discarded_without_executing(self) -> None:
        item = workspace()
        item.transition(WorkspaceState.DISCARDED)
        assert item.state is WorkspaceState.DISCARDED


class TestCrossWorkspaceIsolation:
    """M5.1 item 4: a task must never write into another task's workspace."""

    def ledger(self, tmp_path: Path) -> WorkspaceLedger:
        ledger = WorkspaceLedger()
        ledger.add(workspace("TASK-001", path=tmp_path / "task-001"))
        ledger.add(workspace("TASK-002", path=tmp_path / "task-002"))
        return ledger

    def test_a_task_may_write_its_own_workspace(self, tmp_path: Path) -> None:
        ledger = self.ledger(tmp_path)
        assert ledger.may_write("TASK-001", tmp_path / "task-001" / "src" / "x.py")

    def test_a_task_may_not_write_another_tasks_workspace(self, tmp_path: Path) -> None:
        ledger = self.ledger(tmp_path)
        assert not ledger.may_write("TASK-001", tmp_path / "task-002" / "src" / "x.py")

    def test_the_owner_of_a_path_is_identifiable(self, tmp_path: Path) -> None:
        ledger = self.ledger(tmp_path)
        owner = ledger.owner_of(tmp_path / "task-002" / "deep" / "file.py")
        assert owner is not None
        assert owner.task_id == "TASK-002"

    def test_a_path_in_no_workspace_is_not_this_checks_business(
        self, tmp_path: Path
    ) -> None:
        """The M1 path policy already decides that; this only answers the narrower question."""
        ledger = self.ledger(tmp_path)
        assert ledger.may_write("TASK-001", tmp_path / "elsewhere" / "x.py")

    def test_a_task_cannot_hold_two_active_workspaces(self, tmp_path: Path) -> None:
        ledger = WorkspaceLedger()
        ledger.add(workspace("TASK-001", path=tmp_path / "a"))
        with pytest.raises(WorkspaceError, match="already has an active workspace"):
            ledger.add(workspace("TASK-001", path=tmp_path / "b"))


class TestMergeSafety:
    """M5.1: every condition is checked, and nothing merges by last-write-wins."""

    def test_a_verified_task_may_merge(self) -> None:
        decision = may_merge(workspace(), task_id="TASK-001", verified=True)
        assert decision.allowed

    def test_an_unverified_task_may_not_merge(self) -> None:
        decision = may_merge(workspace(), task_id="TASK-001", verified=False)
        assert decision.refused
        assert "not been verified" in decision.reason

    def test_a_blocking_issue_prevents_merge(self) -> None:
        decision = may_merge(
            workspace(), task_id="TASK-001", verified=True, blocking_issues=1
        )
        assert decision.refused
        assert "blocking verification issue" in decision.reason

    def test_a_workspace_cannot_merge_for_another_task(self) -> None:
        """The check that stops one task promoting another's work."""
        decision = may_merge(workspace("TASK-001"), task_id="TASK-002", verified=True)
        assert decision.refused
        assert "belongs to" in decision.reason

    def test_an_unknown_base_revision_prevents_merge(self) -> None:
        decision = may_merge(workspace(base=""), task_id="TASK-001", verified=True)
        assert decision.refused
        assert "base revision" in decision.reason

    def test_an_already_resolved_workspace_cannot_merge_again(self) -> None:
        decision = may_merge(
            workspace(state=WorkspaceState.MERGED), task_id="TASK-001", verified=True
        )
        assert decision.refused
        assert "already" in decision.reason

    def test_a_refusal_always_carries_a_reason(self) -> None:
        for decision in (
            may_merge(workspace(), task_id="TASK-002", verified=True),
            may_merge(workspace(), task_id="TASK-001", verified=False),
            may_merge(workspace(base=""), task_id="TASK-001", verified=True),
        ):
            assert isinstance(decision, MergeDecision)
            assert decision.refused
            assert decision.reason


class TestQualityStates:
    """M5.1 item 6: generated, complete, verified and runnable are different things."""

    def test_the_states_are_ordered(self) -> None:
        assert QualityState.GENERATED.rank < QualityState.TASK_COMPLETE.rank
        assert QualityState.TASK_COMPLETE.rank < QualityState.VERIFIED.rank
        assert QualityState.VERIFIED.rank < QualityState.INTEGRATED.rank

    def test_repair_exhaustion_is_not_completion(self) -> None:
        """M5.1 item 9: an exhausted budget fails closed."""
        assert TaskOutcome.REPAIR_EXHAUSTED is not TaskOutcome.COMPLETED
        assert TaskOutcome.REPAIR_EXHAUSTED.value == "REPAIR_EXHAUSTED"


class TestFairComparisonSetup:
    """M5.1 item 12: verify the arms share infrastructure *before* comparing them."""

    def test_the_generic_arm_is_a_role_not_a_separate_code_path(self) -> None:
        """It runs through the same executor, so the comparison measures specialisation."""
        assert EngineeringRole.GENERIC in EngineeringRole
        assert agent_for(EngineeringRole.GENERIC) is not None

    def test_the_generic_agent_shares_the_specialised_machinery(self) -> None:
        from edith.agents.coder import CodingAgent
        from edith.engineering.agents import EngineeringInput

        generic = agent_for(EngineeringRole.GENERIC)
        specialised = agent_for(EngineeringRole.BACKEND)
        assert issubclass(generic, CodingAgent)
        assert generic.input_schema is specialised.input_schema is EngineeringInput
        assert generic.output_schema is specialised.output_schema
        assert generic._run is specialised._run

    def test_the_generic_arm_is_not_handicapped_by_a_narrow_scope(self) -> None:
        """A control that fails for lack of permissions measures permissions, not roles."""
        generic = scope_for(EngineeringRole.GENERIC)
        for role in (
            EngineeringRole.BACKEND,
            EngineeringRole.DATABASE,
            EngineeringRole.FRONTEND,
        ):
            for pattern in scope_for(role).write:
                top = pattern.split("/")[0]
                assert any(
                    item.startswith(top) for item in generic.write
                ), f"generic cannot reach {pattern}"

    def test_the_generic_arm_holds_no_extra_privilege_either(self) -> None:
        """Fair means identical, not advantaged."""
        generic = scope_for(EngineeringRole.GENERIC)
        assert "shell.run" not in generic.tools
        assert not any(name.startswith("git.") for name in generic.tools)
        assert generic.tools == scope_for(EngineeringRole.BACKEND).tools

    def test_a_plan_naming_the_generic_agent_resolves_to_it(self) -> None:
        from edith.engineering.ownership import resolve_role

        assert resolve_role("generic") is EngineeringRole.GENERIC
