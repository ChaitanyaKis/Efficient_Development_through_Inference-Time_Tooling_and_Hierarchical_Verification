"""M2 agents: planner translation, coder sanitisation, critic adjudication, debugger output.

Hermetic: the model is faked, because what is under test here is the code around the model,
not the model itself. Live-model behaviour is covered by the benchmarks.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from edith.agents.coder import CoderOutput, CodingAgent, sanitize_content
from edith.agents.critic import CriticOutput, Finding, adjudicate
from edith.agents.debugger import DebuggerOutput
from edith.agents.planner import PlannedStep, PlannerOutput, plan_to_tasks
from edith.errors import FailureCategory
from edith.planning.dag import TaskGraph
from edith.schemas.agent import AgentRequest
from edith.schemas.common import Severity, Verdict
from edith.verification.runner import VerificationOutcome, VerificationReport

from .fakes import FakeProvider
from .tool_fixtures import build_gateway, build_workspace


def outcome(
    *, kind: str = "tests", passed: bool = True, ran: bool = True, exit_code: int = 0
) -> VerificationOutcome:
    """Build a verification outcome."""
    return VerificationOutcome(
        kind=kind,
        command="pytest -q",
        exit_code=exit_code if not passed else 0,
        passed=passed,
        failure_category=None if passed else FailureCategory.TEST_FAILURE,
        unavailable_reason=None if ran else "shell.run is not permitted",
    )


class TestSanitizeContent:
    def test_strips_context_file_marker(self) -> None:
        """The exact corruption observed against the real 3B model."""
        raw = "--- FILE: calculator.py ---\ndef f():\n    return 1\n"
        assert sanitize_content(raw) == "def f():\n    return 1\n"

    def test_strips_markdown_fence(self) -> None:
        raw = "```python\ndef g():\n    pass\n```"
        assert sanitize_content(raw) == "def g():\n    pass\n"

    def test_strips_bare_fence(self) -> None:
        assert sanitize_content("```\nx = 1\n```") == "x = 1\n"

    def test_strips_equals_rule(self) -> None:
        assert sanitize_content("========\ncode = 1\n") == "code = 1\n"

    def test_leaves_clean_code_untouched(self) -> None:
        raw = "def h():\n    pass\n"
        assert sanitize_content(raw) == raw

    def test_does_not_eat_real_code_containing_dashes(self) -> None:
        """A comment rule inside real code must survive."""
        raw = "x = 1\n# ----------------\ny = 2\n"
        assert "# ----------------" in sanitize_content(raw)

    def test_normalizes_line_endings(self) -> None:
        assert "\r" not in sanitize_content("a = 1\r\nb = 2\r\n")

    def test_empty_stays_empty(self) -> None:
        assert sanitize_content("   \n\n") == ""


class TestPlanToTasks:
    def test_translates_steps(self) -> None:
        plan = PlannerOutput(
            goal="add multiply",
            steps=[
                PlannedStep(
                    step=1,
                    title="Add multiply",
                    description="Add a multiply function",
                    files=["calculator.py"],
                    acceptance="test_multiply passes",
                )
            ],
        )
        tasks = plan_to_tasks(plan)
        assert len(tasks) == 1
        assert tasks[0].agent == "coder"
        assert tasks[0].acceptance_criteria == ("test_multiply passes",)
        assert tasks[0].verification[0].kind == "tests"

    def test_scope_is_derived_from_named_files(self) -> None:
        """A step that said it would touch one file cannot write anywhere else."""
        plan = PlannerOutput(
            goal="g",
            steps=[
                PlannedStep(step=1, title="t", description="d", files=["src/calc.py"])
            ],
        )
        scope = plan_to_tasks(plan)[0].scope
        assert "src/calc.py" in scope.write_paths
        assert "src/**" in scope.write_paths
        assert "other/**" not in scope.write_paths

    def test_dependencies_are_remapped_to_task_ids(self) -> None:
        plan = PlannerOutput(
            goal="g",
            steps=[
                PlannedStep(step=1, title="a", description="d"),
                PlannedStep(step=2, title="b", description="d", depends_on=[1]),
            ],
        )
        tasks = plan_to_tasks(plan)
        assert tasks[1].dependencies == (tasks[0].task_id,)
        TaskGraph(tasks)  # must form a valid DAG

    def test_dangling_dependency_is_dropped(self) -> None:
        """A reference to a step that does not exist must not produce a dangling edge."""
        plan = PlannerOutput(
            goal="g",
            steps=[PlannedStep(step=1, title="a", description="d", depends_on=[99])],
        )
        assert plan_to_tasks(plan)[0].dependencies == ()

    def test_self_dependency_is_dropped(self) -> None:
        plan = PlannerOutput(
            goal="g",
            steps=[PlannedStep(step=1, title="a", description="d", depends_on=[1])],
        )
        assert plan_to_tasks(plan)[0].dependencies == ()

    def test_unsafe_paths_are_discarded_from_scope(self) -> None:
        """A traversing or absolute path must not reach TaskScope and fail the whole plan."""
        plan = PlannerOutput(
            goal="g",
            steps=[
                PlannedStep(
                    step=1,
                    title="t",
                    description="d",
                    files=["../../etc/passwd", "C:/Windows/x", "ok.py"],
                )
            ],
        )
        scope = plan_to_tasks(plan)[0].scope
        assert scope.write_paths == ("ok.py",)

    def test_model_cannot_choose_the_agent(self) -> None:
        """PlannedStep has no agent field, so a plan cannot route work to a chosen agent."""
        assert "agent" not in PlannedStep.model_fields

    def test_model_cannot_author_a_verification_command(self) -> None:
        """Verification is a *kind*, selected from operator config, never a command."""
        plan = PlannerOutput(
            goal="g", steps=[PlannedStep(step=1, title="t", description="d")]
        )
        for requirement in plan_to_tasks(plan)[0].verification:
            assert requirement.kind in {"tests", "lint", "typecheck", "build"}


class TestAdjudication:
    def test_failing_tests_override_a_pass_verdict(self) -> None:
        """The core rule: evidence beats the model's opinion."""
        critic = CriticOutput(verdict=Verdict.PASS, reasoning="looks good to me")
        report = VerificationReport(outcomes=[outcome(passed=False, exit_code=1)])
        verdict, reason = adjudicate(critic, report, changes_made=True)
        assert verdict is Verdict.FAIL
        assert "tests failed" in reason

    def test_passing_tests_and_agreeing_critic(self) -> None:
        critic = CriticOutput(verdict=Verdict.PASS)
        report = VerificationReport(outcomes=[outcome(passed=True)])
        verdict, _ = adjudicate(critic, report, changes_made=True)
        assert verdict is Verdict.PASS

    def test_no_changes_is_never_a_pass(self) -> None:
        critic = CriticOutput(verdict=Verdict.PASS)
        report = VerificationReport(outcomes=[outcome(passed=True)])
        verdict, reason = adjudicate(critic, report, changes_made=False)
        assert verdict is Verdict.FAIL
        assert "no files were changed" in reason

    def test_unavailable_verification_is_blocked_not_failed(self) -> None:
        report = VerificationReport(outcomes=[outcome(ran=False)])
        verdict, reason = adjudicate(None, report, changes_made=True)
        assert verdict is Verdict.BLOCKED
        assert "could not run" in reason

    def test_missing_critic_falls_back_to_evidence(self) -> None:
        report = VerificationReport(outcomes=[outcome(passed=True)])
        verdict, _ = adjudicate(None, report, changes_made=True)
        assert verdict is Verdict.PASS

    def test_high_severity_finding_blocks_passing_tests(self) -> None:
        critic = CriticOutput(
            verdict=Verdict.FAIL,
            findings=[
                Finding(
                    severity=Severity.CRITICAL,
                    description="hard-coded credential introduced",
                )
            ],
        )
        report = VerificationReport(outcomes=[outcome(passed=True)])
        verdict, reason = adjudicate(critic, report, changes_made=True)
        assert verdict is Verdict.FAIL
        assert "high-severity" in reason

    def test_low_severity_concern_does_not_block(self) -> None:
        """A vague misgiving must not block completed, verified work forever."""
        critic = CriticOutput(
            verdict=Verdict.FAIL,
            findings=[Finding(severity=Severity.LOW, description="could be tidier")],
        )
        report = VerificationReport(outcomes=[outcome(passed=True)])
        verdict, _ = adjudicate(critic, report, changes_made=True)
        assert verdict is Verdict.PASS


class TestCodingAgent:
    @pytest.fixture
    def workspace(self, tmp_path: Path) -> Path:
        return build_workspace(tmp_path / "ws")

    def _provider(self, edits: list[dict[str, str]]) -> FakeProvider:
        from edith.config.schema import ModelParams

        payload = json.dumps({"edits": edits, "summary": "done", "notes": ""})
        return FakeProvider(ModelParams(model_name="test-model:q4"), [payload])

    def test_writes_through_the_gateway(self, workspace: Path) -> None:
        provider = self._provider([{"path": "src/new.py", "content": "x = 1\n"}])
        agent = CodingAgent(provider=provider, tools=build_gateway(workspace))
        response = agent.execute(
            AgentRequest(payload={"title": "t", "description": "d"})
        )
        assert response.ok
        assert (workspace / "src" / "new.py").read_text(encoding="utf-8") == "x = 1\n"

    def test_sanitises_before_writing(self, workspace: Path) -> None:
        provider = self._provider(
            [{"path": "src/new.py", "content": "--- FILE: src/new.py ---\nx = 1\n"}]
        )
        agent = CodingAgent(provider=provider, tools=build_gateway(workspace))
        agent.execute(AgentRequest(payload={"title": "t", "description": "d"}))
        assert (workspace / "src" / "new.py").read_text(encoding="utf-8") == "x = 1\n"

    def test_denied_write_is_reported_not_worked_around(self, workspace: Path) -> None:
        provider = self._provider([{"path": ".env", "content": "STOLEN=1\n"}])
        agent = CodingAgent(provider=provider, tools=build_gateway(workspace))
        response = agent.execute(
            AgentRequest(payload={"title": "t", "description": "d"})
        )
        output = CoderOutput.model_validate(response.output)
        assert output.rejected_files == [".env"]
        assert not output.made_changes
        assert "API_KEY" in (workspace / ".env").read_text(encoding="utf-8")

    def test_agent_has_no_direct_filesystem_or_process_access(self) -> None:
        """The coding agent must reach the world only through the gateway.

        Parsed with ``ast`` rather than substring-matched, so prose in a docstring cannot
        pass or fail the check -- only real imports and real calls count.
        """
        import ast

        module = ast.parse(
            (Path("src/edith/agents/coder.py")).read_text(encoding="utf-8")
        )

        imported: set[str] = set()
        called: set[str] = set()
        for node in ast.walk(module):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                called.add(node.func.id)

        assert not ({"subprocess", "os", "shutil", "pathlib"} & imported)
        assert "open" not in called


class TestDebuggerOutput:
    def test_guidance_includes_the_diagnosis(self) -> None:
        output = DebuggerOutput(
            diagnosis="subtract returns a + b",
            root_cause="wrong operator",
            suspected_files=["calculator.py"],
            suggested_fix="change + to -",
        )
        guidance = output.as_guidance()
        assert "subtract returns a + b" in guidance
        assert "calculator.py" in guidance
        assert "change + to -" in guidance

    def test_flags_a_wrong_test(self) -> None:
        output = DebuggerOutput(
            diagnosis="d", suggested_fix="f", test_is_wrong=True
        )
        assert "test is incorrect" in output.as_guidance()
