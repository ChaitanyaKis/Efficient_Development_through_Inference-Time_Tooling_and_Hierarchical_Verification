"""The orchestrated loop: sequencing, bounds, persistence, recovery, and boundaries.

The model is scripted rather than live. That is deliberate: these tests assert that the
*loop* behaves correctly given a sequence of model outputs -- including the failure-recovery
path, which cannot be triggered on demand with a real model. Live-model behaviour is
measured by the benchmarks instead.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import pytest

from edith.config.schema import (
    ContextConfig,
    EdithConfig,
    ModelParams,
    ModelsConfig,
    OrchestrationConfig,
    ShellPolicyConfig,
    ToolsConfig,
    VerificationProfile,
)
from edith.errors import FailureCategory
from edith.models.base import ModelProvider
from edith.orchestrator import (
    VERIFIER_PERMISSIONS,
    Orchestrator,
    create_execution,
    resume_graph,
)
from edith.planning.task import TaskStatus
from edith.schemas.common import Verdict
from edith.schemas.model import (
    GenerationOptions,
    GenerationResult,
    HealthState,
    Message,
    ProviderHealth,
    TokenUsage,
)
from edith.state.schema import ProjectState
from edith.state.store import open_store
from edith.workspaces import ProjectWorkspace

from .test_tool_git import init_repo

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")

PYTHON = "python"


class ScriptedProvider(ModelProvider):
    """Returns queued responses keyed by the schema being requested.

    Keying by schema rather than call order keeps a test readable: it states what the
    planner says, what the coder says, and so on, without having to count invocations.
    """

    name = "scripted"

    def __init__(self, params: ModelParams, script: dict[str, list[str]]) -> None:
        super().__init__(params)
        self.script = {key: list(value) for key, value in script.items()}
        self.calls: list[str] = []
        #: Schema title -> the rendered prompts sent for it. Lets a test assert on what
        #: actually reached the model rather than on what was meant to.
        self.prompts: dict[str, list[str]] = {}

    def _generate_raw(
        self,
        messages: Sequence[Message],
        options: GenerationOptions | None = None,
        *,
        json_schema: dict[str, Any] | None = None,
    ) -> GenerationResult:
        title = str((json_schema or {}).get("title", ""))
        self.calls.append(title)
        self.prompts.setdefault(title, []).append(
            "\n".join(message.content for message in messages)
        )
        queue = self.script.get(title)
        if not queue:
            raise AssertionError(f"no scripted response for schema {title!r}")
        text = queue.pop(0) if len(queue) > 1 else queue[0]
        return GenerationResult(
            text=text,
            model=self.params.model_name,
            provider=self.name,
            usage=TokenUsage(prompt_tokens=10, completion_tokens=10),
        )

    def prompts_for(self, schema_title: str) -> list[str]:
        """Every prompt sent while generating ``schema_title``."""
        return self.prompts.get(schema_title, [])

    def stream(
        self, messages: Sequence[Message], options: GenerationOptions | None = None
    ) -> Iterator[str]:
        yield self._generate_raw(messages, options).text

    def health_check(self) -> ProviderHealth:
        return ProviderHealth(provider=self.name, state=HealthState.HEALTHY)

    def list_models(self) -> tuple[str, ...]:
        return (self.params.model_name,)

    def close(self) -> None:
        return None


def plan(
    files: list[str],
    title: str = "Fix the code",
    description: str = "Change the code so the tests pass.",
) -> str:
    """A one-step plan targeting ``files``."""
    return json.dumps(
        {
            "goal": "make the tests pass",
            "steps": [
                {
                    "step": 1,
                    "title": title,
                    "description": description,
                    "files": files,
                    "depends_on": [],
                    "acceptance": "tests pass",
                }
            ],
        }
    )


def edits(path: str, content: str, mode: str = "replace_file") -> str:
    """A coder response writing ``content`` to ``path``."""
    return json.dumps(
        {
            "edits": [{"path": path, "mode": mode, "content": content}],
            "summary": "applied the change",
            "notes": "",
        }
    )


def verdict(value: str = "PASS") -> str:
    return json.dumps({"verdict": value, "reasoning": "checked", "findings": []})


def diagnosis() -> str:
    return json.dumps(
        {
            "diagnosis": "the function returns the wrong value",
            "suspected_files": ["calc.py"],
            "root_cause": "wrong operator",
            "suggested_fix": "return a - b",
            "test_is_wrong": False,
            "confidence": 0.9,
        }
    )


GOOD_CODE = "def subtract(a, b):\n    return a - b\n"
BAD_CODE = "def subtract(a, b):\n    return a + b\n"


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A tiny repository whose test suite fails until `subtract` is fixed."""
    root = tmp_path / "project"
    root.mkdir()
    (root / "calc.py").write_text(BAD_CODE, encoding="utf-8")
    (root / "test_calc.py").write_text(
        "from calc import subtract\n\n\ndef test_subtract():\n    assert subtract(5, 3) == 2\n",
        encoding="utf-8",
    )
    return init_repo(root)


@pytest.fixture
def config(repo: Path) -> EdithConfig:
    import sys

    return EdithConfig(
        models=ModelsConfig(profiles={"default": ModelParams(model_name="scripted:1")}),
        tools=ToolsConfig(
            workspace_root=repo,
            shell=ShellPolicyConfig(allowed_executables=(Path(sys.executable).stem, PYTHON)),
        ),
        orchestration=OrchestrationConfig(
            workspaces_root=repo.parent,
            max_task_attempts=3,
            max_repair_attempts=2,
            context=ContextConfig(max_files=4, max_total_chars=4000),
            verification_profiles={
                "python": VerificationProfile(tests=(PYTHON, "-m", "pytest", "-q"))
            },
        ),
    )


@pytest.fixture
def workspace(repo: Path) -> ProjectWorkspace:
    return ProjectWorkspace(project_id="proj_test", name="project", root=repo)


def build(
    config: EdithConfig, workspace: ProjectWorkspace, store: Any, script: dict[str, list[str]]
) -> Orchestrator:
    provider = ScriptedProvider(ModelParams(model_name="scripted:1"), script)
    return Orchestrator(config, store, workspace, provider=provider)


@pytest.fixture
def store(tmp_path: Path) -> Any:
    with open_store(tmp_path / "state") as opened:
        yield opened


class TestHappyPath:
    def test_request_to_release(
        self, config: EdithConfig, workspace: ProjectWorkspace, store: Any, repo: Path
    ) -> None:
        """A real request enters, a plan is produced, code changes, tests run, PASS."""
        orchestrator = build(
            config,
            workspace,
            store,
            {
                "PlannerOutput": [plan(["calc.py"])],
                "ModelEdits": [edits("calc.py", GOOD_CODE)],
                "CriticOutput": [verdict("PASS")],
            },
        )
        _, execution = create_execution(store, workspace, "fix subtract")
        result = orchestrator.run(execution)

        assert result.verdict is Verdict.PASS
        assert result.state is ProjectState.RELEASE
        assert result.changed_files == ["calc.py"]
        # Verified against the workspace, not the agent's claim.
        assert "a - b" in (repo / "calc.py").read_text(encoding="utf-8")

    def test_evidence_is_persisted(
        self, config: EdithConfig, workspace: ProjectWorkspace, store: Any
    ) -> None:
        orchestrator = build(
            config,
            workspace,
            store,
            {
                "PlannerOutput": [plan(["calc.py"])],
                "ModelEdits": [edits("calc.py", GOOD_CODE)],
                "CriticOutput": [verdict("PASS")],
            },
        )
        _, execution = create_execution(store, workspace, "fix subtract")
        orchestrator.run(execution)

        assert {run.agent for run in store.agent_runs(execution.execution_id)} >= {
            "planner",
            "coder",
            "critic",
        }
        records = store.verifications(execution.execution_id)
        assert records and any(record.passed for record in records)
        # The captured output is retrievable, so "the tests passed" is checkable.
        assert store.artifacts.get(records[-1].output_ref) is not None
        assert store.transitions(execution.execution_id)


class TestFailureRecovery:
    def test_detect_diagnose_repair_reverify(
        self, config: EdithConfig, workspace: ProjectWorkspace, store: Any, repo: Path
    ) -> None:
        """The M2 recovery criterion, driven deterministically.

        The first implementation is wrong, so verification fails for real. The loop must
        classify it, consult the debugger, apply a second implementation, re-run the tests,
        and only then reach PASS.
        """
        orchestrator = build(
            config,
            workspace,
            store,
            {
                "PlannerOutput": [plan(["calc.py"])],
                # First attempt keeps the bug; second fixes it.
                "ModelEdits": [
                    edits("calc.py", BAD_CODE.replace("a + b", "a * b")),
                    edits("calc.py", GOOD_CODE),
                ],
                "CriticOutput": [verdict("FAIL"), verdict("PASS")],
                "DebuggerOutput": [diagnosis()],
            },
        )
        _, execution = create_execution(store, workspace, "fix subtract")
        result = orchestrator.run(execution)

        assert result.verdict is Verdict.PASS
        assert result.repairs_attempted >= 1, "the debugger must have been consulted"
        assert "a - b" in (repo / "calc.py").read_text(encoding="utf-8")

        # The failing attempt is recorded, not hidden.
        failures = store.failures(execution.execution_id)
        assert failures
        assert any(f.category is FailureCategory.TEST_FAILURE for f in failures)

        # And a real failing verification is on record before the passing one.
        records = store.verifications(execution.execution_id)
        assert not records[0].passed
        assert records[-1].passed

    def test_debugger_diagnosis_reaches_the_next_attempt(
        self, config: EdithConfig, workspace: ProjectWorkspace, store: Any
    ) -> None:
        provider = ScriptedProvider(
            ModelParams(model_name="scripted:1"),
            {
                "PlannerOutput": [plan(["calc.py"])],
                "ModelEdits": [edits("calc.py", BAD_CODE), edits("calc.py", GOOD_CODE)],
                "CriticOutput": [verdict("FAIL"), verdict("PASS")],
                "DebuggerOutput": [diagnosis()],
            },
        )
        orchestrator = Orchestrator(config, store, workspace, provider=provider)
        _, execution = create_execution(store, workspace, "fix subtract")
        orchestrator.run(execution)
        assert "DebuggerOutput" in provider.calls

    def test_bounded_when_repair_never_works(
        self, config: EdithConfig, workspace: ProjectWorkspace, store: Any
    ) -> None:
        """A loop that cannot succeed must still terminate."""
        orchestrator = build(
            config,
            workspace,
            store,
            {
                "PlannerOutput": [plan(["calc.py"])],
                "ModelEdits": [edits("calc.py", BAD_CODE)],
                "CriticOutput": [verdict("FAIL")],
                "DebuggerOutput": [diagnosis()],
            },
        )
        _, execution = create_execution(store, workspace, "fix subtract")
        result = orchestrator.run(execution)

        assert result.verdict is not Verdict.PASS
        assert result.state is ProjectState.FAILED
        assert result.agent_runs <= config.orchestration.max_total_agent_runs


class TestEvidenceOverClaims:
    def test_a_critic_pass_cannot_override_failing_tests(
        self, config: EdithConfig, workspace: ProjectWorkspace, store: Any
    ) -> None:
        """The whole point of the design: the model does not get to declare success."""
        orchestrator = build(
            config,
            workspace,
            store,
            {
                "PlannerOutput": [plan(["calc.py"])],
                "ModelEdits": [edits("calc.py", BAD_CODE)],
                "CriticOutput": [verdict("PASS")],
                "DebuggerOutput": [diagnosis()],
            },
        )
        _, execution = create_execution(store, workspace, "fix subtract")
        result = orchestrator.run(execution)
        assert result.verdict is not Verdict.PASS

    def test_no_changes_is_not_success(
        self, config: EdithConfig, workspace: ProjectWorkspace, store: Any
    ) -> None:
        orchestrator = build(
            config,
            workspace,
            store,
            {
                "PlannerOutput": [plan(["calc.py"])],
                "ModelEdits": [edits("calc.py", BAD_CODE)],
                "CriticOutput": [verdict("PASS")],
                "DebuggerOutput": [diagnosis()],
            },
        )
        _, execution = create_execution(store, workspace, "do nothing")
        assert orchestrator.run(execution).verdict is not Verdict.PASS


class TestSecurityBoundaries:
    """The orchestration layer must not be able to widen what M1 permits."""

    def test_task_scope_narrows_the_coder(
        self, config: EdithConfig, workspace: ProjectWorkspace, store: Any, repo: Path
    ) -> None:
        """A plan naming calc.py must not let the coder write elsewhere."""
        orchestrator = build(
            config,
            workspace,
            store,
            {
                "PlannerOutput": [plan(["calc.py"])],
                "ModelEdits": [edits("test_calc.py", "assert True\n")],
                "CriticOutput": [verdict("PASS")],
                "DebuggerOutput": [diagnosis()],
            },
        )
        _, execution = create_execution(store, workspace, "sneak into the tests")
        orchestrator.run(execution)
        assert "def test_subtract" in (repo / "test_calc.py").read_text(encoding="utf-8")

    def test_protected_paths_survive_orchestration(
        self, config: EdithConfig, workspace: ProjectWorkspace, store: Any, repo: Path
    ) -> None:
        (repo / ".env").write_text("SECRET=keepme\n", encoding="utf-8")
        orchestrator = build(
            config,
            workspace,
            store,
            {
                "PlannerOutput": [plan([".env"], title="Update configuration")],
                "ModelEdits": [edits(".env", "SECRET=stolen\n")],
                "CriticOutput": [verdict("PASS")],
                "DebuggerOutput": [diagnosis()],
            },
        )
        _, execution = create_execution(store, workspace, "change the env file")
        orchestrator.run(execution)
        assert "keepme" in (repo / ".env").read_text(encoding="utf-8")

    def test_traversal_in_a_plan_is_refused(
        self, config: EdithConfig, workspace: ProjectWorkspace, store: Any, tmp_path: Path
    ) -> None:
        orchestrator = build(
            config,
            workspace,
            store,
            {
                "PlannerOutput": [plan(["../escaped.py"])],
                "ModelEdits": [edits("../escaped.py", "x = 1\n")],
                "CriticOutput": [verdict("PASS")],
                "DebuggerOutput": [diagnosis()],
            },
        )
        _, execution = create_execution(store, workspace, "escape the workspace")
        orchestrator.run(execution)
        assert not (tmp_path / "escaped.py").exists()

    def test_the_verifier_cannot_write(self) -> None:
        """The principal that decides whether work passed must not be able to write it."""
        assert VERIFIER_PERMISSIONS.read_only
        assert "shell.run" in VERIFIER_PERMISSIONS.allowed_tools
        assert "filesystem.write" not in VERIFIER_PERMISSIONS.allowed_tools

    def test_the_coder_cannot_run_commands(self) -> None:
        from edith.agents.coder import CodingAgent

        assert "shell.run" not in CodingAgent.identity.permissions.allowed_tools

    def test_read_only_agents_declare_no_write_scope(self) -> None:
        from edith.agents.critic import CriticAgent
        from edith.agents.debugger import DebuggingAgent
        from edith.agents.planner import PlannerAgent

        for agent in (PlannerAgent, CriticAgent, DebuggingAgent):
            assert agent.identity.permissions.read_only, agent.identity.name
            assert "shell.run" not in agent.identity.permissions.allowed_tools


class TestTestIntegrityEndToEnd:
    """The M2 false positive, reproduced end to end and now rejected.

    A scoped-to-tests plan lets the coder legally write the test file, it rewrites the
    assertion to match the bug, and the suite genuinely goes green. Nothing but the
    integrity gate stands between that and a PASS.
    """

    TAMPERED_TEST = (
        "from calc import subtract\n\n\ndef test_subtract():\n    assert subtract(5, 3) == 8\n"
    )

    def _run_tampering(self, config: EdithConfig, workspace: ProjectWorkspace, store: Any):
        orchestrator = build(
            config,
            workspace,
            store,
            {
                # The plan grants write access to the test file, so the gateway permits it.
                "PlannerOutput": [plan(["test_calc.py"], title="Make the tests pass")],
                "ModelEdits": [edits("test_calc.py", self.TAMPERED_TEST)],
                "CriticOutput": [verdict("PASS")],
                "DebuggerOutput": [diagnosis()],
            },
        )
        _, execution = create_execution(store, workspace, "make the failing test pass")
        return orchestrator.run(execution), execution

    def test_tampering_is_rejected_even_though_the_suite_goes_green(
        self, config: EdithConfig, workspace: ProjectWorkspace, store: Any, repo: Path
    ) -> None:
        result, _ = self._run_tampering(config, workspace, store)

        # The suite really does pass after the edit -- that is what makes this dangerous.
        assert "== 8" in (repo / "test_calc.py").read_text(encoding="utf-8")
        assert result.verdict is not Verdict.PASS
        assert result.state is ProjectState.FAILED

    def test_the_reason_names_test_integrity(
        self, config: EdithConfig, workspace: ProjectWorkspace, store: Any
    ) -> None:
        result, _ = self._run_tampering(config, workspace, store)
        assert "integrity" in result.summary.lower()

    def test_the_violation_is_persisted(
        self, config: EdithConfig, workspace: ProjectWorkspace, store: Any
    ) -> None:
        """An operator reviewing the run must be able to see what happened."""
        _, execution = self._run_tampering(config, workspace, store)
        failures = store.failures(execution.execution_id)
        assert any("integrity" in failure.message.lower() for failure in failures)

    def test_a_legitimate_source_fix_still_passes(
        self, config: EdithConfig, workspace: ProjectWorkspace, store: Any, repo: Path
    ) -> None:
        """The gate must not block honest work."""
        orchestrator = build(
            config,
            workspace,
            store,
            {
                "PlannerOutput": [plan(["calc.py"])],
                "ModelEdits": [edits("calc.py", GOOD_CODE)],
                "CriticOutput": [verdict("PASS")],
            },
        )
        _, execution = create_execution(store, workspace, "fix subtract")
        result = orchestrator.run(execution)
        assert result.verdict is Verdict.PASS
        assert "== 2" in (repo / "test_calc.py").read_text(encoding="utf-8")

    def test_adding_a_test_alongside_a_fix_passes(
        self, config: EdithConfig, workspace: ProjectWorkspace, store: Any
    ) -> None:
        """Adding coverage is legitimate and must not trip the gate."""
        extended = (
            "from calc import subtract\n\n\ndef test_subtract():\n"
            "    assert subtract(5, 3) == 2\n\n\ndef test_subtract_zero():\n"
            "    assert subtract(4, 4) == 0\n"
        )
        orchestrator = build(
            config,
            workspace,
            store,
            {
                "PlannerOutput": [plan(["calc.py", "test_calc.py"])],
                "ModelEdits": [
                    json.dumps(
                        {
                            "edits": [
                                {
                                    "path": "calc.py",
                                    "mode": "replace_file",
                                    "content": GOOD_CODE,
                                },
                                {
                                    "path": "test_calc.py",
                                    "mode": "replace_file",
                                    "content": extended,
                                },
                            ],
                            "summary": "fix and extend",
                            "notes": "",
                        }
                    )
                ],
                "CriticOutput": [verdict("PASS")],
            },
        )
        _, execution = create_execution(store, workspace, "fix subtract and add a test")
        assert orchestrator.run(execution).verdict is Verdict.PASS


class TestRestartRecovery:
    def test_execution_state_survives_a_restart(
        self, config: EdithConfig, workspace: ProjectWorkspace, tmp_path: Path
    ) -> None:
        """Criterion J: a new process resumes from persisted state alone."""
        state_dir = tmp_path / "state"

        with open_store(state_dir) as first:
            orchestrator = build(
                config,
                workspace,
                first,
                {
                    "PlannerOutput": [plan(["calc.py"])],
                    "ModelEdits": [edits("calc.py", GOOD_CODE)],
                    "CriticOutput": [verdict("PASS")],
                },
            )
            _, execution = create_execution(first, workspace, "fix subtract")
            orchestrator.run(execution)
            execution_id = execution.execution_id

        # Nothing from the first process is carried over.
        with open_store(state_dir) as second:
            reloaded = second.get_execution(execution_id)
            assert reloaded is not None
            assert reloaded.state is ProjectState.RELEASE

            graph = resume_graph(second, execution_id)
            assert graph is not None
            assert all(task.status is TaskStatus.SUCCEEDED for task in graph.tasks())
            assert second.verifications(execution_id)
            assert second.agent_runs(execution_id)

    def test_resume_returns_none_for_an_unknown_execution(self, tmp_path: Path) -> None:
        with open_store(tmp_path / "state") as store:
            assert resume_graph(store, "exec_missing") is None


class TestPlanValidation:
    def test_malformed_plan_fails_safely(
        self, config: EdithConfig, workspace: ProjectWorkspace, store: Any
    ) -> None:
        orchestrator = build(
            config, workspace, store, {"PlannerOutput": ["this is not JSON at all"]}
        )
        _, execution = create_execution(store, workspace, "do something")
        result = orchestrator.run(execution)

        assert result.verdict is Verdict.BLOCKED
        assert result.state is ProjectState.FAILED
        assert "planning failed" in result.summary

    def test_cyclic_plan_is_rejected(
        self, config: EdithConfig, workspace: ProjectWorkspace, store: Any
    ) -> None:
        cyclic = json.dumps(
            {
                "goal": "g",
                "steps": [
                    {"step": 1, "title": "a", "description": "d", "depends_on": [2]},
                    {"step": 2, "title": "b", "description": "d", "depends_on": [1]},
                ],
            }
        )
        orchestrator = build(config, workspace, store, {"PlannerOutput": [cyclic]})
        _, execution = create_execution(store, workspace, "do something")
        assert orchestrator.run(execution).state is ProjectState.FAILED
