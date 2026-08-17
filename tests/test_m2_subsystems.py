"""Context engine, verification runner, retry policy, and workspace isolation."""

from __future__ import annotations

from pathlib import Path

import pytest

from edith.config.schema import (
    ContextConfig,
    EdithConfig,
    ModelParams,
    ModelsConfig,
    OrchestrationConfig,
    VerificationProfile,
)
from edith.context.engine import ContextEngine, keywords
from edith.errors import ConfigurationError, FailureCategory
from edith.policy import FailureAction, decide, is_security_failure, rule_for
from edith.schemas.agent import AgentPermissions
from edith.verification.runner import VerificationRunner
from edith.workspaces import WorkspaceManager, edith_repository_root

from .tool_fixtures import build_config, build_gateway, build_workspace

PYTHON = "python"


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    return build_workspace(tmp_path / "ws")


class TestFailurePolicy:
    @pytest.mark.parametrize(
        ("category", "action"),
        [
            (FailureCategory.MODEL_ERROR, FailureAction.RETRY),
            (FailureCategory.VALIDATION_FAILURE, FailureAction.RETRY),
            (FailureCategory.TIMEOUT, FailureAction.RETRY),
            (FailureCategory.TEST_FAILURE, FailureAction.REPAIR),
            (FailureCategory.BUILD_ERROR, FailureAction.REPAIR),
            (FailureCategory.REQUIREMENT_FAILURE, FailureAction.ESCALATE),
            (FailureCategory.ARCHITECTURE_FAILURE, FailureAction.ESCALATE),
            (FailureCategory.ENVIRONMENT_FAILURE, FailureAction.ESCALATE),
            (FailureCategory.SECURITY_FAILURE, FailureAction.ABORT),
            (FailureCategory.UNKNOWN, FailureAction.ESCALATE),
        ],
    )
    def test_each_category_has_explicit_behaviour(
        self, category: FailureCategory, action: FailureAction
    ) -> None:
        assert decide(category, attempts=1, max_attempts=3) is action

    def test_every_category_is_mapped(self) -> None:
        """An unmapped category would silently inherit a default -- exactly how a security
        failure ends up being retried."""
        from edith.policy import FAILURE_POLICY

        assert set(FAILURE_POLICY) == set(FailureCategory)

    def test_security_failure_is_never_retried_even_on_first_attempt(self) -> None:
        assert decide(FailureCategory.SECURITY_FAILURE, attempts=0, max_attempts=10) is (
            FailureAction.ABORT
        )
        assert not rule_for(FailureCategory.SECURITY_FAILURE).retryable
        assert is_security_failure(FailureCategory.SECURITY_FAILURE)

    def test_budget_exhaustion_downgrades_retry_to_escalate(self) -> None:
        """Bounded loops: a retryable failure stops being retried once the budget is spent."""
        assert decide(FailureCategory.MODEL_ERROR, attempts=3, max_attempts=3) is (
            FailureAction.ESCALATE
        )

    def test_budget_exhaustion_does_not_soften_abort(self) -> None:
        assert decide(FailureCategory.SECURITY_FAILURE, attempts=99, max_attempts=3) is (
            FailureAction.ABORT
        )

    def test_unknown_category_escalates(self) -> None:
        assert decide(None, attempts=1, max_attempts=3) is FailureAction.ESCALATE

    def test_escalation_flags_are_set_where_a_human_is_needed(self) -> None:
        for category in (
            FailureCategory.SECURITY_FAILURE,
            FailureCategory.REQUIREMENT_FAILURE,
            FailureCategory.ENVIRONMENT_FAILURE,
        ):
            assert rule_for(category).escalate_to_human

    def test_every_rule_explains_itself(self) -> None:
        from edith.policy import FAILURE_POLICY

        for category, rule in FAILURE_POLICY.items():
            assert rule.reason, f"{category} has no documented rationale"


class TestContextEngine:
    def test_retrieves_relevant_files(self, workspace: Path) -> None:
        engine = ContextEngine(build_gateway(workspace))
        bundle = engine.build("fix the app main function")
        assert "src/app.py" in bundle.file_paths

    def test_does_not_dump_the_whole_repository(self, workspace: Path) -> None:
        """The entire point of the engine: relevance, not everything."""
        for index in range(40):
            (workspace / "src" / f"unrelated_{index}.py").write_text(
                f"VALUE_{index} = {index}\n", encoding="utf-8"
            )
        engine = ContextEngine(build_gateway(workspace), ContextConfig(max_files=5))
        bundle = engine.build("fix the app main function")
        assert len(bundle.relevant_files) <= 5
        assert bundle.files_considered > 20

    def test_respects_the_character_budget(self, workspace: Path) -> None:
        (workspace / "src" / "huge.py").write_text("x = 1\n" * 5000, encoding="utf-8")
        engine = ContextEngine(
            build_gateway(workspace), ContextConfig(max_total_chars=2000, max_file_chars=800)
        )
        bundle = engine.build("huge")
        assert bundle.estimated_context_chars <= 2000
        assert len(bundle.render()) < 6000

    def test_hint_paths_are_strongly_preferred(self, workspace: Path) -> None:
        engine = ContextEngine(build_gateway(workspace))
        bundle = engine.build("do something", hint_paths=("src/backend/api.py",))
        assert bundle.file_paths[0] == "src/backend/api.py"

    def test_records_a_rationale(self, workspace: Path) -> None:
        """A bad answer is often a retrieval failure; the rationale is how you tell."""
        bundle = ContextEngine(build_gateway(workspace)).build("app main")
        assert bundle.rationale
        assert any("app.py" in line for line in bundle.rationale)

    def test_never_returns_protected_files(self, workspace: Path) -> None:
        bundle = ContextEngine(build_gateway(workspace)).build("api key secret env")
        assert not any(".env" in path for path in bundle.file_paths)
        assert "super-secret-value" not in bundle.render()

    def test_respects_the_agents_read_scope(self, workspace: Path) -> None:
        """The engine is not a side channel to files the agent could not open itself."""
        narrow = AgentPermissions(
            allowed_tools=frozenset({"filesystem.read", "filesystem.search"}),
            allowed_read_paths=("src/**",),
        )
        bundle = ContextEngine(build_gateway(workspace, narrow)).build("guide docs readme")
        assert all(path.startswith("src/") for path in bundle.file_paths)

    def test_render_names_defined_symbols(self, workspace: Path) -> None:
        """Naming what a file defines is what stops the coder silently dropping it."""
        rendered = ContextEngine(build_gateway(workspace)).build("app main helper").render()
        assert "It currently defines:" in rendered
        assert "main()" in rendered

    def test_render_avoids_delimiter_lines(self, workspace: Path) -> None:
        """Regression: a '--- FILE: x ---' delimiter was copied into written files."""
        rendered = ContextEngine(build_gateway(workspace)).build("app").render()
        assert "--- FILE:" not in rendered

    def test_empty_result_is_reported_not_faked(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()
        bundle = ContextEngine(build_gateway(empty)).build("anything")
        assert bundle.relevant_files == []
        assert "no repository context" in bundle.render()

    def test_keywords_drop_stopwords(self) -> None:
        assert "the" not in keywords("fix the multiply function")
        assert "multiply" in keywords("fix the multiply function")


class TestVerificationRunner:
    def _runner(self, workspace: Path, argv: tuple[str, ...]) -> VerificationRunner:
        from edith.config.schema import ShellPolicyConfig

        config = build_config(
            workspace, shell=ShellPolicyConfig(allowed_executables=(PYTHON,))
        )
        permissions = AgentPermissions(
            allowed_tools=frozenset({"shell.run"}), allowed_read_paths=("**",)
        )
        gateway = build_gateway(workspace, permissions, config=config)
        return VerificationRunner(gateway, VerificationProfile(tests=argv))

    def test_passing_command_is_evidence(self, workspace: Path) -> None:
        runner = self._runner(workspace, (PYTHON, "-c", "print('all good')"))
        outcome = runner.run("tests")
        assert outcome.passed and outcome.exit_code == 0
        assert "all good" in outcome.stdout

    def test_failing_command_is_captured_not_raised(self, workspace: Path) -> None:
        runner = self._runner(workspace, (PYTHON, "-c", "import sys; sys.exit(1)"))
        outcome = runner.run("tests")
        assert not outcome.passed
        assert outcome.ran
        assert outcome.failure_category is FailureCategory.TEST_FAILURE

    def test_unconfigured_check_is_unavailable_not_failed(self, workspace: Path) -> None:
        runner = self._runner(workspace, (PYTHON, "-c", "pass"))
        outcome = runner.run("lint")
        assert not outcome.ran
        assert "no 'lint' command is configured" in (outcome.unavailable_reason or "")

    def test_denied_command_is_a_security_failure(self, workspace: Path) -> None:
        """A verifier without shell.run must not silently report a passing suite."""
        permissions = AgentPermissions(
            allowed_tools=frozenset({"filesystem.read"}), allowed_read_paths=("**",)
        )
        gateway = build_gateway(workspace, permissions)
        runner = VerificationRunner(
            gateway, VerificationProfile(tests=(PYTHON, "-c", "pass"))
        )
        outcome = runner.run("tests")
        assert not outcome.ran
        assert outcome.failure_category is FailureCategory.SECURITY_FAILURE

    def test_missing_runner_is_an_environment_failure(self, workspace: Path) -> None:
        """Regression: a missing pytest reported as a failing test suite, sending the
        debugger hunting for a bug in code that never ran."""
        runner = self._runner(
            workspace,
            (
                PYTHON,
                "-c",
                "import sys; sys.stderr.write(\"No module named 'pytest'\"); sys.exit(1)",
            ),
        )
        outcome = runner.run("tests")
        assert not outcome.ran
        assert outcome.failure_category is FailureCategory.ENVIRONMENT_FAILURE

    def test_parses_pytest_counts(self, workspace: Path) -> None:
        runner = self._runner(
            workspace, (PYTHON, "-c", "print('2 failed, 5 passed in 0.3s')")
        )
        outcome = runner.run("tests")
        assert outcome.tests_passed == 5 and outcome.tests_failed == 2

    def test_report_stops_at_the_first_failure(self, workspace: Path) -> None:
        from edith.config.schema import ShellPolicyConfig

        config = build_config(
            workspace, shell=ShellPolicyConfig(allowed_executables=(PYTHON,))
        )
        gateway = build_gateway(
            workspace,
            AgentPermissions(allowed_tools=frozenset({"shell.run"}), allowed_read_paths=("**",)),
            config=config,
        )
        runner = VerificationRunner(
            gateway,
            VerificationProfile(
                tests=(PYTHON, "-c", "import sys; sys.exit(1)"),
                lint=(PYTHON, "-c", "pass"),
            ),
        )
        report = runner.run_all((("tests", None), ("lint", None)))
        assert len(report.outcomes) == 1
        assert not report.passed

    def test_evidence_includes_real_output(self, workspace: Path) -> None:
        runner = self._runner(
            workspace, (PYTHON, "-c", "print('FAILED test_x'); raise SystemExit(1)")
        )
        report = runner.run_all((("tests", None),))
        assert "FAILED test_x" in report.evidence()

    def test_a_model_cannot_author_the_command(self, workspace: Path) -> None:
        """Commands come from operator config; a task selects only a kind."""
        runner = self._runner(workspace, (PYTHON, "-c", "pass"))
        assert runner.run("rm -rf /").unavailable_reason is not None


class TestWorkspaceIsolation:
    def _config(self, root: Path) -> EdithConfig:
        return EdithConfig(
            models=ModelsConfig(profiles={"default": ModelParams(model_name="m")}),
            orchestration=OrchestrationConfig(workspaces_root=root),
        )

    def test_creates_a_workspace(self, tmp_path: Path) -> None:
        manager = WorkspaceManager(self._config(tmp_path / "spaces"))
        workspace = manager.create("project-a", "proj_1")
        assert workspace.root.is_dir()
        assert workspace.root.name == "project-a"

    def test_refuses_the_edith_repository_itself(self, tmp_path: Path) -> None:
        """The guard that stops an autonomous run editing the kernel running it."""
        manager = WorkspaceManager(self._config(tmp_path))
        with pytest.raises(ConfigurationError, match="may not be the Edith repository"):
            manager.adopt(edith_repository_root(), "proj_1")

    def test_refuses_a_directory_inside_the_edith_repository(self) -> None:
        manager = WorkspaceManager(self._config(edith_repository_root() / "src"))
        with pytest.raises(ConfigurationError, match="inside the Edith repository"):
            manager.create("evil", "proj_1")

    def test_refuses_a_parent_of_the_edith_repository(self, tmp_path: Path) -> None:
        manager = WorkspaceManager(self._config(tmp_path))
        with pytest.raises(ConfigurationError, match="may not contain the Edith repository"):
            manager.adopt(edith_repository_root().parent, "proj_1")

    @pytest.mark.parametrize("name", ["../escape", "a/b", "C:evil", "..", ""])
    def test_rejects_unsafe_project_names(self, tmp_path: Path, name: str) -> None:
        manager = WorkspaceManager(self._config(tmp_path / "spaces"))
        with pytest.raises(ConfigurationError):
            manager.path_for(name)

    def test_workspace_reroots_the_tool_layer_only(self, tmp_path: Path) -> None:
        """A workspace narrows *where* tools operate, never *what they may do*."""
        base = build_config(tmp_path)
        manager = WorkspaceManager(self._config(tmp_path / "spaces"))
        workspace = manager.create("p", "proj_1")
        rerooted = workspace.config_for(base)

        assert rerooted.tools.workspace_root == workspace.root
        assert rerooted.tools.paths.protected_patterns == base.tools.paths.protected_patterns
        assert rerooted.tools.shell.allowed_executables == base.tools.shell.allowed_executables

    def test_adopt_requires_an_existing_directory(self, tmp_path: Path) -> None:
        manager = WorkspaceManager(self._config(tmp_path / "spaces"))
        with pytest.raises(ConfigurationError, match="not an existing directory"):
            manager.adopt(tmp_path / "absent", "proj_1")

    def test_relative_root_resolves_against_the_config_directory(self, tmp_path: Path) -> None:
        """So `edith execute` behaves the same wherever it is invoked from."""
        config_dir = tmp_path / "cfg"
        config_dir.mkdir()
        config = EdithConfig(
            models=ModelsConfig(profiles={"default": ModelParams(model_name="m")}),
            orchestration=OrchestrationConfig(workspaces_root=Path("workspaces")),
            config_dir=config_dir,
        )
        assert WorkspaceManager(config).root == (tmp_path / "workspaces").resolve()


class TestVerificationProfileConfig:
    def test_profile_lookup_falls_back(self) -> None:
        settings = OrchestrationConfig(
            verification_profiles={"python": VerificationProfile(tests=("pytest",))}
        )
        assert settings.profile("python").tests == ("pytest",)
        assert settings.profile("nonexistent").tests == ()

    def test_shipped_config_defines_a_python_profile(self, repo_config_dir: Path) -> None:
        from edith.config.loader import load_config

        config = load_config(repo_config_dir)
        profile = config.orchestration.profile("python")
        assert profile.tests, "the shipped python profile must define a test command"
        assert "pytest" in " ".join(profile.tests)

    def test_shipped_workspaces_root_is_outside_the_kernel(self, repo_config_dir: Path) -> None:
        from edith.config.loader import load_config

        config = load_config(repo_config_dir)
        manager = WorkspaceManager(config)
        assert edith_repository_root() != manager.root
        assert not str(manager.root).startswith(str(edith_repository_root()))
