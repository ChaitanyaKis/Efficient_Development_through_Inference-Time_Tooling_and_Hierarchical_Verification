"""CLI behaviour and exit codes.

Exit codes matter: these commands are meant to be usable as gates in scripts and CI.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from edith import __version__
from edith.cli.main import EXIT_CONFIG_ERROR, EXIT_FAILURE, EXIT_OK, app
from edith.config.schema import EdithConfig, ModelParams
from edith.models.registry import register_provider
from edith.schemas.model import HealthState, ProviderHealth

from .fakes import FakeProvider

runner = CliRunner()

ECHO_JSON = json.dumps({"summary": "Edith is local.", "keywords": ["local"], "confidence": 0.8})


@pytest.fixture
def cli_config_dir(config_dir: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point models.yaml at an in-process fake provider so the CLI never needs Ollama."""
    register_provider(
        "cli_fake",
        lambda cfg, params: FakeProvider(params, [ECHO_JSON]),
        replace=True,
    )
    text = (config_dir / "models.yaml").read_text(encoding="utf-8")
    (config_dir / "models.yaml").write_text(
        text.replace("provider: ollama", "provider: cli_fake"), encoding="utf-8"
    )
    return config_dir


class TestVersion:
    def test_prints_version(self) -> None:
        result = runner.invoke(app, ["version"])
        assert result.exit_code == EXIT_OK
        assert __version__ in result.stdout


class TestHelp:
    def test_no_args_shows_help(self) -> None:
        assert "Usage" in runner.invoke(app, []).output

    def test_future_milestone_commands_are_absent(self) -> None:
        """A command that only prints 'not implemented' is worse than no command.

        ``memory`` and ``benchmark`` were added once the milestones behind them existed;
        ``project`` and ``task`` remain absent because nothing yet stands behind them.
        """
        for absent in ("project", "task"):
            assert runner.invoke(app, [absent]).exit_code != EXIT_OK


class TestConfigCommand:
    def test_human_output(self, config_dir: Path) -> None:
        result = runner.invoke(app, ["config", "--config-dir", str(config_dir)])
        assert result.exit_code == EXIT_OK
        assert "default_profile" in result.stdout
        assert "test-model:q4" in result.stdout

    def test_json_output_is_parseable(self, config_dir: Path) -> None:
        result = runner.invoke(app, ["config", "--config-dir", str(config_dir), "--json"])
        payload = json.loads(result.stdout)
        assert payload["models"]["default_profile"] == "default"

    def test_missing_config_dir_exits_two(self, tmp_path: Path) -> None:
        result = runner.invoke(app, ["config", "--config-dir", str(tmp_path / "nope")])
        assert result.exit_code == EXIT_CONFIG_ERROR

    def test_invalid_config_exits_two(self, config_dir: Path) -> None:
        (config_dir / "models.yaml").write_text("profiles: []", encoding="utf-8")
        result = runner.invoke(app, ["config", "--config-dir", str(config_dir)])
        assert result.exit_code == EXIT_CONFIG_ERROR


class TestAgentsCommand:
    def test_lists_the_echo_agent(self, cli_config_dir: Path) -> None:
        result = runner.invoke(app, ["agents", "--config-dir", str(cli_config_dir)])
        assert result.exit_code == EXIT_OK
        assert "echo" in result.stdout

    def test_shows_permission_scope(self, cli_config_dir: Path) -> None:
        result = runner.invoke(app, ["agents", "--config-dir", str(cli_config_dir)])
        assert "read-only" in result.stdout

    def test_json_output(self, cli_config_dir: Path) -> None:
        result = runner.invoke(app, ["agents", "--config-dir", str(cli_config_dir), "--json"])
        payload = json.loads(result.stdout)
        assert {entry["name"] for entry in payload} >= {"echo", "coder", "critic"}


class TestDoctorCommand:
    def test_offline_run_produces_a_report(self, cli_config_dir: Path) -> None:
        result = runner.invoke(
            app, ["doctor", "--config-dir", str(cli_config_dir), "--offline"]
        )
        assert "python" in result.stdout and "Result:" in result.stdout

    def test_json_output_is_parseable(self, cli_config_dir: Path) -> None:
        result = runner.invoke(
            app, ["doctor", "--config-dir", str(cli_config_dir), "--offline", "--json"]
        )
        payload = json.loads(result.stdout)
        assert isinstance(payload["checks"], list) and "resources" in payload

    def test_remediation_is_shown_for_problems(self, cli_config_dir: Path) -> None:
        result = runner.invoke(
            app,
            ["doctor", "--config-dir", str(cli_config_dir), "--offline", "--profile", "ghost"],
        )
        assert result.exit_code == EXIT_FAILURE
        assert "->" in result.stdout  # remediation arrow

    def test_unhealthy_provider_gives_nonzero_exit(self, config_dir: Path) -> None:
        register_provider(
            "cli_down",
            lambda cfg, params: FakeProvider(
                params,
                health=ProviderHealth(
                    provider="fake",
                    state=HealthState.UNAVAILABLE,
                    detail="not running",
                    remediation="start ollama",
                ),
            ),
            replace=True,
        )
        text = (config_dir / "models.yaml").read_text(encoding="utf-8")
        (config_dir / "models.yaml").write_text(
            text.replace("provider: ollama", "provider: cli_down"), encoding="utf-8"
        )
        result = runner.invoke(app, ["doctor", "--config-dir", str(config_dir)])
        assert result.exit_code == EXIT_FAILURE
        assert "start ollama" in result.stdout


class TestRunCommand:
    def test_runs_the_echo_agent(self, cli_config_dir: Path) -> None:
        result = runner.invoke(
            app,
            [
                "run", "echo",
                "--config-dir", str(cli_config_dir),
                "--payload", '{"statement": "Edith is local."}',
            ],
        )
        assert result.exit_code == EXIT_OK
        payload = json.loads(result.stdout)
        assert payload["status"] == "SUCCESS"
        assert payload["output"]["summary"] == "Edith is local."

    def test_payload_from_file(self, cli_config_dir: Path, tmp_path: Path) -> None:
        payload_file = tmp_path / "payload.json"
        payload_file.write_text('{"statement": "from a file"}', encoding="utf-8")
        result = runner.invoke(
            app,
            [
                "run", "echo",
                "--config-dir", str(cli_config_dir),
                "--payload-file", str(payload_file),
            ],
        )
        assert result.exit_code == EXIT_OK

    def test_both_payload_sources_rejected(self, cli_config_dir: Path, tmp_path: Path) -> None:
        payload_file = tmp_path / "p.json"
        payload_file.write_text("{}", encoding="utf-8")
        result = runner.invoke(
            app,
            [
                "run", "echo",
                "--config-dir", str(cli_config_dir),
                "--payload", "{}",
                "--payload-file", str(payload_file),
            ],
        )
        assert result.exit_code == EXIT_CONFIG_ERROR

    def test_unreadable_payload_file(self, cli_config_dir: Path, tmp_path: Path) -> None:
        result = runner.invoke(
            app,
            [
                "run", "echo",
                "--config-dir", str(cli_config_dir),
                "--payload-file", str(tmp_path / "absent.json"),
            ],
        )
        assert result.exit_code == EXIT_CONFIG_ERROR

    def test_malformed_json_payload(self, cli_config_dir: Path) -> None:
        result = runner.invoke(
            app,
            ["run", "echo", "--config-dir", str(cli_config_dir), "--payload", "{not json}"],
        )
        assert result.exit_code == EXIT_CONFIG_ERROR

    def test_non_object_payload_rejected(self, cli_config_dir: Path) -> None:
        result = runner.invoke(
            app, ["run", "echo", "--config-dir", str(cli_config_dir), "--payload", "[1,2]"]
        )
        assert result.exit_code == EXIT_CONFIG_ERROR

    def test_unknown_agent(self, cli_config_dir: Path) -> None:
        result = runner.invoke(app, ["run", "ghost", "--config-dir", str(cli_config_dir)])
        assert result.exit_code == EXIT_CONFIG_ERROR

    def test_agent_failure_gives_nonzero_exit(self, cli_config_dir: Path) -> None:
        """An invalid payload must not exit 0 - scripts depend on this."""
        result = runner.invoke(
            app,
            ["run", "echo", "--config-dir", str(cli_config_dir), "--payload", '{"statement": ""}'],
        )
        assert result.exit_code == EXIT_FAILURE
        assert json.loads(result.stdout)["status"] == "REJECTED"


class TestSelftestCommand:
    def test_passes_with_a_healthy_provider(self, cli_config_dir: Path) -> None:
        result = runner.invoke(app, ["selftest", "--config-dir", str(cli_config_dir)])
        assert result.exit_code == EXIT_OK
        assert "M0 SELF-TEST PASS" in result.stdout

    def test_custom_statement(self, cli_config_dir: Path) -> None:
        result = runner.invoke(
            app,
            ["selftest", "--config-dir", str(cli_config_dir), "--statement", "custom text"],
        )
        assert result.exit_code == EXIT_OK

    def test_aborts_when_the_provider_is_unhealthy(self, config_dir: Path) -> None:
        register_provider(
            "selftest_down",
            lambda cfg, params: FakeProvider(
                params,
                health=ProviderHealth(
                    provider="fake", state=HealthState.UNAVAILABLE, detail="down"
                ),
            ),
            replace=True,
        )
        text = (config_dir / "models.yaml").read_text(encoding="utf-8")
        (config_dir / "models.yaml").write_text(
            text.replace("provider: ollama", "provider: selftest_down"), encoding="utf-8"
        )
        result = runner.invoke(app, ["selftest", "--config-dir", str(config_dir)])
        assert result.exit_code == EXIT_FAILURE
        assert "M0 SELF-TEST PASS" not in result.stdout

    def test_fails_when_the_model_returns_junk(self, config_dir: Path) -> None:
        """A model that cannot produce valid JSON must fail the gate, loudly."""
        register_provider(
            "selftest_junk",
            lambda cfg, params: FakeProvider(params, ["I'm sorry, I can't do that."]),
            replace=True,
        )
        text = (config_dir / "models.yaml").read_text(encoding="utf-8")
        (config_dir / "models.yaml").write_text(
            text.replace("provider: ollama", "provider: selftest_junk"), encoding="utf-8"
        )
        result = runner.invoke(app, ["selftest", "--config-dir", str(config_dir)])
        assert result.exit_code == EXIT_FAILURE
        assert "M0 SELF-TEST FAIL" in result.output


class TestEnvironmentOverridesReachTheCli:
    def test_env_var_changes_resolved_config(
        self, config_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("EDITH__SYSTEM__PROJECT_NAME", "overridden")
        result = runner.invoke(app, ["config", "--config-dir", str(config_dir)])
        assert "overridden" in result.stdout


class TestToolsCommand:
    def test_lists_every_m1_tool(self, cli_config_dir: Path) -> None:
        result = runner.invoke(app, ["tools", "--config-dir", str(cli_config_dir)])
        assert result.exit_code == EXIT_OK
        for tool in ("filesystem.read", "shell.run", "git.commit", "git.worktree"):
            assert tool in result.stdout

    def test_shows_risk_surface(self, cli_config_dir: Path) -> None:
        result = runner.invoke(app, ["tools", "--config-dir", str(cli_config_dir)])
        assert "read+write" in result.stdout

    def test_json_output(self, cli_config_dir: Path) -> None:
        result = runner.invoke(app, ["tools", "--config-dir", str(cli_config_dir), "--json"])
        payload = json.loads(result.stdout)
        assert {entry["name"] for entry in payload} >= {"filesystem.read", "shell.run"}


class TestToolCommand:
    def test_runs_a_tool(self, cli_config_dir: Path, tmp_path: Path) -> None:
        target = tmp_path / "ws"
        target.mkdir()
        (target / "note.txt").write_text("hello from disk\n", encoding="utf-8")
        result = runner.invoke(
            app,
            [
                "tool", "filesystem.read",
                "--config-dir", str(cli_config_dir),
                "--workspace", str(target),
                "--args", '{"path": "note.txt"}',
            ],
        )
        assert result.exit_code == EXIT_OK
        assert "hello from disk" in json.loads(result.stdout)["output"]["content"]

    def test_policy_still_applies_to_the_operator(
        self, cli_config_dir: Path, tmp_path: Path
    ) -> None:
        """Unrestricted permissions do not disable the path policy."""
        target = tmp_path / "ws"
        target.mkdir()
        (target / ".env").write_text("SECRET=x\n", encoding="utf-8")
        result = runner.invoke(
            app,
            [
                "tool", "filesystem.read",
                "--config-dir", str(cli_config_dir),
                "--workspace", str(target),
                "--args", '{"path": ".env"}',
            ],
        )
        assert result.exit_code == EXIT_FAILURE
        assert json.loads(result.stdout)["failure_category"] == "SECURITY_FAILURE"

    def test_failure_exits_nonzero(self, cli_config_dir: Path, tmp_path: Path) -> None:
        target = tmp_path / "ws"
        target.mkdir()
        result = runner.invoke(
            app,
            [
                "tool", "filesystem.read",
                "--config-dir", str(cli_config_dir),
                "--workspace", str(target),
                "--args", '{"path": "absent.txt"}',
            ],
        )
        assert result.exit_code == EXIT_FAILURE

    def test_malformed_args_rejected(self, cli_config_dir: Path, tmp_path: Path) -> None:
        target = tmp_path / "ws"
        target.mkdir()
        result = runner.invoke(
            app,
            [
                "tool", "filesystem.read",
                "--config-dir", str(cli_config_dir),
                "--workspace", str(target),
                "--args", "{not json}",
            ],
        )
        assert result.exit_code == EXIT_CONFIG_ERROR

    def test_missing_workspace_reported(self, cli_config_dir: Path, tmp_path: Path) -> None:
        result = runner.invoke(
            app,
            [
                "tool", "filesystem.read",
                "--config-dir", str(cli_config_dir),
                "--workspace", str(tmp_path / "absent"),
                "--args", '{"path": "x"}',
            ],
        )
        assert result.exit_code == EXIT_CONFIG_ERROR


class TestEnvironmentCommand:
    """``edith environment`` inspects and generates. It never installs."""

    def project(self, root: Path) -> Path:
        root.mkdir(parents=True, exist_ok=True)
        (root / "pyproject.toml").write_text(
            '[project]\nname = "demo"\ndependencies = ["requests>=2.31"]\n', encoding="utf-8"
        )
        (root / "app.py").write_text("import requests\n", encoding="utf-8")
        return root

    def test_inspection_reports_dependencies_as_json(
        self, cli_config_dir: Path, tmp_path: Path
    ) -> None:
        project = self.project(tmp_path / "demo")
        result = runner.invoke(
            app,
            [
                "environment",
                "--config-dir", str(cli_config_dir),
                "--project", str(project),
                "--json",
            ],
        )
        payload = json.loads(result.stdout)
        assert [d["name"] for d in payload["spec"]["dependencies"]] == ["requests"]
        assert payload["written"] == []

    def test_write_generates_artifacts_and_installs_nothing(
        self, cli_config_dir: Path, tmp_path: Path
    ) -> None:
        project = self.project(tmp_path / "demo")
        result = runner.invoke(
            app,
            [
                "environment",
                "--config-dir", str(cli_config_dir),
                "--project", str(project),
                "--write",
                "--json",
            ],
        )
        assert result.exit_code == EXIT_OK
        payload = json.loads(result.stdout)
        assert sorted(payload["written"]) == [
            "requirements.txt",
            "scripts/install.bat",
            "scripts/install.ps1",
            "scripts/install.sh",
        ]
        assert not payload["denied"]

        # Generated, not executed: no environment was created by inspecting the project.
        assert not (project / ".venv").exists()
        script = (project / "scripts" / "install.bat").read_text(encoding="utf-8")
        assert "venv" in script
        assert "--user" not in script


class TestNoCloudDependencies:
    def test_config_endpoint_is_loopback(self, repo_config_dir: Path) -> None:
        """Engineering invariant 1 and 2: no API costs, no hidden cloud dependency."""
        from edith.config.loader import load_config

        cfg: EdithConfig = load_config(repo_config_dir)
        assert cfg.models.ollama.host.startswith(("http://127.0.0.1", "http://localhost"))
        assert cfg.models.ollama.allow_remote is False

    def test_no_cloud_sdk_imported(self) -> None:
        import sys

        import edith.agents
        import edith.cli.main
        import edith.models  # noqa: F401

        forbidden = {"openai", "anthropic", "google.generativeai", "cohere", "mistralai"}
        assert not (forbidden & set(sys.modules))

    def test_model_names_are_not_hard_coded_in_source(self) -> None:
        """Model identity belongs in config; a literal in source defeats the abstraction."""
        source_root = Path(__file__).resolve().parents[1] / "src" / "edith"
        offenders = [
            path.name
            for path in source_root.rglob("*.py")
            if "qwen" in path.read_text(encoding="utf-8").lower()
        ]
        assert offenders == []


class TestModelParamsFixtureSanity:
    def test_fixture_matches_schema(self, model_params: ModelParams) -> None:
        assert model_params.model_name == "test-model:q4"
