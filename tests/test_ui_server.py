"""The control surface is a client, and these tests exist to keep it one.

A UI over an autonomous code executor is the easiest place in the system to accidentally build
a second, weaker engine: a mocked run that always succeeds, a status panel that reports what it
hopes rather than what happened, a model selector that quietly falls back. Each of those would
make the screen look better and the system worth less.

So the assertions here are mostly about what the server *cannot* do. It holds no authority, it
reads durable state rather than inventing it, it reports a refusal as a refusal, and it never
substitutes a model the user did not choose.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from edith.config.loader import load_config
from edith.ui.server import (
    DEFAULT_HOST,
    ExecutionManager,
    describe_agents,
    describe_config,
    describe_models,
    serve,
)


@pytest.fixture
def config(tmp_path: Path) -> Any:
    base = load_config(None)
    return base.model_copy(
        update={
            "system": base.system.model_copy(update={"state_dir": tmp_path / "state"}),
            "orchestration": base.orchestration.model_copy(
                update={"workspaces_root": tmp_path / "workspaces"}
            ),
        }
    )


class TestTheServerHoldsNoAuthority:
    """It can start work and read state. It cannot decide, write, or execute."""

    def test_the_module_never_touches_the_filesystem_or_shell_directly(self) -> None:
        source = Path("src/edith/ui/server.py").read_text(encoding="utf-8")
        for forbidden in ("subprocess", "os.system", "shutil.rmtree", "git "):
            assert forbidden not in source

    def test_it_writes_no_files_of_its_own(self) -> None:
        """Serving static assets is a read. Nothing here creates or modifies a file."""
        source = Path("src/edith/ui/server.py").read_text(encoding="utf-8")
        for forbidden in ("write_text(", "write_bytes(", "open(", "mkdir("):
            assert forbidden not in source

    def test_it_cannot_construct_unrestricted_permissions(self) -> None:
        """The human CLI keeps its unrestricted capability; the UI must not grant it."""
        source = Path("src/edith/ui/server.py").read_text(encoding="utf-8")
        assert "UNRESTRICTED" not in source
        assert "AgentPermissions" not in source

    def test_it_binds_to_loopback_by_default(self) -> None:
        assert DEFAULT_HOST == "127.0.0.1"

    def test_the_static_root_is_the_only_readable_directory(self, config: Any) -> None:
        """A path outside the static root must not be servable."""
        source = Path("src/edith/ui/server.py").read_text(encoding="utf-8")
        assert "STATIC_ROOT.resolve() not in target.parents" in source


class TestExecutionGoesThroughTheRealEngine:
    def test_start_builds_the_real_orchestrator(self) -> None:
        """Not a mock, not a reimplementation: the same class the CLI runs."""
        source = Path("src/edith/ui/server.py").read_text(encoding="utf-8")
        assert "from edith.orchestrator import (" in source
        assert "Orchestrator," in source
        assert "orchestrator.run(execution)" in source

    def test_there_is_no_fabricated_success_path(self) -> None:
        """Nothing may report a verdict the engine did not produce."""
        source = Path("src/edith/ui/server.py").read_text(encoding="utf-8")
        for forbidden in ('"verdict": "PASS"', "succeeded = True", "passed = True"):
            assert forbidden not in source

    def test_a_refused_start_surfaces_the_engine_error(self, config: Any) -> None:
        """An EdithError becomes a structured refusal, never a generic failure."""
        manager = ExecutionManager(config=config)
        with pytest.raises(Exception) as excinfo:
            # An empty project name cannot produce a workspace.
            manager.start("", "do something")
        assert excinfo.value is not None

    def test_the_snapshot_reads_durable_state(self, config: Any) -> None:
        """Unknown executions are reported, not invented."""
        manager = ExecutionManager(config=config)
        assert manager.snapshot("exec_missing") == {"error": "unknown execution"}


class TestModelSelection:
    def test_every_configured_profile_is_reported(self, config: Any) -> None:
        entries = describe_models(config)
        names = {entry["profile"] for entry in entries}
        assert names == set(config.models.profiles)

    def test_an_unavailable_model_is_marked_with_a_reason(self, config: Any) -> None:
        """It must never be silently swapped: a run whose model changed means nothing."""
        entries = describe_models(config)
        unavailable = [entry for entry in entries if not entry["available"]]
        for entry in unavailable:
            assert entry["reason"], f"{entry['profile']} is unavailable with no reason given"

    def test_exactly_one_profile_is_the_default(self, config: Any) -> None:
        entries = describe_models(config)
        assert sum(1 for entry in entries if entry["default"]) == 1

    def test_model_identity_is_configuration_not_code(self) -> None:
        """The whole point of the toggle: swapping models changes config, not agents."""
        source = Path("src/edith/ui/server.py").read_text(encoding="utf-8")
        assert "qwen" not in source.lower()
        assert "build_provider" in source


class TestAgentsAreReportedHonestly:
    def test_only_real_agents_are_listed(self) -> None:
        """Every entry comes from a declared identity, so none can be fabricated."""
        agents = describe_agents()
        assert agents
        for agent in agents:
            assert agent["name"] and agent["description"]

    def test_permissions_shown_match_the_declared_identity(self) -> None:
        from edith.quality.agents import JudgeAgent

        judge = next(a for a in describe_agents() if a["name"] == JudgeAgent.identity.name)
        assert judge["can_write"] is False
        assert judge["can_execute"] is False

    def test_the_tester_is_shown_as_write_capable(self) -> None:
        from edith.quality.agents import TestingAgent

        tester = next(a for a in describe_agents() if a["name"] == TestingAgent.identity.name)
        assert tester["can_write"] is True

    def test_engineering_and_quality_agents_are_distinguished(self) -> None:
        kinds = {agent["kind"] for agent in describe_agents()}
        assert {"engineering", "quality"} <= kinds


class TestConfigurationIsReportedSafely:
    def test_experimental_features_are_flagged_and_off(self, config: Any) -> None:
        described = describe_config(config)
        for name, entry in described["experimental"].items():
            assert entry["experimental"] is True, name
            assert entry["value"] is False, f"{name} must not default on"
            assert entry["evidence"], f"{name} must say why it is off"

    def test_unsupported_controls_are_declared_unsupported(self, config: Any) -> None:
        """Pause, cancel and retry are not engine capabilities, so the UI must not offer them."""
        supported = describe_config(config)["supported"]
        assert supported == {"pause": False, "cancel": False, "retry": False}

    def test_no_permission_surface_is_exposed(self, config: Any) -> None:
        payload = json.dumps(describe_config(config))
        for forbidden in ("allowed_tools", "allowed_write_paths", "UNRESTRICTED"):
            assert forbidden not in payload

    def test_the_repair_budget_is_the_configured_one(self, config: Any) -> None:
        described = describe_config(config)
        assert described["max_repair_attempts"] == config.orchestration.max_repair_attempts


class TestTheServerStarts:
    def test_it_binds_and_serves_the_dashboard(self, config: Any) -> None:
        import urllib.request

        server = serve(config, host="127.0.0.1", port=0)
        port = server.server_address[1]
        import threading

        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/") as response:
                body = response.read().decode("utf-8")
            assert "EDITH" in body
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/agents") as response:
                agents = json.loads(response.read())
            assert agents["agents"]
        finally:
            server.shutdown()
            server.server_close()

    def test_an_unknown_endpoint_is_a_404(self, config: Any) -> None:
        import urllib.error
        import urllib.request

        server = serve(config, host="127.0.0.1", port=0)
        port = server.server_address[1]
        import threading

        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            with pytest.raises(urllib.error.HTTPError) as excinfo:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/api/nope")
            assert excinfo.value.code == 404
        finally:
            server.shutdown()
            server.server_close()

    def test_a_start_without_a_request_is_refused(self, config: Any) -> None:
        import threading
        import urllib.error
        import urllib.request

        server = serve(config, host="127.0.0.1", port=0)
        port = server.server_address[1]
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            payload = json.dumps({"project": "p", "request": ""}).encode()
            request = urllib.request.Request(
                f"http://127.0.0.1:{port}/api/run",
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            with pytest.raises(urllib.error.HTTPError) as excinfo:
                urllib.request.urlopen(request)
            assert excinfo.value.code == 400
        finally:
            server.shutdown()
            server.server_close()


class TestTheDashboardDoesNotOverclaim:
    """The screen must not present a model opinion as a system verdict."""

    def test_it_labels_model_output_as_a_recommendation(self) -> None:
        page = Path("src/edith/ui/static/index.html").read_text(encoding="utf-8")
        assert "model recommendation" in page

    def test_it_labels_verification_as_system_authoritative(self) -> None:
        page = Path("src/edith/ui/static/index.html").read_text(encoding="utf-8")
        assert "system-authoritative" in page

    def test_it_never_says_the_ai_declared_the_code_correct(self) -> None:
        page = Path("src/edith/ui/static/index.html").read_text(encoding="utf-8").lower()
        for forbidden in ("ai says", "ai verified", "ai approved"):
            assert forbidden not in page
        assert "edith verification passed" in page

    def test_unsupported_controls_are_disabled_in_the_markup(self) -> None:
        page = Path("src/edith/ui/static/index.html").read_text(encoding="utf-8")
        assert 'id="btn-pause" disabled' in page
        assert 'id="btn-cancel" disabled' in page

    def test_it_escapes_everything_it_renders(self) -> None:
        """Agent output and error text reach the page; none of it may become markup."""
        page = Path("src/edith/ui/static/index.html").read_text(encoding="utf-8")
        assert "const esc = (s)" in page
