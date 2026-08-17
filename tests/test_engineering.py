"""M5: engineering roles, scopes, assignment, and conflict detection.

The load-bearing claim of M5 is not that five agents can generate code — it is that they
cannot generate it outside their remit. These tests assert the boundary structurally: role
scopes, task-narrowed permissions, forbidden areas, and the collisions that occur when two
tasks claim the same files.

Everything here is offline and model-free.
"""

from __future__ import annotations

from pathlib import Path

from edith.engineering.agents import (
    ENGINEERING_AGENTS,
    EngineeringInput,
    agent_for,
    permissions_for,
)
from edith.engineering.conflicts import (
    ConflictKind,
    detect_conflicts,
    serialise,
)
from edith.engineering.dependency import (
    declare_undeclared,
    reconcile,
    verify_imports,
)
from edith.engineering.ownership import (
    ROLE_SCOPES,
    Assignment,
    EngineeringRole,
    assign,
    assign_plan,
    narrow_scope,
    resolve_role,
    scope_for,
)
from edith.environment.schema import (
    Dependency,
    DependencyOrigin,
    DependencyStatus,
    EnvironmentReport,
    EnvironmentSpec,
)
from edith.product.architecture import (
    Complexity,
    ImplementationPlanDocument,
    PlannedTask,
)


def task(
    task_id: str,
    agent: str = "backend",
    *,
    paths: tuple[str, ...] = (),
    depends_on: tuple[str, ...] = (),
    implements: tuple[str, ...] = (),
) -> PlannedTask:
    return PlannedTask(
        task_id=task_id,
        title=f"Task {task_id}",
        description="Do the thing.",
        agent=agent,
        paths=paths,
        depends_on=depends_on,
        implements=implements,
        complexity=Complexity.SMALL,
    )


def assignment(
    task_id: str, role: EngineeringRole, *paths: str, depends_on: tuple[str, ...] = ()
) -> Assignment:
    return Assignment(
        task=task(task_id, role.value, paths=paths, depends_on=depends_on),
        role=role,
        write_paths=paths or scope_for(role).write,
    )


class TestRoleScopes:
    """M5 item 13: boundaries are declared, not assumed."""

    def test_every_role_has_a_scope(self) -> None:
        assert set(ROLE_SCOPES) == set(EngineeringRole)

    def test_no_engineering_agent_may_run_a_shell(self) -> None:
        """Generating code and deciding it works stay in different hands."""
        for role in EngineeringRole:
            tools = scope_for(role).tools
            assert "shell.run" not in tools
            assert not any(name.startswith("git.") for name in tools)

    def test_no_engineering_agent_has_network_access(self) -> None:
        for role in EngineeringRole:
            assert not permissions_for(role).network_access

    def test_frontend_cannot_write_migrations(self) -> None:
        """A frontend agent that can change the schema can break every other agent's work."""
        scope = scope_for(EngineeringRole.FRONTEND)
        assert "migrations/**" in scope.forbidden
        assert not any(pattern.startswith("migrations") for pattern in scope.write)

    def test_database_cannot_write_business_logic(self) -> None:
        scope = scope_for(EngineeringRole.DATABASE)
        assert "src/backend/**" in scope.forbidden

    def test_devops_cannot_write_source_or_tests(self) -> None:
        scope = scope_for(EngineeringRole.DEVOPS)
        assert "src/**" in scope.forbidden
        assert "tests/**" in scope.forbidden

    def test_dependency_cannot_write_application_code(self) -> None:
        scope = scope_for(EngineeringRole.DEPENDENCY)
        assert "src/**" in scope.forbidden
        assert all(
            not pattern.startswith("src/") for pattern in scope.write
        )

    def test_scopes_are_repo_relative(self) -> None:
        """AgentPermissions rejects absolute or traversing patterns; this proves we comply."""
        for role in EngineeringRole:
            permissions = permissions_for(role)
            for pattern in permissions.allowed_write_paths:
                assert not pattern.startswith(("/", "\\"))
                assert ".." not in pattern.split("/")


class TestRoleResolution:
    def test_known_names_and_aliases_resolve(self) -> None:
        assert resolve_role("frontend") is EngineeringRole.FRONTEND
        assert resolve_role("backend_agent") is EngineeringRole.BACKEND
        assert resolve_role("db") is EngineeringRole.DATABASE
        assert resolve_role("infra") is EngineeringRole.DEVOPS
        assert resolve_role("deps") is EngineeringRole.DEPENDENCY

    def test_an_unknown_agent_is_refused_not_guessed(self) -> None:
        """A plan must not invent an agent with convenient permissions."""
        assert resolve_role("security") is None
        assert resolve_role("") is None


class TestAssignment:
    def test_a_task_is_bound_to_its_role(self) -> None:
        result = assign(task("TASK-001", "backend", paths=("src/backend/api.py",)))
        assert result.assigned
        assert result.role is EngineeringRole.BACKEND

    def test_scope_narrows_to_the_task_not_the_role(self) -> None:
        """The role bounds what an agent may ever touch; the task bounds what it may now."""
        result = assign(task("TASK-001", "backend", paths=("src/backend/api.py",)))
        assert "src/backend/api.py" in result.write_paths
        assert "src/backend/**" in result.write_paths
        assert "tests/backend/**" not in result.write_paths

    def test_a_task_naming_nothing_gets_the_role_ceiling(self) -> None:
        """It still has to be able to create the files it was asked for."""
        result = assign(task("TASK-001", "backend"))
        assert result.write_paths == scope_for(EngineeringRole.BACKEND).write

    def test_an_unknown_agent_produces_an_unassigned_task(self) -> None:
        result = assign(task("TASK-001", "security"))
        assert not result.assigned
        assert "not one of" in result.rejection

    def test_a_task_reaching_outside_its_role_is_refused(self) -> None:
        """M5 item 13, enforced before anything runs."""
        result = assign(
            task("TASK-001", "frontend", paths=("migrations/0001_init.sql",))
        )
        assert not result.assigned
        assert "outside that role's remit" in result.rejection

    def test_paths_outside_the_role_are_dropped_from_the_scope(self) -> None:
        granted = narrow_scope(
            EngineeringRole.BACKEND, ("src/backend/api.py", "src/frontend/app.js")
        )
        assert "src/backend/api.py" in granted
        assert "src/frontend/app.js" not in granted

    def test_traversal_and_absolute_paths_never_enter_a_scope(self) -> None:
        granted = narrow_scope(
            EngineeringRole.BACKEND, ("../../etc/passwd", "/etc/shadow")
        )
        assert granted == scope_for(EngineeringRole.BACKEND).write

    def test_unassignable_tasks_are_reported_not_dropped(self) -> None:
        plan = ImplementationPlanDocument(
            product_name="X",
            goal="g",
            tasks=(task("TASK-001", "backend"), task("TASK-002", "security")),
        )
        assignments = assign_plan(plan)
        assert len(assignments) == 2
        assert sum(1 for item in assignments if item.assigned) == 1

    def test_task_permissions_use_the_narrowed_scope(self) -> None:
        result = assign(task("TASK-001", "backend", paths=("src/backend/api.py",)))
        permissions = result.permissions()
        assert "src/backend/api.py" in permissions.allowed_write_paths
        assert "shell.run" not in permissions.allowed_tools


class TestConflictDetection:
    """M5 item 14: two agents must never silently overwrite each other."""

    def test_two_tasks_naming_the_same_file_conflict(self) -> None:
        conflicts = detect_conflicts(
            (
                assignment("TASK-001", EngineeringRole.BACKEND, "src/backend/api.py"),
                assignment("TASK-002", EngineeringRole.BACKEND, "src/backend/api.py"),
            )
        )
        assert conflicts
        assert conflicts[0].kind is ConflictKind.SAME_FILE
        assert conflicts[0].code == "TASK_CONFLICT"
        assert conflicts[0].scopes == ("src/backend/api.py",)

    def test_a_conflict_names_the_tasks_files_and_agents(self) -> None:
        conflicts = detect_conflicts(
            (
                assignment("TASK-001", EngineeringRole.BACKEND, "src/backend/api.py"),
                assignment("TASK-002", EngineeringRole.BACKEND, "src/backend/api.py"),
            )
        )
        conflict = conflicts[0]
        assert conflict.task_ids == ("TASK-001", "TASK-002")
        assert conflict.agents == (EngineeringRole.BACKEND, EngineeringRole.BACKEND)
        assert "TASK-001" in conflict.render()

    def test_a_dependency_between_the_tasks_removes_the_conflict(self) -> None:
        """The DAG already serialises them; the outcome does not depend on scheduling."""
        conflicts = detect_conflicts(
            (
                assignment("TASK-001", EngineeringRole.BACKEND, "src/backend/api.py"),
                assignment(
                    "TASK-002",
                    EngineeringRole.BACKEND,
                    "src/backend/api.py",
                    depends_on=("TASK-001",),
                ),
            )
        )
        assert conflicts == ()

    def test_disjoint_scopes_do_not_conflict(self) -> None:
        conflicts = detect_conflicts(
            (
                assignment("TASK-001", EngineeringRole.BACKEND, "src/backend/api.py"),
                assignment("TASK-002", EngineeringRole.FRONTEND, "src/frontend/app.js"),
            )
        )
        assert conflicts == ()

    def test_a_cross_role_overlap_is_flagged_as_such(self) -> None:
        """Two roles claiming one area means the architecture drew a boundary wrong.

        Different files, overlapping scopes: the collision is the shared territory, not a
        shared filename, which is the case a per-file check alone would miss.
        """
        left = Assignment(
            task=task("TASK-001", "backend", paths=("src/backend/api.py",)),
            role=EngineeringRole.BACKEND,
            write_paths=("src/backend/**",),
        )
        right = Assignment(
            task=task("TASK-002", "database", paths=("src/backend/models.py",)),
            role=EngineeringRole.DATABASE,
            write_paths=("src/backend/**",),
        )
        conflicts = detect_conflicts((left, right))
        assert conflicts
        assert conflicts[0].kind is ConflictKind.ROLE_OVERLAP
        assert conflicts[0].cross_role

    def test_a_nested_scope_overlaps_its_parent(self) -> None:
        conflicts = detect_conflicts(
            (
                assignment("TASK-001", EngineeringRole.BACKEND, "src/backend/**"),
                assignment("TASK-002", EngineeringRole.BACKEND, "src/backend/api/**"),
            )
        )
        assert conflicts

    def test_unassigned_tasks_cannot_conflict(self) -> None:
        unassigned = Assignment(
            task=task("TASK-002", "security"),
            role=EngineeringRole.BACKEND,
            write_paths=(),
            rejection="unknown agent",
        )
        conflicts = detect_conflicts(
            (assignment("TASK-001", EngineeringRole.BACKEND, "src/backend/api.py"), unassigned)
        )
        assert conflicts == ()

    def test_serialisation_is_deterministic(self) -> None:
        """Two runs of the same plan must produce the same order."""
        assignments = (
            assignment("TASK-003", EngineeringRole.BACKEND, "src/backend/a.py"),
            assignment("TASK-001", EngineeringRole.BACKEND, "src/backend/a.py"),
            assignment("TASK-002", EngineeringRole.FRONTEND, "src/frontend/b.js"),
        )
        conflicts = detect_conflicts(assignments)
        first = [item.task.task_id for item in serialise(assignments, conflicts)]
        second = [item.task.task_id for item in serialise(assignments, conflicts)]
        assert first == second
        assert set(first) == {"TASK-001", "TASK-002", "TASK-003"}


class TestEngineeringAgents:
    def test_every_role_has_an_agent(self) -> None:
        assert set(ENGINEERING_AGENTS) == set(EngineeringRole)

    def test_agents_reuse_the_m2_apply_pipeline(self) -> None:
        """A specialised agent is a different prompt, not a different path to disk.

        Subclassing the coder means the sanitiser, syntax gate, symbol-preservation check
        and gateway write are inherited rather than reimplemented five times.
        """
        from edith.agents.coder import CoderOutput, CodingAgent

        for role in EngineeringRole:
            agent = agent_for(role)
            assert issubclass(agent, CodingAgent)
            assert agent.output_schema is CoderOutput
            # The apply pipeline itself must not be overridden.
            assert agent._run is CodingAgent._run

    def test_agent_permissions_come_from_the_role_scope(self) -> None:
        for role in EngineeringRole:
            identity = agent_for(role).identity
            assert identity.permissions == scope_for(role).permissions()

    def test_agent_names_are_distinct_and_registrable(self) -> None:
        names = {agent_for(role).identity.name for role in EngineeringRole}
        assert len(names) == len(EngineeringRole)
        for name in names:
            assert name.islower()

    def test_the_input_contract_is_scoped_context_not_the_whole_project(self) -> None:
        """M5 item 15: an agent receives its task's material, not the repository."""
        fields = set(EngineeringInput.model_fields)
        assert {"title", "requirements", "architecture", "context", "scope"} <= fields

    def test_the_input_extends_the_coder_contract(self) -> None:
        """Which is what lets these agents inherit the M2 apply pipeline unchanged."""
        from edith.agents.coder import CoderInput

        assert issubclass(EngineeringInput, CoderInput)

    def test_prior_knowledge_is_empty_by_default(self) -> None:
        """Memory stays off unless the M3.2 governor grants it."""
        payload = EngineeringInput(title="t", description="d")
        assert payload.prior_knowledge == ""


class TestDependencyWorkflow:
    """M5 items 6 and 7, over the M3.1 foundation."""

    def write(self, root: Path, relative: str, content: str) -> None:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    def test_an_undeclared_import_is_detected(self, tmp_path: Path) -> None:
        self.write(tmp_path, "pyproject.toml", '[project]\nname = "d"\ndependencies = []\n')
        self.write(tmp_path, "app.py", "import requests\n")

        result = reconcile(tmp_path)
        assert not result.reconciled
        assert result.blocking
        assert [item.name for item in result.undeclared] == ["requests"]

    def test_a_reconciled_project_reports_clean(self, tmp_path: Path) -> None:
        self.write(tmp_path, "pyproject.toml", '[project]\nname = "d"\ndependencies = []\n')
        self.write(tmp_path, "app.py", "import os\nimport json\n")

        result = reconcile(tmp_path)
        assert result.reconciled
        assert not result.blocking

    def test_promotion_declares_only_discovered_imports(self) -> None:
        """Never a package a model suggested -- only what the source actually imports."""
        spec = EnvironmentSpec(
            dependencies=[
                Dependency(
                    name="requests",
                    origin=DependencyOrigin.SOURCE_IMPORT,
                    status=DependencyStatus.UNDECLARED,
                ),
                Dependency(
                    name="guessed",
                    origin=DependencyOrigin.MODEL_SUGGESTION,
                    status=DependencyStatus.UNDECLARED,
                ),
            ]
        )
        promoted = declare_undeclared(spec)
        by_name = {item.name: item for item in promoted.dependencies}
        assert by_name["requests"].origin is DependencyOrigin.MANIFEST
        assert by_name["guessed"].origin is DependencyOrigin.MODEL_SUGGESTION

    def test_import_verification_reports_what_is_not_importable(self) -> None:
        report = EnvironmentReport(
            spec=EnvironmentSpec(
                dependencies=[
                    Dependency(name="requests", status=DependencyStatus.MISSING),
                ]
            )
        )
        ok, detail = verify_imports(report)
        assert not ok
        assert "requests" in detail

    def test_import_verification_passes_when_everything_resolves(self) -> None:
        report = EnvironmentReport(
            spec=EnvironmentSpec(
                dependencies=[
                    Dependency(name="requests", status=DependencyStatus.SATISFIED),
                ]
            )
        )
        ok, _ = verify_imports(report)
        assert ok

    def test_provisioning_goes_through_the_gateway(self, tmp_path: Path) -> None:
        """Install artifacts are executable text; they land through the policy layer."""
        from edith.engineering.dependency import provision_environment

        from .tool_fixtures import build_gateway

        self.write(
            tmp_path, "pyproject.toml", '[project]\nname = "d"\ndependencies = []\n'
        )
        self.write(tmp_path, "app.py", "import os\n")

        gateway = build_gateway(
            tmp_path, permissions_for(EngineeringRole.DEPENDENCY)
        )
        outcome = provision_environment(gateway, tmp_path)

        assert outcome.ok, f"denied={outcome.denied} error={outcome.error}"
        assert (tmp_path / "requirements.txt").is_file()
        assert (tmp_path / "scripts" / "install.bat").is_file()
        assert (tmp_path / "scripts" / "install.ps1").is_file()

    def test_a_write_outside_the_dependency_scope_is_denied(self, tmp_path: Path) -> None:
        from edith.engineering.dependency import provision_environment
        from edith.schemas.agent import AgentPermissions

        from .tool_fixtures import build_gateway

        self.write(
            tmp_path, "pyproject.toml", '[project]\nname = "d"\ndependencies = []\n'
        )
        gateway = build_gateway(
            tmp_path,
            AgentPermissions(
                allowed_tools=frozenset({"filesystem.write"}),
                allowed_write_paths=("requirements.txt",),
            ),
        )
        outcome = provision_environment(gateway, tmp_path)

        assert not outcome.ok
        assert any("scripts/" in path for path in outcome.denied)
        assert not (tmp_path / "scripts").exists()

    def test_nothing_is_installed_by_provisioning(self, tmp_path: Path) -> None:
        """Dependency installation stays an execution boundary."""
        from edith.engineering.dependency import provision_environment

        from .tool_fixtures import build_gateway

        self.write(
            tmp_path, "pyproject.toml", '[project]\nname = "d"\ndependencies = []\n'
        )
        gateway = build_gateway(tmp_path, permissions_for(EngineeringRole.DEPENDENCY))
        provision_environment(gateway, tmp_path)

        assert not (tmp_path / ".venv").exists()
        script = (tmp_path / "scripts" / "install.bat").read_text(encoding="utf-8")
        assert "--user" not in script
        assert "sudo" not in script
