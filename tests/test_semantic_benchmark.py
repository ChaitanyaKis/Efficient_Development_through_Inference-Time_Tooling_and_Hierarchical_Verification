"""M7: the benchmark's ground truth must be beyond the model's reach.

The experiment only means something if the model under evaluation cannot influence what counts
as correct. Three things have to hold, and none of them is guaranteed by the reviewers being
well-behaved:

* the acceptance tests are authored by hand and live in the repository, not generated;
* no quality principal can write to them, or to anything else;
* acceptance is decided by a separate process after the merge, so no verdict from the Judge
  can reach it.

The last one is why :func:`benchmarks.run_semantic.acceptance` shells out rather than importing
anything from the quality layer.
"""

from __future__ import annotations

import ast
from pathlib import Path

from benchmarks.semantic import TASKS, Category, by_category

from edith.quality.principals import (
    JUDGE,
    QUALITY_PERMISSIONS,
    REVIEWER,
    SECURITY,
    TESTER,
    may_write,
)
from edith.schemas.agent import AgentPermissions
from edith.tools.schemas import ToolCall

from .tool_fixtures import build_gateway


class TestTheBenchmarkIsWellFormed:
    def test_there_are_at_least_twelve_tasks(self) -> None:
        assert len(TASKS) >= 12

    def test_every_category_has_at_least_three(self) -> None:
        for category, tasks in by_category().items():
            assert len(tasks) >= 3, f"{category} has only {len(tasks)}"

    def test_all_four_categories_are_represented(self) -> None:
        assert set(by_category()) == set(Category)

    def test_task_ids_are_unique(self) -> None:
        assert len({task.task_id for task in TASKS}) == len(TASKS)

    def test_every_task_has_an_independent_acceptance_test(self) -> None:
        for task in TASKS:
            assert task.acceptance.strip(), f"{task.task_id} has no acceptance test"
            assert "def test_" in task.acceptance

    def test_every_acceptance_test_parses(self) -> None:
        """A ground truth that does not run is not a ground truth."""
        for task in TASKS:
            ast.parse(task.acceptance)

    def test_every_acceptance_test_imports_the_generated_module(self) -> None:
        """Otherwise it could pass without the implementation existing -- M5's vacuous check."""
        for task in TASKS:
            module = task.path.removesuffix(".py").replace("/", ".")
            assert module in task.acceptance, f"{task.task_id} never imports {module}"

    def test_requirements_do_not_leak_the_acceptance_assertions(self) -> None:
        """The coder is told the behaviour, never shown the test that grades it."""
        for task in TASKS:
            assert "assert" not in task.requirement


class TestGroundTruthIsBeyondTheModel:
    def test_no_quality_principal_can_write_to_tests(self, tmp_path: Path) -> None:
        """Item 2 and 3: a reviewer cannot weaken what judges it."""
        (tmp_path / "tests").mkdir()
        original = "def test_x():\n    assert compute() == 42\n"
        (tmp_path / "tests" / "test_acc.py").write_text(original, encoding="utf-8")
        for name, permissions in QUALITY_PERMISSIONS.items():
            if permissions == TESTER:
                continue  # the tester writes tests by design; it is not a reviewer
            gateway = build_gateway(tmp_path, permissions)
            result = gateway.execute(
                ToolCall(
                    tool="filesystem.write",
                    arguments={"path": "tests/test_acc.py", "content": "def test_x(): pass\n"},
                )
            )
            assert not result.ok, f"{name} could rewrite an acceptance test"
        assert (tmp_path / "tests" / "test_acc.py").read_text(encoding="utf-8") == original

    def test_the_reviewers_hold_no_write_scope_at_all(self) -> None:
        for permissions in (REVIEWER, SECURITY, JUDGE):
            assert not may_write(permissions)

    def test_acceptance_runs_outside_the_quality_layer(self) -> None:
        """Item 4: no Judge verdict can reach the acceptance decision.

        Asserted structurally -- the runner's acceptance path must not import the quality
        layer, so there is no route by which a verdict could influence it.
        """
        source = Path("benchmarks/run_semantic.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        function = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "acceptance"
        )
        body = ast.dump(function)
        assert "quality" not in body
        assert "Judge" not in body
        assert "subprocess" in body, "acceptance must run in its own process"

    def test_a_reviewer_cannot_modify_an_implementation(self, tmp_path: Path) -> None:
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "a.py").write_text("VALUE = 1\n", encoding="utf-8")
        gateway = build_gateway(tmp_path, REVIEWER)
        result = gateway.execute(
            ToolCall(
                tool="filesystem.write",
                arguments={"path": "src/a.py", "content": "VALUE = 2\n"},
            )
        )
        assert not result.ok
        assert (tmp_path / "src" / "a.py").read_text(encoding="utf-8") == "VALUE = 1\n"

    def test_the_tester_still_cannot_reach_the_implementation(self, tmp_path: Path) -> None:
        """The one principal with write scope is still bounded to tests."""
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "a.py").write_text("VALUE = 1\n", encoding="utf-8")
        gateway = build_gateway(tmp_path, TESTER)
        result = gateway.execute(
            ToolCall(
                tool="filesystem.write",
                arguments={"path": "src/a.py", "content": "VALUE = 2\n"},
            )
        )
        assert not result.ok

    def test_an_unscoped_principal_is_not_silently_permitted(self, tmp_path: Path) -> None:
        gateway = build_gateway(tmp_path, AgentPermissions(allowed_tools=frozenset()))
        result = gateway.execute(
            ToolCall(tool="filesystem.write", arguments={"path": "x.py", "content": "x"})
        )
        assert not result.ok
