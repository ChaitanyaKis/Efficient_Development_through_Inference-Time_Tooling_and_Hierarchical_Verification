"""Planner fan-out: one request becomes many single-function tasks.

The measurement this exists for: a four-function request ran seventeen repairs and delivered
nothing, while the same work scoped one function at a time delivered reliably. The cause was
upstream of the loop -- the planner emitted a single task carrying four implementations -- so
the fix belongs in planning, and the invariant worth asserting is that no task reaching the
coding agent ever describes more than one function.

Two properties carry the design and are tested directly:

**Phase B is pure.** The decomposition itself must not be a source of nondeterminism, so the
same function list and request always produce byte-identical tasks, with no model involved.

**A dropped function fails loudly.** The worst available outcome is a run that completes with
an operation silently missing, discovered by whoever uses the result.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from edith.agents.fanout import (
    DEFAULT_CASES,
    FunctionList,
    FunctionSpec,
    PlanFanOutError,
    assembly_name,
    assert_coverage,
    cases_for,
    enforce_single_function,
    fan_out,
    signatures_in,
    specs_to_steps,
)
from edith.agents.planner import PlannedStep, PlannerOutput, plan_to_tasks
from edith.orchestrator import _needs_fanout
from edith.planning import TaskGraph

REQUEST = (
    "Build a calculator with add(a, b), subtract(a, b), multiply(a, b) and "
    "divide(a, b) which raises ValueError when b is 0."
)


def specs(*names: str) -> list[FunctionSpec]:
    return [
        FunctionSpec(name=name, signature=f"{name}(a, b)", behaviour=f"returns the {name}")
        for name in names
    ]


class TestTheInvariant:
    """No task reaching the coding agent may describe more than one function."""

    def test_a_four_signature_spec_is_split(self) -> None:
        """The exact shape that ran seventeen repairs and delivered nothing."""
        bundled = PlannedStep(
            step=1,
            title="Implement the calculator",
            description=(
                "Implement add(a, b), subtract(a, b), multiply(a, b) and divide(a, b) "
                "in src/backend/calculator.py"
            ),
            files=["src/backend/calculator.py"],
            depends_on=[],
            acceptance="all four work",
        )
        result = enforce_single_function([bundled])
        assert len(result) == 4
        for step in result:
            assert len(signatures_in(step.description)) == 1

    def test_a_single_signature_step_is_left_alone(self) -> None:
        """An ordinary request already has the working shape and pays nothing."""
        steps = specs_to_steps(specs("add"), REQUEST)
        assert enforce_single_function(steps) == [
            step.model_copy(update={"step": 1}) for step in steps
        ]

    def test_every_generated_task_names_exactly_one_function(self) -> None:
        steps = enforce_single_function(
            specs_to_steps(specs("add", "subtract", "multiply", "divide"), REQUEST)
        )
        for step in steps:
            if step.title.startswith("Assemble "):
                continue
            assert len(signatures_in(step.description)) == 1, step.description

    def test_splitting_renumbers_without_collision(self) -> None:
        bundled = PlannedStep(
            step=1,
            title="Implement two",
            description="Implement alpha(x) and beta(y)",
            files=[],
            depends_on=[],
            acceptance="",
        )
        result = enforce_single_function([bundled, bundled.model_copy(update={"step": 2})])
        numbers = [step.step for step in result]
        assert numbers == sorted(numbers)
        assert len(numbers) == len(set(numbers))

    def test_the_split_survives_translation_into_tasks(self) -> None:
        """The invariant has to hold at the boundary the coder actually reads."""
        steps = enforce_single_function(
            specs_to_steps(specs("add", "subtract"), REQUEST, assembly_module="calculator")
        )
        tasks = plan_to_tasks(PlannerOutput(goal="g", steps=steps))
        for task in tasks:
            if task.title.startswith("Assemble "):
                continue
            assert len(signatures_in(task.description)) == 1


class TestPhaseBIsPure:
    def test_the_same_input_produces_the_same_tasks(self) -> None:
        first = specs_to_steps(specs("add", "divide"), REQUEST, assembly_module="calculator")
        second = specs_to_steps(specs("add", "divide"), REQUEST, assembly_module="calculator")
        assert first == second

    def test_it_calls_no_model(self) -> None:
        """Asserted structurally: there is no provider to call."""
        import ast
        from pathlib import Path

        tree = ast.parse(Path("src/edith/agents/fanout.py").read_text(encoding="utf-8"))
        function = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "specs_to_steps"
        )
        body = ast.dump(function)
        for forbidden in ("provider", "structured_generate", "execute"):
            assert forbidden not in body

    def test_the_task_shape_is_the_measured_one(self) -> None:
        """Every detail in the template was learned from a specific failure."""
        step = specs_to_steps(specs("divide"), REQUEST)[0]
        assert "containing one function" in step.description
        assert "from src.backend.divide import divide" in step.description
        # An earlier wording asked only for a file "whose first line is" the import, and
        # three of four generated test files then contained exactly that line and nothing
        # else -- passing collection while asserting nothing.
        assert "must contain a test function, not only the import line" in step.description
        assert "def test_divide():" in step.description

    def test_the_template_still_names_exactly_one_implementation(self) -> None:
        """The test function the template demands must not count against the invariant."""
        step = specs_to_steps(specs("divide"), REQUEST)[0]
        assert set(signatures_in(step.description)) == {"divide"}

    def test_boundary_cases_are_used_when_the_analyser_fires(self) -> None:
        """Reusing the one supported mechanism rather than inventing expected values."""
        spec = FunctionSpec(
            name="fee",
            signature="fee(days)",
            behaviour="charges nothing until a payment is more than 3 days late",
        )
        rendered = cases_for(spec, "no fee until more than 3 days late")
        assert "> 3" in rendered
        assert "4" in rendered

    def test_the_assertion_list_is_left_open_when_nothing_fires(self) -> None:
        """A fabricated expected value would be worse than none."""
        spec = FunctionSpec(name="add", signature="add(a, b)", behaviour="returns a plus b")
        assert cases_for(spec, "add two numbers") == DEFAULT_CASES


class TestCoverageCrossCheck:
    def test_a_dropped_operation_fails_the_plan(self) -> None:
        """Silently missing behaviour is the worst outcome available."""
        with pytest.raises(PlanFanOutError, match="omits"):
            assert_coverage(REQUEST, specs("add", "subtract", "multiply"))

    def test_the_error_names_what_is_missing(self) -> None:
        with pytest.raises(PlanFanOutError) as excinfo:
            assert_coverage(REQUEST, specs("add"))
        assert "divide" in str(excinfo.value)

    def test_a_complete_plan_passes(self) -> None:
        assert_coverage(REQUEST, specs("add", "subtract", "multiply", "divide"))

    def test_extra_functions_are_not_an_error(self) -> None:
        """Planning a helper the request did not name is a judgement call, not a defect."""
        assert_coverage(REQUEST, specs("add", "subtract", "multiply", "divide", "negate"))

    def test_prose_without_call_syntax_is_not_penalised(self) -> None:
        """Only operations named in call form are checked."""
        assert_coverage("build something that adds and subtracts", specs("add"))


class TestAssemblyTask:
    def test_it_is_ordered_last(self) -> None:
        steps = enforce_single_function(
            specs_to_steps(specs("add", "subtract"), REQUEST, assembly_module="calculator")
        )
        assert steps[-1].title == "Assemble calculator"
        assert steps[-1].step == max(step.step for step in steps)

    def test_it_depends_on_every_implementation(self) -> None:
        steps = enforce_single_function(
            specs_to_steps(specs("add", "subtract", "divide"), REQUEST, assembly_module="calc")
        )
        implementations = [s.step for s in steps if not s.title.startswith("Assemble ")]
        assert sorted(steps[-1].depends_on) == sorted(implementations)

    def test_the_dag_orders_it_last(self) -> None:
        """Asserted through the real graph, not just the step numbers."""
        steps = enforce_single_function(
            specs_to_steps(specs("add", "subtract"), REQUEST, assembly_module="calculator")
        )
        tasks = plan_to_tasks(PlannerOutput(goal="g", steps=steps))
        graph = TaskGraph(tasks)
        last = graph.get(graph.topological_order()[-1])
        assert last is not None
        assert last.title == "Assemble calculator"

    def test_it_contains_no_logic(self) -> None:
        steps = specs_to_steps(specs("add", "divide"), REQUEST, assembly_module="calculator")
        assembly = steps[-1].description
        assert "__all__" in assembly
        assert "ONLY these import lines" in assembly
        assert "def " not in assembly

    def test_a_single_function_request_gets_no_assembly(self) -> None:
        """Nothing to assemble; the extra task would be pure overhead."""
        steps = specs_to_steps(specs("add"), REQUEST, assembly_module=None)
        assert all(not step.title.startswith("Assemble ") for step in steps)

    def test_the_module_name_is_derived_not_hardcoded(self) -> None:
        assert assembly_name("Build a calculator with add(a, b)") == "calculator"
        assert assembly_name("Build an inventory with count(x)") == "inventory"

    def test_a_request_offering_no_noun_still_yields_a_name(self) -> None:
        assert assembly_name("add(a, b)").isidentifier()


class TestFanOutIsOnlyUsedWhenNeeded:
    def step(self, description: str, files: list[str] | None = None) -> PlannedStep:
        return PlannedStep(
            step=1,
            title="t",
            description=description,
            files=files or [],
            depends_on=[],
            acceptance="",
        )

    def test_a_multi_function_step_triggers_it(self, tmp_path: Path) -> None:
        plan = PlannerOutput(goal="g", steps=[self.step("Implement add(a, b) and subtract(a, b)")])
        assert _needs_fanout(plan, "do the thing", tmp_path) is True

    def test_a_multi_function_request_triggers_it(self, tmp_path: Path) -> None:
        """The correction a real run forced.

        On a four-function calculator the planner produced five steps that all wrote to one
        shared file with no tests, and two failed fighting over it. Counting signatures per
        step saw nothing wrong, so the request is read as well.
        """
        plan = PlannerOutput(goal="g", steps=[self.step("Create the calculator class")])
        assert _needs_fanout(plan, REQUEST, tmp_path) is True

    def test_a_single_function_request_does_not(self, tmp_path: Path) -> None:
        plan = PlannerOutput(goal="g", steps=[self.step("Implement add(a, b)")])
        assert _needs_fanout(plan, "Add an add(a, b) function", tmp_path) is False

    def test_an_existing_project_is_left_alone(self, tmp_path: Path) -> None:
        """Fan-out imposes a greenfield layout, which is wrong for a repair."""
        (tmp_path / "inventory.py").write_text("x = 1\n", encoding="utf-8")
        plan = PlannerOutput(
            goal="g",
            steps=[self.step("Fix add(a, b) and remove(a, b)", files=["inventory.py"])],
        )
        assert _needs_fanout(plan, REQUEST, tmp_path) is False

    def test_the_repair_benchmark_request_does_not_trigger_it(self, tmp_path: Path) -> None:
        """A guard against fan-out hijacking the existing live benchmarks."""
        request = (
            "The calculator module is missing a multiply function. Add a multiply(a, b) "
            "function to calculator.py that returns the product of its two arguments."
        )
        plan = PlannerOutput(goal="g", steps=[self.step("Add multiply(a, b)")])
        assert _needs_fanout(plan, request, tmp_path) is False


class TestPhaseAFailsSafely:
    def _agent(self, output: Any, *, ok: bool = True) -> Any:
        class _Response:
            def __init__(self) -> None:
                self.ok = ok
                self.error = "" if ok else "model unreachable"
                self.failure_category = None
                self.output = output

        class _Agent:
            def execute(self, request: object) -> object:
                return _Response()

        return _Agent()

    def test_an_empty_function_list_is_refused(self) -> None:
        agent = self._agent(FunctionList(functions=[]).model_dump())
        with pytest.raises(PlanFanOutError, match="named no functions"):
            fan_out(agent, REQUEST)

    def test_a_failed_phase_a_is_refused(self) -> None:
        agent = self._agent({}, ok=False)
        with pytest.raises(PlanFanOutError, match="fan-out planning failed"):
            fan_out(agent, REQUEST)

    def test_malformed_output_raises_a_validation_error(self) -> None:
        """Handled by the strict schema, as malformed model output is everywhere else."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            FunctionList.model_validate({"functions": [{"name": "not an identifier"}]})

    def test_a_signature_without_arguments_is_refused(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            FunctionSpec(name="add", signature="add", behaviour="adds")

    def test_a_complete_fan_out_produces_one_step_per_function(self) -> None:
        agent = self._agent(
            FunctionList(functions=specs("add", "subtract", "multiply", "divide")).model_dump()
        )
        plan = fan_out(agent, REQUEST)
        implementations = [s for s in plan.steps if not s.title.startswith("Assemble ")]
        assert len(implementations) == 4
        assert plan.steps[-1].title.startswith("Assemble ")


class TestRunLevelRepairCap:
    def test_the_cap_exists_and_is_bounded(self) -> None:
        from edith.config.loader import load_config

        settings = load_config(None).orchestration
        assert settings.max_total_repairs >= settings.max_repair_attempts

    def test_the_per_task_budget_is_unchanged(self) -> None:
        """Fan-out must not quietly alter what a single task is allowed."""
        from edith.config.loader import load_config

        assert load_config(None).orchestration.max_repair_attempts == 2

    def test_the_orchestrator_consults_the_run_cap(self) -> None:
        from pathlib import Path

        source = Path("src/edith/orchestrator.py").read_text(encoding="utf-8")
        assert "self._run_repairs < self.settings.max_total_repairs" in source
        assert "self._run_repairs += 1" in source

    def test_the_cap_only_counts_actual_repairs(self) -> None:
        """Environment faults never enter the repair branch, so they cannot consume it."""
        from pathlib import Path

        source = Path("src/edith/orchestrator.py").read_text(encoding="utf-8")
        marker = source.index("self._run_repairs += 1")
        window = source[marker - 700 : marker]
        assert "action is FailureAction.REPAIR" in window
