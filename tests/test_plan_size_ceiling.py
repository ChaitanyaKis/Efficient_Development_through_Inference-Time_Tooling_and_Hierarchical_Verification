"""A larger request must plan, not crash, and must not be cut off half-built.

Both defects surfaced on the first genuinely multi-function request -- a six-function
statistics library -- and each capped the system well below what its architecture supports.

**The plan ceiling was a model-distrust limit applied to trustworthy structure.** Steps were
bounded at six because a small model left unbounded emits a dozen vague stages. But fan-out
does not come from the model: Phase B builds one step per function plus an assembly step,
deterministically. Six functions needed seven steps, the schema refused the seventh, and the
ValidationError killed the execution instead of declining the fan-out.

**The run budget was flat where the work is not.** A total agent-run ceiling of 40 exists to
stop a runaway loop, but applied unscaled it also caps how much may be asked for: the seven
tasks spent it on the first five and the run was cut off with two never attempted.

Neither is about the model being weak. Both are the harness refusing work it could do.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from edith.agents.fanout import MAX_FUNCTIONS, FunctionSpec, specs_to_steps
from edith.agents.planner import MAX_PLAN_STEPS, PlannedStep, PlannerOutput
from edith.config.loader import load_config


def specs(count: int) -> list[FunctionSpec]:
    return [
        FunctionSpec(name=f"fn{i}", signature=f"fn{i}(values)", behaviour="does a thing")
        for i in range(1, count + 1)
    ]


class TestAPlanCanHoldAFannedOutRequest:
    def test_six_functions_plus_assembly_fit(self) -> None:
        """The exact request that crashed: 6 functions, 7 steps."""
        steps = specs_to_steps(specs(6), "six functions", assembly_module="statistics")
        assert len(steps) == 7
        assert PlannerOutput(goal="g", steps=steps).steps == steps

    def test_the_fan_out_cap_cannot_exceed_the_plan_ceiling(self) -> None:
        """The two were picked independently once, at 12 against 6, and the run died."""
        assert MAX_FUNCTIONS + 1 <= MAX_PLAN_STEPS

    def test_a_full_fan_out_still_validates(self) -> None:
        steps = specs_to_steps(specs(MAX_FUNCTIONS), "many", assembly_module="library")
        assert PlannerOutput(goal="g", steps=steps)

    def test_the_ceiling_is_still_bounded(self) -> None:
        """Unbounded fan-out is its own failure mode; this is larger, not absent."""
        with pytest.raises(ValidationError):
            PlannedStep(
                step=MAX_PLAN_STEPS + 1,
                title="too far",
                description="d",
                files=[],
                depends_on=[],
                acceptance="",
            )


class TestTheRunBudgetScalesWithThePlan:
    @pytest.fixture
    def settings(self) -> object:
        return load_config(None).orchestration

    def test_a_seven_task_plan_gets_more_than_the_flat_floor(self, settings: object) -> None:
        floor = settings.max_total_agent_runs
        scaled = max(floor, 7 * settings.agent_runs_per_task)
        assert scaled > floor

    def test_the_floor_still_applies_to_a_small_plan(self, settings: object) -> None:
        """A one-task run must not get a smaller budget than it had before."""
        scaled = max(settings.max_total_agent_runs, 1 * settings.agent_runs_per_task)
        assert scaled == settings.max_total_agent_runs

    def test_the_per_task_allowance_covers_a_task_that_exhausts_repairs(
        self, settings: object
    ) -> None:
        """Measured at roughly seven agent runs for such a task."""
        assert settings.agent_runs_per_task >= 7

    def test_the_orchestrator_uses_the_scaled_ceiling(self) -> None:
        from pathlib import Path

        source = Path("src/edith/orchestrator.py").read_text(encoding="utf-8")
        assert "self._runs >= self._effective_budget()" in source
        assert "self._runs >= self.settings.max_total_agent_runs" not in source

    def test_the_budget_falls_back_before_a_plan_exists(self) -> None:
        """_effective_budget is read on paths that run before the plan is sized."""
        from pathlib import Path

        source = Path("src/edith/orchestrator.py").read_text(encoding="utf-8")
        assert "return self._run_budget or self.settings.max_total_agent_runs" in source
