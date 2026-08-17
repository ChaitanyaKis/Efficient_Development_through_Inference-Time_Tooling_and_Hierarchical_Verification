"""M5.2: a failure the agent cannot fix must not be charged to its repair budget.

The live isolated task exposed this. Verification timed out with ``tests: NOT RUN``; the
executor treated the rejection as ordinary, spent a repair attempt regenerating code that
had never been tested, timed out again, and reported ``REPAIR_EXHAUSTED`` -- which reads as
"this agent cannot write working code" when in fact nothing had ever been run against it.

Two distinct wrongs: the budget is consumed, and the *attribution* is false. The fix is a
single explicit set, :data:`REPAIRABLE_FAILURES`, applied at the one place the loop decides
to go around again.

The bar is that the split stays honest in both directions. M5.2 said it explicitly: do not
turn "retry everything" into "never retry anything." A genuine test failure is still the
agent's problem and must still be repaired.
"""

from __future__ import annotations

import pytest

from edith.engineering.executor import REPAIRABLE_FAILURES
from edith.errors import FailureCategory

#: Failures that describe the environment or the policy, never the generated code.
NOT_THE_AGENTS_FAULT = (
    FailureCategory.TIMEOUT,
    FailureCategory.ENVIRONMENT_FAILURE,
    FailureCategory.DEPENDENCY_FAILURE,
    FailureCategory.SECURITY_FAILURE,
    FailureCategory.CONFIGURATION_ERROR,
    FailureCategory.TOOL_ERROR,
)


class TestTheRepairableSet:
    @pytest.mark.parametrize("category", NOT_THE_AGENTS_FAULT)
    def test_environment_and_policy_failures_are_not_repairable(
        self, category: FailureCategory
    ) -> None:
        assert category not in REPAIRABLE_FAILURES

    def test_a_timeout_is_not_repairable(self) -> None:
        """The exact category the live run produced, called out on its own.

        ``tests: NOT RUN [TIMEOUT]`` says the code was never executed. There is no evidence
        about the code in that result, so there is nothing for the agent to repair.
        """
        assert FailureCategory.TIMEOUT not in REPAIRABLE_FAILURES

    def test_a_genuine_test_failure_is_still_repairable(self) -> None:
        """The other half of the bar: this must not become 'never retry anything'."""
        assert FailureCategory.TEST_FAILURE in REPAIRABLE_FAILURES

    def test_code_that_does_not_import_is_still_repairable(self) -> None:
        assert FailureCategory.CODE_FAILURE in REPAIRABLE_FAILURES

    def test_malformed_model_output_is_still_repairable(self) -> None:
        """M5.1's real failure shape: the model got the envelope wrong. That is fixable."""
        assert FailureCategory.VALIDATION_FAILURE in REPAIRABLE_FAILURES

    def test_the_set_is_a_strict_subset_of_the_taxonomy(self) -> None:
        """A guard against someone widening this back to 'everything is repairable'."""
        every = set(FailureCategory)
        assert every > REPAIRABLE_FAILURES
        assert len(REPAIRABLE_FAILURES) < len(every)

    def test_every_repairable_category_is_a_real_category(self) -> None:
        assert set(FailureCategory) >= REPAIRABLE_FAILURES
