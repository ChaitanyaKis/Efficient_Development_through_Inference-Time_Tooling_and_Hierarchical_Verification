"""The timeline must not let an agent's run status read as its verdict.

From a real demo run. The screen showed, in three consecutive rows:

    verifier  tests FAIL - exit 1
    critic    SUCCESS - task_01
    executor  CODE_FAILURE - REPAIR - tests failed with exit code 1

Every line is accurate and the middle one is unreadable. ``SUCCESS`` there is the *agent run*
status -- the critic executed without erroring -- while its verdict was FAIL, which is why
repair follows. Anyone watching reads it as the critic approving a task it had just rejected,
and then cannot explain why a passed task went to repair.

The same word doing two jobs, which is the defect ``DEFERRED`` was split out to fix in the
task statuses. The outcome of a step is already carried by the verifier and executor rows, so
the fix is for the agent row to claim only what it knows: that the agent ran.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest


@pytest.fixture
def page() -> str:
    return Path("src/edith/ui/static/index.html").read_text(encoding="utf-8")


class TestAnAgentRunRowSaysOnlyThatItRan:
    def test_the_raw_status_is_not_printed(self, page: str) -> None:
        """`${r.status}` in the timeline row is what produced `critic SUCCESS`."""
        assert "msg: `${r.status}" not in page

    def test_the_row_goes_through_the_label_helper(self, page: str) -> None:
        assert "msg: `${agentRan(r.status)}" in page

    def test_success_becomes_ran(self, page: str) -> None:
        assert 'if (s === "SUCCESS") return "ran";' in page

    def test_failure_becomes_errored(self, page: str) -> None:
        """An agent that crashed is still worth seeing; it just is not a verdict."""
        assert 'if (s === "FAILURE") return "errored";' in page

    def test_an_unknown_status_still_renders(self, page: str) -> None:
        """A status this build has not seen must not blank the row."""
        helper = page.split("function agentRan(status) {")[1].split("}")[0]
        assert "return s.toLowerCase() || \"ran\"" in helper


class TestTheOutcomeIsStillVisible:
    """Relabelling must not remove information, only stop it being misread."""

    def test_verification_rows_still_report_pass_or_fail(self, page: str) -> None:
        assert "SNAP.verifications" in page

    def test_a_failed_agent_run_is_still_marked_bad(self, page: str) -> None:
        """The colour still keys off the real status, not the softened label."""
        assert 'bad: /FAIL|ERROR/.test(String(r.status))' in page

    def test_the_helper_explains_why_it_exists(self, page: str) -> None:
        """This one is not obvious from the code, and reverts easily without the reason."""
        block = page[: page.index("function agentRan(status) {")]
        assert "not what it decided" in block


class TestNoOtherRowLaundersAStatus:
    def test_the_word_success_is_never_shown_for_an_agent_run(self, page: str) -> None:
        rendered = re.findall(r"msg: `\$\{([a-zA-Z_.]+)", page)
        assert "r.status" not in rendered
