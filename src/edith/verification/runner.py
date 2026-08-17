"""The verification runner: executes real checks and returns real evidence.

CLAUDE.md invariant 6: no agent may claim tests passed without evidence. This module is
what makes that enforceable -- it runs the command through the M1 ``shell.run`` tool and
returns the exit code and captured output. Nothing here asks a model whether the tests
passed, and no agent can produce a :class:`VerificationOutcome` without a command having
actually run.

Commands come from the configured :class:`~edith.config.schema.VerificationProfile`, never
from a model or a plan. A task selects a *kind* (``tests``, ``lint``, ...); the argv is
operator-controlled configuration.
"""

from __future__ import annotations

import re

from pydantic import Field

from edith.config.schema import VerificationProfile
from edith.environment.classify import classify_failure
from edith.errors import FailureCategory
from edith.observability.logging import get_logger
from edith.schemas.common import EdithModel
from edith.tools.gateway import ToolGateway
from edith.tools.schemas import ToolCall

logger = get_logger(__name__)

#: Kind -> failure classification when that check fails. Lets the retry policy distinguish
#: "the code is wrong" from "the code does not compile".
_FAILURE_BY_KIND: dict[str, FailureCategory] = {
    "tests": FailureCategory.TEST_FAILURE,
    "build": FailureCategory.BUILD_ERROR,
    "lint": FailureCategory.BUILD_ERROR,
    "typecheck": FailureCategory.BUILD_ERROR,
}

#: pytest's summary line, e.g. "3 failed, 12 passed in 0.42s".
_PYTEST_PASSED_RE = re.compile(r"(\d+) passed")
_PYTEST_FAILED_RE = re.compile(r"(\d+) failed")
_PYTEST_ERROR_RE = re.compile(r"(\d+) error")

#: How much captured output to hand a model. The full text is always preserved as an
#: artifact; this is only what fits in a prompt.
SUMMARY_CHARS = 3000


class VerificationOutcome(EdithModel):
    """Evidence from one executed check."""

    kind: str
    command: str
    exit_code: int
    passed: bool
    duration_seconds: float = 0.0
    stdout: str = ""
    stderr: str = ""
    truncated: bool = False
    tests_passed: int | None = None
    tests_failed: int | None = None
    failure_category: FailureCategory | None = None
    #: Why the run failed, in the four-way taxonomy. Set for any non-zero exit.
    diagnosis: str = ""
    #: False when the project code never executed, so the run says nothing about it.
    code_executed: bool = True
    #: Set when the check could not be run at all, as opposed to running and failing.
    unavailable_reason: str | None = None

    @property
    def ran(self) -> bool:
        """Whether the command actually executed."""
        return self.unavailable_reason is None

    def evidence_summary(self, limit: int = SUMMARY_CHARS) -> str:
        """A compact, model-readable summary of what happened.

        Prefers the *tail* of the output: a test runner puts its failure summary at the end,
        and that is the part a Critic or Debugger needs.
        """
        if not self.ran:
            label = f" [{self.failure_category}]" if self.failure_category else ""
            summary = f"{self.kind}: NOT RUN{label} ({self.unavailable_reason})"
            if not self.code_executed:
                summary += (
                    "\nThe project's code never executed, so this run says nothing about "
                    "whether it is correct."
                )
            return summary
        header = (
            f"{self.kind}: {'PASSED' if self.passed else 'FAILED'} "
            f"(exit={self.exit_code}, {self.duration_seconds:.1f}s)"
        )
        if self.tests_passed is not None or self.tests_failed is not None:
            header += f" [passed={self.tests_passed} failed={self.tests_failed}]"
        if self.diagnosis:
            # Stated plainly, because "the tests failed" and "the code never imported" call
            # for different work, and a reader of this summary is deciding which to do.
            header += f"\n{self.failure_category}: {self.diagnosis}"
            if not self.code_executed:
                header += (
                    "\nThe project's code never executed, so this run says nothing about "
                    "whether it is correct."
                )

        body = (self.stdout + ("\n" + self.stderr if self.stderr else "")).strip()
        if len(body) > limit:
            body = "... (earlier output omitted) ...\n" + body[-limit:]
        return f"{header}\n{body}" if body else header


class VerificationReport(EdithModel):
    """The combined result of every check run for a task."""

    outcomes: list[VerificationOutcome] = Field(default_factory=list)

    @property
    def passed(self) -> bool:
        """True when every check that ran passed, and at least one ran."""
        executed = [outcome for outcome in self.outcomes if outcome.ran]
        return bool(executed) and all(outcome.passed for outcome in executed)

    @property
    def failures(self) -> list[VerificationOutcome]:
        """Checks that ran and failed."""
        return [outcome for outcome in self.outcomes if outcome.ran and not outcome.passed]

    @property
    def failure_category(self) -> FailureCategory | None:
        """Classification of the first failure, for the retry policy.

        Falls back to a check that could not run. A missing package leaves no *executed*
        failure to point at, and returning ``None`` there would erase the diagnosis and let
        the policy treat a known environment problem as an unclassified one.
        """
        failures = self.failures
        if failures:
            return failures[0].failure_category
        for outcome in self.outcomes:
            if not outcome.ran and outcome.failure_category is not None:
                return outcome.failure_category
        return None

    def evidence(self, limit: int = SUMMARY_CHARS) -> str:
        """Rendered evidence for every check, for a Critic or Debugger prompt."""
        if not self.outcomes:
            return "(no verification was configured for this task)"
        budget = max(limit // max(len(self.outcomes), 1), 400)
        return "\n\n".join(outcome.evidence_summary(budget) for outcome in self.outcomes)


class VerificationRunner:
    """Runs configured checks through the M1 tool gateway."""

    def __init__(
        self,
        gateway: ToolGateway,
        profile: VerificationProfile,
        *,
        timeout_seconds: float | None = None,
        local_modules: frozenset[str] | None = None,
    ) -> None:
        """
        Args:
            gateway: Permission-scoped gateway. The calling agent must hold ``shell.run``.
            profile: Operator-configured commands.
            timeout_seconds: Per-command budget; ``None`` uses the shell policy default.
            local_modules: Modules the project defines itself, so a failed import of one is
                classified as the project's own defect rather than a missing package.
        """
        self.gateway = gateway
        self.profile = profile
        self.timeout_seconds = timeout_seconds
        self.local_modules = local_modules or frozenset()

    def run(self, kind: str, selector: str | None = None) -> VerificationOutcome:
        """Execute one check.

        Args:
            kind: ``tests``, ``lint``, ``typecheck``, or ``build``.
            selector: Optional extra argument, e.g. a test path. Appended as a single argv
                element, so it cannot inject a second command.
        """
        argv = list(self.profile.command_for(kind))
        if not argv:
            return VerificationOutcome(
                kind=kind,
                command="",
                exit_code=-1,
                passed=False,
                unavailable_reason=f"no {kind!r} command is configured for this project",
            )
        if selector:
            argv.append(selector)

        arguments: dict[str, object] = {"argv": argv}
        if self.timeout_seconds is not None:
            arguments["timeout_seconds"] = self.timeout_seconds

        result = self.gateway.execute(ToolCall(tool="shell.run", arguments=arguments))
        command = " ".join(argv)

        if not result.ok:
            # The command could not be run: denied, missing executable, or timed out. That
            # is an environment/tool problem, not a failing test suite, and the distinction
            # matters to the retry policy.
            category = (
                FailureCategory.SECURITY_FAILURE
                if result.denied
                else (result.failure_category or FailureCategory.TOOL_ERROR)
            )
            logger.warning(
                "verification.unavailable", kind=kind, command=command, error=result.error
            )
            return VerificationOutcome(
                kind=kind,
                command=command,
                exit_code=-1,
                passed=False,
                failure_category=category,
                unavailable_reason=result.error or "the command could not be executed",
            )

        output = result.output
        exit_code = int(output["exit_code"])
        passed = exit_code == 0
        stdout = str(output.get("stdout", ""))
        stderr = str(output.get("stderr", ""))
        counts = _parse_test_counts(stdout + "\n" + stderr) if kind == "tests" else (None, None)

        # Classify what a non-zero exit actually means. A missing package is not a failing
        # test, and sending the Debugger after code that never imported burns the whole
        # repair budget on a guess.
        #
        # M2.1 treated *any* unimportable module as a missing runner, which mislabelled a
        # missing application dependency as broken tooling. The classifier makes that
        # distinction properly, so it owns the decision here.
        diagnosis = None
        if not passed:
            diagnosis = classify_failure(
                f"{stdout}\n{stderr}",
                exit_code=exit_code,
                local_modules=self.local_modules,
            )

        # When the project's code never executed, the check *ran* but *verified nothing*.
        # Recording that as a plain failure would let "the tests are failing" stand in for
        # "the tests never got the chance to run".
        unavailable_reason = None
        if diagnosis is not None and not diagnosis.code_executed:
            unavailable_reason = diagnosis.reason
            logger.warning(
                "verification.did_not_verify",
                kind=kind,
                category=str(diagnosis.category),
                subject=diagnosis.subject,
                reason=diagnosis.reason,
            )

        outcome = VerificationOutcome(
            kind=kind,
            command=command,
            exit_code=exit_code,
            passed=passed,
            duration_seconds=float(output.get("duration_seconds", 0.0)),
            stdout=stdout,
            stderr=stderr,
            truncated=bool(output.get("stdout_truncated")) or bool(output.get("stderr_truncated")),
            tests_passed=counts[0],
            tests_failed=counts[1],
            failure_category=(
                None
                if passed
                else (
                    diagnosis.category
                    if diagnosis is not None
                    else _FAILURE_BY_KIND.get(kind, FailureCategory.UNKNOWN)
                )
            ),
            diagnosis=diagnosis.reason if diagnosis else "",
            code_executed=diagnosis.code_executed if diagnosis else True,
            unavailable_reason=unavailable_reason,
        )
        logger.info(
            "verification.completed",
            kind=kind,
            passed=passed,
            exit_code=exit_code,
            tests_failed=outcome.tests_failed,
            category=str(outcome.failure_category) if outcome.failure_category else "",
            diagnosis=outcome.diagnosis,
        )
        return outcome

    def run_all(self, requirements: tuple[tuple[str, str | None], ...]) -> VerificationReport:
        """Run several checks in order, stopping at the first hard failure.

        Stopping early is deliberate: if the tests fail, a lint report adds noise to the
        Debugger's prompt without adding information about the actual defect.
        """
        outcomes: list[VerificationOutcome] = []
        for kind, selector in requirements:
            outcome = self.run(kind, selector)
            outcomes.append(outcome)
            if outcome.ran and not outcome.passed:
                break
        return VerificationReport(outcomes=outcomes)


def _parse_test_counts(text: str) -> tuple[int | None, int | None]:
    """Extract passed/failed counts from a pytest summary line, when present."""
    passed = _PYTEST_PASSED_RE.search(text)
    failed = _PYTEST_FAILED_RE.search(text)
    errors = _PYTEST_ERROR_RE.search(text)

    failed_count: int | None = None
    if failed or errors:
        failed_count = int(failed.group(1)) if failed else 0
        failed_count += int(errors.group(1)) if errors else 0
    return (int(passed.group(1)) if passed else None, failed_count)
