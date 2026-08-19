"""Telling four different failures apart.

A verification command exits non-zero. That single fact means one of four very different
things, and treating them alike wastes the repair budget chasing the wrong problem:

``ENVIRONMENT_FAILURE``
    The toolchain itself is broken or absent -- no interpreter, no pytest, no runner. The
    code was never executed, so nothing was learned about it. Escalate to a human.

``DEPENDENCY_FAILURE``
    A package the project needs is not installed. The code may be perfectly correct. Fix
    the environment, not the source.

``CODE_FAILURE``
    The project's own code could not be imported or parsed -- a syntax error, a bad import,
    a missing name. Real, and the Debugger's job.

``TEST_FAILURE``
    Everything ran and an assertion did not hold. The only case that says something about
    correctness.

M2 collapsed the first two into ``TEST_FAILURE`` and sent the Debugger hunting for a bug in
code that had never executed. Classification is deterministic pattern matching on real
output; no model is consulted, because a model asked "why did this fail" will always have
an opinion.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from edith.errors import FailureCategory
from edith.observability.logging import get_logger

logger = get_logger(__name__)

#: A module that failed to import, with the name captured.
#:
#: The quotes are optional because the two forms come from different places and only one of
#: them is a traceback: an interpreter invoked as ``python -m pytest`` with pytest absent
#: prints a bare ``No module named pytest`` and exits 1. Requiring the quoted traceback form
#: is precisely how a missing test runner gets reported as a failing test.
_MISSING_MODULE_RE = re.compile(
    r"No module named ['\"]?([A-Za-z0-9_.]+)['\"]?"
)
_CANNOT_IMPORT_NAME_RE = re.compile(
    r"ImportError:\s*cannot import name ['\"]([A-Za-z0-9_]+)['\"]"
)

#: Tooling whose absence means the *runner* is broken, not the project.
_TOOLCHAIN_MODULES = frozenset({"pytest", "pip", "setuptools", "mypy", "ruff", "tox", "nox"})

#: Output that means the interpreter or runner never started.
_ENVIRONMENT_MARKERS: tuple[str, ...] = (
    "is not recognized as an internal or external command",
    "command not found",
    "no such file or directory: python",
    "can't open file",
    "was not found on path",
    "not allowlisted",
    "the command could not be executed",
    "python was not found",
    "no python at",
)

#: Output that means the project's own code is broken.
_CODE_MARKERS: tuple[str, ...] = (
    "syntaxerror",
    "indentationerror",
    "taberror",
    "nameerror",
    "attributeerror",
    "typeerror",
    "collection error",
    "errors during collection",
)

#: Output that means a test ran and failed.
_TEST_MARKERS: tuple[str, ...] = (
    "assertionerror",
    "failed",
    "assert ",
)


@dataclass(frozen=True)
class FailureDiagnosis:
    """What a failed verification command actually means."""

    category: FailureCategory
    reason: str
    #: The package or module implicated, when one could be identified.
    subject: str = ""
    #: Whether the code under test ever executed. When false, the run says nothing about
    #: correctness and a repair attempt would be guesswork.
    code_executed: bool = True
    remediation: str = ""

    @property
    def is_environment(self) -> bool:
        """Whether the toolchain, rather than the project, is at fault."""
        return self.category is FailureCategory.ENVIRONMENT_FAILURE

    @property
    def is_dependency(self) -> bool:
        """Whether a missing package is at fault."""
        return self.category is FailureCategory.DEPENDENCY_FAILURE


def classify_failure(
    output: str,
    *,
    exit_code: int = 1,
    local_modules: frozenset[str] | None = None,
) -> FailureDiagnosis:
    """Diagnose a failed verification command from its real output.

    Order matters and encodes precedence: a missing runner masks everything downstream, a
    missing dependency masks the code, and only once both are excluded does a failing
    assertion mean what it appears to mean.

    Args:
        output: Combined stdout and stderr from the command.
        exit_code: The command's exit status.
        local_modules: Modules the project defines itself. A failed import of one of these
            is the project's own bug, not a missing third-party package.
    """
    lowered = output.lower()
    local = local_modules or frozenset()

    if any(marker in lowered for marker in _ENVIRONMENT_MARKERS):
        return FailureDiagnosis(
            category=FailureCategory.ENVIRONMENT_FAILURE,
            reason="the verification command could not be started",
            code_executed=False,
            remediation="Check the configured interpreter and that the runner is installed.",
        )

    missing = _MISSING_MODULE_RE.search(output)
    if missing:
        module = missing.group(1).split(".")[0]
        if module in _TOOLCHAIN_MODULES:
            return FailureDiagnosis(
                category=FailureCategory.ENVIRONMENT_FAILURE,
                reason=f"the test runner is not installed: {module!r} is missing",
                subject=module,
                code_executed=False,
                remediation=(
                    f"Install the toolchain into the project environment: "
                    f"python -m pip install {module}"
                ),
            )
        if module in local:
            return FailureDiagnosis(
                category=FailureCategory.CODE_FAILURE,
                reason=f"the project's own module {module!r} could not be imported",
                subject=module,
                remediation="The module is missing, misnamed, or not on the import path.",
            )
        return FailureDiagnosis(
            category=FailureCategory.DEPENDENCY_FAILURE,
            reason=f"a required package is not installed: {module!r}",
            subject=module,
            code_executed=False,
            remediation=(
                f"Add {module!r} to the dependency manifest and install it into the "
                "project-local environment."
            ),
        )

    cannot_import = _CANNOT_IMPORT_NAME_RE.search(output)
    if cannot_import:
        return FailureDiagnosis(
            category=FailureCategory.CODE_FAILURE,
            reason=f"a name could not be imported: {cannot_import.group(1)!r}",
            subject=cannot_import.group(1),
            remediation="The definition is missing, renamed, or was dropped by an edit.",
        )

    # An assertion failure often mentions a type error in its diff, so a genuine test
    # failure takes precedence when both appear.
    if "assertionerror" not in lowered and any(marker in lowered for marker in _CODE_MARKERS):
        return FailureDiagnosis(
            category=FailureCategory.CODE_FAILURE,
            reason="the project's code failed to load or execute",
            remediation="Fix the syntax or name error before re-running verification.",
        )

    if any(marker in lowered for marker in _TEST_MARKERS):
        return FailureDiagnosis(
            category=FailureCategory.TEST_FAILURE,
            reason="the tests ran and at least one assertion failed",
            remediation="Correct the implementation so the assertion holds.",
        )

    if exit_code == 5:
        # pytest's "no tests collected". Nothing was verified, so nothing was proven -- and
        # ``code_executed=False`` keeps that fact attached to the result, so no later stage can
        # read this as evidence about the code.
        #
        # It is not an environment fault: pytest ran perfectly and found nothing to run. The
        # project simply has no tests for this change, which is missing work rather than a
        # broken machine. Classifying it as ENVIRONMENT_FAILURE escalated it to a human and
        # left a greenfield project permanently unverifiable -- every task failed, forever,
        # because the first one had nothing to test it.
        #
        # TEST_FAILURE is the honest label: the verification contract is unsatisfied, the
        # remedy is inside the agent's authority, and the repair loop can act on it. This does
        # not make an unverified change pass; it asks for the tests that would verify it.
        return FailureDiagnosis(
            category=FailureCategory.TEST_FAILURE,
            reason="the project has no tests, so this change is unverified",
            code_executed=False,
            remediation=(
                "Add a test file under tests/ that imports the changed module and asserts "
                "its documented behaviour."
            ),
        )

    return FailureDiagnosis(
        category=FailureCategory.TEST_FAILURE,
        reason=f"verification exited with code {exit_code}",
    )


def missing_modules(output: str) -> list[str]:
    """Every distinct module named in an import failure."""
    return sorted(
        {match.split(".")[0] for match in _MISSING_MODULE_RE.findall(output)}
    )
