"""The scaffold gate: a generated test must agree with known-correct behaviour, or be discarded.

M8 measured what happens without this. Requirement-derived tests were 98% mechanically valid
and killed 12/12 mutants, yet only 4 of 12 suites passed a *known-correct* implementation. The
tests were not discriminating between right and wrong; they were failing nearly everything,
which kills mutants trivially. Turned loose on real tasks they blocked 32 of 36 runs.

The gate here is the cheapest possible correction. A test that asserts something a correct
implementation does not satisfy is, by definition, asserting the wrong thing -- and that is
decidable by execution, with no model involved:

    generated test + known-correct scaffold -> pytest -> pass? retain : discard

Two design choices matter.

**The gate is per test, not per suite.** M8 discarded or kept whole suites, so one wrong
assertion condemned three good ones. Running each test against the scaffold separately keeps
the survivors, which is the only way a partially-wrong generator can still contribute.

**The scaffold is data, never a special case.** :func:`gate_tests` takes the scaffold source as
an argument and knows nothing about which task it belongs to. There is no task id in this
module, and a test asserts that: a gate that recognised individual benchmark tasks would be
measuring its own hard-coding rather than a mechanism.

What the gate cannot do is establish that a surviving test is *useful*. A test that asserts
something trivially true passes any scaffold. Strength still has to be measured against a
known-incorrect implementation, which is a separate control and deliberately not part of the
gate -- a gate with access to the wrong implementation would be scoring itself.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from edith.observability.logging import get_logger
from edith.quality.testgen import GENERATED_TEST_DIR, GeneratedTest, TestProvenance

logger = get_logger(__name__)

#: Budget for one scaffold probe. A generated test that hangs is discarded, not waited on.
PROBE_TIMEOUT_SECONDS = 60.0


class GateOutcome(StrEnum):
    """Why a generated test was retained or discarded by the scaffold gate."""

    #: Ran against the scaffold and passed. May participate in verification.
    RETAINED = "RETAINED"
    #: Ran against the scaffold and failed: it asserts behaviour the requirement does not have.
    CONTRADICTS_SCAFFOLD = "CONTRADICTS_SCAFFOLD"
    #: Could not run at all -- import error, collection error, crash.
    DID_NOT_EXECUTE = "DID_NOT_EXECUTE"
    #: Exceeded the probe budget.
    TIMED_OUT = "TIMED_OUT"
    #: Never reached the gate; M8 validation had already rejected it.
    ALREADY_INVALID = "ALREADY_INVALID"


@dataclass(frozen=True)
class GatedTest:
    """One generated test and what the scaffold gate decided about it."""

    test: GeneratedTest
    outcome: GateOutcome
    detail: str = ""

    @property
    def retained(self) -> bool:
        return self.outcome is GateOutcome.RETAINED

    @property
    def authoritative(self) -> bool:
        """Only a test that survived both M8 validation and this gate may participate."""
        return self.retained and self.test.authoritative


@dataclass(frozen=True)
class SuiteVerdict:
    """The gate's decision over one requirement's generated tests."""

    gated: tuple[GatedTest, ...] = ()

    @property
    def retained(self) -> tuple[GeneratedTest, ...]:
        return tuple(item.test for item in self.gated if item.authoritative)

    @property
    def discarded(self) -> tuple[GatedTest, ...]:
        return tuple(item for item in self.gated if not item.authoritative)

    @property
    def usable(self) -> bool:
        """Whether anything survived. An empty suite must not block a coder."""
        return bool(self.retained)

    def count(self, outcome: GateOutcome) -> int:
        return sum(1 for item in self.gated if item.outcome is outcome)


def _probe_root(scaffold: str, module_path: str) -> Path:
    """Build a throwaway tree containing only the scaffold and the package markers."""
    root = Path(tempfile.mkdtemp(prefix="scaffold-"))
    target = root / module_path
    target.parent.mkdir(parents=True, exist_ok=True)
    # Package markers along the whole path, so the generated import resolves.
    current = root
    for part in Path(module_path).parent.parts:
        current = current / part
        (current / "__init__.py").write_text("", encoding="utf-8")
    target.write_text(scaffold, encoding="utf-8")
    (root / GENERATED_TEST_DIR).mkdir(parents=True, exist_ok=True)
    return root


def run_against_scaffold(
    test: GeneratedTest,
    *,
    scaffold: str,
    module_path: str,
    timeout_seconds: float = PROBE_TIMEOUT_SECONDS,
) -> tuple[GateOutcome, str]:
    """Execute one generated test against a known-correct implementation.

    The probe runs in its own temporary tree containing nothing but the scaffold, so a test
    cannot accidentally pass by importing the real project, and cannot reach anything belonging
    to the task it was generated for.
    """
    root = _probe_root(scaffold, module_path)
    try:
        path = root / GENERATED_TEST_DIR / "test_probe.py"
        path.write_text(test.source, encoding="utf-8")
        try:
            completed = subprocess.run(  # noqa: S603 - fixed argv, shell=False
                [sys.executable, "-m", "pytest", str(path), "-q", "-p", "no:cacheprovider"],
                cwd=str(root),
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            return (GateOutcome.TIMED_OUT, f"exceeded {timeout_seconds}s")
        except OSError as exc:
            return (GateOutcome.DID_NOT_EXECUTE, f"{type(exc).__name__}: {exc}")

        output = (completed.stdout + completed.stderr)[-600:]
        if completed.returncode == 0:
            return (GateOutcome.RETAINED, "")
        # pytest exits 2+ for collection/usage errors, 1 for genuine assertion failures. The
        # distinction matters: a test that could not run says nothing about the requirement,
        # while one that ran and failed contradicts a correct implementation.
        if completed.returncode >= 2 or "ERROR" in output:
            return (GateOutcome.DID_NOT_EXECUTE, output)
        return (GateOutcome.CONTRADICTS_SCAFFOLD, output)
    finally:
        shutil.rmtree(root, ignore_errors=True)


def gate_tests(
    tests: tuple[GeneratedTest, ...],
    *,
    scaffold: str,
    module_path: str,
    timeout_seconds: float = PROBE_TIMEOUT_SECONDS,
) -> SuiteVerdict:
    """Retain only the generated tests a known-correct implementation actually satisfies.

    Generic by construction: the scaffold arrives as an argument and this module contains no
    task identifiers, no requirement text, and no expected values. Swapping in a different
    benchmark changes nothing here.

    Tests that M8 validation already rejected are carried through as ``ALREADY_INVALID`` rather
    than dropped, so the two rejection stages stay separately measurable.
    """
    gated: list[GatedTest] = []
    for test in tests:
        if not test.authoritative:
            gated.append(
                GatedTest(
                    test=test,
                    outcome=GateOutcome.ALREADY_INVALID,
                    detail=test.reason.value if test.reason else "",
                )
            )
            continue
        outcome, detail = run_against_scaffold(
            test,
            scaffold=scaffold,
            module_path=module_path,
            timeout_seconds=timeout_seconds,
        )
        gated.append(GatedTest(test=test, outcome=outcome, detail=detail))

    verdict = SuiteVerdict(gated=tuple(gated))
    logger.info(
        "testgate.decided",
        total=len(gated),
        retained=len(verdict.retained),
        contradicts=verdict.count(GateOutcome.CONTRADICTS_SCAFFOLD),
        did_not_execute=verdict.count(GateOutcome.DID_NOT_EXECUTE),
        timed_out=verdict.count(GateOutcome.TIMED_OUT),
    )
    return verdict


def provenance_of(verdict: SuiteVerdict, test: GeneratedTest) -> TestProvenance:
    """The provenance a test carries after gating.

    A test discarded by the gate is advisory whatever it was before: it survived syntax and
    shape checks, and then disagreed with known-correct behaviour, which is a stronger reason to
    distrust it than anything M8 could detect.
    """
    for item in verdict.gated:
        if item.test is test:
            return (
                TestProvenance.REQUIREMENT_DERIVED_TEST
                if item.authoritative
                else TestProvenance.MODEL_ADVISORY_TEST
            )
    return TestProvenance.MODEL_ADVISORY_TEST
