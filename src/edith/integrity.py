"""Test integrity: detecting when an agent moves the goalposts.

M2 shipped a verification architecture that reduced to *code + current test results*. That
is not sufficient, and it failed in exactly the way it had to: asked to repair a broken
``subtract``, the coding agent rewrote ``assert subtract(5, 3) == 2`` into ``== 8``, the
suite went green, and Edith's own Critic returned PASS. The external harness caught it.

The root cause was not a bad prompt. It was that **the only defense was model judgement**.
There was no deterministic rule anywhere that said "an assertion's expected value changed".

This module supplies that rule. It compares each test file against its baseline with the
AST -- not a regex, and not an LLM -- and reports what changed structurally: tests removed,
assertions dropped, expected values altered, skips introduced. The orchestrator treats an
unexplained weakening as a verification failure regardless of what the suite or the Critic
says.

Adding tests remains entirely legitimate. Only *weakening existing ones* is blocked, and
even then it is permitted when the agent declares it -- so changing a test that really was
wrong stays possible, but never silent.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from pydantic import Field

from edith.observability.logging import get_logger
from edith.schemas.common import EdithModel, Severity

logger = get_logger(__name__)

#: Directory and filename markers that identify a test file.
_TEST_DIR_MARKERS = ("tests/", "test/", "spec/", "__tests__/")
_TEST_NAME_PREFIXES = ("test_",)
_TEST_NAME_SUFFIXES = (
    "_test.py",
    "_test.go",
    ".test.js",
    ".test.jsx",
    ".test.ts",
    ".test.tsx",
    ".spec.js",
    ".spec.jsx",
    ".spec.ts",
    ".spec.tsx",
)

_FIXTURE_MARKERS = ("fixtures/", "testdata/", "conftest.py")
_CONFIG_SUFFIXES = (".toml", ".cfg", ".ini", ".yaml", ".yml", ".json")
_DOC_SUFFIXES = (".md", ".rst", ".txt")

#: Decorators and calls that disable a test.
_SKIP_MARKERS = ("skip", "skipif", "xfail")


class FileKind(StrEnum):
    """What a changed path represents.

    The Judge must not treat "the implementation changed" and "the test changed" as the
    same event, so classification happens before any judgement.
    """

    SOURCE = "SOURCE"
    TEST = "TEST"
    FIXTURE = "FIXTURE"
    CONFIG = "CONFIG"
    DOC = "DOC"
    UNKNOWN = "UNKNOWN"


def classify_path(path: str) -> FileKind:
    """Classify a repository-relative path."""
    lowered = path.replace("\\", "/").lower()
    name = lowered.rsplit("/", 1)[-1]

    if any(marker in f"/{lowered}" for marker in _FIXTURE_MARKERS) and not name.startswith(
        _TEST_NAME_PREFIXES
    ):
        return FileKind.FIXTURE
    if (
        name.startswith(_TEST_NAME_PREFIXES)
        or lowered.endswith(_TEST_NAME_SUFFIXES)
        or any(marker in f"/{lowered}" for marker in _TEST_DIR_MARKERS)
    ):
        return FileKind.TEST
    if lowered.endswith(_DOC_SUFFIXES):
        return FileKind.DOC
    if lowered.endswith(_CONFIG_SUFFIXES):
        return FileKind.CONFIG
    if "." in name:
        return FileKind.SOURCE
    return FileKind.UNKNOWN


@dataclass(frozen=True)
class TestFunction:
    """A single test and the shape of its assertions."""

    name: str
    #: ``ast.dump`` of each assertion's expression, so a changed expected value shows up as
    #: a removed entry rather than being missed by a textual comparison.
    assertions: tuple[str, ...]
    #: Human-readable source of each assertion, for evidence.
    assertion_source: tuple[str, ...]
    skipped: bool = False

    @property
    def assertion_count(self) -> int:
        """How many assertions this test makes."""
        return len(self.assertions)


def _is_skip_decorator(node: ast.expr) -> bool:
    """Whether a decorator disables the test it is attached to."""
    parts: list[str] = []
    current: ast.expr | None = node.func if isinstance(node, ast.Call) else node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
    return any(part.lower() in _SKIP_MARKERS for part in parts)


def _contains_skip_call(node: ast.AST) -> bool:
    """Whether a test body calls ``pytest.skip()`` and friends."""
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            function = child.func
            if isinstance(function, ast.Attribute) and function.attr.lower() in _SKIP_MARKERS:
                return True
            if isinstance(function, ast.Name) and function.id.lower() in _SKIP_MARKERS:
                return True
    return False


def extract_tests(content: str) -> dict[str, TestFunction]:
    """Extract test functions and their assertions from Python source.

    Returns an empty mapping for unparseable content: a syntax error is a separate problem,
    already caught by the coder's syntax gate, and guessing here would produce noise.
    """
    try:
        module = ast.parse(content)
    except (SyntaxError, ValueError):
        return {}

    lines = content.split("\n")
    found: dict[str, TestFunction] = {}

    def visit(nodes: list[ast.stmt], prefix: str = "") -> None:
        for node in nodes:
            if isinstance(node, ast.ClassDef):
                visit(node.body, f"{prefix}{node.name}.")
                continue
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if not node.name.startswith("test"):
                continue

            dumps: list[str] = []
            sources: list[str] = []
            for child in ast.walk(node):
                if isinstance(child, ast.Assert):
                    dumps.append(ast.dump(child.test))
                    start = (child.lineno or 1) - 1
                    end = child.end_lineno or child.lineno or 1
                    sources.append(" ".join(part.strip() for part in lines[start:end]))

            skipped = any(_is_skip_decorator(item) for item in node.decorator_list) or (
                _contains_skip_call(node)
            )
            found[f"{prefix}{node.name}"] = TestFunction(
                name=f"{prefix}{node.name}",
                assertions=tuple(dumps),
                assertion_source=tuple(sources),
                skipped=skipped,
            )

    visit(module.body)
    return found


class IntegrityFinding(EdithModel):
    """One structural change to a test file."""

    severity: Severity
    kind: str
    path: str
    detail: str
    evidence: str = ""

    @property
    def blocking(self) -> bool:
        """Whether this finding should fail verification on its own."""
        return self.severity in {Severity.HIGH, Severity.CRITICAL}


class IntegrityReport(EdithModel):
    """The result of comparing changed files against their baseline."""

    findings: list[IntegrityFinding] = Field(default_factory=list)
    source_files_changed: list[str] = Field(default_factory=list)
    test_files_changed: list[str] = Field(default_factory=list)
    tests_added: list[str] = Field(default_factory=list)
    tests_removed: list[str] = Field(default_factory=list)
    assertions_removed: int = 0
    skips_added: list[str] = Field(default_factory=list)
    #: Set when the agent declared and justified its test changes.
    justification: str = ""
    #: True when the baseline could not be established, so nothing was actually compared.
    baseline_unavailable: bool = False

    @property
    def blocking_findings(self) -> list[IntegrityFinding]:
        """Findings severe enough to fail verification."""
        return [finding for finding in self.findings if finding.blocking]

    @property
    def tampered(self) -> bool:
        """Whether existing tests were weakened without justification."""
        return bool(self.blocking_findings) and not self.justification.strip()

    @property
    def tests_were_modified(self) -> bool:
        """Whether any test file changed at all, weakened or not."""
        return bool(self.test_files_changed)

    def summary(self) -> str:
        """A compact, model-readable description of what changed."""
        if self.baseline_unavailable:
            return "test integrity: NOT CHECKED (no baseline was available)"
        if not self.findings:
            scope = (
                f"{len(self.source_files_changed)} source file(s) changed; "
                f"existing tests unchanged"
            )
            return f"test integrity: OK ({scope})"

        lines = [f"test integrity: {len(self.findings)} finding(s)"]
        for finding in self.findings:
            lines.append(f"  [{finding.severity}] {finding.path}: {finding.detail}")
            if finding.evidence:
                lines.append(f"      evidence: {finding.evidence}")
        return "\n".join(lines)


@dataclass
class _Delta:
    """Working state while comparing one file."""

    path: str
    findings: list[IntegrityFinding] = field(default_factory=list)
    tests_added: list[str] = field(default_factory=list)
    tests_removed: list[str] = field(default_factory=list)
    assertions_removed: int = 0
    skips_added: list[str] = field(default_factory=list)


def compare_test_file(path: str, baseline: str, current: str) -> _Delta:
    """Compare one test file against its baseline, structurally.

    Detects, in order of severity: a test that disappeared, a test that became skipped, and
    a test that kept its name but lost or altered assertions. The last case is the one that
    matters most -- it is what "change the expected value to match the bug" looks like.
    """
    delta = _Delta(path=path)
    before = extract_tests(baseline)
    after = extract_tests(current)

    for name in sorted(set(before) - set(after)):
        delta.tests_removed.append(name)
        delta.findings.append(
            IntegrityFinding(
                severity=Severity.CRITICAL,
                kind="test_removed",
                path=path,
                detail=f"test {name!r} was deleted",
                evidence=f"it previously made {before[name].assertion_count} assertion(s)",
            )
        )

    for name in sorted(set(after) - set(before)):
        delta.tests_added.append(name)

    for name in sorted(set(before) & set(after)):
        original, updated = before[name], after[name]

        if updated.skipped and not original.skipped:
            delta.skips_added.append(name)
            delta.findings.append(
                IntegrityFinding(
                    severity=Severity.HIGH,
                    kind="test_skipped",
                    path=path,
                    detail=f"test {name!r} was marked as skipped",
                    evidence="a skipped test cannot fail, so it proves nothing",
                )
            )

        # Assertions present before but not after: either deleted outright, or altered --
        # a changed expected value produces a different AST and so appears as a removal.
        lost = [
            index
            for index, dump in enumerate(original.assertions)
            if dump not in set(updated.assertions)
        ]
        if not lost:
            continue

        delta.assertions_removed += len(lost)
        weakened = updated.assertion_count < original.assertion_count
        examples = "; ".join(original.assertion_source[index] for index in lost[:3])
        replacement = (
            "; ".join(
                source
                for source, dump in zip(
                    updated.assertion_source, updated.assertions, strict=False
                )
                if dump not in set(original.assertions)
            )[:200]
            or "(nothing equivalent was added)"
        )
        delta.findings.append(
            IntegrityFinding(
                severity=Severity.HIGH,
                kind="assertion_removed" if weakened else "assertion_changed",
                path=path,
                detail=(
                    f"test {name!r} lost {len(lost)} assertion(s)"
                    if weakened
                    else f"test {name!r} had {len(lost)} assertion(s) altered"
                ),
                evidence=f"was: {examples[:200]} | now: {replacement}",
            )
        )

    return delta


def build_report(
    baselines: dict[str, str],
    current: dict[str, str],
    changed_paths: list[str],
    *,
    justification: str = "",
    baseline_unavailable: bool = False,
) -> IntegrityReport:
    """Assemble an :class:`IntegrityReport` for a set of changed paths.

    Args:
        baselines: path -> content at the baseline ref. A missing entry means the file is
            new, which is never a weakening.
        current: path -> content now. A missing entry means the file was deleted.
        changed_paths: every path reported as changed.
        justification: the agent's declared reason for touching tests, if any.
        baseline_unavailable: set when no baseline could be established at all.
    """
    report = IntegrityReport(
        justification=justification, baseline_unavailable=baseline_unavailable
    )

    for path in sorted(changed_paths):
        kind = classify_path(path)
        if kind is FileKind.SOURCE:
            report.source_files_changed.append(path)
            continue
        if kind is not FileKind.TEST:
            continue

        report.test_files_changed.append(path)
        baseline = baselines.get(path)
        if baseline is None:
            # A brand-new test file: adding tests is always legitimate.
            continue

        if path not in current:
            report.findings.append(
                IntegrityFinding(
                    severity=Severity.CRITICAL,
                    kind="test_file_deleted",
                    path=path,
                    detail="an existing test file was deleted",
                    evidence="deleting the tests cannot make the code correct",
                )
            )
            continue

        delta = compare_test_file(path, baseline, current[path])
        report.findings.extend(delta.findings)
        report.tests_added.extend(delta.tests_added)
        report.tests_removed.extend(delta.tests_removed)
        report.assertions_removed += delta.assertions_removed
        report.skips_added.extend(delta.skips_added)

    if report.findings:
        logger.warning(
            "integrity.findings",
            count=len(report.findings),
            tampered=report.tampered,
            paths=report.test_files_changed,
        )
    return report


#: Directories that never contain the project's own tests.
_NON_TEST_DIRS = frozenset(
    {".git", ".venv", "venv", "__pycache__", ".pytest_cache", ".edith", "node_modules"}
)


def _module_paths_for(changed: tuple[str, ...]) -> set[str]:
    """The importable names a changed Python file could be referred to by.

    A test may import ``src.backend.mathops``, ``backend.mathops`` or plain ``mathops``
    depending on the project's layout, so any of them counts as exercising the change.
    """
    names: set[str] = set()
    for relative in changed:
        posix = relative.replace("\\", "/")
        if not posix.endswith(".py"):
            continue
        parts = posix[:-3].split("/")
        if not parts:
            continue
        names.add(parts[-1])
        for start in range(len(parts)):
            names.add(".".join(parts[start:]))
    return names


def tests_exercise_changes(
    project_root: Path, changed: tuple[str, ...]
) -> str | None:
    """Return a reason when the test suite never imports what the task changed.

    Green tests are only evidence if they ran against the change. A suite that passes without
    importing the changed module proves the suite works, not the code -- which is the vacuous
    verification M5 caught in its own benchmark and M8 caught in generated tests, arriving
    here by a third route: a coder that writes ``assert 1 + 1 == 2`` in a file named after the
    module it never imports.

    Deterministic and import-based: no model, no execution, no judgement. It answers only
    "was the changed code reachable from the tests", and stays silent when the task changed no
    Python source or when the project has no tests to inspect.
    """
    targets = _module_paths_for(
        tuple(item for item in changed if not _is_test_path(item))
    )
    if not targets:
        return None

    test_sources: list[str] = []
    for path in project_root.rglob("*.py"):
        if any(part in _NON_TEST_DIRS for part in path.parts):
            continue
        if not _is_test_path(str(path.relative_to(project_root))):
            continue
        try:
            test_sources.append(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError):
            continue
    if not test_sources:
        return None

    imported: set[str] = set()
    for source in test_sources:
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
                imported.add(node.module.split(".")[-1])
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    imported.add(alias.name)
                    imported.add(alias.name.split(".")[-1])

    if imported & targets:
        return None
    changed_modules = ", ".join(sorted({t for t in targets if "." not in t})) or "the change"
    return (
        f"the tests passed but never import {changed_modules}, so they verify nothing about "
        f"this change. Import the changed module in a test and assert its behaviour."
    )


def _is_test_path(relative: str) -> bool:
    """Whether a repository-relative path is one of the project's tests."""
    posix = relative.replace("\\", "/")
    name = posix.rsplit("/", 1)[-1]
    return (
        posix.startswith("tests/")
        or "/tests/" in posix
        or name.startswith("test_")
        or name.endswith("_test.py")
    )
