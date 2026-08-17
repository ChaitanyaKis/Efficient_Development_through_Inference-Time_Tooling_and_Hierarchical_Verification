"""Deterministic quality and security checks.

M6's governing rule is that LLM judgement must never replace a deterministic check that could
answer the same question. This module is where that rule is cashed in: everything here is a
pure function over source text or an AST, so it is reproducible, fast, and cannot be talked out
of a finding.

The checks are AST-based wherever the question is structural. Grepping for ``eval`` finds it in
a comment and misses ``builtins.eval``; parsing does neither. Where a regex is used -- secret
literals, mainly -- it is because the question really is textual.

Every finding carries :class:`FindingOrigin.DETERMINISTIC`, which is what lets the adjudicator
treat it as blocking. A model's agreement adds nothing to these and its disagreement removes
nothing.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Callable, Iterator

from edith.quality.artifacts import (
    FindingOrigin,
    QualityFinding,
    ReviewEvidence,
)
from edith.schemas.common import Severity

#: Calls that execute arbitrary code or commands. The value is the severity to report.
_DANGEROUS_CALLS: dict[str, Severity] = {
    "eval": Severity.CRITICAL,
    "exec": Severity.CRITICAL,
    "compile": Severity.HIGH,
    "__import__": Severity.HIGH,
}

#: ``module.attribute`` calls that are unsafe regardless of how the module was imported.
_DANGEROUS_ATTRS: dict[tuple[str, str], Severity] = {
    ("os", "system"): Severity.CRITICAL,
    ("os", "popen"): Severity.CRITICAL,
    ("pickle", "loads"): Severity.HIGH,
    ("pickle", "load"): Severity.HIGH,
    ("yaml", "load"): Severity.HIGH,
    ("marshal", "loads"): Severity.HIGH,
    ("subprocess", "getoutput"): Severity.HIGH,
}

#: Secret-shaped assignments. Deliberately narrow: a name that *means* credential, assigned a
#: non-trivial literal. Broader patterns flood the report and get ignored, which is worse.
_SECRET_NAMES = re.compile(
    r"(?i)\b(password|passwd|secret|api_key|apikey|access_key|token|private_key|credential)\b"
)

#: A literal long enough to be a real credential rather than a placeholder or a flag.
_MIN_SECRET_LENGTH = 8

#: Placeholders that are obviously not credentials, so flagging them is pure noise.
_PLACEHOLDERS = frozenset(
    {
        "", "none", "null", "changeme", "your_password_here", "xxx", "todo",
        "example", "placeholder", "redacted", "dummy", "test", "fake",
        "<password>", "...", "secret", "password",
    }
)

#: Logging calls, for the "secret reaches the log" check.
_LOG_METHODS = frozenset(
    {"debug", "info", "warning", "warn", "error", "critical", "exception", "log"}
)


def _call_name(node: ast.Call) -> str:
    """The bare name of a called function, if it is a plain name."""
    if isinstance(node.func, ast.Name):
        return node.func.id
    return ""


def _attr_pair(node: ast.Call) -> tuple[str, str] | None:
    """``(module, attribute)`` for a ``module.attr()`` call."""
    func = node.func
    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
        return (func.value.id, func.attr)
    return None


def _evidence(source: str, path: str, line: int, detail: str) -> ReviewEvidence:
    return ReviewEvidence(source=source, detail=detail[:4000], file=path, line=line)


def _finding(
    *,
    category: str,
    severity: Severity,
    summary: str,
    path: str,
    line: int,
    detail: str,
    scanner: str,
    repairable: bool,
) -> QualityFinding:
    return QualityFinding(
        category=category,
        severity=severity,
        summary=summary,
        evidence=(_evidence(scanner, path, line, detail),),
        affected_files=(path,),
        origin=FindingOrigin.DETERMINISTIC,
        repairable=repairable,
        confidence=1.0,
    )


def scan_security(path: str, content: str) -> tuple[QualityFinding, ...]:
    """Find security defects in one Python file.

    Covers the M6 Part 4 items that are decidable from the source alone: command injection,
    unsafe deserialisation, secret literals, secrets reaching logs, path traversal, and
    disabled TLS verification. Authentication and authorization are *not* here, because
    whether an endpoint should require auth is not answerable from syntax -- that is left to
    the model reviewer, whose finding is advisory by construction.

    A file that does not parse yields a single CRITICAL finding rather than an empty result: a
    scanner that silently reports "clean" for input it could not read is worse than no scanner.
    """
    try:
        tree = ast.parse(content)
    except SyntaxError as exc:
        return (
            _finding(
                category="security-scan",
                severity=Severity.CRITICAL,
                summary="the file could not be parsed, so it was not security-scanned",
                path=path,
                line=max(1, exc.lineno or 1),
                detail=f"SyntaxError: {exc.msg}",
                scanner="ast.parse",
                repairable=True,
            ),
        )

    findings: list[QualityFinding] = []
    lines = content.splitlines()

    def line_text(number: int) -> str:
        return lines[number - 1].strip() if 1 <= number <= len(lines) else ""

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            findings.extend(_scan_call(node, path, line_text))
        elif isinstance(node, ast.Assign):
            findings.extend(_scan_assignment(node, path, line_text))

    findings.extend(_scan_secret_logging(tree, path, line_text))
    return tuple(findings)


def _scan_call(
    node: ast.Call, path: str, line_text: Callable[[int], str]
) -> Iterator[QualityFinding]:
    text = line_text(node.lineno)

    name = _call_name(node)
    if name in _DANGEROUS_CALLS:
        yield _finding(
            category="code-injection",
            severity=_DANGEROUS_CALLS[name],
            summary=f"{name}() executes arbitrary code",
            path=path,
            line=node.lineno,
            detail=text,
            scanner="ast:dangerous-call",
            repairable=True,
        )

    pair = _attr_pair(node)
    if pair in _DANGEROUS_ATTRS:
        module, attribute = pair
        category = (
            "command-injection"
            if module == "os" or attribute == "getoutput"
            else "unsafe-deserialization"
        )
        yield _finding(
            category=category,
            severity=_DANGEROUS_ATTRS[pair],
            summary=f"{module}.{attribute}() is unsafe on untrusted input",
            path=path,
            line=node.lineno,
            detail=text,
            scanner="ast:dangerous-attribute",
            repairable=True,
        )

    # shell=True turns any argument into shell syntax, which is the whole M1 argv rule.
    for keyword in node.keywords:
        if (
            keyword.arg == "shell"
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value is True
        ):
            yield _finding(
                category="command-injection",
                severity=Severity.CRITICAL,
                summary="shell=True passes the command through a shell",
                path=path,
                line=node.lineno,
                detail=text,
                scanner="ast:shell-true",
                repairable=True,
            )
        if (
            keyword.arg == "verify"
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value is False
        ):
            yield _finding(
                category="insecure-configuration",
                severity=Severity.HIGH,
                summary="TLS certificate verification is disabled",
                path=path,
                line=node.lineno,
                detail=text,
                scanner="ast:verify-false",
                repairable=True,
            )

    # Path traversal: a literal '..' segment reaching a filesystem call.
    if name == "open" or _attr_pair(node) in {("os", "remove"), ("shutil", "rmtree")}:
        for argument in node.args:
            if (
                isinstance(argument, ast.Constant)
                and isinstance(argument.value, str)
                and ".." in argument.value.replace("\\", "/").split("/")
            ):
                    yield _finding(
                        category="path-traversal",
                        severity=Severity.HIGH,
                        summary="a filesystem path escapes its directory with '..'",
                        path=path,
                        line=node.lineno,
                        detail=text,
                        scanner="ast:path-traversal",
                        repairable=True,
                    )


def _is_secret_literal(value: object) -> bool:
    """Whether a literal looks like a real credential rather than a placeholder."""
    if not isinstance(value, str):
        return False
    stripped = value.strip()
    if len(stripped) < _MIN_SECRET_LENGTH:
        return False
    if stripped.lower() in _PLACEHOLDERS:
        return False
    # An environment lookup or format placeholder is not a literal secret.
    return not (stripped.startswith("${") or stripped.startswith("{"))


def _scan_assignment(
    node: ast.Assign, path: str, line_text: Callable[[int], str]
) -> Iterator[QualityFinding]:
    if not isinstance(node.value, ast.Constant):
        return
    if not _is_secret_literal(node.value.value):
        return
    for target in node.targets:
        name = target.id if isinstance(target, ast.Name) else getattr(target, "attr", "")
        if name and _SECRET_NAMES.search(name):
            yield _finding(
                category="secret-exposure",
                severity=Severity.CRITICAL,
                summary=f"a credential appears to be hardcoded in {name}",
                path=path,
                line=node.lineno,
                detail=f"{name} = <redacted literal>",
                scanner="ast:hardcoded-secret",
                repairable=True,
            )


def _scan_secret_logging(
    tree: ast.AST, path: str, line_text: Callable[[int], str]
) -> Iterator[QualityFinding]:
    """A secret-named variable passed to a logging call.

    The redactor in :mod:`edith.observability.logging` protects EDITH's own logs; generated
    code has no such guarantee, so this is checked structurally.
    """
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        pair = _attr_pair(node)
        method = pair[1] if pair else ""
        if method not in _LOG_METHODS and not (
            isinstance(node.func, ast.Attribute) and node.func.attr in _LOG_METHODS
        ):
            continue
        arguments = [*node.args, *(keyword.value for keyword in node.keywords)]
        for argument in arguments:
            names = [
                inner.id
                for inner in ast.walk(argument)
                if isinstance(inner, ast.Name)
            ]
            names.extend(
                inner.attr
                for inner in ast.walk(argument)
                if isinstance(inner, ast.Attribute)
            )
            if any(_SECRET_NAMES.search(name) for name in names):
                yield _finding(
                    category="sensitive-logging",
                    severity=Severity.HIGH,
                    summary="a credential-named value is written to the log",
                    path=path,
                    line=node.lineno,
                    detail=line_text(node.lineno),
                    scanner="ast:secret-logging",
                    repairable=True,
                )
                return


def scan_review(path: str, content: str) -> tuple[QualityFinding, ...]:
    """Deterministic code-review checks: the structural subset of M6 Part 3.

    Only what an AST can decide. "Is this maintainable?" is left to the model, whose opinion is
    advisory; "does this swallow every exception?" is decided here and blocks if severe enough.
    """
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return ()

    findings: list[QualityFinding] = []
    lines = content.splitlines()

    def text(number: int) -> str:
        return lines[number - 1].strip() if 1 <= number <= len(lines) else ""

    for node in ast.walk(tree):
        # A bare or blanket except that does nothing hides failures, which CLAUDE.md forbids
        # outright ("Never hide failures").
        if isinstance(node, ast.ExceptHandler):
            swallowed = all(isinstance(item, ast.Pass) for item in node.body)
            blanket = node.type is None or (
                isinstance(node.type, ast.Name) and node.type.id in {"Exception", "BaseException"}
            )
            if swallowed and blanket:
                findings.append(
                    _finding(
                        category="error-handling",
                        severity=Severity.HIGH,
                        summary="an exception handler silently discards every error",
                        path=path,
                        line=node.lineno,
                        detail=text(node.lineno),
                        scanner="ast:swallowed-exception",
                        repairable=True,
                    )
                )
        # Unreachable code after an unconditional return.
        if isinstance(node, ast.FunctionDef):
            findings.extend(_scan_dead_code(node, path, text))

    return tuple(findings)


def _scan_dead_code(
    node: ast.FunctionDef, path: str, text: Callable[[int], str]
) -> Iterator[QualityFinding]:
    body = node.body
    for index, statement in enumerate(body[:-1]):
        if isinstance(statement, ast.Return | ast.Raise):
            following = body[index + 1]
            yield _finding(
                category="dead-code",
                severity=Severity.MEDIUM,
                summary=f"code after an unconditional return in {node.name}() is unreachable",
                path=path,
                line=following.lineno,
                detail=text(following.lineno),
                scanner="ast:dead-code",
                repairable=True,
            )
            return
