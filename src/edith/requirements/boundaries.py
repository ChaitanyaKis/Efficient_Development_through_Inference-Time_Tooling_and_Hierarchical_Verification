"""Deterministic boundary detection: finding the thresholds a requirement leaves implicit.

Every false PASS surviving M7 through M9 was the same defect. BIZ-003 says a fee applies when
a payment is "more than 3 days late", and the model implemented ``days >= 3``. Three
milestones of test generation could not catch it, because the generated tests inherited the
same misreading as the code.

The insight M9 ended on is that this is not a testing problem. ``more than 3`` and ``3 or
more`` differ by one character in the operator and by one word in English, and the difference
is decidable *lexically* -- no model required, no implementation needed, before a line is
written.

So this module reads requirement text and nothing else. It has no access to source, to tests,
or to the model's opinion, and it runs before implementation. What it produces is evidence:
the number, the unit, the operator the words actually license, and the neighbouring cases that
distinguish a correct implementation from an off-by-one.

**Uncertainty is a result, not a failure.** "after 3 days" genuinely does not specify whether
day three counts. The detector returns ``CLARIFICATION_REQUIRED`` rather than guessing,
because a guess here is exactly the defect being hunted. M10's rule: false uncertainty is
safer than silently choosing the wrong operator.

**A number is not a threshold.** "Version 3 of the API" and "store the value 500 as the
default" contain numbers and no boundary at all. Detection therefore keys on *comparison
language* adjacent to a quantity, never on the quantity alone -- a detector that saw thresholds
everywhere would be worse than none, since every requirement would demand clarification.
"""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from enum import StrEnum

from pydantic import Field

from edith.schemas.common import EdithModel


class Operator(StrEnum):
    """The comparison a requirement's wording licenses."""

    GT = ">"
    GTE = ">="
    LT = "<"
    LTE = "<="
    EQ = "=="
    #: Two-sided; carried by :class:`BoundaryCondition.upper`.
    BETWEEN = "between"

    @property
    def inclusive(self) -> bool:
        """Whether the stated quantity itself satisfies the condition."""
        return self in {Operator.GTE, Operator.LTE, Operator.EQ, Operator.BETWEEN}


class BoundaryStatus(StrEnum):
    """How confidently the boundary was determined."""

    #: The wording licenses exactly one operator. Safe to implement.
    EXPLICIT = "EXPLICIT"
    #: Comparison language is present but does not fix the operator. Must not be guessed.
    CLARIFICATION_REQUIRED = "CLARIFICATION_REQUIRED"
    #: A quantity appears with comparison-like wording the detector does not model.
    BOUNDARY_UNCERTAIN = "BOUNDARY_UNCERTAIN"
    #: A human decided the operator; their decision is authoritative.
    RESOLVED = "RESOLVED"

    @property
    def implementable(self) -> bool:
        """Whether a coder may proceed on this boundary."""
        return self in {BoundaryStatus.EXPLICIT, BoundaryStatus.RESOLVED}


#: Phrases that fix an operator unambiguously, longest first so "no more than" beats "more
#: than". Every entry is a phrase a requirement author chose deliberately; the mapping is the
#: whole point of the module and deliberately conservative.
_EXPLICIT_PHRASES: tuple[tuple[str, Operator], ...] = (
    ("no less than", Operator.GTE),
    ("not less than", Operator.GTE),
    ("no more than", Operator.LTE),
    ("not more than", Operator.LTE),
    ("greater than or equal to", Operator.GTE),
    ("less than or equal to", Operator.LTE),
    ("greater than", Operator.GT),
    ("less than", Operator.LT),
    ("more than", Operator.GT),
    ("fewer than", Operator.LT),
    ("at least", Operator.GTE),
    ("at most", Operator.LTE),
    ("minimum of", Operator.GTE),
    ("maximum of", Operator.LTE),
    ("or more", Operator.GTE),
    ("or greater", Operator.GTE),
    ("or higher", Operator.GTE),
    ("or less", Operator.LTE),
    ("or fewer", Operator.LTE),
    ("or lower", Operator.LTE),
    ("up to", Operator.LTE),
    ("over", Operator.GT),
    ("under", Operator.LT),
    ("above", Operator.GT),
    ("below", Operator.LT),
    ("beyond", Operator.GT),
)

#: Phrases that signal a threshold without fixing which side of it counts. These are the
#: genuinely ambiguous cases -- "after 3 days" is the shape that produced BIZ-003 -- and the
#: correct output is a question, not a decision.
_AMBIGUOUS_PHRASES: tuple[str, ...] = (
    "after",
    "before",
    "within",
    "past",
    "by",
    "from",
    "once",
    "reaches",
    "hits",
    "exceeds the limit of",
)

#: Two-sided forms.
_RANGE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"between\s+(?P<low>\d+(?:\.\d+)?)\s*(?P<unit>[a-z%₹$€£]*)\s*and\s+(?P<high>\d+(?:\.\d+)?)",
        re.IGNORECASE,
    ),
    re.compile(
        r"from\s+(?P<low>\d+(?:\.\d+)?)\s*(?P<unit>[a-z%₹$€£]*)\s*to\s+(?P<high>\d+(?:\.\d+)?)",
        re.IGNORECASE,
    ),
)

#: A quantity: an optional currency mark, a number, an optional unit word or symbol.
_QUANTITY = re.compile(
    r"(?P<currency>[₹$€£])?\s*(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>%|[a-zA-Z]+)?"
)

#: Words that can follow a number but are never units. Extracting one of these means the match
#: was not a measured quantity, which is a strong signal the "threshold" is spurious.
_NON_UNITS = frozenset(
    {
        "for", "and", "or", "of", "in", "on", "at", "to", "from", "with", "per",
        "the", "a", "an", "as", "is", "are", "be", "when", "if", "then", "that",
    }
)

#: Units whose domain is continuous, where neighbouring integers are not meaningful.
_CONTINUOUS_UNITS = frozenset(
    {"kg", "g", "lb", "kb", "mb", "gb", "seconds", "second", "ms", "percent", "%"}
)

#: Words that mark a number as an identifier or a label rather than a threshold. These are the
#: false-positive controls: "version 3", "protocol 2", "id 1001" are not business rules.
_LABEL_WORDS = frozenset(
    {
        "version", "v", "id", "identifier", "number", "no", "code", "port",
        "protocol", "revision", "release", "year", "status", "http", "error",
        "index", "level", "step", "phase", "chapter", "page", "line", "column",
    }
)


class BoundaryCase(EdithModel):
    """One neighbouring value and whether the condition holds there.

    These are what actually catch an off-by-one. A requirement saying ``> 3`` and an
    implementation saying ``>= 3`` agree on every input except one, so the case at the
    boundary is the entire test.
    """

    value: str = Field(min_length=1, max_length=40)
    satisfies: bool
    note: str = Field(default="", max_length=200)


class BoundaryCondition(EdithModel):
    """One threshold found in a requirement, with the evidence that licensed it.

    ``boundary_id`` and ``status`` are system-assigned. A model may propose an interpretation
    (see :func:`check_proposal`) but may never write these fields, because the whole purpose of
    the layer is that lexical evidence outranks the model's reading of the same sentence.
    """

    boundary_id: str = Field(min_length=1, max_length=80)
    requirement_id: str = Field(min_length=1, max_length=120)
    quantity: str = Field(min_length=1, max_length=40)
    unit: str = Field(default="", max_length=40)
    operator: Operator | None = None
    #: Present only for two-sided conditions.
    upper: str = Field(default="", max_length=40)
    status: BoundaryStatus = BoundaryStatus.BOUNDARY_UNCERTAIN
    #: The exact phrase that produced this reading. Quoted from the requirement, never paraphrased.
    evidence: str = Field(min_length=1, max_length=400)
    #: The question to put to a human when the wording does not fix the operator.
    question: str = Field(default="", max_length=400)
    cases: tuple[BoundaryCase, ...] = Field(default=(), max_length=8)

    @property
    def blocking(self) -> bool:
        """Whether implementation must not proceed until this is resolved."""
        return not self.status.implementable

    def condition(self, variable: str = "value") -> str:
        """The condition as code, for the plan. Empty when the boundary is unresolved."""
        if not self.status.implementable or self.operator is None:
            return ""
        if self.operator is Operator.BETWEEN:
            return f"{self.quantity} <= {variable} <= {self.upper}"
        return f"{variable} {self.operator.value} {self.quantity}"


def _is_integer(value: str) -> bool:
    try:
        return Decimal(value) == Decimal(value).to_integral_value()
    except InvalidOperation:
        return False


def _continuous(unit: str) -> bool:
    return unit.lower() in _CONTINUOUS_UNITS


def expand_cases(
    quantity: str, operator: Operator, unit: str = "", upper: str = ""
) -> tuple[BoundaryCase, ...]:
    """Derive the neighbouring cases that distinguish this operator from its off-by-one twin.

    Only for integer domains. Neighbouring whole numbers around ``10 kg`` are not meaningful --
    the interesting nearby value is 9.99, not 9 -- so a continuous unit yields the boundary
    itself and nothing invented around it.
    """
    if operator is Operator.BETWEEN:
        return (
            BoundaryCase(value=quantity, satisfies=True, note="lower bound, inclusive"),
            BoundaryCase(value=upper, satisfies=True, note="upper bound, inclusive"),
        )

    def holds(candidate: Decimal, threshold: Decimal) -> bool:
        return {
            Operator.GT: candidate > threshold,
            Operator.GTE: candidate >= threshold,
            Operator.LT: candidate < threshold,
            Operator.LTE: candidate <= threshold,
            Operator.EQ: candidate == threshold,
        }[operator]

    try:
        threshold = Decimal(quantity)
    except InvalidOperation:
        return ()

    if _continuous(unit) or not _is_integer(quantity):
        return (
            BoundaryCase(
                value=quantity,
                satisfies=holds(threshold, threshold),
                note="at the boundary; neighbours omitted for a continuous quantity",
            ),
        )

    values = [threshold - 1, threshold, threshold + 1]
    return tuple(
        BoundaryCase(
            value=str(value),
            satisfies=holds(value, threshold),
            note="at the boundary" if value == threshold else "",
        )
        for value in values
    )


def _quantity_near(text: str, index: int, window: int = 40) -> tuple[str, str] | None:
    """The first quantity following a phrase, with its unit."""
    match = _QUANTITY.search(text, index, index + window)
    if match is None:
        return None
    unit = (match.group("unit") or "").strip()
    if match.group("currency"):
        unit = match.group("currency")
    if unit.lower() in _LABEL_WORDS:
        return None
    if unit.lower() in _NON_UNITS:
        unit = ""
    return (match.group("value"), unit)


def _label_context(text: str, position: int) -> bool:
    """Whether a number is preceded by a word marking it as an identifier, not a threshold."""
    prefix = text[max(0, position - 24) : position].lower()
    words = re.findall(r"[a-z]+", prefix)
    return bool(words) and words[-1] in _LABEL_WORDS


def detect_boundaries(
    requirement: str, *, requirement_id: str
) -> tuple[BoundaryCondition, ...]:
    """Find every threshold in one requirement, deterministically.

    Reads the requirement and nothing else: no implementation, no tests, no model. Called
    before the coder runs, so its output can gate the plan rather than explain a failure.
    """
    text = requirement.strip()
    lowered = text.lower()
    found: list[BoundaryCondition] = []
    consumed: list[tuple[int, int]] = []

    def overlaps(start: int, end: int) -> bool:
        return any(start < seen_end and end > seen_start for seen_start, seen_end in consumed)

    for pattern in _RANGE_PATTERNS:
        for match in pattern.finditer(text):
            low, high = match.group("low"), match.group("high")
            unit = (match.group("unit") or "").strip()
            consumed.append(match.span())
            found.append(
                BoundaryCondition(
                    boundary_id=f"{requirement_id}-B{len(found) + 1}",
                    requirement_id=requirement_id,
                    quantity=low,
                    upper=high,
                    unit=unit,
                    operator=Operator.BETWEEN,
                    status=BoundaryStatus.EXPLICIT,
                    evidence=match.group(0),
                    cases=expand_cases(low, Operator.BETWEEN, unit, high),
                )
            )

    for phrase, operator in _EXPLICIT_PHRASES:
        start = 0
        while True:
            index = lowered.find(phrase, start)
            if index == -1:
                break
            start = index + len(phrase)
            if overlaps(index, index + len(phrase)):
                continue
            # "or more" trails its quantity; every other phrase leads it.
            if phrase.startswith("or "):
                head = text[max(0, index - 40) : index]
                matches = list(_QUANTITY.finditer(head))
                quantity = None
                if matches:
                    last = matches[-1]
                    unit = (last.group("unit") or "").strip()
                    if last.group("currency"):
                        unit = last.group("currency")
                    if unit.lower() not in _LABEL_WORDS:
                        quantity = (last.group("value"), unit)
            else:
                quantity = _quantity_near(text, start)
            if quantity is None:
                continue
            value, unit = quantity
            consumed.append((index, start))
            found.append(
                BoundaryCondition(
                    boundary_id=f"{requirement_id}-B{len(found) + 1}",
                    requirement_id=requirement_id,
                    quantity=value,
                    unit=unit,
                    operator=operator,
                    status=BoundaryStatus.EXPLICIT,
                    evidence=text[max(0, index - 10) : start + 24].strip(),
                    cases=expand_cases(value, operator, unit),
                )
            )

    for phrase in _AMBIGUOUS_PHRASES:
        # The quantity must follow the phrase directly, allowing only an article. "after 3
        # days" is a boundary; "after that the fee is 1.5 per day" is not, and a loose window
        # would turn every later number in a sentence into a spurious clarification request.
        pattern = re.compile(
            rf"\b{re.escape(phrase)}\s+(?:the\s+|a\s+|an\s+)?(?=[₹$€£]?\s*\d)",
            re.IGNORECASE,
        )
        for match in pattern.finditer(text):
            index = match.start()
            if overlaps(index, match.end()):
                continue
            quantity = _quantity_near(text, match.end(), window=24)
            if quantity is None:
                continue
            value, unit = quantity
            if _label_context(text, text.lower().find(value, match.end())):
                continue
            consumed.append((index, match.end()))
            found.append(
                BoundaryCondition(
                    boundary_id=f"{requirement_id}-B{len(found) + 1}",
                    requirement_id=requirement_id,
                    quantity=value,
                    unit=unit,
                    operator=None,
                    status=BoundaryStatus.CLARIFICATION_REQUIRED,
                    evidence=text[max(0, index - 10) : match.end() + 24].strip(),
                    question=(
                        f'"{phrase} {value}{(" " + unit) if unit else ""}" does not say whether '
                        f"{value} itself counts. Is the condition > {value} or >= {value}?"
                    ),
                )
            )

    return tuple(found)


def unresolved(conditions: tuple[BoundaryCondition, ...]) -> tuple[BoundaryCondition, ...]:
    """Boundaries that must be settled before implementation may proceed."""
    return tuple(item for item in conditions if item.blocking)


def check_proposal(
    condition: BoundaryCondition, proposed: Operator
) -> tuple[bool, str]:
    """Judge a model's proposed operator against the lexical evidence.

    The model may propose; it may not decide. A proposal contradicting an ``EXPLICIT`` reading
    is rejected outright -- "more than 3" cannot become ``>= 3`` because a model preferred it.
    Where the wording is genuinely ambiguous the proposal is *still* rejected, because guessing
    is the defect: only a human resolution settles those.
    """
    if condition.status is BoundaryStatus.EXPLICIT:
        if proposed is condition.operator:
            return (True, "proposal matches the lexical evidence")
        return (
            False,
            f"proposal {proposed.value} contradicts "
            f"{condition.operator.value if condition.operator else '?'} "
            f"licensed by {condition.evidence!r}",
        )
    if condition.status is BoundaryStatus.RESOLVED:
        if proposed is condition.operator:
            return (True, "proposal matches the human resolution")
        return (False, "proposal contradicts a human resolution")
    return (
        False,
        "the wording does not fix an operator; a human resolution is required",
    )


def resolve(
    condition: BoundaryCondition, *, operator: Operator, decided_by: str
) -> BoundaryCondition:
    """Record a human decision about an ambiguous boundary.

    Returns a new condition rather than editing the original: the requirement text and the
    detector's reading of it are both preserved, and the resolution is an additional artifact
    layered on top. Nothing about the source requirement is silently rewritten.
    """
    if not decided_by.strip():
        raise ValueError("a resolution must record who made it")
    return condition.model_copy(
        update={
            "operator": operator,
            "status": BoundaryStatus.RESOLVED,
            "question": "",
            "evidence": f"{condition.evidence} [resolved by {decided_by}: {operator.value}]",
            "cases": expand_cases(condition.quantity, operator, condition.unit),
        }
    )


def render_for_plan(conditions: tuple[BoundaryCondition, ...]) -> str:
    """State the resolved boundaries for the implementer, in the plan.

    This is the intervention M10 measures: the coder is told the operator and the neighbouring
    cases explicitly, instead of being left to infer them from prose that three milestones of
    evidence say it infers wrongly.
    """
    lines: list[str] = []
    for item in conditions:
        if not item.status.implementable:
            continue
        lines.append(f"BOUNDARY {item.boundary_id}: {item.condition()}")
        for case in item.cases:
            verdict = "satisfies the condition" if case.satisfies else "does not satisfy it"
            unit = f" {item.unit}" if item.unit else ""
            lines.append(f"  - {case.value}{unit} {verdict}")
    return "\n".join(lines)
