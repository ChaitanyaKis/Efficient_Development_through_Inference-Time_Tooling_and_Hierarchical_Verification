"""M10: a labelled corpus for boundary detection, with human-authored ground truth.

Three kinds of entry, and the third is the one that keeps the detector honest:

``TRUE_BOUNDARY``   the wording licenses exactly one operator
``AMBIGUOUS``       comparison language without a decidable operator; must ask, never guess
``NO_BOUNDARY``     no threshold at all, including sentences full of numbers

The ``NO_BOUNDARY`` set exists because a detector that sees a threshold in every number is
useless: every requirement would demand clarification and the layer would be ignored. Version
numbers, identifiers, ports, years and default values all appear here deliberately.

Every label is written by hand. The detector never contributed to them.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from edith.requirements.boundaries import Operator


class Label(StrEnum):
    TRUE_BOUNDARY = "TRUE_BOUNDARY"
    AMBIGUOUS = "AMBIGUOUS"
    NO_BOUNDARY = "NO_BOUNDARY"


class Kind(StrEnum):
    """What sort of threshold, for per-category reporting."""

    DURATION = "duration"
    QUANTITY = "quantity"
    PERCENTAGE = "percentage"
    MONETARY = "monetary"
    COUNT = "count"
    RANGE = "range"
    UPPER_BOUND = "upper_bound"
    NONE = "none"


@dataclass(frozen=True)
class BoundarySample:
    """One requirement sentence and what a person says it means."""

    sample_id: str
    text: str
    label: Label
    kind: Kind = Kind.NONE
    #: Expected operator, for TRUE_BOUNDARY only.
    operator: Operator | None = None
    #: Expected threshold, for TRUE_BOUNDARY only.
    quantity: str = ""


SAMPLES: tuple[BoundarySample, ...] = (
    # -- the defect that survived M7, M8 and M9 ------------------------------------------
    BoundarySample(
        "BIZ-003",
        "No fee applies until a payment is more than 3 days late.",
        Label.TRUE_BOUNDARY, Kind.DURATION, Operator.GT, "3",
    ),
    # -- duration -------------------------------------------------------------------------
    BoundarySample(
        "DUR-001",
        "A session expires after being idle for at least 30 minutes.",
        Label.TRUE_BOUNDARY, Kind.DURATION, Operator.GTE, "30",
    ),
    BoundarySample(
        "DUR-002",
        "Trial accounts are cancelled if unused for over 14 days.",
        Label.TRUE_BOUNDARY, Kind.DURATION, Operator.GT, "14",
    ),
    # -- quantity -------------------------------------------------------------------------
    BoundarySample(
        "QTY-001",
        "Orders weighing 10 kg or more ship free.",
        Label.TRUE_BOUNDARY, Kind.QUANTITY, Operator.GTE, "10",
    ),
    BoundarySample(
        "QTY-002",
        "Reject an upload of more than 25 MB.",
        Label.TRUE_BOUNDARY, Kind.QUANTITY, Operator.GT, "25",
    ),
    # -- percentage -----------------------------------------------------------------------
    BoundarySample(
        "PCT-001",
        "Flag an account when the error rate is above 5%.",
        Label.TRUE_BOUNDARY, Kind.PERCENTAGE, Operator.GT, "5",
    ),
    # -- monetary -------------------------------------------------------------------------
    BoundarySample(
        "MON-001",
        "Orders under $50 incur a handling charge.",
        Label.TRUE_BOUNDARY, Kind.MONETARY, Operator.LT, "50",
    ),
    # -- count ----------------------------------------------------------------------------
    BoundarySample(
        "CNT-001",
        "Give up after no more than 3 retries.",
        Label.TRUE_BOUNDARY, Kind.COUNT, Operator.LTE, "3",
    ),
    # -- range ----------------------------------------------------------------------------
    BoundarySample(
        "RNG-001",
        "Accept a score between 1 and 10.",
        Label.TRUE_BOUNDARY, Kind.RANGE, Operator.BETWEEN, "1",
    ),
    # -- upper bound ----------------------------------------------------------------------
    BoundarySample(
        "UPB-001",
        "Return up to 100 results per page.",
        Label.TRUE_BOUNDARY, Kind.UPPER_BOUND, Operator.LTE, "100",
    ),
    # -- deliberately ambiguous -----------------------------------------------------------
    BoundarySample(
        "AMB-001",
        "Users receive a discount after 3 days.",
        Label.AMBIGUOUS, Kind.DURATION,
    ),
    BoundarySample(
        "AMB-002",
        "Escalate the alert once 5 failures are seen.",
        Label.AMBIGUOUS, Kind.COUNT,
    ),
    # -- numbers that are not thresholds --------------------------------------------------
    BoundarySample("CTL-001", "Version 3 of the API returns a list of users.", Label.NO_BOUNDARY),
    BoundarySample("CTL-002", "Process ID 1001 owns the lock.", Label.NO_BOUNDARY),
    BoundarySample("CTL-003", "Use protocol version 2 for all requests.", Label.NO_BOUNDARY),
    BoundarySample("CTL-004", "Store the value 500 as the default.", Label.NO_BOUNDARY),
    BoundarySample("CTL-005", "Display the year 2026 in the footer.", Label.NO_BOUNDARY),
    BoundarySample("CTL-006", "Listen on port 8080 for incoming traffic.", Label.NO_BOUNDARY),
    BoundarySample(
        "CTL-007", "Return HTTP status 404 when the record is absent.", Label.NO_BOUNDARY
    ),
    BoundarySample("CTL-008", "Error code 42 indicates a parse failure.", Label.NO_BOUNDARY),
    BoundarySample(
        "CTL-009", "Section 7 of the specification describes the format.", Label.NO_BOUNDARY
    ),
    BoundarySample("CTL-010", "Revision 12 introduced the new schema.", Label.NO_BOUNDARY),
    # -- no numbers at all ----------------------------------------------------------------
    BoundarySample("CTL-011", "The service returns a list of users.", Label.NO_BOUNDARY),
    BoundarySample("CTL-012", "Persist the record and return its identifier.", Label.NO_BOUNDARY),
)


def by_label(label: Label) -> tuple[BoundarySample, ...]:
    return tuple(sample for sample in SAMPLES if sample.label is label)


def by_kind() -> dict[Kind, tuple[BoundarySample, ...]]:
    return {
        kind: tuple(sample for sample in SAMPLES if sample.kind is kind)
        for kind in Kind
    }
