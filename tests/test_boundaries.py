"""M10: deterministic boundary detection, and the rules that keep it useful.

Every false PASS that survived M7, M8 and M9 was one defect: BIZ-003 says "more than 3 days
late" and the model implemented ``>= 3``. Generated tests could not catch it because they
inherited the same misreading as the code.

The difference between ``> 3`` and ``>= 3`` is decidable from the words alone, before any code
exists, with no model involved. That is what this layer does, and two properties make it worth
having rather than merely present.

**It refuses to guess.** "after 3 days" does not say whether day three counts, so the answer is
a question. Guessing there would reproduce the exact defect being hunted.

**It does not see thresholds everywhere.** "Version 3", "port 8080", "status 404" contain
numbers and no business rule. A detector that flagged those would make every requirement
require clarification and the layer would be switched off -- so the false-positive controls are
tested as seriously as the true positives.
"""

from __future__ import annotations

import pytest
from benchmarks.boundary_corpus import SAMPLES, BoundarySample, Kind, Label, by_label

from edith.requirements.boundaries import (
    BoundaryCondition,
    BoundaryStatus,
    Operator,
    check_proposal,
    detect_boundaries,
    expand_cases,
    render_for_plan,
    resolve,
    unresolved,
)


def explicit(text: str, requirement_id: str = "REQ-001") -> list[BoundaryCondition]:
    return [
        item
        for item in detect_boundaries(text, requirement_id=requirement_id)
        if item.status is BoundaryStatus.EXPLICIT
    ]


class TestTheDefectThatSurvivedThreeMilestones:
    """BIZ-003, stated as its own test because it is the whole reason M10 exists."""

    def test_more_than_three_days_is_strictly_greater(self) -> None:
        found = explicit("No fee applies until a payment is more than 3 days late.")
        assert len(found) == 1
        assert found[0].operator is Operator.GT
        assert found[0].quantity == "3"
        assert found[0].unit == "days"

    def test_the_boundary_cases_distinguish_the_off_by_one(self) -> None:
        """``> 3`` and ``>= 3`` agree on every input except one. That one is the test."""
        condition = explicit("a payment is more than 3 days late")[0]
        cases = {case.value: case.satisfies for case in condition.cases}
        assert cases == {"2": False, "3": False, "4": True}

    def test_the_plan_text_states_the_condition_and_the_cases(self) -> None:
        rendered = render_for_plan(
            tuple(explicit("No fee until a payment is more than 3 days late."))
        )
        assert "value > 3" in rendered
        assert "3 days does not satisfy" in rendered
        assert "4 days satisfies" in rendered


class TestOperatorsAreReadFromTheWords:
    @pytest.mark.parametrize(
        ("text", "operator", "quantity"),
        [
            ("orders of 10 kg or more ship free", Operator.GTE, "10"),
            ("reject an upload of more than 25 MB", Operator.GT, "25"),
            ("give up after no more than 3 retries", Operator.LTE, "3"),
            ("accounts idle for at least 30 minutes expire", Operator.GTE, "30"),
            ("orders under 50 dollars incur a charge", Operator.LT, "50"),
            ("flag an account when the error rate is above 5%", Operator.GT, "5"),
            ("return up to 100 results", Operator.LTE, "100"),
            ("cancel accounts unused for over 14 days", Operator.GT, "14"),
            ("allow fewer than 8 connections", Operator.LT, "8"),
            ("a minimum of 2 approvals is needed", Operator.GTE, "2"),
        ],
    )
    def test_phrase_maps_to_operator(
        self, text: str, operator: Operator, quantity: str
    ) -> None:
        found = explicit(text)
        assert found, f"no boundary found in {text!r}"
        assert found[0].operator is operator
        assert found[0].quantity == quantity

    def test_no_more_than_beats_more_than(self) -> None:
        """Longest-phrase-first, or the negation is inverted into its opposite."""
        assert explicit("no more than 3 retries")[0].operator is Operator.LTE

    def test_a_range_is_two_sided(self) -> None:
        found = explicit("accept a score between 1 and 10")
        assert found[0].operator is Operator.BETWEEN
        assert found[0].quantity == "1"
        assert found[0].upper == "10"
        assert found[0].condition() == "1 <= value <= 10"


class TestItRefusesToGuess:
    @pytest.mark.parametrize(
        "text",
        [
            "users receive a discount after 3 days",
            "escalate the alert once 5 failures are seen",
            "retry the request within 10 minutes",
        ],
    )
    def test_ambiguous_wording_asks_rather_than_decides(self, text: str) -> None:
        found = detect_boundaries(text, requirement_id="REQ-001")
        assert found
        assert found[0].status is BoundaryStatus.CLARIFICATION_REQUIRED
        assert found[0].operator is None
        assert found[0].question

    def test_an_ambiguous_boundary_blocks_implementation(self) -> None:
        found = detect_boundaries("a discount after 3 days", requirement_id="R")
        assert unresolved(found)
        assert found[0].blocking

    def test_an_unresolved_boundary_yields_no_condition(self) -> None:
        """There is no code to emit for a question."""
        found = detect_boundaries("a discount after 3 days", requirement_id="R")
        assert found[0].condition() == ""

    def test_the_question_names_both_candidate_operators(self) -> None:
        found = detect_boundaries("a discount after 3 days", requirement_id="R")
        assert "> 3" in found[0].question
        assert ">= 3" in found[0].question


class TestANumberIsNotAThreshold:
    """A detector that flags every number makes every requirement need clarification."""

    @pytest.mark.parametrize("sample", by_label(Label.NO_BOUNDARY), ids=lambda s: s.sample_id)
    def test_control_sentences_produce_no_boundary(self, sample: BoundarySample) -> None:
        found = detect_boundaries(sample.text, requirement_id=sample.sample_id)
        assert found == (), f"{sample.sample_id}: spurious boundary in {sample.text!r}"

    def test_a_sentence_with_no_numbers_produces_nothing(self) -> None:
        assert detect_boundaries("The service returns a list of users.", requirement_id="R") == ()


class TestTheCorpus:
    """Accuracy over the whole labelled corpus, per sample so failures name themselves."""

    @pytest.mark.parametrize("sample", SAMPLES, ids=lambda s: s.sample_id)
    def test_the_detector_agrees_with_the_human_label(self, sample: BoundarySample) -> None:
        found = detect_boundaries(sample.text, requirement_id=sample.sample_id)
        explicit_found = [i for i in found if i.status is BoundaryStatus.EXPLICIT]
        clarify = [i for i in found if i.status is BoundaryStatus.CLARIFICATION_REQUIRED]

        if sample.label is Label.TRUE_BOUNDARY:
            assert explicit_found, f"missed a boundary in {sample.text!r}"
            assert explicit_found[0].operator is sample.operator
            assert explicit_found[0].quantity == sample.quantity
        elif sample.label is Label.AMBIGUOUS:
            assert clarify and not explicit_found
        else:
            assert not found

    def test_the_corpus_covers_every_required_kind(self) -> None:
        kinds = {sample.kind for sample in SAMPLES if sample.label is Label.TRUE_BOUNDARY}
        assert kinds >= {
            Kind.DURATION,
            Kind.QUANTITY,
            Kind.PERCENTAGE,
            Kind.MONETARY,
            Kind.COUNT,
            Kind.RANGE,
            Kind.UPPER_BOUND,
        }

    def test_the_corpus_has_at_least_ten_false_positive_controls(self) -> None:
        assert len(by_label(Label.NO_BOUNDARY)) >= 10

    def test_the_corpus_has_an_ambiguous_case(self) -> None:
        assert by_label(Label.AMBIGUOUS)


class TestCaseExpansion:
    @pytest.mark.parametrize(
        ("operator", "expected"),
        [
            (Operator.GT, {"2": False, "3": False, "4": True}),
            (Operator.GTE, {"2": False, "3": True, "4": True}),
            (Operator.LT, {"2": True, "3": False, "4": False}),
            (Operator.LTE, {"2": True, "3": True, "4": False}),
        ],
    )
    def test_neighbours_are_derived_for_integers(
        self, operator: Operator, expected: dict[str, bool]
    ) -> None:
        cases = expand_cases("3", operator, "days")
        assert {case.value: case.satisfies for case in cases} == expected

    def test_a_continuous_quantity_gets_no_invented_neighbours(self) -> None:
        """The interesting value near 10 kg is 9.99, not 9, so nothing is fabricated."""
        cases = expand_cases("10", Operator.GTE, "kg")
        assert len(cases) == 1
        assert cases[0].value == "10"

    def test_a_decimal_threshold_gets_no_invented_neighbours(self) -> None:
        assert len(expand_cases("2.5", Operator.GT, "")) == 1


class TestTheModelCannotOverrideTheWords:
    """The model may propose. It may not decide."""

    def test_a_proposal_matching_the_evidence_is_accepted(self) -> None:
        condition = explicit("more than 3 days late")[0]
        ok, reason = check_proposal(condition, Operator.GT)
        assert ok, reason

    def test_the_exact_m7_misreading_is_rejected(self) -> None:
        """``more than 3`` cannot become ``>= 3`` because a model preferred it."""
        condition = explicit("more than 3 days late")[0]
        ok, reason = check_proposal(condition, Operator.GTE)
        assert not ok
        assert "contradicts" in reason

    def test_a_proposal_on_ambiguous_wording_is_rejected(self) -> None:
        """Guessing is the defect; only a human resolves these."""
        condition = detect_boundaries("a discount after 3 days", requirement_id="R")[0]
        ok, reason = check_proposal(condition, Operator.GT)
        assert not ok
        assert "human" in reason


class TestHumanResolution:
    def test_a_resolution_makes_the_boundary_implementable(self) -> None:
        condition = detect_boundaries("a discount after 3 days", requirement_id="R")[0]
        settled = resolve(condition, operator=Operator.GT, decided_by="product-owner")
        assert settled.status is BoundaryStatus.RESOLVED
        assert not settled.blocking
        assert settled.condition() == "value > 3"

    def test_a_resolution_records_who_made_it(self) -> None:
        condition = detect_boundaries("a discount after 3 days", requirement_id="R")[0]
        settled = resolve(condition, operator=Operator.GTE, decided_by="product-owner")
        assert "product-owner" in settled.evidence

    def test_a_resolution_does_not_rewrite_the_original(self) -> None:
        """The requirement text and the detector's reading of it are both preserved."""
        condition = detect_boundaries("a discount after 3 days", requirement_id="R")[0]
        resolve(condition, operator=Operator.GT, decided_by="owner")
        assert condition.status is BoundaryStatus.CLARIFICATION_REQUIRED
        assert condition.operator is None

    def test_an_anonymous_resolution_is_refused(self) -> None:
        condition = detect_boundaries("a discount after 3 days", requirement_id="R")[0]
        with pytest.raises(ValueError, match="who made it"):
            resolve(condition, operator=Operator.GT, decided_by="  ")

    def test_a_resolved_boundary_still_rejects_a_contradicting_proposal(self) -> None:
        condition = detect_boundaries("a discount after 3 days", requirement_id="R")[0]
        settled = resolve(condition, operator=Operator.GT, decided_by="owner")
        ok, _ = check_proposal(settled, Operator.GTE)
        assert not ok


class TestTheDetectorSeesOnlyText:
    def test_it_takes_no_implementation_argument(self) -> None:
        import ast
        from pathlib import Path

        tree = ast.parse(Path("src/edith/requirements/boundaries.py").read_text(encoding="utf-8"))
        function = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "detect_boundaries"
        )
        names = {arg.arg for arg in function.args.args + function.args.kwonlyargs}
        assert names == {"requirement", "requirement_id"}

    def test_it_imports_no_model_no_judge_and_no_tests(self) -> None:
        """Asserted over imports rather than text: the module may *discuss* the Judge in a
        docstring, but it must not be able to reach one."""
        import ast
        from pathlib import Path

        tree = ast.parse(Path("src/edith/requirements/boundaries.py").read_text(encoding="utf-8"))
        modules: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                modules.add(node.module)
            elif isinstance(node, ast.Import):
                modules.update(alias.name for alias in node.names)
        forbidden = ("edith.quality", "edith.agents", "edith.models", "edith.engineering")
        assert not [name for name in modules if name.startswith(forbidden)]

    def test_it_is_deterministic(self) -> None:
        text = "orders of 10 kg or more ship free"
        first = detect_boundaries(text, requirement_id="R")
        second = detect_boundaries(text, requirement_id="R")
        assert first == second


class TestPlanRendering:
    def test_unresolved_boundaries_contribute_nothing_to_the_plan(self) -> None:
        found = detect_boundaries("a discount after 3 days", requirement_id="R")
        assert render_for_plan(found) == ""

    def test_an_empty_set_renders_empty(self) -> None:
        assert render_for_plan(()) == ""
