"""Requirement-derived tests: written from the requirement, before the implementation exists.

M7 measured EDITH accepting three implementations per arm that independent tests later
rejected. The platform's own verification ran, passed, and said nothing useful -- because the
only tests it had were the ones the coder wrote, and a coder that misreads a requirement writes
tests agreeing with its misreading. The tests and the code shared a single point of failure.

So the tests here are generated from the *requirement text alone*, by an agent that never sees
the implementation, before the coder runs. That ordering is the whole idea, and it is enforced
structurally: :class:`TestGenInput` has no field through which source code could arrive.

Three rules keep generated tests from quietly becoming their own authority.

**Provenance is assigned by the system.** A model cannot mark its own test authoritative;
:class:`TestProvenance` is set here, never parsed from model output.

**A test earns authority by validation, not by generation.** Deterministic checks run before a
generated test may participate, and one that fails any of them is advisory and never decides
anything.

**Green generated tests are not a verdict.** They join the existing verification contract; they
do not replace independent acceptance. M8 is meant to shrink the false-pass gap, not to paper
over it with a suite that shares the model's blind spots.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar

from pydantic import BaseModel, Field

from edith.agents.base import Agent
from edith.models.base import ModelProvider
from edith.observability.logging import get_logger
from edith.quality.principals import TESTGEN
from edith.schemas.agent import AgentIdentity, AgentRequest, Capability
from edith.schemas.common import EdithModel
from edith.schemas.model import Message, Role

logger = get_logger(__name__)

#: Where requirement-derived tests are written. Isolated from hand-written tests so a generated
#: file can never overwrite a human acceptance test.
GENERATED_TEST_DIR = "tests/generated"

#: How many cases one requirement may produce. A 3B model asked for more returns padding.
MAX_CASES = 4

#: Calls that mutate state. A test that writes is not a test, and this shape would not be
#: caught by any permission check because the suite runs as the verifier, not as an agent.
_MUTATING_CALLS = frozenset(
    {"open", "write_text", "write_bytes", "remove", "unlink", "rmtree", "mkdir", "system"}
)


class TestProvenance(StrEnum):
    """Where a test came from, and therefore how much authority it carries.

    Not bookkeeping. A human acceptance test *defines* correctness; a requirement-derived test
    is a hypothesis about correctness that survived validation; a model advisory test is
    neither. Collapsing them would let a model that misread the requirement decide what the
    requirement meant.
    """

    #: Written by a person. Defines correctness. Never generated, never modified by an agent.
    HUMAN_ACCEPTANCE_TEST = "HUMAN_ACCEPTANCE_TEST"
    #: Generated from the requirement before implementation, and validated. May participate.
    REQUIREMENT_DERIVED_TEST = "REQUIREMENT_DERIVED_TEST"
    #: Generated but unvalidated, or produced with sight of the implementation. Advisory only.
    MODEL_ADVISORY_TEST = "MODEL_ADVISORY_TEST"

    @property
    def authoritative(self) -> bool:
        """Whether a test with this provenance may decide the verification contract."""
        return self in {
            TestProvenance.HUMAN_ACCEPTANCE_TEST,
            TestProvenance.REQUIREMENT_DERIVED_TEST,
        }


class InvalidReason(StrEnum):
    """Why a generated test was refused. Recorded so the failure mode stays measurable."""

    SYNTAX = "SYNTAX"
    NO_TEST_FUNCTION = "NO_TEST_FUNCTION"
    NO_ASSERTION = "NO_ASSERTION"
    VACUOUS = "VACUOUS"
    UNRESOLVED_REQUIREMENT = "UNRESOLVED_REQUIREMENT"
    WRITES_SOURCE = "WRITES_SOURCE"
    IMPORTS_QUALITY_LAYER = "IMPORTS_QUALITY_LAYER"
    DOES_NOT_IMPORT_TARGET = "DOES_NOT_IMPORT_TARGET"
    ESCAPES_TEST_WORKSPACE = "ESCAPES_TEST_WORKSPACE"


class ModelTestCase(EdithModel):
    """One test case as the model states it. Three flat fields, deliberately.

    No requirement id, no provenance, no authority -- those are system-owned. M4.1 showed this
    model failing large schemas outright, so the request is the smallest thing still runnable.
    """

    name: str = Field(min_length=1, max_length=60)
    #: One line describing what behaviour this checks. Becomes the docstring, not control flow.
    intent: str = Field(min_length=1, max_length=200)
    #: The body of a pytest function, without the ``def`` line.
    body: str = Field(min_length=1, max_length=1200)


class TestCaseSet(EdithModel):
    """What the generator returns for one requirement."""

    cases: list[ModelTestCase] = Field(default_factory=list, max_length=MAX_CASES)


class TestGenInput(EdithModel):
    """What the generator is shown.

    There is deliberately no field for source, a diff, or an implementation path beyond the
    module name to import. The anti-circularity rule is a property of the type rather than a
    convention someone has to remember.
    """

    requirement_id: str = Field(min_length=1, max_length=120)
    requirement: str = Field(min_length=1, max_length=2000)
    acceptance_criteria: tuple[str, ...] = Field(default=(), max_length=10)
    #: The dotted module the tests must import, derived by the system from the task's paths.
    module: str = Field(min_length=1, max_length=200)


@dataclass(frozen=True)
class GeneratedTest:
    """One requirement-derived test, validated or rejected."""

    name: str
    requirement_id: str
    module: str
    source: str
    provenance: TestProvenance
    valid: bool
    reason: InvalidReason | None = None

    @property
    def authoritative(self) -> bool:
        """Whether this test may participate in the verification contract."""
        return self.valid and self.provenance.authoritative


TESTGEN_PROMPT = """You write pytest tests from a requirement, before any code exists.

You have NOT seen the implementation and must not assume how it is written. Test only the
behaviour the requirement describes.

Rules:
- Each case is one pytest function body, using plain assert statements.
- The module is already importable; its functions are in scope.
- Assert concrete expected values. Never write assert True, and never assert only that a call
  did not raise.
- Cover the stated behaviour first, then an edge case the requirement mentions.
- If the requirement says an error is raised, assert that with pytest.raises.

Return at most four cases. Fewer good cases beat more weak ones."""


class TestGeneratorAgent(Agent):
    """Writes tests from a requirement. Never sees the implementation.

    Writes only inside ``tests/generated/**``, holds no shell and no git, and cannot reach
    ``src/**`` -- so it can neither implement the thing it is testing nor edit a human
    acceptance test into agreeing with it.
    """

    identity: ClassVar[AgentIdentity] = AgentIdentity(
        name="test_generator",
        description="Derives pytest cases from a requirement before implementation exists.",
        capabilities=frozenset({Capability.TESTING}),
        permissions=TESTGEN,
    )
    input_schema: ClassVar[type[BaseModel]] = TestGenInput
    output_schema: ClassVar[type[BaseModel]] = TestCaseSet

    def _run(self, payload: BaseModel, request: AgentRequest) -> BaseModel:
        assert isinstance(payload, TestGenInput)  # noqa: S101 - guaranteed by validate_input
        provider: ModelProvider = self.require_provider()
        criteria = "\n".join(f"- {item}" for item in payload.acceptance_criteria)
        messages = [
            Message(role=Role.SYSTEM, content=TESTGEN_PROMPT),
            Message(
                role=Role.USER,
                content=(
                    f"MODULE TO IMPORT: {payload.module}\n\n"
                    f"REQUIREMENT:\n{payload.requirement}\n\n"
                    + (f"ACCEPTANCE CRITERIA:\n{criteria}\n" if criteria else "")
                ),
            ),
        ]
        return provider.structured_generate(messages, TestCaseSet)


def render_test(case: ModelTestCase, *, requirement_id: str, module: str) -> str:
    """Assemble one model case into a runnable pytest function.

    The import line and the provenance header are written by EDITH, not by the model, so a case
    cannot import something the system did not intend or claim authority it was not given.
    """
    safe = "".join(char if char.isalnum() or char == "_" else "_" for char in case.name)
    body = "\n".join(f"    {line}" for line in case.body.strip().splitlines())
    return (
        f"# provenance: {TestProvenance.REQUIREMENT_DERIVED_TEST.value}\n"
        f"# requirement: {requirement_id}\n"
        "import pytest  # noqa: F401\n"
        f"from {module} import *  # noqa: F403\n\n\n"
        f"def test_{safe}():\n"
        f'    """{case.intent}"""\n'
        f"{body}\n"
    )


def _is_vacuous(tree: ast.AST) -> bool:
    """Whether every assertion is trivially true.

    ``assert True`` is the shape M2.1 already refuses in hand-written tests. A generated suite
    full of it would pass any implementation -- precisely the blind spot M8 exists to close.
    """
    asserts = [node for node in ast.walk(tree) if isinstance(node, ast.Assert)]
    if not asserts:
        return True
    for node in asserts:
        test = node.test
        if isinstance(test, ast.Constant) and bool(test.value) is True:
            continue
        return False
    return True


def _imported_modules(tree: ast.AST) -> set[str]:
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
        elif isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
    return modules


def validate_test(
    source: str,
    *,
    requirement_id: str,
    module: str,
    known_requirements: frozenset[str],
) -> InvalidReason | None:
    """Run every deterministic check a generated test must survive.

    Returns the first failure, or ``None`` when the test may be trusted to *execute*. Passing
    validation does not make a test correct -- only runnable, non-vacuous, and confined.
    """
    if requirement_id not in known_requirements:
        return InvalidReason.UNRESOLVED_REQUIREMENT
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return InvalidReason.SYNTAX

    if not any(
        isinstance(node, ast.FunctionDef) and node.name.startswith("test_")
        for node in ast.walk(tree)
    ):
        return InvalidReason.NO_TEST_FUNCTION
    if not any(isinstance(node, ast.Assert) for node in ast.walk(tree)):
        return InvalidReason.NO_ASSERTION
    if _is_vacuous(tree):
        return InvalidReason.VACUOUS

    modules = _imported_modules(tree)
    if module not in modules:
        return InvalidReason.DOES_NOT_IMPORT_TARGET
    if any(
        name.startswith(("edith.quality", "edith.agents", "edith.engineering"))
        for name in modules
    ):
        return InvalidReason.IMPORTS_QUALITY_LAYER

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = (
                node.func.attr
                if isinstance(node.func, ast.Attribute)
                else getattr(node.func, "id", "")
            )
            if name in _MUTATING_CALLS:
                return InvalidReason.WRITES_SOURCE
    if ".." in source.replace("\\", "/"):
        return InvalidReason.ESCAPES_TEST_WORKSPACE
    return None


def generate_tests(
    agent: TestGeneratorAgent,
    *,
    requirement_id: str,
    requirement: str,
    module: str,
    acceptance_criteria: tuple[str, ...] = (),
    known_requirements: frozenset[str] | None = None,
) -> tuple[GeneratedTest, ...]:
    """Generate and validate requirement-derived tests for one requirement.

    Note what this signature does not accept: no source, no diff, no implementation path.

    A generation failure yields an empty tuple rather than raising. No tests is a measurable
    outcome; an exception here would abort a task for a reason unrelated to its code.
    """
    known = known_requirements or frozenset({requirement_id})
    try:
        response = agent.execute(
            AgentRequest(
                payload=TestGenInput(
                    requirement_id=requirement_id,
                    requirement=requirement,
                    acceptance_criteria=acceptance_criteria,
                    module=module,
                ).model_dump()
            )
        )
        if not response.ok:
            raise RuntimeError(response.error or "test generation failed")
        parsed = TestCaseSet.model_validate(response.output)
    except Exception as exc:  # noqa: BLE001 - generation failure must not abort the task
        logger.warning(
            "testgen.failed",
            requirement_id=requirement_id,
            error=f"{type(exc).__name__}: {exc}",
        )
        return ()

    generated: list[GeneratedTest] = []
    for case in parsed.cases:
        source = render_test(case, requirement_id=requirement_id, module=module)
        reason = validate_test(
            source,
            requirement_id=requirement_id,
            module=module,
            known_requirements=known,
        )
        generated.append(
            GeneratedTest(
                name=case.name,
                requirement_id=requirement_id,
                module=module,
                source=source,
                # Assigned here, by the system. A test that failed validation is advisory no
                # matter how confidently the model wrote it.
                provenance=(
                    TestProvenance.REQUIREMENT_DERIVED_TEST
                    if reason is None
                    else TestProvenance.MODEL_ADVISORY_TEST
                ),
                valid=reason is None,
                reason=reason,
            )
        )
    logger.info(
        "testgen.generated",
        requirement_id=requirement_id,
        cases=len(generated),
        valid=sum(1 for item in generated if item.valid),
    )
    return tuple(generated)


def module_for(path: str) -> str:
    """The dotted module a test should import for a task's target path."""
    return path.removesuffix(".py").replace("\\", "/").replace("/", ".")
