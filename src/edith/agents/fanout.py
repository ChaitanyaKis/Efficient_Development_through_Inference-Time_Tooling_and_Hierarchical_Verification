"""Planner fan-out: one request becomes many single-function tasks.

Measured behaviour on the reference local model: a request scoped to one function and one
test file converges reliably; a request bundling four functions did not converge in seventeen
repair attempts and delivered nothing. The failure signature was consistent -- functions
defined twice with the second silently winning, error handling copied onto the wrong
function, and test files that re-implemented the module locally instead of importing it.

The cause is upstream of the orchestration. The planner emitted a single task carrying four
implementations, and the model was then asked to hold all four in one response. So the fix is
in the planning stage, not the loop: split the request first, and never ask the model to write
more than one function at a time.

Two phases, and the split between them is the whole point:

**Phase A asks the model for almost nothing.** A flat list of name, signature and a one-line
behaviour. No bodies, no tests, no error semantics. This is the smallest useful thing a 3B
model can be asked for, and its output is validated against a strict schema so a malformed
answer becomes :class:`FailureCategory.VALIDATION_FAILURE` rather than an executed instruction.

**Phase B calls no model at all.** It formats each entry into the exact task shape that was
measured to work, populating expected cases from the requirement-boundary analyser where that
fires. Being pure means the same list in always produces the same tasks out -- the fan-out
itself cannot become a source of nondeterminism.

A deterministic validator then guarantees the invariant the change exists for: no task
reaching the coding agent describes more than one function. A spec that somehow carries two is
split rather than rejected, because failing the run would be a worse outcome than doing the
split the request needed anyway.
"""

from __future__ import annotations

import re
from typing import ClassVar

from pydantic import BaseModel, Field, field_validator

from edith.agents.base import Agent
from edith.errors import EdithError, FailureCategory
from edith.models.base import ModelProvider
from edith.observability.logging import get_logger
from edith.requirements.boundaries import BoundaryStatus, detect_boundaries
from edith.schemas.agent import (
    AgentIdentity,
    AgentPermissions,
    AgentRequest,
    Capability,
)
from edith.schemas.common import EdithModel
from edith.schemas.model import Message, Role

from .planner import PlannedStep, PlannerOutput

logger = get_logger(__name__)

#: Upper bound on fan-out. A request naming more functions than this is more likely a
#: misparse than a real plan, and a hundred single-function runs is not a useful default.
MAX_FUNCTIONS = 12

#: The measured-reliable task shape. Three details in it are load-bearing, each learned from a
#: specific failure: "one function", because bundling four did not converge; the explicit
#: import line, because left to itself the model redefines the functions inside the test file
#: rather than importing them, so the tests pass while verifying nothing; and the explicit
#: demand for a test *function*, because an earlier wording that asked only for a file "whose
#: first line is" the import produced three test files containing exactly that line and
#: nothing else.
TASK_TEMPLATE = (
    "Create src/backend/{name}.py containing one function {signature} that {behaviour}.\n"
    "\n"
    "Also create tests/test_{name}.py with exactly this structure:\n"
    "\n"
    "from src.backend.{name} import {name}\n"
    "\n"
    "def test_{name}():\n"
    "    <two or more assert statements that CALL {name}>\n"
    "\n"
    "The test file must contain a test function, not only the import line.\n"
    "The assertions must cover: {cases}"
)

#: What to ask for when the boundary analyser finds nothing to pin down.
DEFAULT_CASES = "at least two concrete input/output cases for the described behaviour"

#: An identifier followed by an argument list: how a function is named in prose.
_SIGNATURE_RE = re.compile(r"\b([a-z_][a-z0-9_]*)\s*\(([^)]*)\)")

#: Words that look like calls in prose but are not the project's functions.
_NOT_FUNCTIONS = frozenset(
    {
        "def", "return", "assert", "raises", "print", "int", "str", "float", "bool",
        "list", "dict", "tuple", "set", "len", "range", "type", "e", "g", "eg", "ie",
        "src", "tests", "import", "from", "pytest", "self", "if", "for", "in", "is",
    }
)


class PlanFanOutError(EdithError):
    """The request could not be fanned out into single-function tasks."""

    category = FailureCategory.REQUIREMENT_FAILURE


class FunctionSpec(EdithModel):
    """One function as Phase A names it. Three flat fields, deliberately.

    No body, no tests, no error semantics: those are what the model gets wrong when asked for
    several at once, and Phase B does not need them to build the task.
    """

    name: str = Field(min_length=1, max_length=60)
    signature: str = Field(min_length=1, max_length=200)
    behaviour: str = Field(min_length=1, max_length=300)

    @field_validator("name")
    @classmethod
    def _identifier(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned.isidentifier():
            raise ValueError(f"{value!r} is not a valid Python identifier")
        return cleaned

    @field_validator("signature")
    @classmethod
    def _has_arguments(cls, value: str) -> str:
        cleaned = value.strip()
        if "(" not in cleaned or ")" not in cleaned:
            raise ValueError(f"{value!r} is not a function signature")
        return cleaned


class FunctionList(EdithModel):
    """Phase A's whole output."""

    functions: list[FunctionSpec] = Field(default_factory=list, max_length=MAX_FUNCTIONS)


class FanOutInput(EdithModel):
    """What Phase A is shown: the request, and nothing else."""

    request: str = Field(min_length=1, max_length=4000)


FANOUT_PROMPT = """You are decomposing a software request into the functions it needs.

Return a flat list. For each function give only:
- name: a valid Python identifier
- signature: the call form, for example add(a, b)
- behaviour: ONE line saying what it returns or raises

Do NOT write any code. Do NOT write tests. Do NOT describe error handling in detail.
Do NOT group two functions into one entry.

List every function the request explicitly asks for, and nothing more."""


class FanOutAgent(Agent):
    """Phase A. Asks for a list of function signatures and nothing else.

    Read-only: it proposes names, it does not write. The tasks it leads to are built without a
    model, so this is the only place the model influences the shape of the fan-out.
    """

    identity: ClassVar[AgentIdentity] = AgentIdentity(
        name="fanout_planner",
        description="Lists the functions a request needs, as name, signature and behaviour.",
        capabilities=frozenset({Capability.PLANNING}),
        permissions=AgentPermissions(
            allowed_tools=frozenset({"filesystem.read", "filesystem.search"}),
            allowed_read_paths=("**",),
        ),
    )
    input_schema: ClassVar[type[BaseModel]] = FanOutInput
    output_schema: ClassVar[type[BaseModel]] = FunctionList

    def _run(self, payload: BaseModel, request: AgentRequest) -> BaseModel:
        assert isinstance(payload, FanOutInput)  # noqa: S101 - guaranteed by validate_input
        provider: ModelProvider = self.require_provider()
        messages = [
            Message(role=Role.SYSTEM, content=FANOUT_PROMPT),
            Message(role=Role.USER, content=f"REQUEST:\n{payload.request}"),
        ]
        return provider.structured_generate(messages, FunctionList)


def signatures_in(text: str) -> dict[str, str]:
    """Every distinct function named in a piece of prose, mapped to its argument list.

    Used two ways: to count signatures in a task spec (the validator's invariant) and to find
    the operations a request names explicitly (the coverage cross-check). Both need the same
    answer to "which functions does this text name", so it is computed once here.
    """
    found: dict[str, str] = {}
    for match in _SIGNATURE_RE.finditer(text):
        name, arguments = match.group(1), match.group(2).strip()
        if name in _NOT_FUNCTIONS:
            continue
        # A test function is not one of the implementation functions being counted. The task
        # template names ``def test_divide():`` explicitly, and counting that as a second
        # signature would make every generated task violate the single-function invariant it
        # was built to satisfy.
        if name.startswith("test_"):
            continue
        # First mention wins: the template repeats the name in the import line without args.
        if name not in found or (not found[name] and arguments):
            found[name] = arguments
    return found


def cases_for(spec: FunctionSpec, request: str) -> str:
    """The assertion list to hand the coding agent.

    Reuses the requirement-boundary analyser, which is the one supported mechanism for pinning
    down what a requirement actually demands: where it finds an explicit threshold it also
    derives the neighbouring values that distinguish a correct implementation from an
    off-by-one, and those are exactly the cases worth asserting.

    Where it finds nothing, the assertion list is left to the coding agent rather than
    invented here. A fabricated expected value would be worse than none.
    """
    conditions = [
        condition
        for condition in detect_boundaries(
            f"{spec.behaviour} {request}", requirement_id=spec.name
        )
        if condition.status is BoundaryStatus.EXPLICIT and condition.cases
    ]
    if not conditions:
        return DEFAULT_CASES

    fragments: list[str] = []
    for condition in conditions[:2]:
        unit = f" {condition.unit}" if condition.unit else ""
        for case in condition.cases:
            verdict = "satisfies" if case.satisfies else "does not satisfy"
            fragments.append(f"{case.value}{unit} {verdict} {condition.condition()}")
    return "; ".join(fragments) or DEFAULT_CASES


def specs_to_steps(
    functions: list[FunctionSpec],
    request: str,
    *,
    assembly_module: str | None = None,
) -> list[PlannedStep]:
    """Phase B. Turn a function list into task steps, calling no model.

    Pure: the same list and request always produce the same steps. That property is what stops
    the fan-out itself becoming a source of nondeterminism, and it is asserted in the tests.

    When ``assembly_module`` is given, a final step is appended that depends on every other
    and creates a module containing only imports and ``__all__`` -- no logic, so there is
    nothing in it for the model to get wrong.

    That step carries its own test, and must. Its module is pure re-exports, so nothing else
    will ever import it, and the vacuous-verification check then fails the run over a changed
    file that no test exercises -- the check being exactly right about a file the fan-out
    should not have produced bare. Asserting the names are importable and callable is also the
    only thing worth testing about a re-export module: a broken or misspelled export is the
    one way it can fail.
    """
    steps: list[PlannedStep] = []
    for index, spec in enumerate(functions, start=1):
        description = TASK_TEMPLATE.format(
            name=spec.name,
            signature=spec.signature,
            behaviour=spec.behaviour.rstrip("."),
            cases=cases_for(spec, request),
        )
        steps.append(
            PlannedStep(
                step=index,
                title=f"Implement {spec.name}",
                description=description,
                files=[f"src/backend/{spec.name}.py", f"tests/test_{spec.name}.py"],
                depends_on=[],
                acceptance=f"{spec.name} behaves as described and its test passes",
            )
        )

    if assembly_module and steps:
        names = [spec.name for spec in functions]
        exports = ", ".join(f'"{name}"' for name in names)
        imports = "\n".join(
            f"from src.backend.{name} import {name}" for name in names
        )
        joined = ", ".join(names)
        assertions = "\n".join(f"    assert callable({name})" for name in names)
        steps.append(
            PlannedStep(
                step=len(steps) + 1,
                title=f"Assemble {assembly_module}",
                description=(
                    f"Create src/backend/{assembly_module}.py containing ONLY these import "
                    f"lines and an __all__ list. Write no other code and no logic.\n"
                    f"{imports}\n"
                    f"__all__ = [{exports}]\n"
                    f"\n"
                    f"Also create tests/test_{assembly_module}.py containing exactly:\n"
                    f"\n"
                    f"from src.backend.{assembly_module} import {joined}\n"
                    f"\n"
                    f"def test_{assembly_module}_exports():\n"
                    f"{assertions}"
                ),
                files=[
                    f"src/backend/{assembly_module}.py",
                    f"tests/test_{assembly_module}.py",
                ],
                depends_on=[step.step for step in steps],
                acceptance=f"{assembly_module} re-exports every function",
            )
        )
    return steps


def enforce_single_function(steps: list[PlannedStep]) -> list[PlannedStep]:
    """The invariant this whole change exists for: one function per task.

    A step describing several functions is split rather than rejected. Rejecting would fail a
    run over a plan defect the split fixes anyway, and the measured evidence is that a
    multi-function task does not converge -- so letting one through is the failure, not the
    splitting.

    The assembly step is exempt by construction: it names every function because it imports
    them, and contains no implementation for the model to conflate.
    """
    result: list[PlannedStep] = []
    counter = 0
    for step in steps:
        if step.title.startswith("Assemble "):
            counter += 1
            result.append(step.model_copy(update={"step": counter}))
            continue

        named = signatures_in(step.description)
        if len(named) <= 1:
            counter += 1
            result.append(step.model_copy(update={"step": counter}))
            continue

        logger.warning(
            "fanout.split",
            title=step.title,
            functions=sorted(named),
            reason="a task naming several functions does not converge",
        )
        for name, arguments in sorted(named.items()):
            counter += 1
            result.append(
                PlannedStep(
                    step=counter,
                    title=f"Implement {name}",
                    description=TASK_TEMPLATE.format(
                        name=name,
                        signature=f"{name}({arguments})",
                        behaviour=f"is described by: {step.title}",
                        cases=DEFAULT_CASES,
                    ),
                    files=[f"src/backend/{name}.py", f"tests/test_{name}.py"],
                    depends_on=[],
                    acceptance=f"{name} behaves as described and its test passes",
                )
            )

    # Re-point the assembly step at the renumbered implementation steps.
    return [
        step.model_copy(
            update={
                "depends_on": [
                    other.step for other in result if not other.title.startswith("Assemble ")
                ]
            }
        )
        if step.title.startswith("Assemble ")
        else step
        for step in result
    ]


def assert_coverage(request: str, functions: list[FunctionSpec]) -> None:
    """Fail loudly when Phase A dropped an operation the request named explicitly.

    A dropped function is the worst available outcome: the run completes, every task it did
    plan verifies, and the missing behaviour is discovered by whoever uses the result. Only
    operations the request names in call form are checked, so a request written in prose is
    not penalised for wording the analyser cannot read.

    Raises:
        PlanFanOutError: A named operation is absent from the plan.
    """
    requested = signatures_in(request)
    planned = {spec.name for spec in functions}
    missing = sorted(set(requested) - planned)
    if missing:
        raise PlanFanOutError(
            f"the plan omits {len(missing)} operation(s) the request named: "
            f"{', '.join(missing)}",
            details={"missing": missing, "planned": sorted(planned)},
        )


def assembly_name(request: str) -> str:
    """A module name for the assembly step, derived from the request.

    Derived rather than configured so nothing in production carries a benchmark's vocabulary.
    Falls back to ``package`` when the request offers no usable noun.
    """
    for word in re.findall(r"[A-Za-z][A-Za-z0-9_]{2,}", request.lower()):
        if word.isidentifier() and word not in _NOT_FUNCTIONS and word not in {
            "create", "build", "make", "write", "implement", "function", "functions",
            "add", "also", "which", "that", "the", "and", "with", "returns", "raises",
            "containing", "whose", "first", "line", "asserts", "python", "file",
        }:
            return str(word)
    return "package"


def fan_out(
    agent: FanOutAgent,
    request: str,
    *,
    goal: str = "",
    assemble: bool = True,
) -> PlannerOutput:
    """Decompose a request into single-function steps.

    One model call, in Phase A. Everything after it is deterministic.

    Raises:
        PlanFanOutError: Phase A produced nothing usable, or dropped a named operation.
    """
    response = agent.execute(
        AgentRequest(payload=FanOutInput(request=request).model_dump())
    )
    if not response.ok:
        raise PlanFanOutError(
            f"fan-out planning failed: {response.error}",
            details={"category": str(response.failure_category)},
        )

    parsed = FunctionList.model_validate(response.output)
    if not parsed.functions:
        raise PlanFanOutError("fan-out planning named no functions")

    assert_coverage(request, parsed.functions)

    module = assembly_name(request) if assemble and len(parsed.functions) > 1 else None
    # Never let the assembly module collide with a function's own module.
    if module in {spec.name for spec in parsed.functions}:
        module = None

    steps = enforce_single_function(
        specs_to_steps(parsed.functions, request, assembly_module=module)
    )
    logger.info(
        "fanout.planned",
        functions=len(parsed.functions),
        steps=len(steps),
        assembly=module or "",
    )
    return PlannerOutput(goal=goal or request[:200], steps=steps)
