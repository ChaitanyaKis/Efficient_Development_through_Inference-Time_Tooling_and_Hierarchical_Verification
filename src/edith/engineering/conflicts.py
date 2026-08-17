"""Detecting when two engineering tasks would fight over the same files.

M5 item 14: if two agents need to modify the same file, do not silently overwrite. Return a
``TASK_CONFLICT`` naming the tasks, the files, the agents, and the scopes.

The detection is deterministic and runs *before* anything executes, which is the only point
where it is cheap. Two tasks whose write scopes overlap are not necessarily wrong — a
backend task adding an endpoint and another adding a service may both live under
``src/backend/**`` and never touch the same file. What matters is whether they can be
*ordered*: if one depends on the other, the DAG already serialises them and there is nothing
to resolve. A conflict is an overlap between tasks that are otherwise free to run in either
order, because that is where the outcome depends on which ran last.

M5 executes sequentially on this hardware, so a conflict does not corrupt anything today.
It is still reported, because "these two tasks are order-dependent and nobody said so" is a
defect in the plan that becomes a race the moment execution parallelises.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from edith.observability.logging import get_logger

from .ownership import Assignment, EngineeringRole

logger = get_logger(__name__)


class ConflictKind(StrEnum):
    """What sort of collision was found."""

    #: Two tasks may write the same file and neither depends on the other.
    OVERLAPPING_SCOPE = "OVERLAPPING_SCOPE"
    #: Two tasks name exactly the same file.
    SAME_FILE = "SAME_FILE"
    #: Different roles claim the same area, which is a plan-level mistake.
    ROLE_OVERLAP = "ROLE_OVERLAP"


@dataclass(frozen=True)
class TaskConflict:
    """Two tasks that cannot safely run in an arbitrary order."""

    kind: ConflictKind
    task_ids: tuple[str, str]
    agents: tuple[EngineeringRole, EngineeringRole]
    #: The patterns or files both tasks claim.
    scopes: tuple[str, ...]
    detail: str = ""

    @property
    def code(self) -> str:
        """The stable code a caller matches on."""
        return "TASK_CONFLICT"

    @property
    def cross_role(self) -> bool:
        """Whether the two tasks belong to different roles.

        Worse than a within-role overlap: two backend tasks touching one file is a planning
        detail, whereas a backend task and a database task touching one file means the
        architecture put a boundary in the wrong place.
        """
        return self.agents[0] is not self.agents[1]

    def render(self) -> str:
        left, right = self.task_ids
        return (
            f"[{self.code}/{self.kind}] {left} ({self.agents[0].value}) and "
            f"{right} ({self.agents[1].value}) both claim "
            f"{', '.join(self.scopes[:3])}"
        )


def _normalise(pattern: str) -> str:
    """Reduce a write pattern to the directory it governs."""
    cleaned = pattern.replace("\\", "/").strip()
    return cleaned[:-3] if cleaned.endswith("/**") else cleaned


def _overlaps(left: str, right: str) -> bool:
    """Whether two write patterns can reach a common path."""
    first, second = _normalise(left), _normalise(right)
    if first == second:
        return True
    return first.startswith(f"{second}/") or second.startswith(f"{first}/")


def _shared_scopes(left: Assignment, right: Assignment) -> tuple[str, ...]:
    """Every pattern pair between two assignments that can reach a common path."""
    found = {
        first
        for first in left.write_paths
        for second in right.write_paths
        if _overlaps(first, second)
    }
    return tuple(sorted(found))


def _shared_files(left: Assignment, right: Assignment) -> tuple[str, ...]:
    """Files both tasks explicitly named."""
    return tuple(sorted(set(left.task.paths) & set(right.task.paths)))


def _ordered(left: Assignment, right: Assignment) -> bool:
    """Whether the DAG already forces one of these tasks to precede the other.

    Direct dependency only. A transitive chain also orders them, and computing the closure
    would be more correct -- but a plan that relies on a three-hop chain to keep two writers
    apart is fragile enough to be worth reporting anyway.
    """
    return (
        right.task.task_id in left.task.depends_on
        or left.task.task_id in right.task.depends_on
    )


def detect_conflicts(assignments: tuple[Assignment, ...]) -> tuple[TaskConflict, ...]:
    """Find every pair of tasks that could write the same file in an undefined order.

    Only assigned tasks are compared: an unassigned task will not execute, so it cannot
    collide with anything.
    """
    executable = [item for item in assignments if item.assigned]
    conflicts: list[TaskConflict] = []

    for index, left in enumerate(executable):
        for right in executable[index + 1 :]:
            if _ordered(left, right):
                # The DAG serialises these, so the outcome does not depend on scheduling.
                continue

            files = _shared_files(left, right)
            if files:
                conflicts.append(
                    TaskConflict(
                        kind=ConflictKind.SAME_FILE,
                        task_ids=(left.task.task_id, right.task.task_id),
                        agents=(left.role, right.role),
                        scopes=files,
                        detail=(
                            "both tasks name the same file and neither depends on the "
                            "other, so the result depends on which runs last"
                        ),
                    )
                )
                continue

            shared = _shared_scopes(left, right)
            if not shared:
                continue

            kind = (
                ConflictKind.ROLE_OVERLAP
                if left.role is not right.role
                else ConflictKind.OVERLAPPING_SCOPE
            )
            conflicts.append(
                TaskConflict(
                    kind=kind,
                    task_ids=(left.task.task_id, right.task.task_id),
                    agents=(left.role, right.role),
                    scopes=shared,
                    detail=(
                        "the write scopes overlap and neither task depends on the other"
                    ),
                )
            )

    if conflicts:
        logger.warning(
            "engineering.conflicts",
            total=len(conflicts),
            cross_role=sum(1 for item in conflicts if item.cross_role),
            pairs=[item.task_ids for item in conflicts[:5]],
        )
    return tuple(conflicts)


def serialise(
    assignments: tuple[Assignment, ...], conflicts: tuple[TaskConflict, ...]
) -> tuple[Assignment, ...]:
    """Order conflicting tasks deterministically so execution is reproducible.

    M5 item 14 says the orchestrator must resolve *or serialise* conflicting tasks. Since M5
    runs sequentially anyway, serialising is the honest resolution: the conflict is reported,
    and the order is made a property of the task ids rather than of dictionary iteration, so
    two runs of the same plan produce the same result.

    This does not *fix* the conflict. It makes it harmless today and keeps it visible.
    """
    conflicted = {
        task_id for conflict in conflicts for task_id in conflict.task_ids
    }
    return tuple(
        sorted(
            assignments,
            key=lambda item: (
                item.task.task_id in conflicted,
                item.task.task_id,
            ),
        )
    )
