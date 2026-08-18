"""M7: twelve tasks whose defects deterministic gates cannot see.

Every task here is chosen so that a wrong implementation still *parses*, still *imports*, and
still passes an AST security scan. Subtraction where addition was asked for is invisible to
every check EDITH had before M6.1; only running an independently written test, or reading the
code and understanding the intent, catches it.

Ground truth is authored here, by hand, before any generation:

- ``requirement`` is what the coder is told.
- ``acceptance`` is what decides correctness, and the coder never sees it.

Neither the coder nor the reviewer nor the Judge writes any part of the ground truth, which is
what makes "accepted" mean something. The Judge cannot override these tests: they run as a
separate process after the task has already been merged.

Categories, three tasks each:

    SEMANTIC      the operation itself is wrong
    CONTRACT      the signature or return shape is wrong
    EDGE          the boundary or empty case is wrong
    BUSINESS      the rule is wrong
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum


class Category(StrEnum):
    """What kind of defect a task is designed to expose."""

    SEMANTIC = "semantic"
    CONTRACT = "contract"
    EDGE = "edge"
    BUSINESS = "business"


@dataclass(frozen=True)
class BenchmarkTask:
    """One task, its requirement, and the independent test that judges it."""

    task_id: str
    category: Category
    path: str
    requirement: str
    #: Human-authored. Imports the generated module and asserts the intended behaviour.
    acceptance: str
    #: Human-authored known-correct implementation, used only by M9's scaffold gate.
    #:
    #: Never shown to the generator or the coder. It exists so a generated test can be checked
    #: against behaviour that is known to satisfy the requirement -- M8 measured suites that
    #: were mechanically valid and semantically wrong, and only execution against a correct
    #: implementation distinguishes those.
    scaffold: str = ""
    #: Human-authored known-INCORRECT implementation, used only as the strength control.
    #:
    #: Deliberately separate from the gate: a gate with sight of the wrong implementation
    #: would be scoring itself.
    mutant: str = ""


_BASE: tuple[BenchmarkTask, ...] = (
    # -- semantic correctness: the operation is the thing that goes wrong ----------------
    BenchmarkTask(
        "SEM-001", Category.SEMANTIC, "src/backend/totals.py",
        "Implement running_total(values: list[int]) -> list[int] returning the cumulative "
        "sums, so [1, 2, 3] becomes [1, 3, 6].",
        "from src.backend.totals import running_total\n\n"
        "def test_cumulative():\n    assert running_total([1, 2, 3]) == [1, 3, 6]\n\n"
        "def test_single():\n    assert running_total([5]) == [5]\n\n"
        "def test_empty():\n    assert running_total([]) == []\n",
    ),
    BenchmarkTask(
        "SEM-002", Category.SEMANTIC, "src/backend/ranking.py",
        "Implement top_scores(scores: list[int], n: int) -> list[int] returning the n "
        "highest scores in descending order.",
        "from src.backend.ranking import top_scores\n\n"
        "def test_descending():\n    assert top_scores([3, 9, 1, 7], 2) == [9, 7]\n\n"
        "def test_all():\n    assert top_scores([2, 1], 5) == [2, 1]\n",
    ),
    BenchmarkTask(
        "SEM-003", Category.SEMANTIC, "src/backend/money.py",
        "Implement apply_discount(price: float, percent: float) -> float returning the price "
        "after subtracting the given percentage, rounded to two decimal places.",
        "from src.backend.money import apply_discount\n\n"
        "def test_discount():\n    assert apply_discount(100.0, 10.0) == 90.0\n\n"
        "def test_rounding():\n    assert apply_discount(9.99, 50.0) == 5.0\n\n"
        "def test_zero():\n    assert apply_discount(50.0, 0.0) == 50.0\n",
    ),
    # -- API / contract correctness: the shape is wrong ----------------------------------
    BenchmarkTask(
        "API-001", Category.CONTRACT, "src/backend/lookup.py",
        "Implement find_user(users: dict[str, dict], user_id: str) -> dict | None returning "
        "the user record, or None when the id is absent. Do not raise on a missing id.",
        "from src.backend.lookup import find_user\n\n"
        "def test_found():\n    assert find_user({'a': {'n': 1}}, 'a') == {'n': 1}\n\n"
        "def test_missing_returns_none():\n    assert find_user({'a': {}}, 'zz') is None\n",
    ),
    BenchmarkTask(
        "API-002", Category.CONTRACT, "src/backend/response.py",
        "Implement make_response(data: dict, ok: bool) -> dict returning a dict with keys "
        "'status' set to 'ok' or 'error', and 'data' set to the given data.",
        "from src.backend.response import make_response\n\n"
        "def test_ok():\n    r = make_response({'x': 1}, True)\n"
        "    assert r['status'] == 'ok' and r['data'] == {'x': 1}\n\n"
        "def test_error():\n    assert make_response({}, False)['status'] == 'error'\n",
    ),
    BenchmarkTask(
        "API-003", Category.CONTRACT, "src/backend/pager.py",
        "Implement page(items: list, size: int) -> list[list] splitting items into chunks of "
        "at most size, preserving order.",
        "from src.backend.pager import page\n\n"
        "def test_chunks():\n    assert page([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]\n\n"
        "def test_exact():\n    assert page([1, 2], 2) == [[1, 2]]\n\n"
        "def test_empty():\n    assert page([], 3) == []\n",
    ),
    # -- edge cases: the boundary is wrong -----------------------------------------------
    BenchmarkTask(
        "EDGE-001", Category.EDGE, "src/backend/clamp.py",
        "Implement clamp(value: int, low: int, high: int) -> int returning value limited to "
        "the inclusive range low..high.",
        "from src.backend.clamp import clamp\n\n"
        "def test_inside():\n    assert clamp(5, 1, 10) == 5\n\n"
        "def test_low():\n    assert clamp(0, 1, 10) == 1\n\n"
        "def test_boundary():\n    assert clamp(10, 1, 10) == 10\n",
    ),
    BenchmarkTask(
        "EDGE-002", Category.EDGE, "src/backend/window.py",
        "Implement last_n(items: list, n: int) -> list returning the final n items, or every "
        "item when n exceeds the length. Return an empty list when n is zero.",
        "from src.backend.window import last_n\n\n"
        "def test_tail():\n    assert last_n([1, 2, 3, 4], 2) == [3, 4]\n\n"
        "def test_over():\n    assert last_n([1, 2], 9) == [1, 2]\n\n"
        "def test_zero():\n    assert last_n([1, 2], 0) == []\n",
    ),
    BenchmarkTask(
        "EDGE-003", Category.EDGE, "src/backend/average.py",
        "Implement safe_average(values: list[float]) -> float returning the mean, and exactly "
        "0.0 for an empty list rather than raising.",
        "from src.backend.average import safe_average\n\n"
        "def test_mean():\n    assert safe_average([2.0, 4.0]) == 3.0\n\n"
        "def test_empty_is_zero():\n    assert safe_average([]) == 0.0\n",
    ),
    # -- business logic: the rule is wrong -----------------------------------------------
    BenchmarkTask(
        "BIZ-001", Category.BUSINESS, "src/backend/shipping.py",
        "Implement shipping_cost(weight_kg: float) -> float. Orders of 10 kg or more ship "
        "free. Anything lighter costs 5.0.",
        "from src.backend.shipping import shipping_cost\n\n"
        "def test_light():\n    assert shipping_cost(2.0) == 5.0\n\n"
        "def test_free():\n    assert shipping_cost(15.0) == 0.0\n\n"
        "def test_boundary_is_free():\n    assert shipping_cost(10.0) == 0.0\n",
    ),
    BenchmarkTask(
        "BIZ-002", Category.BUSINESS, "src/backend/access.py",
        "Implement can_edit(role: str, owner: bool) -> bool. Admins may always edit. Everyone "
        "else may edit only what they own.",
        "from src.backend.access import can_edit\n\n"
        "def test_admin():\n    assert can_edit('admin', False) is True\n\n"
        "def test_owner():\n    assert can_edit('user', True) is True\n\n"
        "def test_other():\n    assert can_edit('user', False) is False\n",
    ),
    BenchmarkTask(
        "BIZ-003", Category.BUSINESS, "src/backend/billing.py",
        "Implement late_fee(days_late: int) -> float. No fee until a payment is more than 3 "
        "days late; after that the fee is 1.5 per late day, counting every late day.",
        "from src.backend.billing import late_fee\n\n"
        "def test_grace():\n    assert late_fee(3) == 0.0\n\n"
        "def test_after_grace():\n    assert late_fee(4) == 6.0\n\n"
        "def test_none():\n    assert late_fee(0) == 0.0\n",
    ),
)


#: Known-correct implementations. Used only by M9's scaffold gate, and never shown to the
#: generator or the coder. Each is the simplest implementation that satisfies its
#: requirement, so a generated test that fails one is asserting the wrong thing.
_SCAFFOLDS: dict[str, str] = {
    "SEM-001": (
        'def running_total(values):\n'
        '    out = []\n'
        '    total = 0\n'
        '    for value in values:\n'
        '        total += value\n'
        '        out.append(total)\n'
        '    return out\n'
    ),
    "SEM-002": (
        'def top_scores(scores, n):\n'
        '    return sorted(scores, reverse=True)[:n]\n'
    ),
    "SEM-003": (
        'def apply_discount(price, percent):\n'
        '    return round(price - price * percent / 100.0, 2)\n'
    ),
    "API-001": (
        'def find_user(users, user_id):\n'
        '    return users.get(user_id)\n'
    ),
    "API-002": (
        'def make_response(data, ok):\n'
        "    return {'status': 'ok' if ok else 'error', 'data': data}\n"
    ),
    "API-003": (
        'def page(items, size):\n'
        '    return [items[i:i + size] for i in range(0, len(items), size)]\n'
    ),
    "EDGE-001": (
        'def clamp(value, low, high):\n'
        '    return max(low, min(high, value))\n'
    ),
    "EDGE-002": (
        'def last_n(items, n):\n'
        '    if n <= 0:\n'
        '        return []\n'
        '    return items[-n:]\n'
    ),
    "EDGE-003": (
        'def safe_average(values):\n'
        '    if not values:\n'
        '        return 0.0\n'
        '    return sum(values) / len(values)\n'
    ),
    "BIZ-001": (
        'def shipping_cost(weight_kg):\n'
        '    return 0.0 if weight_kg >= 10 else 5.0\n'
    ),
    "BIZ-002": (
        'def can_edit(role, owner):\n'
        "    return role == 'admin' or owner\n"
    ),
    "BIZ-003": (
        'def late_fee(days_late):\n'
        '    if days_late <= 3:\n'
        '        return 0.0\n'
        '    return round(days_late * 1.5, 2)\n'
    ),
}

#: Known-INCORRECT implementations, used only as the strength control. Kept away from the
#: gate entirely: a gate that could see the wrong implementation would be scoring itself.
#: Each differs from its scaffold by exactly the defect its category names.
_MUTANTS: dict[str, str] = {
    "SEM-001": (
        'def running_total(values):\n'
        '    return list(values)\n'
    ),
    "SEM-002": (
        'def top_scores(scores, n):\n'
        '    return sorted(scores)[:n]\n'
    ),
    "SEM-003": (
        'def apply_discount(price, percent):\n'
        '    return round(price + price * percent / 100.0, 2)\n'
    ),
    "API-001": (
        'def find_user(users, user_id):\n'
        '    return users[user_id]\n'
    ),
    "API-002": (
        'def make_response(data, ok):\n'
        "    return {'status': ok, 'data': data}\n"
    ),
    "API-003": (
        'def page(items, size):\n'
        '    return [items[i:i + size] for i in range(0, len(items), size + 1)]\n'
    ),
    "EDGE-001": (
        'def clamp(value, low, high):\n'
        '    return max(low, min(high - 1, value))\n'
    ),
    "EDGE-002": (
        'def last_n(items, n):\n'
        '    return items[-n:]\n'
    ),
    "EDGE-003": (
        'def safe_average(values):\n'
        '    return sum(values) / len(values)\n'
    ),
    "BIZ-001": (
        'def shipping_cost(weight_kg):\n'
        '    return 0.0 if weight_kg > 10 else 5.0\n'
    ),
    "BIZ-002": (
        'def can_edit(role, owner):\n'
        '    return owner\n'
    ),
    "BIZ-003": (
        'def late_fee(days_late):\n'
        '    if days_late < 3:\n'
        '        return 0.0\n'
        '    return round(days_late * 1.5, 2)\n'
    ),
}


#: The benchmark, with the correctness oracles attached.
TASKS: tuple[BenchmarkTask, ...] = tuple(
    replace(task, scaffold=_SCAFFOLDS[task.task_id], mutant=_MUTANTS[task.task_id])
    for task in _BASE
)


def by_category() -> dict[Category, tuple[BenchmarkTask, ...]]:
    """Tasks grouped for per-category reporting."""
    return {
        category: tuple(task for task in TASKS if task.category is category)
        for category in Category
    }
