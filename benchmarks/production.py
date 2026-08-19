"""M13: an independent benchmark, written after M10-M12 and never used to tune them.

The semantic benchmark in :mod:`benchmarks.semantic` shaped three milestones of work. Reusing
it to declare EDITH finished would measure how well EDITH fits the thing it was built against,
which is the one number that cannot be trusted.

So every task here is new. The requirements were written for this milestone, the acceptance
tests with them, and nothing in M10's boundary tables, M11's characterisation or M12's repair
fix was derived from any of them.

Ground truth rules, unchanged from M7 and enforced by the runner:

- ``requirement`` is all the coder is told;
- ``acceptance`` is hand-written, never shown to the coder, and runs in a separate process
  against the merged tree after the task is done;
- nothing in the quality or review pipeline can see or influence it.

``ambiguous=True`` marks a requirement that genuinely permits more than one reading. Those are
scored separately: a model choosing a defensible interpretation is not a defect, and counting
it as one would make the benchmark punish honesty.

``hidden_edge`` records an edge case the requirement implies but does not spell out. It exists
so a passing result can be distinguished from a result that only passes the obvious cases.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Area(StrEnum):
    """What kind of work the task represents."""

    CRUD = "crud"
    API = "api"
    DATABASE = "database"
    AUTH = "auth"
    FILE_IO = "file_io"
    BUSINESS_RULE = "business_rule"
    BOUNDARY = "boundary"
    TRANSFORM = "transform"
    ERROR_HANDLING = "error_handling"
    MULTI_FILE = "multi_file"
    DEPENDENCY = "dependency"
    AMBIGUOUS = "ambiguous"


@dataclass(frozen=True)
class ProductionTask:
    """One independent task and the human-authored test that judges it."""

    task_id: str
    area: Area
    paths: tuple[str, ...]
    requirement: str
    acceptance: str
    #: True when the requirement genuinely permits more than one correct implementation.
    ambiguous: bool = False
    #: An edge case the requirement implies but does not state outright.
    hidden_edge: str = ""


TASKS: tuple[ProductionTask, ...] = (
    ProductionTask(
        "PRD-001", Area.CRUD, ("src/backend/notes.py",),
        "Implement a NoteStore class with add(title, body) returning a new integer id starting "
        "at 1 and increasing, get(note_id) returning the note as a dict with keys 'title' and "
        "'body' or None when absent, and delete(note_id) returning True when a note was "
        "removed and False otherwise.",
        "from src.backend.notes import NoteStore\n\n"
        "def test_add_returns_increasing_ids():\n    s = NoteStore()\n"
        "    assert s.add('a', 'x') == 1\n    assert s.add('b', 'y') == 2\n\n"
        "def test_get_returns_the_note():\n    s = NoteStore()\n    i = s.add('a', 'x')\n"
        "    assert s.get(i) == {'title': 'a', 'body': 'x'}\n\n"
        "def test_get_missing_is_none():\n    assert NoteStore().get(99) is None\n\n"
        "def test_delete_reports_whether_it_removed():\n    s = NoteStore()\n"
        "    i = s.add('a', 'x')\n    assert s.delete(i) is True\n"
        "    assert s.delete(i) is False\n",
        hidden_edge="ids must not be reused after a delete",
    ),
    ProductionTask(
        "PRD-002", Area.API, ("src/backend/routes.py",),
        "Implement handle(method: str, path: str) -> tuple[int, dict] returning an HTTP "
        "status code and a body. GET /health returns 200 and {'status': 'ok'}. Any unknown path "
        "returns 404 and {'error': 'not found'}. Any known path with an unsupported method "
        "returns 405 and {'error': 'method not allowed'}.",
        "from src.backend.routes import handle\n\n"
        "def test_health():\n    assert handle('GET', '/health') == (200, {'status': 'ok'})\n\n"
        "def test_unknown_path_is_404():\n"
        "    assert handle('GET', '/nope')[0] == 404\n\n"
        "def test_wrong_method_is_405():\n"
        "    assert handle('POST', '/health')[0] == 405\n",
        hidden_edge="405 takes precedence over 404 only for a path that exists",
    ),
    ProductionTask(
        "PRD-003", Area.DATABASE, ("src/backend/repo.py",),
        "Implement a UserRepo class backed by an in-memory dict. save(user) takes a dict with "
        "an 'email' key and stores it keyed by email, returning the stored dict. find_by_email"
        "(email) returns the stored dict or None. Saving the same email twice must overwrite "
        "rather than duplicate, and count() returns the number of distinct users.",
        "from src.backend.repo import UserRepo\n\n"
        "def test_save_and_find():\n    r = UserRepo()\n    r.save({'email': 'a@b.c'})\n"
        "    assert r.find_by_email('a@b.c') == {'email': 'a@b.c'}\n\n"
        "def test_missing_is_none():\n    assert UserRepo().find_by_email('x@y.z') is None\n\n"
        "def test_overwrite_does_not_duplicate():\n    r = UserRepo()\n"
        "    r.save({'email': 'a@b.c', 'n': 1})\n    r.save({'email': 'a@b.c', 'n': 2})\n"
        "    assert r.count() == 1\n    assert r.find_by_email('a@b.c')['n'] == 2\n",
        hidden_edge="overwrite must replace the record, not merge it",
    ),
    ProductionTask(
        "PRD-004", Area.AUTH, ("src/backend/authz.py",),
        "Implement may_delete(role: str, is_owner: bool, is_locked: bool) -> bool. An admin may "
        "delete anything that is not locked. A regular user may delete only their own item, and "
        "only when it is not locked. Nobody may delete a locked item.",
        "from src.backend.authz import may_delete\n\n"
        "def test_admin_can_delete_others():\n"
        "    assert may_delete('admin', False, False) is True\n\n"
        "def test_owner_can_delete_own():\n"
        "    assert may_delete('user', True, False) is True\n\n"
        "def test_user_cannot_delete_others():\n"
        "    assert may_delete('user', False, False) is False\n\n"
        "def test_locked_blocks_admin_too():\n"
        "    assert may_delete('admin', True, True) is False\n",
        hidden_edge="the lock overrides the admin role, not the other way round",
    ),
    ProductionTask(
        "PRD-005", Area.FILE_IO, ("src/backend/textfile.py",),
        "Implement count_lines(path: str) -> int returning the number of lines in a UTF-8 text "
        "file, and raising FileNotFoundError when the path does not exist. An empty file has "
        "zero lines.",
        "import pytest\nfrom src.backend.textfile import count_lines\n\n"
        "def test_counts_lines(tmp_path):\n    p = tmp_path / 'a.txt'\n"
        "    p.write_text('one\\ntwo\\nthree\\n', encoding='utf-8')\n"
        "    assert count_lines(str(p)) == 3\n\n"
        "def test_empty_file_is_zero(tmp_path):\n    p = tmp_path / 'e.txt'\n"
        "    p.write_text('', encoding='utf-8')\n    assert count_lines(str(p)) == 0\n\n"
        "def test_missing_file_raises(tmp_path):\n"
        "    with pytest.raises(FileNotFoundError):\n"
        "        count_lines(str(tmp_path / 'nope.txt'))\n",
        hidden_edge="a file with no trailing newline still counts its last line",
    ),
    ProductionTask(
        "PRD-006", Area.BUSINESS_RULE, ("src/backend/loyalty.py",),
        "Implement tier(points: int) -> str. Fewer than 100 points is 'bronze'. From 100 up to "
        "but not including 500 is 'silver'. 500 or more is 'gold'. Negative points are treated "
        "as zero.",
        "from src.backend.loyalty import tier\n\n"
        "def test_bronze():\n    assert tier(0) == 'bronze'\n    assert tier(99) == 'bronze'\n\n"
        "def test_silver():\n    assert tier(100) == 'silver'\n    assert tier(499) == 'silver'\n\n"
        "def test_gold():\n    assert tier(500) == 'gold'\n\n"
        "def test_negative_is_bronze():\n    assert tier(-5) == 'bronze'\n",
        hidden_edge="both tier boundaries are inclusive-below, exclusive-above",
    ),
    ProductionTask(
        "PRD-007", Area.BOUNDARY, ("src/backend/quota.py",),
        "Implement over_quota(used_mb: float, limit_mb: float) -> bool returning True only when "
        "usage is strictly more than the limit. Exactly at the limit is not over quota.",
        "from src.backend.quota import over_quota\n\n"
        "def test_under():\n    assert over_quota(4.0, 5.0) is False\n\n"
        "def test_exactly_at_limit_is_not_over():\n"
        "    assert over_quota(5.0, 5.0) is False\n\n"
        "def test_over():\n    assert over_quota(5.1, 5.0) is True\n",
        hidden_edge="equality is the whole test",
    ),
    ProductionTask(
        "PRD-008", Area.TRANSFORM, ("src/backend/slugify.py",),
        "Implement slugify(title: str) -> str returning a lowercase slug where runs of "
        "non-alphanumeric characters become a single hyphen, with no leading or trailing "
        "hyphen. An empty or purely symbolic title returns an empty string.",
        "from src.backend.slugify import slugify\n\n"
        "def test_basic():\n    assert slugify('Hello World') == 'hello-world'\n\n"
        "def test_runs_collapse():\n    assert slugify('a  --  b') == 'a-b'\n\n"
        "def test_trimmed():\n    assert slugify('  Hi!  ') == 'hi'\n\n"
        "def test_symbols_only_is_empty():\n    assert slugify('!!!') == ''\n",
        hidden_edge="a symbols-only title must not produce a lone hyphen",
    ),
    ProductionTask(
        "PRD-009", Area.ERROR_HANDLING, ("src/backend/parsing.py",),
        "Implement parse_config(text: str) -> dict turning lines of 'key=value' into a dict. "
        "Blank lines and lines beginning with # are ignored. A line with no '=' raises "
        "ValueError naming the offending line. Values keep any '=' after the first one.",
        "import pytest\nfrom src.backend.parsing import parse_config\n\n"
        "def test_parses():\n    assert parse_config('a=1\\nb=2') == {'a': '1', 'b': '2'}\n\n"
        "def test_ignores_comments_and_blanks():\n"
        "    assert parse_config('# c\\n\\na=1') == {'a': '1'}\n\n"
        "def test_value_keeps_later_equals():\n"
        "    assert parse_config('k=a=b') == {'k': 'a=b'}\n\n"
        "def test_bad_line_raises():\n"
        "    with pytest.raises(ValueError):\n        parse_config('nope')\n",
        hidden_edge="split on the first '=' only",
    ),
    ProductionTask(
        "PRD-010", Area.MULTI_FILE, ("src/backend/money2.py", "src/backend/invoice.py"),
        "Create two modules. src/backend/money2.py provides to_cents(amount: float) -> int "
        "rounding to the nearest cent. src/backend/invoice.py provides total_cents(lines: "
        "list[float]) -> int which imports to_cents from src.backend.money2 and sums the "
        "converted values.",
        "from src.backend.invoice import total_cents\nfrom src.backend.money2 import to_cents\n\n"
        "def test_to_cents():\n    assert to_cents(1.005) in (100, 101)\n"
        "    assert to_cents(2.5) == 250\n\n"
        "def test_total_uses_conversion():\n"
        "    assert total_cents([1.0, 2.5]) == 350\n\n"
        "def test_empty_total_is_zero():\n    assert total_cents([]) == 0\n",
        hidden_edge="invoice must import from money2 rather than reimplementing conversion",
    ),
    ProductionTask(
        "PRD-011", Area.DEPENDENCY, ("src/backend/timeutil.py",),
        "Implement iso_day(timestamp: float) -> str returning the UTC calendar date of a Unix "
        "timestamp in YYYY-MM-DD form, using only the Python standard library.",
        "from src.backend.timeutil import iso_day\n\n"
        "def test_epoch():\n    assert iso_day(0) == '1970-01-01'\n\n"
        "def test_known_date():\n    assert iso_day(1700000000) == '2023-11-14'\n",
        hidden_edge="must be UTC, not local time",
    ),
    ProductionTask(
        "PRD-012", Area.AMBIGUOUS, ("src/backend/trim.py",),
        "Implement shorten(text: str, limit: int) -> str which shortens text that is longer "
        "than the limit so the result fits within it.",
        "from src.backend.trim import shorten\n\n"
        "def test_short_text_is_unchanged():\n"
        "    assert shorten('abc', 10) == 'abc'\n\n"
        "def test_long_text_fits_the_limit():\n"
        "    assert len(shorten('abcdefghij', 5)) <= 5\n",
        ambiguous=True,
        hidden_edge="whether an ellipsis is added, and whether it counts toward the limit",
    ),
)


def by_area() -> dict[Area, tuple[ProductionTask, ...]]:
    return {area: tuple(t for t in TASKS if t.area is area) for area in Area}
