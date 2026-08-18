"""M6.2: the quality layer wired into the real engineering loop.

The integration adds one way for a task to be rejected, so these tests exist to bound it. The
deterministic gates must still decide; the model must remain advisory; and turning the feature
off must cost exactly zero model calls, because the default is off and a default that quietly
spends inference is not off.

The subtle one is 7. A Judge that cannot be reached is *infrastructure*, and classifying that
as a coder defect would repeat the M5.2 mistake of attributing an environment fault to the
agent's work. It contributes nothing and the deterministic verdict stands.
"""

from __future__ import annotations

from pathlib import Path

from edith.config.loader import load_config
from edith.config.schema import ModelParams
from edith.engineering.executor import EngineeringExecutor, _render_quality
from edith.errors import FailureCategory
from edith.quality.artifacts import (
    FindingOrigin,
    QualityFinding,
    QualityReport,
    QualityVerdict,
    ReviewEvidence,
)
from edith.schemas.common import Severity
from edith.workspaces import ProjectWorkspace

from .fakes import FakeProvider

CLEAN = "def add(a: int, b: int) -> int:\n    return a + b\n"
INJECTION = "import os\n\n\ndef run(cmd: str) -> None:\n    os.system(cmd)\n"


class _CountingProvider(FakeProvider):
    """A provider that records whether it was asked for anything at all."""

    def __init__(self) -> None:
        super().__init__(ModelParams(model_name="t"), ['{"findings": []}'] * 40)


def executor_for(root: Path, *, model_review: bool) -> EngineeringExecutor:
    base = load_config(None)
    config = base.model_copy(
        update={
            "tools": base.tools.model_copy(update={"workspace_root": root}),
            "orchestration": base.orchestration.model_copy(
                update={"model_quality_review": model_review}
            ),
        }
    )
    return EngineeringExecutor(
        config,
        ProjectWorkspace(project_id="p", name="n", root=root),
        provider=_CountingProvider(),
        isolate=False,
    )


def write(root: Path, relative: str, content: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


class TestTheFlagControlsTheModel:
    def test_the_default_is_off(self) -> None:
        """M6.2 item 2: enabling by default would slow every task for an unproven benefit."""
        assert load_config(None).orchestration.model_quality_review is False

    def test_disabled_costs_zero_model_calls(self, tmp_path: Path) -> None:
        write(tmp_path, "src/a.py", CLEAN)
        executor = executor_for(tmp_path, model_review=False)
        report = executor._quality("T1", ("src/a.py",), report=None, root=tmp_path)
        assert report is not None
        assert executor._provider.calls == []  # type: ignore[union-attr]

    def test_enabled_invokes_the_reviewers(self, tmp_path: Path) -> None:
        write(tmp_path, "src/a.py", CLEAN)
        executor = executor_for(tmp_path, model_review=True)
        executor._quality("T1", ("src/a.py",), report=None, root=tmp_path)
        assert executor._provider.calls  # type: ignore[union-attr]

    def test_a_deterministic_block_skips_the_model_entirely(self, tmp_path: Path) -> None:
        """Item 3: no model call can change a verdict that is already BLOCKED."""
        write(tmp_path, "src/a.py", INJECTION)
        executor = executor_for(tmp_path, model_review=True)
        report = executor._quality("T1", ("src/a.py",), report=None, root=tmp_path)
        assert report is not None
        assert report.verdict() is QualityVerdict.BLOCKED
        assert executor._provider.calls == []  # type: ignore[union-attr]


class TestTheDeterministicGatesStillDecide:
    def test_a_critical_scanner_finding_rejects_the_task(self, tmp_path: Path) -> None:
        write(tmp_path, "src/a.py", INJECTION)
        executor = executor_for(tmp_path, model_review=False)
        report = executor._quality("T1", ("src/a.py",), report=None, root=tmp_path)
        assert report is not None and not report.verdict().merges

    def test_clean_code_passes_without_the_model(self, tmp_path: Path) -> None:
        write(tmp_path, "src/a.py", CLEAN)
        executor = executor_for(tmp_path, model_review=False)
        report = executor._quality("T1", ("src/a.py",), report=None, root=tmp_path)
        assert report is not None and report.verdict().merges

    def test_unreadable_changes_produce_no_report(self, tmp_path: Path) -> None:
        """No sources means no evidence, and no evidence means no finding."""
        executor = executor_for(tmp_path, model_review=False)
        assert executor._quality("T1", ("gone.py",), report=None, root=tmp_path) is None


class TestRepairRoutingUsesTheExistingPolicy:
    def test_a_repairable_verdict_maps_to_code_failure(self) -> None:
        """So it enters M5.2's existing budget rather than a second policy."""
        from edith.engineering.executor import REPAIRABLE_FAILURES

        assert FailureCategory.CODE_FAILURE in REPAIRABLE_FAILURES

    def test_a_blocked_verdict_maps_to_a_non_repairable_category(self) -> None:
        from edith.engineering.executor import REPAIRABLE_FAILURES

        assert FailureCategory.SECURITY_FAILURE not in REPAIRABLE_FAILURES

    def test_repair_evidence_puts_deterministic_findings_first(self) -> None:
        """An agent shown an opinion beside a real defect tends to address the opinion."""
        evidence = (ReviewEvidence(source="s", detail="d"),)
        report = QualityReport(
            findings=(
                QualityFinding(
                    category="style",
                    severity=Severity.HIGH,
                    summary="model opinion",
                    evidence=evidence,
                    origin=FindingOrigin.MODEL,
                ),
                QualityFinding(
                    category="command-injection",
                    severity=Severity.HIGH,
                    summary="real defect",
                    evidence=evidence,
                    origin=FindingOrigin.DETERMINISTIC,
                ),
            )
        )
        rendered = _render_quality(report)
        assert rendered.index("real defect") < len(rendered)
        assert "real defect" in rendered

    def test_only_blocking_findings_become_repair_evidence(self) -> None:
        evidence = (ReviewEvidence(source="s", detail="d"),)
        report = QualityReport(
            findings=(
                QualityFinding(
                    category="advisory",
                    severity=Severity.LOW,
                    summary="minor nit",
                    evidence=evidence,
                    origin=FindingOrigin.DETERMINISTIC,
                ),
            )
        )
        assert "minor nit" not in _render_quality(report)


class TestModelFailureIsNotACoderDefect:
    def test_a_judge_that_raises_does_not_block(self, tmp_path: Path) -> None:
        """Item 7: infrastructure failure must not be attributed to the agent's code."""
        write(tmp_path, "src/a.py", CLEAN)
        executor = executor_for(tmp_path, model_review=True)

        class _Broken(_CountingProvider):
            def structured_generate(self, *args: object, **kwargs: object) -> object:
                raise RuntimeError("ollama is down")

        executor._provider = _Broken()
        report = executor._quality("T1", ("src/a.py",), report=None, root=tmp_path)
        assert report is not None
        assert report.verdict().merges, "a broken reviewer must not fail good code"

    def test_a_model_finding_alone_never_blocks(self, tmp_path: Path) -> None:
        """Item 4 and 8: the adjudicator decides, and MODEL origin is not blocking."""
        report = QualityReport(
            findings=(
                QualityFinding(
                    category="correctness",
                    severity=Severity.CRITICAL,
                    summary="model is certain",
                    evidence=(ReviewEvidence(source="model", detail="hunch"),),
                    origin=FindingOrigin.MODEL,
                ),
            )
        )
        assert report.verdict() is QualityVerdict.PASS_WITH_ADVISORIES

    def test_judge_pass_cannot_override_a_deterministic_failure(self) -> None:
        """Item 5, restated at the integration boundary."""
        report = QualityReport(
            findings=(
                QualityFinding(
                    category="command-injection",
                    severity=Severity.CRITICAL,
                    summary="os.system",
                    evidence=(ReviewEvidence(source="ast", detail="os.system(cmd)"),),
                    origin=FindingOrigin.DETERMINISTIC,
                ),
            ),
            judge_verdict=QualityVerdict.PASS,
        )
        assert report.verdict() is QualityVerdict.BLOCKED


class TestAVerifiedTaskReachesMain:
    """M6.2's real finding: merge moved the workspace's *state* and nothing else.

    Every task reported COMPLETED and merged while main received no files, so six of six tasks
    "succeeded" and none passed an independently written acceptance test. The state machine was
    correct and the tree was empty, which is why no existing test caught it -- they all asserted
    on the ledger rather than on the filesystem.
    """

    def test_merging_copies_the_changed_files_into_the_main_tree(
        self, tmp_path: Path
    ) -> None:
        from edith.engineering.isolation import (
            TaskWorkspace,
            WorkspaceState,
            merge_workspace,
        )

        main = tmp_path / "main"
        work = tmp_path / "wt"
        (main / "src").mkdir(parents=True)
        (work / "src").mkdir(parents=True)
        (work / "src" / "new.py").write_text("VALUE = 1\n", encoding="utf-8")

        workspace = TaskWorkspace(
            workspace_id="task-t1",
            task_id="T1",
            execution_id="e",
            path=work,
            base_revision="abc123",
            state=WorkspaceState.VERIFIED,
        )
        decision = merge_workspace(
            None,  # type: ignore[arg-type]
            workspace,
            task_id="T1",
            verified=True,
            destination=main,
            changed_files=("src/new.py",),
        )
        assert not decision.refused
        assert (main / "src" / "new.py").read_text(encoding="utf-8") == "VALUE = 1\n"

    def test_a_refused_merge_leaves_main_untouched(self, tmp_path: Path) -> None:
        from edith.engineering.isolation import (
            TaskWorkspace,
            WorkspaceState,
            merge_workspace,
        )

        main = tmp_path / "main"
        work = tmp_path / "wt"
        main.mkdir()
        work.mkdir()
        (work / "leak.py").write_text("x = 1\n", encoding="utf-8")
        workspace = TaskWorkspace(
            workspace_id="task-t1",
            task_id="T1",
            execution_id="e",
            path=work,
            base_revision="abc",
            state=WorkspaceState.EXECUTING,
        )
        decision = merge_workspace(
            None,  # type: ignore[arg-type]
            workspace,
            task_id="T1",
            verified=False,
            destination=main,
            changed_files=("leak.py",),
        )
        assert decision.refused
        assert not (main / "leak.py").exists(), "an unverified task must not reach main"

    def test_a_traversing_path_is_refused_during_merge(self, tmp_path: Path) -> None:
        """A worktree path is not trusted merely because the workspace was verified."""
        from edith.engineering.isolation import (
            TaskWorkspace,
            WorkspaceState,
            merge_workspace,
        )

        main = tmp_path / "main"
        work = tmp_path / "wt"
        main.mkdir()
        work.mkdir()
        (tmp_path / "outside.py").write_text("escaped\n", encoding="utf-8")
        workspace = TaskWorkspace(
            workspace_id="task-t1",
            task_id="T1",
            execution_id="e",
            path=work,
            base_revision="abc",
            state=WorkspaceState.VERIFIED,
        )
        merge_workspace(
            None,  # type: ignore[arg-type]
            workspace,
            task_id="T1",
            verified=True,
            destination=main,
            changed_files=("../outside.py",),
        )
        assert (tmp_path / "outside.py").read_text(encoding="utf-8") == "escaped\n"
