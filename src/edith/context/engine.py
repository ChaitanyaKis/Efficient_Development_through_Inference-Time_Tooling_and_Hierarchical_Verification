"""The Context Engine: the smallest useful context for a task.

The question this subsystem answers is not "what is in the repository" but *"what is the
minimum this agent needs to do this task correctly"*. On a 3B model with an 8k window that
is not an optimisation, it is the difference between working and not working.

Retrieval is lexical and structural: path relevance, filename relevance, term overlap,
import relationships, and test pairing. No embeddings and no vector database (CLAUDE.md
explicitly warns against starting there). :class:`Retriever` is the seam where an embedding
ranker can be added later without touching the agents.

Everything the engine reads goes through the M1 tool gateway, so the Context Engine has
exactly the same filesystem permissions as the agent it is serving -- it cannot become a
side channel to files the agent could not open itself.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Protocol

from pydantic import Field

from edith.config.schema import ContextConfig
from edith.observability.logging import get_logger
from edith.schemas.common import EdithModel
from edith.tools.gateway import ToolGateway
from edith.tools.schemas import ToolCall

logger = get_logger(__name__)

#: Extensions worth reading as source.
_SOURCE_SUFFIXES = frozenset(
    {".py", ".js", ".ts", ".tsx", ".jsx", ".go", ".rs", ".java", ".cs", ".rb", ".php"}
)
_DOC_SUFFIXES = frozenset({".md", ".rst", ".txt"})

#: Documents that describe the project rather than implement it.
_DOC_NAMES = frozenset(
    {"readme", "architecture", "system", "design", "api", "contributing", "adr"}
)

#: Words too common to carry signal when matching a request to a file.
_STOPWORDS = frozenset(
    {
        "a", "an", "the", "and", "or", "to", "of", "in", "for", "on", "with", "that",
        "this", "it", "is", "are", "be", "add", "make", "new", "use", "using", "should",
        "must", "please", "function", "method", "code", "file", "files", "test", "tests",
        "implement", "create", "update", "change", "fix", "so", "when", "then", "if",
    }
)

_WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")
_PY_SYMBOL_RE = re.compile(r"^\s*(?:def|class)\s+([A-Za-z_][A-Za-z0-9_]*)", re.MULTILINE)


def _defined_symbols(content: str) -> list[str]:
    """Top-level function and class names in a Python source excerpt.

    Naming them in the prompt is what stops the coder quietly dropping existing functions
    when it rewrites a file.
    """
    return [f"{name}()" for name in _PY_SYMBOL_RE.findall(content)[:12]]


def keywords(text: str) -> set[str]:
    """Extract lower-cased content words from free text."""
    return {
        word.lower()
        for word in _WORD_RE.findall(text)
        if word.lower() not in _STOPWORDS
    }


class ContextFile(EdithModel):
    """One file included in a bundle."""

    path: str
    content: str
    score: float = 0.0
    truncated: bool = False
    reason: str = ""


class ContextBundle(EdithModel):
    """The assembled context handed to an agent.

    ``rationale`` exists so a human reviewing a failed run can see *why* the model was shown
    what it was shown -- a bad answer is often a retrieval failure, not a reasoning failure.
    """

    task_summary: str = ""
    relevant_files: list[ContextFile] = Field(default_factory=list)
    relevant_symbols: list[str] = Field(default_factory=list)
    relevant_tests: list[str] = Field(default_factory=list)
    relevant_docs: list[str] = Field(default_factory=list)
    rationale: list[str] = Field(default_factory=list)
    estimated_context_chars: int = 0
    files_considered: int = 0
    truncated: bool = False
    #: Set when retrieval produced suspiciously little to work with. The M2 ``**/*`` glob
    #: bug silently returned an empty bundle for every task and the loop carried on
    #: regardless, so an under-populated bundle is now an explicit, inspectable condition
    #: rather than something a caller has to notice for itself.
    degraded: bool = False
    degraded_reason: str = ""

    @property
    def usable(self) -> bool:
        """Whether this bundle gives an agent enough to work with."""
        return not self.degraded

    def render(self) -> str:
        """Render the bundle as prompt text.

        Each file is introduced by a sentence rather than a delimiter line. A small model
        shown ``--- FILE: x.py ---`` above some code reliably copies that line into any file
        it writes; a prose introduction is far less likely to be mistaken for content.
        """
        if not self.relevant_files:
            return "(no repository context was retrieved)"
        blocks = []
        for entry in self.relevant_files:
            suffix = "\n(file truncated here)" if entry.truncated else ""
            symbols = _defined_symbols(entry.content)
            defines = f" It currently defines: {', '.join(symbols)}." if symbols else ""
            blocks.append(
                f"File `{entry.path}` contains the following code.{defines}\n\n"
                f"{entry.content}{suffix}"
            )
        return "\n\n\n".join(blocks)

    @property
    def file_paths(self) -> list[str]:
        """Paths of every included file."""
        return [entry.path for entry in self.relevant_files]


@dataclass
class Candidate:
    """A scored retrieval candidate."""

    path: str
    score: float = 0.0
    reasons: list[str] = field(default_factory=list)
    #: Points earned from the *query* rather than from structure. Documentation and test
    #: pairing score a little for every task, so total score alone cannot distinguish
    #: "relevant to this request" from "always mildly present".
    query_hits: int = 0

    def add(self, points: float, reason: str, *, from_query: bool = False) -> None:
        """Add to the score and record why."""
        self.score += points
        self.reasons.append(reason)
        if from_query:
            self.query_hits += 1


class Retriever(Protocol):
    """Pluggable ranking strategy.

    The seam for future embedding-based retrieval: implement this and pass it to
    :class:`ContextEngine`. Agents are unaffected.
    """

    def score(self, candidates: dict[str, Candidate], query: str) -> None:
        """Adjust candidate scores in place."""


class LexicalRetriever:
    """Default ranker: path, filename, term overlap, and content matches."""

    def __init__(self, gateway: ToolGateway) -> None:
        self.gateway = gateway

    def score(self, candidates: dict[str, Candidate], query: str) -> None:
        """Score by filename and path overlap with the request's keywords."""
        terms = keywords(query)
        if not terms:
            return
        for candidate in candidates.values():
            lowered = candidate.path.lower()
            stem = lowered.rsplit("/", 1)[-1].rsplit(".", 1)[0]
            stem_parts = set(re.split(r"[_\-.]", stem)) - {""}

            for term in terms:
                if term == stem:
                    candidate.add(10.0, f"filename matches '{term}'", from_query=True)
                elif term in stem_parts:
                    candidate.add(6.0, f"filename contains '{term}'", from_query=True)
                elif term in lowered:
                    candidate.add(2.0, f"path mentions '{term}'", from_query=True)

    def score_content(
        self, candidates: dict[str, Candidate], query: str, contents: dict[str, str]
    ) -> None:
        """Add points for query terms appearing in file content."""
        terms = keywords(query)
        for path, text in contents.items():
            candidate = candidates.get(path)
            if candidate is None:
                continue
            lowered = text.lower()
            hits = sum(1 for term in terms if term in lowered)
            if hits:
                candidate.add(
                    min(hits * 1.5, 9.0),
                    f"content mentions {hits} query term(s)",
                    from_query=True,
                )


class ContextEngine:
    """Builds a :class:`ContextBundle` for a task, under a character budget."""

    def __init__(
        self,
        gateway: ToolGateway,
        config: ContextConfig | None = None,
        retriever: Retriever | None = None,
    ) -> None:
        """
        Args:
            gateway: Permission-scoped tool gateway. All reads go through it.
            config: Budget settings.
            retriever: Ranking strategy; defaults to :class:`LexicalRetriever`.
        """
        self.gateway = gateway
        self.config = config or ContextConfig()
        self._lexical = LexicalRetriever(gateway)
        self.retriever: Retriever = retriever or self._lexical

    # -- Repository scan -------------------------------------------------------------

    def _list_files(self) -> list[str]:
        """List candidate source files via the gateway.

        Uses ``filesystem.search``, so protected and out-of-scope files are already absent.
        """
        result = self.gateway.execute(
            ToolCall(
                tool="filesystem.search",
                arguments={"name_pattern": "**/*", "max_results": 200},
            )
        )
        if not result.ok:
            logger.warning("context.scan_failed", error=result.error)
            return []
        return [match["path"] for match in result.output.get("matches", [])]

    def _read(self, path: str) -> str | None:
        """Read a file through the gateway, returning ``None`` when unavailable."""
        result = self.gateway.execute(
            ToolCall(tool="filesystem.read", arguments={"path": path})
        )
        if not result.ok:
            return None
        content = result.output.get("content", "")
        return content if isinstance(content, str) else None

    # -- Classification --------------------------------------------------------------

    @staticmethod
    def _is_test(path: str) -> bool:
        lowered = path.lower()
        name = lowered.rsplit("/", 1)[-1]
        return (
            name.startswith("test_")
            or name.endswith("_test.py")
            or ".test." in name
            or ".spec." in name
            or lowered.startswith("tests/")
            or "/tests/" in lowered
        )

    @staticmethod
    def _is_doc(path: str) -> bool:
        lowered = path.lower()
        suffix = "." + lowered.rsplit(".", 1)[-1] if "." in lowered else ""
        if suffix not in _DOC_SUFFIXES:
            return False
        stem = lowered.rsplit("/", 1)[-1].rsplit(".", 1)[0]
        return stem in _DOC_NAMES or "/docs/" in lowered or lowered.startswith("docs/")

    @staticmethod
    def _is_source(path: str) -> bool:
        suffix = "." + path.rsplit(".", 1)[-1] if "." in path else ""
        return suffix in _SOURCE_SUFFIXES

    @staticmethod
    def _test_partner(path: str) -> str:
        """The conventional test filename for a source path."""
        name = path.rsplit("/", 1)[-1]
        stem = name.rsplit(".", 1)[0]
        return f"test_{stem}"

    # -- Bundle construction ---------------------------------------------------------

    def build(
        self,
        query: str,
        *,
        hint_paths: tuple[str, ...] = (),
        task_summary: str = "",
    ) -> ContextBundle:
        """Assemble the smallest useful context for ``query``.

        Args:
            query: The task description or user request driving retrieval.
            hint_paths: Paths the planner or task named explicitly. These are strongly
                boosted -- an explicit instruction outranks a heuristic guess.
            task_summary: Short description recorded on the bundle.
        """
        all_paths = self._list_files()
        candidates: dict[str, Candidate] = {
            path: Candidate(path=path) for path in all_paths if self._is_indexable(path)
        }

        for hint in hint_paths:
            normalized = hint.replace("\\", "/").lstrip("./")
            candidate = candidates.get(normalized)
            if candidate is None:
                candidate = Candidate(path=normalized)
                candidates[normalized] = candidate
            candidate.add(50.0, "named explicitly by the task")

        self.retriever.score(candidates, query)
        self._score_structure(candidates, query)

        # Read the strongest candidates, then let their content refine the ranking. Reading
        # a bounded shortlist keeps this cheap on a constrained machine.
        shortlist = sorted(
            candidates.values(), key=lambda c: (-c.score, c.path)
        )[: self.config.max_files * 3]
        contents: dict[str, str] = {}
        for candidate in shortlist:
            if candidate.score <= 0:
                continue
            text = self._read(candidate.path)
            if text is not None:
                contents[candidate.path] = text
        self._lexical.score_content(candidates, query, contents)

        ranked = [
            candidate
            for candidate in sorted(candidates.values(), key=lambda c: (-c.score, c.path))
            if candidate.score > 0
        ]
        return self._assemble(ranked, contents, task_summary or query, len(candidates))

    def _is_indexable(self, path: str) -> bool:
        """Whether a path is worth considering at all."""
        if self._is_source(path) or self._is_doc(path):
            return True
        return path.lower().endswith((".toml", ".cfg", ".ini", ".yaml", ".yml", ".json"))

    def _score_structure(self, candidates: dict[str, Candidate], query: str) -> None:
        """Apply structural signals: test pairing, doc relevance, and depth."""
        terms = keywords(query)
        source_stems = {
            candidate.path.rsplit("/", 1)[-1].rsplit(".", 1)[0]
            for candidate in candidates.values()
            if self._is_source(candidate.path) and not self._is_test(candidate.path)
        }

        for candidate in candidates.values():
            path = candidate.path
            if self._is_test(path):
                if not self.config.include_tests:
                    candidate.score = 0.0
                    continue
                # A test pairing with a source file the request mentions is highly
                # relevant: it usually *is* the acceptance criterion.
                stem = path.rsplit("/", 1)[-1].rsplit(".", 1)[0]
                partner = stem.removeprefix("test_").removesuffix("_test")
                if partner in source_stems and partner in terms:
                    candidate.add(8.0, f"tests '{partner}', which the request names")
                elif partner in source_stems:
                    candidate.add(3.0, f"tests '{partner}'")
            elif self._is_doc(path):
                if not self.config.include_docs:
                    candidate.score = 0.0
                    continue
                candidate.add(1.0, "project documentation")

    def _assemble(
        self,
        ranked: list[Candidate],
        contents: dict[str, str],
        task_summary: str,
        considered: int,
    ) -> ContextBundle:
        """Fill the bundle up to the configured budget."""
        files: list[ContextFile] = []
        symbols: list[str] = []
        tests: list[str] = []
        docs: list[str] = []
        rationale: list[str] = []
        used = 0
        truncated = False

        for candidate in ranked:
            if len(files) >= self.config.max_files:
                truncated = True
                break
            text = contents.get(candidate.path)
            if text is None:
                text = self._read(candidate.path)
                if text is None:
                    continue

            excerpt, was_cut = self._fit(text, self.config.max_file_chars)
            if used + len(excerpt) > self.config.max_total_chars:
                remaining = self.config.max_total_chars - used
                if remaining < 200:
                    truncated = True
                    break
                excerpt, was_cut = self._fit(text, remaining)

            files.append(
                ContextFile(
                    path=candidate.path,
                    content=excerpt,
                    score=round(candidate.score, 2),
                    truncated=was_cut,
                    reason="; ".join(candidate.reasons[:3]),
                )
            )
            used += len(excerpt)
            rationale.append(
                f"{candidate.path} (score {candidate.score:.1f}): "
                f"{'; '.join(candidate.reasons[:2]) or 'general relevance'}"
            )

            if self._is_test(candidate.path):
                tests.append(candidate.path)
            elif self._is_doc(candidate.path):
                docs.append(candidate.path)
            if candidate.path.endswith(".py"):
                symbols.extend(
                    f"{candidate.path}:{name}" for name in _PY_SYMBOL_RE.findall(text)[:12]
                )

        query_hits = sum(1 for candidate in ranked if candidate.query_hits)
        degraded, reason = self._assess(files, considered, used, query_hits)
        bundle = ContextBundle(
            task_summary=task_summary,
            relevant_files=files,
            relevant_symbols=symbols[:40],
            relevant_tests=tests,
            relevant_docs=docs,
            rationale=rationale,
            estimated_context_chars=used,
            files_considered=considered,
            truncated=truncated,
            degraded=degraded,
            degraded_reason=reason,
        )
        if degraded:
            logger.warning(
                "context.degraded", reason=reason, considered=considered, files=len(files)
            )
        logger.info(
            "context.built",
            files=len(files),
            considered=considered,
            chars=used,
            truncated=truncated,
        )
        return bundle

    @staticmethod
    def _assess(
        files: list[ContextFile], considered: int, used: int, query_hits: int = 1
    ) -> tuple[bool, str]:
        """Decide whether a bundle is too thin to be trusted.

        Distinguishes the three ways retrieval can under-deliver, because they have
        different causes and different fixes:

        - nothing was indexed at all (the scan failed, or the workspace is empty);
        - files existed but none scored (the query matched nothing);
        - a file was selected but almost no content came back (a read failed).
        """
        if considered == 0:
            return (True, "no files were indexed; the repository scan returned nothing")
        if not files or query_hits == 0:
            return (
                True,
                f"{considered} file(s) were indexed but none matched the task",
            )
        if used < 50:
            return (True, f"only {used} characters of content were retrieved")
        return (False, "")

    @staticmethod
    def _fit(text: str, limit: int) -> tuple[str, bool]:
        """Cut ``text`` to ``limit`` characters on a line boundary where possible."""
        if len(text) <= limit:
            return text, False
        clipped = text[:limit]
        newline = clipped.rfind("\n")
        if newline > limit // 2:
            clipped = clipped[:newline]
        return clipped, True
