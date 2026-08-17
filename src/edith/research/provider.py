"""Provider-neutral research retrieval, plus the cache.

:class:`ResearchProvider` is the seam. The Research Agent talks to this interface and never
to a specific engine, so swapping or adding a backend touches nothing above it.

Two providers ship: :class:`OfflineProvider`, which is honest about being unable to reach
the network, and :class:`DuckDuckGoProvider`, which needs no API key and therefore no money
(CLAUDE.md: $0 budget, no paid services).
"""

from __future__ import annotations

import hashlib
import json
import re
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from html import unescape
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import httpx

from edith.errors import EdithError, FailureCategory
from edith.observability.logging import get_logger
from edith.schemas.common import utc_now

from .extract import MAX_EXTRACT_CHARS, classify_source, extract_text, neutralize
from .schema import RetrievalStatus, SearchHit, Source

logger = get_logger(__name__)

DEFAULT_TIMEOUT_SECONDS = 15.0
DEFAULT_CACHE_TTL_HOURS = 24.0
MAX_FETCH_BYTES = 2_000_000

#: Hard ceiling on a stored excerpt, matching the Source schema's field limit.
MAX_EXCERPT_CHARS = MAX_EXTRACT_CHARS

#: Sent so operators of the sites being read can identify the traffic.
USER_AGENT = "Edith-Research/0.1 (local autonomous engineering assistant)"


class ResearchUnavailableError(EdithError):
    """Research cannot run: offline, no provider, or the provider is unreachable."""

    category = FailureCategory.ENVIRONMENT_FAILURE


class SourceUnavailableError(EdithError):
    """One source could not be retrieved."""

    category = FailureCategory.TOOL_ERROR


class ResearchTimeoutError(EdithError):
    """A retrieval exceeded its budget."""

    category = FailureCategory.TIMEOUT
    default_retryable = True


class ResearchCache:
    """Content-addressed cache of fetched sources.

    Keyed by URL, storing the body alongside metadata: timestamp, content hash, status. A
    repeated question does not re-fetch the same page, which matters both for politeness to
    the sites being read and for reproducibility of a report.
    """

    def __init__(self, root: Path, ttl_hours: float = DEFAULT_CACHE_TTL_HOURS) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.ttl = timedelta(hours=ttl_hours)

    @staticmethod
    def key_for(url: str) -> str:
        """Stable cache key for a URL."""
        return hashlib.sha256(url.encode("utf-8")).hexdigest()

    def _paths(self, url: str) -> tuple[Path, Path]:
        key = self.key_for(url)
        shard = self.root / key[:2]
        return (shard / f"{key}.json", shard / f"{key}.body")

    def get(self, url: str) -> dict[str, Any] | None:
        """Return cached metadata and body, or ``None`` when absent or stale."""
        meta_path, body_path = self._paths(url)
        if not meta_path.is_file() or not body_path.is_file():
            return None
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            fetched = datetime.fromisoformat(meta["retrieved_at"])
        except (OSError, ValueError, KeyError):
            logger.warning("research.cache_unreadable", url=url[:120])
            return None

        if utc_now() - fetched > self.ttl:
            logger.info("research.cache_stale", url=url[:120])
            return None

        try:
            meta["body"] = body_path.read_text(encoding="utf-8")
        except OSError:
            return None
        logger.info("research.cache_hit", url=url[:120])
        entry: dict[str, Any] = meta
        return entry

    def put(self, url: str, body: str, *, status: str, title: str = "") -> str:
        """Store a fetched body and return its content hash."""
        meta_path, body_path = self._paths(url)
        body_path.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
        body_path.write_text(body, encoding="utf-8")
        meta_path.write_text(
            json.dumps(
                {
                    "url": url,
                    "title": title,
                    "retrieved_at": utc_now().isoformat(),
                    "content_hash": digest,
                    "status": status,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return digest

    def invalidate(self, url: str) -> bool:
        """Drop one entry."""
        removed = False
        for path in self._paths(url):
            if path.is_file():
                path.unlink()
                removed = True
        return removed

    def clear(self) -> int:
        """Drop every entry, returning how many bodies were removed."""
        removed = 0
        for path in self.root.rglob("*.body"):
            path.unlink()
            removed += 1
        for path in self.root.rglob("*.json"):
            path.unlink()
        return removed


class ResearchProvider(ABC):
    """A source of external information."""

    #: Stable identifier used in logs and on results.
    name: str = "abstract"

    @abstractmethod
    def search(self, query: str, *, limit: int = 5) -> list[SearchHit]:
        """Find candidate sources for a query."""

    @abstractmethod
    def fetch(self, url: str) -> Source:
        """Retrieve one source. Never raises for an expected failure."""

    @abstractmethod
    def health_check(self) -> tuple[bool, str]:
        """Return ``(available, detail)``."""

    def extract(self, source: Source) -> Source:
        """Normalize a fetched source's content.

        Neutralisation happens here so every provider inherits it -- a backend added later
        cannot forget to defang what it retrieved.

        Re-truncated afterwards: annotating a line makes it *longer*, so a page that was
        just inside the excerpt limit can exceed it once defanged. Trimming here rather
        than raising the limit keeps the prompt budget honest.
        """
        safe, detections = neutralize(source.excerpt)
        if detections:
            source.excerpt = safe[:MAX_EXCERPT_CHARS]
        return source


class OfflineProvider(ResearchProvider):
    """The provider used when research is disabled or the network is unavailable.

    Returns nothing and says why. It never fabricates a result -- an empty, honest answer is
    the only correct behaviour when retrieval is impossible.
    """

    name = "offline"

    def __init__(self, reason: str = "research is disabled or the network is unavailable") -> None:
        self.reason = reason

    def search(self, query: str, *, limit: int = 5) -> list[SearchHit]:
        """Return nothing; research is unavailable."""
        logger.info("research.offline_search", query=query[:120])
        return []

    def fetch(self, url: str) -> Source:
        """Return an unavailable source rather than inventing content."""
        return Source(
            url=url,
            status=RetrievalStatus.UNAVAILABLE,
            error=self.reason,
            tier=classify_source(url),
        )

    def health_check(self) -> tuple[bool, str]:
        """Always unavailable, with the configured reason."""
        return (False, self.reason)


class DuckDuckGoProvider(ResearchProvider):
    """Search and fetch over plain HTTP, with no API key.

    DuckDuckGo's HTML endpoint needs no account and no payment, which is what makes it
    usable under a $0 budget. It is one implementation of the interface, not a dependency of
    the architecture.
    """

    name = "duckduckgo"
    SEARCH_URL = "https://html.duckduckgo.com/html/"

    def __init__(
        self,
        cache: ResearchCache,
        *,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        client: httpx.Client | None = None,
    ) -> None:
        self.cache = cache
        self.timeout_seconds = timeout_seconds
        self._owns_client = client is None
        self._client = client or httpx.Client(
            timeout=timeout_seconds,
            follow_redirects=True,
            headers={"User-Agent": USER_AGENT},
        )

    _RESULT_RE = re.compile(
        r'<a[^>]+class="[^"]*result__a[^"]*"[^>]+href="([^"]+)"[^>]*>(.*?)</a>',
        re.DOTALL | re.IGNORECASE,
    )
    _TAG_RE = re.compile(r"<[^>]+>")

    def search(self, query: str, *, limit: int = 5) -> list[SearchHit]:
        """Search, returning ranked candidate sources."""
        try:
            response = self._client.post(self.SEARCH_URL, data={"q": query})
            response.raise_for_status()
        except httpx.HTTPError as exc:
            logger.warning("research.search_failed", query=query[:120], error=type(exc).__name__)
            return []

        hits: list[SearchHit] = []
        for rank, match in enumerate(self._RESULT_RE.finditer(response.text)):
            if len(hits) >= limit:
                break
            url = self._clean_url(unescape(match.group(1)))
            if not url:
                continue
            title = unescape(self._TAG_RE.sub("", match.group(2))).strip()
            hits.append(
                SearchHit(url=url, title=title[:500], rank=rank, provider=self.name)
            )
        logger.info("research.search", query=query[:120], hits=len(hits))
        return hits

    @staticmethod
    def _clean_url(raw: str) -> str:
        """Unwrap DuckDuckGo's redirect wrapper to the real destination."""
        if raw.startswith("//"):
            raw = f"https:{raw}"
        parsed = urlparse(raw)
        if "duckduckgo.com" in (parsed.hostname or "") and parsed.path.startswith("/l"):
            target = parse_qs(parsed.query).get("uddg")
            if target:
                return target[0]
        return raw if raw.startswith(("http://", "https://")) else ""

    def fetch(self, url: str) -> Source:
        """Retrieve one source, using the cache when it is fresh."""
        tier = classify_source(url)

        cached = self.cache.get(url)
        if cached is not None:
            title, text = extract_text(cached["body"])
            return self.extract(
                Source(
                    url=url,
                    title=cached.get("title") or title,
                    tier=tier,
                    status=RetrievalStatus.CACHED,
                    content_hash=cached.get("content_hash", ""),
                    content_reference=self.cache.key_for(url),
                    excerpt=text,
                )
            )

        try:
            response = self._client.get(url)
            response.raise_for_status()
        except httpx.TimeoutException:
            return Source(
                url=url, tier=tier, status=RetrievalStatus.TIMEOUT,
                error=f"timed out after {self.timeout_seconds}s",
            )
        except httpx.HTTPStatusError as exc:
            status = (
                RetrievalStatus.BLOCKED
                if exc.response.status_code in {401, 403, 429}
                else RetrievalStatus.UNAVAILABLE
            )
            return Source(
                url=url, tier=tier, status=status,
                error=f"HTTP {exc.response.status_code}",
            )
        except httpx.HTTPError as exc:
            return Source(
                url=url, tier=tier, status=RetrievalStatus.UNAVAILABLE,
                error=type(exc).__name__,
            )

        body = response.text[:MAX_FETCH_BYTES]
        title, text = extract_text(body)
        if not text.strip():
            return Source(
                url=url, tier=tier, title=title, status=RetrievalStatus.PARSE_FAILURE,
                error="no readable text could be extracted",
            )

        digest = self.cache.put(url, body, status="OK", title=title)
        logger.info("research.fetched", url=url[:120], tier=str(tier), chars=len(text))
        return self.extract(
            Source(
                url=url, title=title, tier=tier, status=RetrievalStatus.OK,
                content_hash=digest, content_reference=self.cache.key_for(url),
                excerpt=text,
            )
        )

    def health_check(self) -> tuple[bool, str]:
        """Probe the search endpoint."""
        try:
            response = self._client.head(self.SEARCH_URL, timeout=5.0)
        except httpx.HTTPError as exc:
            return (False, f"search endpoint unreachable: {type(exc).__name__}")
        return (response.status_code < 500, f"HTTP {response.status_code}")

    def close(self) -> None:
        """Release the HTTP client if this provider created it."""
        if self._owns_client:
            self._client.close()
