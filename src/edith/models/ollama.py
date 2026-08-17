"""Ollama implementation of :class:`ModelProvider`.

This is the only module in Edith that knows Ollama's HTTP shape. Everything above it works
against :class:`~edith.models.base.ModelProvider` and the provider-neutral schemas.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator, Sequence
from typing import Any

import httpx

from edith.config.schema import ModelParams, OllamaProviderConfig
from edith.errors import (
    ModelNotFoundError,
    ProviderError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)
from edith.observability.logging import get_logger
from edith.schemas.model import (
    GenerationOptions,
    GenerationResult,
    HealthState,
    Message,
    ProviderHealth,
    StructuredMode,
    TokenUsage,
)

from .base import ModelProvider

logger = get_logger(__name__)

PROVIDER_NAME = "ollama"

_INSTALL_HINT = (
    "Ollama is not reachable at {endpoint}. Install it from https://ollama.com/download "
    "(or `winget install Ollama.Ollama`) and ensure the service is running."
)
_PULL_HINT = "Model {model!r} is not present. Run: ollama pull {model}"

#: Nanoseconds per second; Ollama reports durations in ns.
_NS_PER_S = 1_000_000_000


#: Depth guard for schema inlining; also breaks self-referential schemas.
_MAX_SCHEMA_DEPTH = 12


def normalize_model_name(name: str) -> str:
    """Normalize an Ollama model reference so ``foo`` and ``foo:latest`` compare equal."""
    return name if ":" in name else f"{name}:latest"


def inline_schema_refs(schema: dict[str, Any]) -> dict[str, Any]:
    """Return a JSON Schema with ``$ref``/``$defs`` resolved inline.

    Pydantic emits nested models as ``$defs`` plus ``$ref`` pointers. Ollama compiles the
    schema into a sampling grammar and does not follow those pointers -- it rejects the
    request with ``failed to parse grammar``. Inlining keeps constrained decoding working
    for any schema with a nested model, which is most of the interesting ones.
    """
    definitions = schema.get("$defs", {})

    def resolve(node: Any, depth: int) -> Any:
        if depth > _MAX_SCHEMA_DEPTH:
            # A recursive schema cannot be fully inlined; leave the remainder as a
            # permissive object rather than looping forever.
            return {"type": "object"}
        if isinstance(node, dict):
            reference = node.get("$ref")
            if isinstance(reference, str) and reference.startswith("#/$defs/"):
                target = definitions.get(reference.split("/")[-1])
                if target is None:
                    return {"type": "object"}
                merged = {**resolve(target, depth + 1)}
                # Preserve sibling keys such as "description" that sat beside the $ref.
                merged.update(
                    {k: resolve(v, depth + 1) for k, v in node.items() if k != "$ref"}
                )
                return merged
            return {
                key: resolve(value, depth + 1)
                for key, value in node.items()
                if key != "$defs"
            }
        if isinstance(node, list):
            return [resolve(item, depth + 1) for item in node]
        return node

    resolved = resolve(schema, 0)
    return resolved if isinstance(resolved, dict) else schema


def _is_grammar_rejection(response: httpx.Response) -> bool:
    """Whether a 400 indicates the runtime could not compile the schema grammar."""
    if response.status_code != 400:
        return False
    body = response.text.lower()
    return "grammar" in body or "sampler" in body


class OllamaProvider(ModelProvider):
    """Local model provider backed by an Ollama runtime.

    Synchronous by design: CLAUDE.md forbids unnecessary async complexity, and on a 6 GB
    GPU inference is serialized anyway.
    """

    name = PROVIDER_NAME

    def __init__(
        self,
        config: OllamaProviderConfig,
        params: ModelParams,
        *,
        client: httpx.Client | None = None,
    ) -> None:
        """
        Args:
            config: Endpoint and timeout settings.
            params: Model profile driving inference parameters.
            client: Injected HTTP client, primarily for tests. When omitted, one is created
                and owned (and closed) by this provider.
        """
        super().__init__(params)
        self.config = config
        self._structured_detail = ""
        self._owns_client = client is None
        self._client = client or httpx.Client(
            base_url=config.host,
            timeout=httpx.Timeout(
                config.timeout_seconds, connect=config.connect_timeout_seconds
            ),
        )

    # -- Request construction ------------------------------------------------------

    def _resolve(self, options: GenerationOptions | None) -> dict[str, Any]:
        """Merge per-call options over the profile into Ollama's ``options`` block."""
        params = self.params
        opts = options or GenerationOptions()
        stop = opts.stop if opts.stop is not None else params.stop
        resolved: dict[str, Any] = {
            "temperature": (
                opts.temperature if opts.temperature is not None else params.temperature
            ),
            "top_p": opts.top_p if opts.top_p is not None else params.top_p,
            "num_predict": (
                opts.max_output_tokens
                if opts.max_output_tokens is not None
                else params.max_output_tokens
            ),
            "num_ctx": (
                opts.context_length
                if opts.context_length is not None
                else params.context_length
            ),
        }
        seed = opts.seed if opts.seed is not None else params.seed
        if seed is not None:
            resolved["seed"] = seed
        if stop:
            resolved["stop"] = list(stop)
        return resolved

    def _payload(
        self,
        messages: Sequence[Message],
        options: GenerationOptions | None,
        *,
        stream: bool,
        json_schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.params.model_name,
            "messages": [{"role": str(m.role), "content": m.content} for m in messages],
            "stream": stream,
            "options": self._resolve(options),
            "keep_alive": self.params.keep_alive,
        }
        if json_schema is not None:
            # Ollama constrains decoding to a JSON Schema when `format` is a schema object.
            # Refs must be inlined first: its grammar compiler does not follow them.
            payload["format"] = inline_schema_refs(json_schema)
        return payload

    # -- Error translation ---------------------------------------------------------

    def _wrap_transport_error(self, exc: Exception) -> ProviderError:
        """Translate httpx failures into Edith's classified error hierarchy."""
        endpoint = self.config.host
        if isinstance(exc, httpx.TimeoutException):
            return ProviderTimeoutError(
                f"Ollama request timed out after {self.config.timeout_seconds}s",
                details={"endpoint": endpoint, "model": self.params.model_name},
            )
        return ProviderUnavailableError(
            _INSTALL_HINT.format(endpoint=endpoint),
            details={"endpoint": endpoint, "cause": type(exc).__name__},
        )

    def _raise_for_status(self, response: httpx.Response) -> None:
        """Convert a non-2xx Ollama response into a classified error."""
        if response.is_success:
            return
        body = response.text[:500]
        if response.status_code == 404:
            raise ModelNotFoundError(
                _PULL_HINT.format(model=self.params.model_name),
                details={"model": self.params.model_name, "status": response.status_code},
            )
        raise ProviderError(
            f"Ollama returned HTTP {response.status_code}: {body}",
            retryable=response.status_code >= 500,
            details={"status": response.status_code, "model": self.params.model_name},
        )

    # -- ModelProvider surface -----------------------------------------------------

    def _generate_raw(
        self,
        messages: Sequence[Message],
        options: GenerationOptions | None = None,
        *,
        json_schema: dict[str, Any] | None = None,
    ) -> GenerationResult:
        payload = self._payload(messages, options, stream=False, json_schema=json_schema)
        started = time.monotonic()
        try:
            response = self._client.post("/api/chat", json=payload)
        except httpx.HTTPError as exc:
            raise self._wrap_transport_error(exc) from exc

        if json_schema is not None and _is_grammar_rejection(response):
            # The runtime could not compile this schema into a sampling grammar. Degrade to
            # generic JSON mode rather than failing the call: the caller validates the
            # result against the schema anyway, so correctness does not depend on the
            # runtime enforcing it -- only reliability does.
            logger.warning(
                "ollama.grammar_unsupported",
                model=self.params.model_name,
                fallback="format=json",
            )
            # Record what is actually true now, so diagnostics and agents stop believing
            # the schema is being enforced during decoding.
            self._structured_mode = StructuredMode.JSON_MODE
            self._structured_detail = (
                "the runtime rejected this JSON Schema as a sampling grammar; output is "
                "constrained to valid JSON only and validated locally"
            )
            payload["format"] = "json"
            try:
                response = self._client.post("/api/chat", json=payload)
            except httpx.HTTPError as exc:
                raise self._wrap_transport_error(exc) from exc

        self._raise_for_status(response)

        try:
            data = response.json()
        except ValueError as exc:
            raise ProviderError(
                "Ollama returned a non-JSON response body",
                retryable=True,
                details={"model": self.params.model_name},
            ) from exc

        if json_schema is not None and self._structured_mode is StructuredMode.UNKNOWN:
            self._structured_mode = StructuredMode.NATIVE
            self._structured_detail = "the runtime compiled the schema into a grammar"

        text = (data.get("message") or {}).get("content", "")
        duration = time.monotonic() - started
        total_ns = data.get("total_duration")
        result = GenerationResult(
            text=text,
            model=data.get("model", self.params.model_name),
            provider=self.name,
            usage=TokenUsage(
                prompt_tokens=int(data.get("prompt_eval_count") or 0),
                completion_tokens=int(data.get("eval_count") or 0),
            ),
            duration_seconds=duration,
            finish_reason=data.get("done_reason"),
            metadata={
                "runtime_seconds": round(total_ns / _NS_PER_S, 3) if total_ns else None,
                "constrained": json_schema is not None,
            },
        )
        logger.debug(
            "ollama.generate",
            model=result.model,
            prompt_tokens=result.usage.prompt_tokens,
            completion_tokens=result.usage.completion_tokens,
            duration_seconds=round(duration, 3),
        )
        return result

    def stream(
        self, messages: Sequence[Message], options: GenerationOptions | None = None
    ) -> Iterator[str]:
        """Yield response chunks as they are produced."""
        if not messages:
            raise ValueError("messages must not be empty")
        payload = self._payload(messages, options, stream=True)
        try:
            with self._client.stream("POST", "/api/chat", json=payload) as response:
                self._raise_for_status(response)
                for line in response.iter_lines():
                    if not line.strip():
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        # A malformed frame must not abort a long generation.
                        logger.warning("ollama.stream.bad_frame", model=self.params.model_name)
                        continue
                    if chunk.get("error"):
                        raise ProviderError(
                            f"Ollama stream error: {chunk['error']}",
                            retryable=True,
                            details={"model": self.params.model_name},
                        )
                    piece = (chunk.get("message") or {}).get("content", "")
                    if piece:
                        yield piece
                    if chunk.get("done"):
                        break
        except httpx.HTTPError as exc:
            raise self._wrap_transport_error(exc) from exc

    def list_models(self) -> tuple[str, ...]:
        """Return the model names present in the local Ollama runtime."""
        try:
            response = self._client.get("/api/tags")
        except httpx.HTTPError as exc:
            raise self._wrap_transport_error(exc) from exc
        self._raise_for_status(response)
        try:
            data = response.json()
        except ValueError as exc:
            raise ProviderError(
                "Ollama /api/tags returned a non-JSON body", retryable=True
            ) from exc
        models = data.get("models") or []
        return tuple(
            str(entry.get("name") or entry.get("model", ""))
            for entry in models
            if entry.get("name") or entry.get("model")
        )

    def health_check(self) -> ProviderHealth:
        """Report runtime reachability and configured-model presence.

        Returns a structured result for every expected failure rather than raising, so the
        doctor command can render all findings in one pass.
        """
        endpoint = self.config.host
        configured = self.params.model_name
        started = time.monotonic()
        try:
            available = self.list_models()
        except ProviderTimeoutError as exc:
            return ProviderHealth(
                provider=self.name,
                state=HealthState.UNAVAILABLE,
                detail=exc.message,
                remediation="Ollama is running but slow to respond. Check GPU/CPU load.",
                endpoint=endpoint,
                configured_model=configured,
            )
        except ProviderUnavailableError as exc:
            return ProviderHealth(
                provider=self.name,
                state=HealthState.UNAVAILABLE,
                detail=exc.message,
                remediation="Install Ollama and start it, then re-run `edith doctor`.",
                endpoint=endpoint,
                configured_model=configured,
            )
        except ProviderError as exc:
            return ProviderHealth(
                provider=self.name,
                state=HealthState.UNAVAILABLE,
                detail=exc.message,
                remediation="Inspect the Ollama server logs.",
                endpoint=endpoint,
                configured_model=configured,
            )

        latency_ms = (time.monotonic() - started) * 1000
        normalized = {normalize_model_name(name) for name in available}
        present = normalize_model_name(configured) in normalized

        return ProviderHealth(
            provider=self.name,
            state=HealthState.HEALTHY if present else HealthState.DEGRADED,
            detail=(
                f"Ollama reachable; model {configured!r} is available."
                if present
                else f"Ollama reachable but model {configured!r} is not pulled."
            ),
            remediation=None if present else _PULL_HINT.format(model=configured),
            endpoint=endpoint,
            available_models=available,
            configured_model=configured,
            configured_model_present=present,
            latency_ms=round(latency_ms, 2),
            structured_mode=self._structured_mode,
            structured_detail=self._structured_detail
            or "not yet exercised; the first structured call determines this",
        )

    def close(self) -> None:
        """Close the HTTP client if this provider created it."""
        if self._owns_client:
            self._client.close()
