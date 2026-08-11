"""Domain schema validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from edith.errors import FailureCategory
from edith.schemas.agent import (
    AgentIdentity,
    AgentPermissions,
    AgentRequest,
    AgentResponse,
    AgentStatus,
    Capability,
    TaskRef,
)
from edith.schemas.common import EdithModel, Severity, Timestamped, Verdict, new_id, utc_now
from edith.schemas.model import (
    GenerationOptions,
    GenerationResult,
    HealthState,
    Message,
    ProviderHealth,
    Role,
    TokenUsage,
)


class TestCommon:
    def test_new_id_is_prefixed_and_unique(self) -> None:
        first, second = new_id("task"), new_id("task")
        assert first.startswith("task_") and first != second

    def test_utc_now_is_timezone_aware(self) -> None:
        assert utc_now().tzinfo is not None

    def test_extra_fields_rejected(self) -> None:
        """An LLM inventing a field is a validation failure, not silent data loss."""

        class Sample(EdithModel):
            value: int

        with pytest.raises(ValidationError):
            Sample.model_validate({"value": 1, "hallucinated": "x"})

    def test_touch_advances_updated_at(self) -> None:
        item = Timestamped()
        original = item.updated_at
        item.touch()
        assert item.updated_at >= original

    def test_enums_are_string_valued(self) -> None:
        assert Severity.CRITICAL == "CRITICAL"
        assert Verdict.PASS == "PASS"  # noqa: S105 - a verdict, not a credential


class TestModelSchemas:
    def test_message_roles(self) -> None:
        assert Message(role=Role.SYSTEM, content="x").role is Role.SYSTEM

    def test_invalid_role_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Message(role="wizard", content="x")  # type: ignore[arg-type]

    def test_token_usage_total(self) -> None:
        assert TokenUsage(prompt_tokens=10, completion_tokens=5).total_tokens == 15

    def test_negative_tokens_rejected(self) -> None:
        with pytest.raises(ValidationError):
            TokenUsage(prompt_tokens=-1)

    def test_generation_options_default_to_none(self) -> None:
        """None means 'use the profile value' - it must not coerce to a hidden default."""
        options = GenerationOptions()
        assert options.temperature is None and options.max_output_tokens is None

    @pytest.mark.parametrize(("field", "value"), [("temperature", 3.0), ("top_p", -0.1)])
    def test_generation_options_bounds(self, field: str, value: float) -> None:
        with pytest.raises(ValidationError):
            GenerationOptions(**{field: value})

    def test_generation_result_defaults(self) -> None:
        result = GenerationResult(text="hi", model="m", provider="fake")
        assert result.usage.total_tokens == 0 and result.duration_seconds == 0.0

    def test_provider_health_ok_property(self) -> None:
        assert ProviderHealth(provider="p", state=HealthState.HEALTHY).ok
        assert not ProviderHealth(provider="p", state=HealthState.DEGRADED).ok
        assert not ProviderHealth(provider="p", state=HealthState.UNAVAILABLE).ok


class TestAgentPermissions:
    def test_default_is_read_only(self) -> None:
        assert AgentPermissions().read_only

    def test_write_scope_makes_it_writable(self) -> None:
        assert not AgentPermissions(allowed_write_paths=("src/frontend/**",)).read_only

    @pytest.mark.parametrize("pattern", ["/etc/passwd", "C:/Windows/**", "\\\\server\\share"])
    def test_absolute_paths_rejected(self, pattern: str) -> None:
        """Permissions are repo-relative; an absolute pattern would escape the sandbox."""
        with pytest.raises(ValidationError):
            AgentPermissions(allowed_write_paths=(pattern,))

    @pytest.mark.parametrize("pattern", ["../../etc", "src/../../../secrets", "a/../..\\b"])
    def test_traversal_rejected(self, pattern: str) -> None:
        with pytest.raises(ValidationError):
            AgentPermissions(allowed_read_paths=(pattern,))

    def test_legitimate_patterns_accepted(self) -> None:
        perms = AgentPermissions(
            allowed_read_paths=("src/**", "tests/**"),
            allowed_write_paths=("src/backend/**",),
            allowed_tools=frozenset({"filesystem.read"}),
        )
        assert "src/**" in perms.allowed_read_paths

    def test_network_access_defaults_off(self) -> None:
        assert AgentPermissions().network_access is False


class TestAgentIdentity:
    def test_valid_identity(self) -> None:
        identity = AgentIdentity(
            name="planner",
            description="Plans work",
            capabilities=frozenset({Capability.PLANNING}),
        )
        assert identity.name == "planner" and identity.version == "0.1.0"

    @pytest.mark.parametrize("name", ["Planner", "1planner", "plan-ner", "plan ner", ""])
    def test_invalid_names_rejected(self, name: str) -> None:
        with pytest.raises(ValidationError):
            AgentIdentity(name=name, description="d")

    def test_description_required(self) -> None:
        with pytest.raises(ValidationError):
            AgentIdentity(name="planner", description="")


class TestAgentRequestResponse:
    def test_request_generates_ids(self) -> None:
        request = AgentRequest()
        assert request.request_id.startswith("req_")
        assert request.task.task_id.startswith("task_")

    def test_request_carries_payload_and_context(self) -> None:
        request = AgentRequest(payload={"statement": "x"}, context={"files": []})
        assert request.payload["statement"] == "x" and request.context == {"files": []}

    def test_task_ref_accepts_explicit_ids(self) -> None:
        ref = TaskRef(task_id="task_x", project_id="proj_y", title="t")
        assert (ref.task_id, ref.project_id) == ("task_x", "proj_y")

    def test_success_response(self) -> None:
        response = AgentResponse(
            request_id="req_1", agent="echo", status=AgentStatus.SUCCESS, output={"a": 1}
        )
        assert response.ok and response.failure_category is None

    def test_failure_response_carries_category(self) -> None:
        response = AgentResponse(
            request_id="req_1",
            agent="echo",
            status=AgentStatus.FAILURE,
            error="boom",
            failure_category=FailureCategory.MODEL_ERROR,
        )
        assert not response.ok
        assert response.failure_category is FailureCategory.MODEL_ERROR

    def test_negative_duration_rejected(self) -> None:
        with pytest.raises(ValidationError):
            AgentResponse(
                request_id="r", agent="a", status=AgentStatus.SUCCESS, duration_seconds=-1.0
            )

    def test_response_round_trips_through_json(self) -> None:
        """Responses are persisted and shipped between subsystems; serialization must hold."""
        original = AgentResponse(
            request_id="req_1",
            agent="echo",
            status=AgentStatus.SUCCESS,
            output={"summary": "s", "confidence": 0.5},
        )
        restored = AgentResponse.model_validate_json(original.model_dump_json())
        assert restored.output == original.output and restored.status is original.status
