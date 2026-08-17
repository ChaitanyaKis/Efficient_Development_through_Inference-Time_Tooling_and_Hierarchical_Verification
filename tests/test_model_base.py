"""The provider abstraction: JSON extraction, structured generation, and bounded repair."""

from __future__ import annotations

import json

import pytest
from pydantic import BaseModel, Field

from edith.config.schema import ModelParams
from edith.errors import ProviderError, StructuredOutputError
from edith.models.base import extract_json_object, render_schema_instruction
from edith.schemas.model import Message, Role

from .fakes import FakeProvider


class Target(BaseModel):
    """Schema used to exercise structured generation."""

    name: str
    count: int = Field(ge=0)
    tags: list[str] = Field(default_factory=list)


VALID = json.dumps({"name": "edith", "count": 2, "tags": ["local"]})


def messages() -> list[Message]:
    return [Message(role=Role.USER, content="produce the object")]


class TestExtractJsonObject:
    def test_bare_object(self) -> None:
        assert json.loads(extract_json_object('{"a": 1}'))["a"] == 1

    def test_json_code_fence(self) -> None:
        text = 'Here you go:\n```json\n{"a": 1}\n```\nHope that helps!'
        assert json.loads(extract_json_object(text))["a"] == 1

    def test_unlabelled_code_fence(self) -> None:
        assert json.loads(extract_json_object('```\n{"a": 1}\n```'))["a"] == 1

    def test_surrounding_prose(self) -> None:
        """Small quantized models add prose despite instructions; recovery must be local."""
        assert json.loads(extract_json_object('Sure! {"a": 1} Let me know.'))["a"] == 1

    def test_nested_braces(self) -> None:
        text = '{"outer": {"inner": {"deep": 1}}, "x": 2}'
        assert json.loads(extract_json_object(text))["outer"]["inner"]["deep"] == 1

    def test_braces_inside_strings_do_not_confuse_the_scanner(self) -> None:
        text = '{"note": "use {braces} carefully", "n": 1}'
        assert json.loads(extract_json_object(text))["note"] == "use {braces} carefully"

    def test_escaped_quote_inside_string(self) -> None:
        text = '{"note": "she said \\"hi\\"", "n": 1}'
        assert json.loads(extract_json_object(text))["n"] == 1

    def test_array_payload(self) -> None:
        assert json.loads(extract_json_object("result: [1, 2, 3]")) == [1, 2, 3]

    @pytest.mark.parametrize("text", ["", "no json at all", "{unbalanced", "}{"])
    def test_unrecoverable_input_raises(self, text: str) -> None:
        with pytest.raises(ValueError, match="no balanced JSON object"):
            extract_json_object(text)


class TestStructuredGenerate:
    def test_valid_output_first_try(self, model_params: ModelParams) -> None:
        provider = FakeProvider(model_params, [VALID])
        result = provider.structured_generate(messages(), Target)
        assert isinstance(result, Target)
        assert result.name == "edith" and result.count == 2
        assert len(provider.calls) == 1

    def test_json_schema_is_passed_to_the_provider(self, model_params: ModelParams) -> None:
        """Constrained decoding is requested, not merely hoped for."""
        provider = FakeProvider(model_params, [VALID])
        provider.structured_generate(messages(), Target)
        schema = provider.calls[0]["json_schema"]
        assert schema is not None and "properties" in schema

    def test_recovers_from_prose_wrapping(self, model_params: ModelParams) -> None:
        provider = FakeProvider(model_params, [f"Certainly!\n```json\n{VALID}\n```"])
        assert provider.structured_generate(messages(), Target).name == "edith"
        assert len(provider.calls) == 1  # repaired locally, no extra inference

    def test_repairs_invalid_output(self, model_params: ModelParams) -> None:
        invalid = json.dumps({"name": "edith", "count": -5})
        provider = FakeProvider(model_params, [invalid, VALID])
        result = provider.structured_generate(messages(), Target)
        assert result.count == 2
        assert len(provider.calls) == 2

    def test_repair_prompt_includes_the_validation_error(self, model_params: ModelParams) -> None:
        provider = FakeProvider(model_params, [json.dumps({"name": "e"}), VALID])
        provider.structured_generate(messages(), Target)
        repair_turn = provider.calls[1]["messages"][-1][1]
        assert "count" in repair_turn and "not valid" in repair_turn

    def test_raises_after_exhausting_repairs(self, model_params: ModelParams) -> None:
        provider = FakeProvider(model_params, ["not json at all"])
        with pytest.raises(StructuredOutputError) as excinfo:
            provider.structured_generate(messages(), Target, max_repair_attempts=2)
        assert len(provider.calls) == 3  # bounded: 1 initial + 2 repairs
        assert excinfo.value.details["attempts"] == 3
        assert excinfo.value.retryable is True

    def test_zero_repairs_means_one_attempt(self, model_params: ModelParams) -> None:
        provider = FakeProvider(model_params, ["garbage"])
        with pytest.raises(StructuredOutputError):
            provider.structured_generate(messages(), Target, max_repair_attempts=0)
        assert len(provider.calls) == 1

    def test_model_claiming_success_is_not_enough(self, model_params: ModelParams) -> None:
        """CLAUDE.md: never trust an LLM's own claim of correctness."""
        provider = FakeProvider(
            model_params, ['{"status": "valid", "message": "output conforms to schema"}']
        )
        with pytest.raises(StructuredOutputError):
            provider.structured_generate(messages(), Target, max_repair_attempts=0)

    def test_negative_repair_budget_rejected(self, model_params: ModelParams) -> None:
        provider = FakeProvider(model_params, [VALID])
        with pytest.raises(ValueError, match="max_repair_attempts"):
            provider.structured_generate(messages(), Target, max_repair_attempts=-1)

    def test_empty_messages_rejected(self, model_params: ModelParams) -> None:
        provider = FakeProvider(model_params, [VALID])
        with pytest.raises(ValueError, match="must not be empty"):
            provider.structured_generate([], Target)

    def test_provider_errors_propagate(self, model_params: ModelParams) -> None:
        """A transport failure is not a validation failure and must not be silently repaired."""
        provider = FakeProvider(model_params, raises=ProviderError("runtime exploded"))
        with pytest.raises(ProviderError, match="runtime exploded"):
            provider.structured_generate(messages(), Target)


class TestProviderSurface:
    def test_generate_rejects_empty_messages(self, model_params: ModelParams) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            FakeProvider(model_params).generate([])

    def test_supports_tools_reflects_profile(self) -> None:
        assert not FakeProvider(ModelParams(model_name="m")).supports_tools()
        assert FakeProvider(ModelParams(model_name="m", supports_tools=True)).supports_tools()

    def test_context_manager_closes(self, model_params: ModelParams) -> None:
        provider = FakeProvider(model_params)
        with provider:
            pass
        assert provider.closed

    def test_stream_yields_text(self, model_params: ModelParams) -> None:
        provider = FakeProvider(model_params, ["chunk"])
        assert list(provider.stream(messages())) == ["chunk"]


class TestSchemaInstruction:
    def test_includes_schema_and_forbids_prose(self) -> None:
        instruction = render_schema_instruction(Target)
        assert "ONLY the JSON object" in instruction
        assert '"count"' in instruction
