"""M5.2: structured-output reliability for ``ModelEdits``, without weakening it.

M5.1's benchmark failed on a shape error, repeatedly and in both arms' code path: the model
returned ``{"replace_file": {...}}`` where ``{"edits": [{"mode": "replace_file", ...}]}`` was
required. The repair prompt told it the output was invalid but never told it what the outer
object should look like.

The invariant these tests defend is the one that matters:

    MODEL OUTPUT -> repair prompt -> STRICT SCHEMA -> TOOL GATEWAY -> EXECUTION

never

    MODEL OUTPUT -> TRUST -> FILESYSTEM

So the fix is a better *prompt*, not a looser schema and not a normaliser that guesses. A
hoisted-key response still fails validation; it just gets a clearer second chance.
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from edith.agents.coder import EditMode, FileEdit, ModelEdits
from edith.config.schema import ModelParams
from edith.errors import StructuredOutputError
from edith.models.base import render_envelope_hint
from edith.schemas.model import Message, Role

from .fakes import FakeProvider

PARAMS = ModelParams(model_name="test-model:q4")

#: The exact malformed shape observed in the M5.1 benchmark.
HOISTED = json.dumps(
    {"replace_file": {"path": "src/database/store.py", "content": "x = 1\n"}}
)
CANONICAL = json.dumps(
    {
        "edits": [
            {"path": "src/a.py", "mode": "replace_file", "content": "x = 1\n"}
        ],
        "summary": "done",
        "notes": "",
    }
)


def messages() -> list[Message]:
    return [Message(role=Role.USER, content="do the thing")]


class TestEnvelopeHint:
    """The fix: tell the model the shape, derived from the schema itself."""

    def test_the_hint_names_the_required_top_level_keys(self) -> None:
        hint = render_envelope_hint(ModelEdits)
        assert '"edits"' in hint
        assert "required" in hint

    def test_optional_keys_are_marked_as_such(self) -> None:
        hint = render_envelope_hint(ModelEdits)
        assert "optional" in hint

    def test_the_hint_forbids_extra_top_level_keys(self) -> None:
        """Which is exactly what the observed failure did."""
        assert "Do not put any other key at the top level" in render_envelope_hint(
            ModelEdits
        )

    def test_the_hint_is_derived_not_hand_written(self) -> None:
        """A new structured output gets the same help without anyone remembering."""
        from edith.agents.planner import PlannerOutput

        hint = render_envelope_hint(PlannerOutput)
        assert '"steps"' in hint
        assert '"edits"' not in hint

    def test_a_schema_with_no_properties_produces_no_hint(self) -> None:
        from edith.schemas.common import EdithModel

        class Empty(EdithModel):
            pass

        assert render_envelope_hint(Empty) == ""


class TestRepairPromptCarriesTheHint:
    def test_the_repair_turn_shows_the_required_envelope(self) -> None:
        """The model gets the shape on its second chance, not just an error string."""
        provider = FakeProvider(PARAMS, [HOISTED, CANONICAL])
        result = provider.structured_generate(messages(), ModelEdits, max_repair_attempts=2)

        assert result.edits[0].path == "src/a.py"
        repair_prompt = "\n".join(
            content for _, content in provider.calls[1]["messages"]
        )
        assert "top-level keys must be exactly" in repair_prompt
        assert '"edits"' in repair_prompt

    def test_a_model_that_never_complies_still_fails(self) -> None:
        """The hint is help, not permission. Strict validation is unchanged."""
        provider = FakeProvider(PARAMS, [HOISTED])
        with pytest.raises(StructuredOutputError):
            provider.structured_generate(messages(), ModelEdits, max_repair_attempts=2)

    def test_repair_remains_bounded(self) -> None:
        provider = FakeProvider(PARAMS, [HOISTED])
        with pytest.raises(StructuredOutputError):
            provider.structured_generate(messages(), ModelEdits, max_repair_attempts=1)
        assert len(provider.calls) == 2, "one initial attempt plus one repair"


class TestModelEditsRemainsStrict:
    """M5.2 forbids weakening the schema to accommodate the model."""

    def test_the_hoisted_shape_is_still_invalid(self) -> None:
        with pytest.raises(ValidationError):
            ModelEdits.model_validate(json.loads(HOISTED))

    def test_unknown_top_level_keys_are_still_refused(self) -> None:
        with pytest.raises(ValidationError):
            ModelEdits.model_validate(
                {"edits": [{"path": "a.py", "mode": "replace_file", "content": "x"}],
                 "surprise": 1}
            )

    def test_edits_is_still_required_and_non_empty(self) -> None:
        with pytest.raises(ValidationError):
            ModelEdits.model_validate({"summary": "nothing to do"})
        with pytest.raises(ValidationError):
            ModelEdits.model_validate({"edits": []})

    def test_an_unknown_edit_mode_is_still_refused(self) -> None:
        with pytest.raises(ValidationError):
            FileEdit.model_validate(
                {"path": "a.py", "mode": "rm -rf", "content": "x"}
            )

    def test_every_declared_mode_still_validates(self) -> None:
        for mode in EditMode:
            edit = FileEdit.model_validate(
                {"path": "a.py", "mode": mode.value, "content": "x = 1\n"}
            )
            assert edit.mode is mode

    def test_multiple_edits_validate(self) -> None:
        parsed = ModelEdits.model_validate(
            {
                "edits": [
                    {"path": "a.py", "mode": "replace_file", "content": "x = 1\n"},
                    {"path": "b.py", "mode": "append", "content": "y = 2\n"},
                ]
            }
        )
        assert len(parsed.edits) == 2


class TestRecoveryCannotBypassEnforcement:
    """The whole point: a valid-looking response is still only a *proposal*."""

    def test_a_schema_valid_edit_is_not_permission_to_write(self, tmp_path) -> None:
        """Validation and authorisation are different layers, and both still run."""
        from edith.schemas.agent import AgentPermissions
        from edith.tools.schemas import ToolCall

        from .tool_fixtures import build_gateway

        (tmp_path / "src").mkdir()
        gateway = build_gateway(
            tmp_path,
            AgentPermissions(
                allowed_tools=frozenset({"filesystem.write"}),
                allowed_write_paths=("src/**",),
            ),
        )
        edit = ModelEdits.model_validate(
            {"edits": [{"path": "secrets.env", "mode": "replace_file", "content": "K=1"}]}
        ).edits[0]

        result = gateway.execute(
            ToolCall(
                tool="filesystem.write",
                arguments={"path": edit.path, "content": edit.content},
            )
        )
        assert not result.ok
        assert result.denied
        assert not (tmp_path / "secrets.env").exists()

    def test_a_traversing_path_is_refused_by_the_gateway(self, tmp_path) -> None:
        from edith.schemas.agent import AgentPermissions
        from edith.tools.schemas import ToolCall

        from .tool_fixtures import build_gateway

        gateway = build_gateway(
            tmp_path,
            AgentPermissions(
                allowed_tools=frozenset({"filesystem.write"}),
                allowed_write_paths=("**",),
            ),
        )
        result = gateway.execute(
            ToolCall(
                tool="filesystem.write",
                arguments={"path": "../escaped.txt", "content": "x"},
            )
        )
        assert not result.ok
        assert not (tmp_path.parent / "escaped.txt").exists()
