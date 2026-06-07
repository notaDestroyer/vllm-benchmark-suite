"""Tests for the structured-output workload scoring.

Exercises the pure :func:`score_structured_response` against valid JSON,
malformed JSON, schema-violating JSON, correct/incorrect tool calls and a
truncated streaming fragment, plus the workload's run/summarize via a
mocked HTTP boundary.
"""

from __future__ import annotations

import asyncio
import json

from vllm_benchmark.config import BenchmarkConfig
from vllm_benchmark.core.backends.base import ServerInfo
from vllm_benchmark.core.workloads import structured as struct_mod
from vllm_benchmark.core.workloads.structured import (
    StructuredWorkload,
    load_structured_probes,
    score_structured_response,
)

CONTACT_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "email": {"type": "string"},
        "phone": {"type": "string"},
    },
    "required": ["name", "email"],
    "additionalProperties": False,
}


# ---------------------------------------------------------------------------
# JSON-schema scoring
# ---------------------------------------------------------------------------


def test_valid_json_and_schema():
    text = json.dumps({"name": "Jane Doe", "email": "jane@example.com"})
    s = score_structured_response(text, CONTACT_SCHEMA)
    assert s["valid_json"] is True
    assert s["schema_valid"] is True
    assert s["adherent"] is True


def test_invalid_json_fails():
    s = score_structured_response("this is not json at all", CONTACT_SCHEMA)
    assert s["valid_json"] is False
    assert s["schema_valid"] is False
    assert s["adherent"] is False


def test_schema_violating_json_fails():
    # Valid JSON but missing the required "email" field.
    text = json.dumps({"name": "Jane Doe"})
    s = score_structured_response(text, CONTACT_SCHEMA)
    assert s["valid_json"] is True
    assert s["schema_valid"] is False
    assert s["adherent"] is False


def test_additional_properties_violation():
    text = json.dumps({"name": "Jane", "email": "j@x.com", "extra": 1})
    s = score_structured_response(text, CONTACT_SCHEMA)
    assert s["valid_json"] is True
    assert s["schema_valid"] is False


def test_fenced_json_is_salvaged():
    text = "Here you go:\n```json\n{\"name\": \"Jane\", \"email\": \"j@x.com\"}\n```"
    s = score_structured_response(text, CONTACT_SCHEMA)
    assert s["valid_json"] is True
    assert s["schema_valid"] is True


def test_embedded_json_span_is_salvaged():
    text = "Sure! {\"name\": \"Jane\", \"email\": \"j@x.com\"} hope that helps."
    s = score_structured_response(text, CONTACT_SCHEMA)
    assert s["valid_json"] is True
    assert s["schema_valid"] is True


def test_truncated_streaming_fragment_fails():
    # A streaming response cut off mid-object never parses.
    fragment = '{"name": "Jane", "email": "jane@exa'
    s = score_structured_response(fragment, CONTACT_SCHEMA)
    assert s["valid_json"] is False
    assert s["adherent"] is False


def test_no_schema_requires_only_valid_json():
    s = score_structured_response('{"anything": true}', None)
    assert s["valid_json"] is True
    assert s["schema_valid"] is True
    assert s["adherent"] is True


# ---------------------------------------------------------------------------
# Tool-call scoring
# ---------------------------------------------------------------------------


def test_correct_tool_call():
    tool_calls = [
        {"function": {"name": "get_current_weather", "arguments": '{"location": "Paris"}'}}
    ]
    s = score_structured_response(
        "", None, expected_tool="get_current_weather",
        tool_calls=tool_calls, required_args=["location"],
    )
    assert s["tool_correct"] is True
    assert s["adherent"] is True


def test_wrong_tool_selected():
    tool_calls = [
        {"function": {"name": "send_email", "arguments": '{"to": "a", "body": "b"}'}}
    ]
    s = score_structured_response(
        "", None, expected_tool="get_current_weather",
        tool_calls=tool_calls, required_args=["location"],
    )
    assert s["tool_correct"] is False
    assert s["adherent"] is False


def test_correct_tool_missing_required_arg():
    tool_calls = [
        {"function": {"name": "get_current_weather", "arguments": '{"unit": "celsius"}'}}
    ]
    s = score_structured_response(
        "", None, expected_tool="get_current_weather",
        tool_calls=tool_calls, required_args=["location"],
    )
    assert s["tool_correct"] is False


def test_no_tool_call_at_all():
    s = score_structured_response(
        "I can't help with that", None, expected_tool="get_current_weather",
        tool_calls=None, required_args=["location"],
    )
    assert s["tool_correct"] is False


def test_tool_args_as_dict():
    tool_calls = [
        {"function": {"name": "get_current_weather", "arguments": {"location": "Paris"}}}
    ]
    s = score_structured_response(
        "", None, expected_tool="get_current_weather",
        tool_calls=tool_calls, required_args=["location"],
    )
    assert s["tool_correct"] is True


# ---------------------------------------------------------------------------
# applicable() / probe set
# ---------------------------------------------------------------------------


def test_applicable_generate_only():
    wl = StructuredWorkload()
    assert wl.applicable(ServerInfo(backend="vllm", task="generate")) is True
    assert wl.applicable(ServerInfo(backend="vllm", task="unknown")) is True
    assert wl.applicable(ServerInfo(backend="vllm", task="embed")) is False


def test_probe_set_loads():
    probes = load_structured_probes()
    assert probes["schema_probes"]
    assert probes["tool_probes"]
    assert "_README" not in probes


# ---------------------------------------------------------------------------
# run() / summarize() with mocked HTTP boundary
# ---------------------------------------------------------------------------


def test_run_and_summarize(monkeypatch):
    async def fake_post(session, endpoint, body, request_timeout):
        # Schema probes get a valid contact/order; tool probes get a call.
        if "tools" in body:
            return {
                "success": True,
                "duration": 0.08,
                "content": "",
                "tool_calls": [
                    {"function": {"name": "get_current_weather", "arguments": '{"location": "Paris"}'}}
                ],
            }
        # Return an object that satisfies the contact OR order schema; the
        # extract_contact schema needs name+email, order needs sku+qty+price.
        return {
            "success": True,
            "duration": 0.07,
            "content": json.dumps(
                {
                    "name": "Jane",
                    "email": "j@x.com",
                    "sku": "A19",
                    "quantity": 2,
                    "unit_price": 4.5,
                }
            ),
            "tool_calls": None,
        }

    monkeypatch.setattr(struct_mod, "_post_chat", fake_post)
    config = BenchmarkConfig()
    wl = StructuredWorkload()
    cells = asyncio.run(wl.run(config, "some-model"))
    assert cells
    summary = wl.summarize(cells)
    # The order schema forbids additionalProperties, so adherence may be <1;
    # tool correctness should be perfect.
    assert summary["tool_call_correctness"] == 1.0
    assert summary["schema_adherence_rate"] is not None
    assert summary["best_ttft"] is not None
