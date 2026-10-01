"""Regression cases found during the T1 terminal-contract review."""
import json
from types import SimpleNamespace

import pytest
from google.genai import types

from api_helpers import execute_gemini_call
from message_processing import process_gemini_response_to_openai_dict
from models import OpenAIRequest
from test_sdk_execution_terminal import _FakeClient


async def _run_stream(chunks, **kwargs):
    request = OpenAIRequest(model="gemini-3.7-flash", stream=True,
                            messages=[{"role": "user", "content": "hi"}])
    response = await execute_gemini_call(
        _FakeClient(chunks), request.model, lambda _: [], {}, request, **kwargs)
    lines = [line async for line in response.body_iterator]
    payloads = [json.loads(line[6:]) for line in lines
                if line.startswith("data: {")]
    return lines, payloads


async def test_sdk_default_candidate_index_completes_without_protocol_error():
    chunk = types.GenerateContentResponse(candidates=[types.Candidate(
        content=types.Content(parts=[types.Part(text="hello")], role="model"),
        finish_reason=types.FinishReason.STOP)])
    lines, payloads = await _run_stream([chunk])
    assert not any("error" in payload for payload in payloads)
    assert [choice["finish_reason"] for payload in payloads
            for choice in payload.get("choices", []) if choice.get("finish_reason")] == ["stop"]
    assert lines.count("data: [DONE]\n\n") == 1


async def test_empty_candidate_safety_filter_is_not_invalid_content():
    raw = types.GenerateContentResponse(candidates=[types.Candidate(
        content=types.Content(parts=[], role="model"), finish_reason=types.FinishReason.SAFETY)])
    request = OpenAIRequest(model="gemini-3.7-flash", stream=False,
                            messages=[{"role": "user", "content": "hi"}])
    response = await execute_gemini_call(_FakeClient(raw), request.model, lambda _: [], {}, request)
    payload = json.loads(response.body)
    assert response.status_code == 200
    assert payload["choices"][0]["finish_reason"] == "content_filter"


def test_default_prompt_feedback_preserves_independent_legacy_text():
    raw = SimpleNamespace(candidates=[], text="legacy", usage_metadata=None,
                          prompt_feedback=SimpleNamespace(block_reason="UNSPECIFIED"))
    response = process_gemini_response_to_openai_dict(raw, "gemini-3.7-flash")
    assert response["choices"][0]["message"]["content"] == "legacy"
    assert response["choices"][0]["finish_reason"] == "stop"


async def test_partial_real_tool_is_flushed_before_independent_stop_finish():
    partial = types.Part(function_call=types.FunctionCall(
        name="lookup", args={"city": "上海"}, will_continue=True))
    chunks = [types.GenerateContentResponse(candidates=[types.Candidate(
        content=types.Content(parts=[partial], role="model"))]),
        types.GenerateContentResponse(candidates=[types.Candidate(
            content=types.Content(parts=[types.Part(function_call=types.FunctionCall(
                partial_args=[types.PartialArg(json_path="$.city", string_value="上海", will_continue=True)]))], role="model"))]),
        types.GenerateContentResponse(candidates=[types.Candidate(
            content=types.Content(parts=[], role="model"), finish_reason=types.FinishReason.STOP)])]
    lines, payloads = await _run_stream(chunks, synthetic_tool_name="v2o_emit_review")
    calls = [call for payload in payloads for choice in payload.get("choices", [])
             for call in choice.get("delta", {}).get("tool_calls", [])]
    assert len(calls) == 1
    assert json.loads(calls[0]["function"]["arguments"]) == {"city": "上海"}
    assert not any("error" in payload for payload in payloads)
    assert [choice["finish_reason"] for payload in payloads
            for choice in payload.get("choices", []) if choice.get("finish_reason")] == ["tool_calls"]
    assert lines.count("data: [DONE]\n\n") == 1
