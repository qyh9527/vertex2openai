"""SDK terminal-generation behavior at the execute_gemini_call boundary."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi.responses import JSONResponse
from google.genai import types

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))

import api_helpers as helpers
from models import OpenAIRequest
from routes import chat_api


class _FakeModels:
    def __init__(self, response):
        self.response = response
        self.calls = 0

    async def generate_content(self, **kwargs):
        self.calls += 1
        return self.response

    async def generate_content_stream(self, **kwargs):
        self.calls += 1

        async def chunks():
            for chunk in self.response:
                yield chunk

        return chunks()


class _FakeClient:
    def __init__(self, response):
        self.aio = SimpleNamespace(models=_FakeModels(response))


def _response(finish_reason, *, parts=None, prompt_feedback=None):
    candidate = SimpleNamespace(
        content=SimpleNamespace(role="model", parts=parts or [types.Part(text="partial")]),
        finish_reason=finish_reason,
    )
    result = SimpleNamespace(
        candidates=[candidate],
        prompt_feedback=prompt_feedback,
        usage_metadata=SimpleNamespace(
            prompt_token_count=2, candidates_token_count=1, total_token_count=3,
            cached_content_token_count=0,
        ),
    )
    return result


def _chunk(finish_reason=None, *, parts=None, index=0, prompt_feedback=None, usage=None):
    candidates = []
    if finish_reason is not None or parts is not None:
        candidates = [SimpleNamespace(
            index=index,
            content=SimpleNamespace(role="model", parts=parts or []),
            finish_reason=finish_reason,
        )]
    return SimpleNamespace(candidates=candidates, prompt_feedback=prompt_feedback,
                           usage_metadata=usage)


@pytest.mark.asyncio
@pytest.mark.parametrize("reason,finish", [
    ("ESCALATION", "content_filter"), ("MAX_TOKENS", "length"),
])
@pytest.mark.parametrize("channel", ["express", "vertex"])
@pytest.mark.parametrize("fake", [False, True])
async def test_stream_filter_and_length_remain_terminal_with_tool_calls(reason, finish, channel, fake):
    tool_part = types.Part(function_call=types.FunctionCall(name="lookup", args={}))
    client = _FakeClient(
        _response(reason, parts=[tool_part]) if fake
        else [_chunk(reason, parts=[tool_part])])
    request = OpenAIRequest(
        model="gemini-3.6-flash", messages=[{"role": "user", "content": "hi"}],
        tools=[{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}],
        stream=True,
    )

    response = await helpers.execute_gemini_call(
        client, "gemini-3.6-flash", lambda _: [], {}, request,
        channel_name=channel, force_fake_streaming=fake,
    )
    lines = [line async for line in response.body_iterator]
    payloads = [json.loads(line[6:]) for line in lines
                if line.startswith("data: {") and '"choices"' in line]
    finishes = [choice["finish_reason"] for payload in payloads
                for choice in payload.get("choices", []) if choice.get("finish_reason")]
    tool_calls = ([tc for payload in payloads for choice in payload.get("choices", [])
                   for tc in (choice.get("delta", {}).get("tool_calls") or [])]
                  + [tc for payload in payloads for choice in payload.get("choices", [])
                     for tc in (choice.get("message", {}).get("tool_calls") or [])])

    assert finish in finishes
    assert tool_calls
    assert lines.count("data: [DONE]\n\n") == 1
    assert not any('"error"' in line for line in lines)


@pytest.mark.asyncio
@pytest.mark.parametrize("channel", ["express", "vertex"])
async def test_nonstream_filter_reason_wins_over_tool_call(channel):

    tool_part = types.Part(function_call=types.FunctionCall(name="lookup", args={}))
    client = _FakeClient(_response("ESCALATION", parts=[tool_part]))
    request = OpenAIRequest(
        model="gemini-3.6-flash", messages=[{"role": "user", "content": "hi"}],
        tools=[{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}],
        stream=False,
    )

    response = await helpers.execute_gemini_call(
        client, "gemini-3.6-flash", lambda _: [], {}, request, channel_name=channel,
    )

    body = json.loads(response.body)
    assert body["choices"][0]["finish_reason"] == "content_filter"
    assert body["choices"][0]["message"]["tool_calls"]


@pytest.mark.asyncio
@pytest.mark.parametrize("channel", ["express", "vertex"])
async def test_nonstream_max_tokens_reason_wins_over_tool_call(channel):
    tool_part = types.Part(function_call=types.FunctionCall(name="lookup", args={}))
    client = _FakeClient(_response("MAX_TOKENS", parts=[tool_part]))
    request = OpenAIRequest(
        model="gemini-3.6-flash", messages=[{"role": "user", "content": "hi"}],
        tools=[{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}],
        stream=False,
    )

    response = await helpers.execute_gemini_call(
        client, "gemini-3.6-flash", lambda _: [], {}, request, channel_name=channel,
    )

    body = json.loads(response.body)
    assert body["choices"][0]["finish_reason"] == "length"
    assert body["choices"][0]["message"]["tool_calls"]


@pytest.mark.asyncio
@pytest.mark.parametrize("channel", ["express", "vertex"])
async def test_stream_pup_failure_emits_one_error_and_done_without_retry(channel, monkeypatch):
    client = _FakeClient([
        _chunk(parts=[types.Part(text="partial")]),
        _chunk("PUP_LIMITED_DISABLED", parts=[types.Part(text="hidden")]),
    ])
    monkeypatch.setattr(helpers, "get_retry_settings", lambda *_: (2, 0))
    monkeypatch.setattr(helpers, "stats", Mock())
    request = OpenAIRequest(
        model="gemini-3.6-flash", messages=[{"role": "user", "content": "hi"}],
        stream=True, stream_options={"include_usage": True},
    )

    response = await helpers.execute_gemini_call(
        client, "gemini-3.6-flash", lambda _: [], {}, request,
        channel_name=channel, prefill_text="prefill ",
    )
    lines = [line async for line in response.body_iterator]
    payloads = [json.loads(line[6:]) for line in lines
                if line.startswith("data: ") and line.strip() != "data: [DONE]"]
    contents = [choice["delta"].get("content") for payload in payloads
                for choice in payload.get("choices", [])]

    assert "prefill " in contents
    assert "partial" in contents
    assert "hidden" not in contents
    errors = [payload["error"] for payload in payloads if "error" in payload]
    assert len(errors) == 1
    assert errors[0]["category"] == "credential_permanent"
    assert errors[0]["upstream_finish_reason"] == "PUP_LIMITED_DISABLED"
    assert lines.count("data: [DONE]\n\n") == 1
    assert not any(payload.get("choices") == [] for payload in payloads)
    assert client.aio.models.calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("channel", ["express", "vertex"])
async def test_stream_pup_with_synthetic_part_is_checked_before_transform(channel):
    synthetic_name = "v2o_emit_terminal"
    synthetic_part = types.Part(function_call=types.FunctionCall(
        name=synthetic_name, args={"content": "must-not-leak"}))
    client = _FakeClient([_chunk(
        "PUP_LIMITED_DISABLED", parts=[types.Part(text="partial"), synthetic_part])])
    request = OpenAIRequest(
        model="gemini-3.6-flash", messages=[{"role": "user", "content": "hi"}],
        stream=True,
    )

    response = await helpers.execute_gemini_call(
        client, "gemini-3.6-flash", lambda _: [], {}, request,
        channel_name=channel, synthetic_tool_name=synthetic_name,
    )
    lines = [line async for line in response.body_iterator]
    body = "".join(lines)

    assert "must-not-leak" not in body
    assert "partial" not in body
    assert '"upstream_finish_reason": "PUP_LIMITED_DISABLED"' in body
    assert lines.count("data: [DONE]\n\n") == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("channel", ["express", "vertex"])
async def test_stream_missing_final_reason_is_protocol_error(channel):
    client = _FakeClient([_chunk(parts=[types.Part(text="partial")])])
    request = OpenAIRequest(
        model="gemini-3.6-flash", messages=[{"role": "user", "content": "hi"}],
        stream=True,
    )

    response = await helpers.execute_gemini_call(
        client, "gemini-3.6-flash", lambda _: [], {}, request, channel_name=channel,
    )
    lines = [line async for line in response.body_iterator]
    errors = [json.loads(line[6:])["error"] for line in lines
              if line.startswith("data: {") and '"error"' in line]

    assert len(errors) == 1
    assert errors[0]["category"] == "empty_or_protocol"
    assert errors[0]["upstream_finish_reason"] == ""
    assert lines.count("data: [DONE]\n\n") == 1
    assert not any('"finish_reason": "stop"' in line for line in lines)


@pytest.mark.asyncio
@pytest.mark.parametrize("channel", ["express", "vertex"])
async def test_nonstream_unspecified_prompt_feedback_is_not_a_block(channel):
    prompt_feedback = SimpleNamespace(block_reason="UNSPECIFIED", block_reason_message=None)
    client = _FakeClient(_response("STOP", prompt_feedback=prompt_feedback))
    request = OpenAIRequest(
        model="gemini-3.6-flash", messages=[{"role": "user", "content": "hi"}],
        stream=False,
    )

    response = await helpers.execute_gemini_call(
        client, "gemini-3.6-flash", lambda _: [], {}, request, channel_name=channel,
    )

    assert response.status_code == 200
    assert json.loads(response.body)["choices"][0]["finish_reason"] == "stop"


@pytest.mark.asyncio
@pytest.mark.parametrize("channel", ["express", "vertex"])
async def test_nonstream_prompt_feedback_safety_without_candidates_is_content_filter(channel):
    prompt_feedback = SimpleNamespace(block_reason="SAFETY", block_reason_message="blocked")
    prompt_feedback = SimpleNamespace(block_reason="SAFETY", block_reason_message="blocked")
    client = _FakeClient(SimpleNamespace(
        candidates=[], text=None, prompt_feedback=prompt_feedback, usage_metadata=None,
    ))
    request = OpenAIRequest(
        model="gemini-3.6-flash", messages=[{"role": "user", "content": "hi"}],
        stream=False,
    )

    response = await helpers.execute_gemini_call(
        client, "gemini-3.6-flash", lambda _: [], {}, request, channel_name=channel,
    )

    body = json.loads(response.body)
    assert response.status_code == 200
    assert body["choices"][0]["finish_reason"] == "content_filter"


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", ["PUP_LIMITED_DISABLED", "NEW_UNKNOWN_REASON"])
async def test_fake_hybrid_generation_failure_is_one_visible_error_not_failover(reason, monkeypatch):
    client = _FakeClient(_response(reason))
    request = OpenAIRequest(
        model="fake-gemini-3.6-flash",
        messages=[{"role": "user", "content": "hi"}], stream=True,
    )
    monkeypatch.setattr(
        chat_api, "_dispatch",
        lambda *args, **kwargs: pytest.fail("generation protocol failure must not fail over"),
    )

    response = await helpers.execute_gemini_call(
        client, "gemini-3.6-flash", lambda _: [], {}, request,
        channel_name="express", failover_mode=True, force_fake_streaming=True,
    )
    wrapper = chat_api._stream_with_failover(
        response, ["vertex"], request, SimpleNamespace(), True, "express")
    lines = [line async for line in wrapper]
    errors = [json.loads(line[6:])["error"] for line in lines
              if line.startswith("data: {") and '"error"' in line]

    assert len(errors) == 1
    assert errors[0]["category"] == ("credential_permanent" if reason == "PUP_LIMITED_DISABLED"
                                      else "empty_or_protocol")
    assert errors[0]["upstream_finish_reason"] == reason
    assert lines.count("data: [DONE]\n\n") == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", ["PUP_LIMITED_DISABLED", "NEW_UNKNOWN_REASON"])
async def test_real_hybrid_generation_failure_is_one_visible_error_not_failover(reason, monkeypatch):
    client = _FakeClient([
        _chunk(parts=[types.Part(text="visible prefix")]),
        _chunk(reason, parts=[types.Part(text="must-not-leak")]),
    ])
    request = OpenAIRequest(
        model="gemini-3.6-flash", messages=[{"role": "user", "content": "hi"}],
        stream=True,
    )
    monkeypatch.setattr(
        chat_api, "_dispatch",
        lambda *args, **kwargs: pytest.fail("generation protocol failure must not fail over"),
    )

    response = await helpers.execute_gemini_call(
        client, "gemini-3.6-flash", lambda _: [], {}, request,
        channel_name="express", failover_mode=True,
    )
    wrapper = chat_api._stream_with_failover(
        response, ["vertex"], request, SimpleNamespace(), True, "express")
    lines = [line async for line in wrapper]
    errors = [json.loads(line[6:])["error"] for line in lines
              if line.startswith("data: {") and '"error"' in line]
    output = "".join(lines)

    assert "visible prefix" in output
    assert "must-not-leak" not in output
    assert len(errors) == 1
    assert errors[0]["category"] == ("credential_permanent" if reason == "PUP_LIMITED_DISABLED"
                                      else "empty_or_protocol")
    assert errors[0]["upstream_finish_reason"] == reason
    assert lines.count("data: [DONE]\n\n") == 1
    assert client.aio.models.calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("channel", ["express", "vertex"])
async def test_nonstream_permanent_generation_end_is_explicit_error(channel, monkeypatch):

    client = _FakeClient(_response("PUP_LIMITED_DISABLED"))
    stats = Mock()
    monkeypatch.setattr(helpers, "stats", stats)
    request = OpenAIRequest(
        model="gemini-3.6-flash", messages=[{"role": "user", "content": "hi"}],
        stream=False,
    )

    response = await helpers.execute_gemini_call(
        client, "gemini-3.6-flash", lambda _: [], {}, request, channel_name=channel,
    )

    assert isinstance(response, JSONResponse)
    assert response.status_code == 502
    error = json.loads(response.body)["error"]
    assert error["category"] == "credential_permanent"
    assert error["upstream_finish_reason"] == "PUP_LIMITED_DISABLED"
    assert error["type"] == "upstream_account_disabled"
    assert client.aio.models.calls == 1
    stats.add_tokens.assert_not_called()
