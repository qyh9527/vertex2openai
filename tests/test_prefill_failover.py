"""Prefill stays local until a stream can commit upstream output."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from google.genai import types

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))

import api_helpers as helpers
from models import OpenAIRequest
from routes import chat_api


class _FakeModels:
    def __init__(self, attempts, *, fail_at_await=False):
        self.attempts = attempts
        self.fail_at_await = fail_at_await
        self.calls = 0

    async def generate_content_stream(self, **kwargs):
        events = self.attempts[min(self.calls, len(self.attempts) - 1)]
        self.calls += 1
        if self.fail_at_await and isinstance(events[0], Exception):
            raise events[0]

        async def chunks():
            for event in events:
                if isinstance(event, Exception):
                    raise event
                yield event

        return chunks()


class _FakeClient:
    def __init__(self, attempts, **kwargs):
        self.aio = SimpleNamespace(models=_FakeModels(attempts, **kwargs))


def _chunk(finish_reason=None, *, parts=None, usage=None, prompt_feedback=None):
    candidates = []
    if finish_reason is not None or parts is not None:
        candidates = [types.Candidate(
            content=types.Content(role="model", parts=parts or []),
            finish_reason=finish_reason,
        )]
    return types.GenerateContentResponse(
        candidates=candidates, usage_metadata=usage, prompt_feedback=prompt_feedback)


def _request():
    return OpenAIRequest(model="gemini-3.6-flash",
                         messages=[{"role": "user", "content": "hi"}], stream=True)


async def _execute(client, request=None, **kwargs):
    return await helpers.execute_gemini_call(
        client, "gemini-3.6-flash", lambda _: [], {}, request or _request(),
        channel_name=kwargs.pop("channel_name", "express"),
        prefill_text="prefix ", failover_mode=True, **kwargs)


def _payloads(lines):
    return [json.loads(line[6:]) for line in lines
            if line.startswith("data: ") and line.strip() != "data: [DONE]"]


def _contents(lines):
    return [choice["delta"]["content"] for payload in _payloads(lines)
            for choice in payload.get("choices", [])
            if choice.get("delta", {}).get("content")]


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    monkeypatch.setattr(helpers, "get_retry_settings", lambda *_: (1, 0))
    monkeypatch.setattr(helpers, "stats", Mock())
    breaker = Mock()
    breaker.is_cooling.return_value = False
    monkeypatch.setattr(chat_api, "breaker", breaker)
    monkeypatch.setattr(chat_api, "_current_credential_id", lambda _: "")


async def _dispatch(monkeypatch, primary, fallback):
    calls = []

    def upstream(channel, client):
        async def chat_completions(request, fastapi_request, **kwargs):
            calls.append(channel)
            return await _execute(client, request, channel_name=channel)
        return SimpleNamespace(chat_completions=chat_completions)

    monkeypatch.setattr(chat_api, "CHANNELS", {
        "express": upstream("express", primary),
        "vertex": upstream("vertex", fallback),
    })
    response = await chat_api._dispatch(["express", "vertex"], _request(), None, True)
    lines = [line async for line in response.body_iterator]
    return calls, lines


@pytest.mark.asyncio
@pytest.mark.parametrize("code", [429, 503])
@pytest.mark.parametrize("fail_at_await", [False, True])
async def test_prefill_unstarted_failure_switches_real_dispatch(monkeypatch, code, fail_at_await):
    primary = _FakeClient([[RuntimeError(f"{code} old Express failure")]],
                          fail_at_await=fail_at_await)
    fallback = _FakeClient([[_chunk("STOP", parts=[types.Part(text="answer")])]])
    calls, lines = await _dispatch(monkeypatch, primary, fallback)
    assert calls == ["express", "vertex"]
    assert primary.aio.models.calls == 2
    assert fallback.aio.models.calls == 1
    assert _contents(lines) == ["prefix ", "answer"]
    assert lines.count("data: [DONE]\n\n") == 1
    assert not any("old Express failure" in line or '"error"' in line for line in lines)


@pytest.mark.asyncio
async def test_usage_and_heartbeat_do_not_commit_prefill(monkeypatch):
    usage = types.GenerateContentResponseUsageMetadata(prompt_token_count=2)
    primary = _FakeClient([[_chunk(), _chunk(usage=usage), RuntimeError("429 busy")]])
    fallback = _FakeClient([[_chunk("STOP", parts=[types.Part(text="answer")])]])
    calls, lines = await _dispatch(monkeypatch, primary, fallback)
    assert calls == ["express", "vertex"]
    assert _contents(lines) == ["prefix ", "answer"]
    assert lines.count("data: [DONE]\n\n") == 1


@pytest.mark.asyncio
async def test_retry_success_preserves_prefill_deduplication():
    client = _FakeClient([[RuntimeError("429 busy")], [
        _chunk(parts=[types.Part(text="pre")]),
        _chunk("STOP", parts=[types.Part(text="fix answer")]),
    ]])
    response = await _execute(client)
    lines = [line async for line in response.body_iterator]
    assert _contents(lines) == ["prefix ", "answer"]
    assert client.aio.models.calls == 2
    assert lines.count("data: [DONE]\n\n") == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["text", "tool", "reasoning"])
async def test_committed_output_does_not_retry_or_switch(monkeypatch, kind):
    part = {"text": types.Part(text="answer"),
            "tool": types.Part(function_call=types.FunctionCall(name="lookup", args={})),
            "reasoning": types.Part(text="thinking", thought=True)}[kind]
    primary = _FakeClient([[_chunk(parts=[part]), RuntimeError("429 busy")]])
    fallback = _FakeClient([[_chunk("STOP", parts=[types.Part(text="wrong")])]])
    calls, lines = await _dispatch(monkeypatch, primary, fallback)
    assert calls == ["express"]
    assert primary.aio.models.calls == 1
    assert fallback.aio.models.calls == 0
    assert _contents(lines).count("prefix ") == 1
    assert lines.count("data: [DONE]\n\n") == 1
    assert any('"error"' in line for line in lines)
    if kind == "text":
        assert _contents(lines) == ["prefix ", "answer"]
    else:
        field = "tool_calls" if kind == "tool" else "reasoning_content"
        effective = [p for p in _payloads(lines)
                     if any(c.get("delta", {}).get(field) for c in p.get("choices", []))]
        assert effective
        assert lines.index(next(line for line in lines if '"prefix "' in line)) < lines.index(
            next(line for line in lines if f'"{field}"' in line))


@pytest.mark.asyncio
async def test_synthetic_content_follows_prefill():
    name = "v2o_emit_terminal"
    client = _FakeClient([[_chunk("STOP", parts=[types.Part(function_call=types.FunctionCall(
        name=name, args={"content": "answer"}))])]])
    response = await _execute(client, synthetic_tool_name=name)
    lines = [line async for line in response.body_iterator]
    assert _contents(lines) == ["prefix ", "answer"]
    assert lines.count("data: [DONE]\n\n") == 1


@pytest.mark.asyncio
async def test_pending_real_tool_flush_follows_prefill():
    part = types.Part(function_call=types.FunctionCall(name="lookup", will_continue=True))
    client = _FakeClient([[_chunk(parts=[part])]])
    response = await _execute(client, synthetic_tool_name="v2o_emit_terminal")
    lines = [line async for line in response.body_iterator]
    assert _contents(lines) == []
    assert not any('"tool_calls"' in line for line in lines)
    assert any('"error"' in line for line in lines)
    assert lines.count("data: [DONE]\n\n") == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("threshold", [0, 3, 10000])
async def test_filtered_tail_and_side_buffer_follow_prefill(monkeypatch, threshold):
    original = helpers.app_state.get_setting
    monkeypatch.setattr(helpers.app_state, "get_setting", lambda key, *args:
                        threshold if key == helpers.SETTING_SIDE_BUFFER else original(key, *args))
    client = _FakeClient([[
        _chunk(parts=[types.Part(text="pre")]),
        _chunk(parts=[types.Part(text="amble")]),
        _chunk("STOP"),
    ]])
    response = await _execute(client, synthetic_tool_name="v2o_emit_terminal")
    lines = [line async for line in response.body_iterator]
    assert "".join(_contents(lines)) == "prefix preamble"
    assert _contents(lines)[0] == "prefix "
    assert _contents(lines).count("prefix ") == 1
    assert lines.count("data: [DONE]\n\n") == 1


@pytest.mark.asyncio
async def test_normal_empty_stop_keeps_prefill():
    response = await _execute(_FakeClient([[_chunk("STOP")]]))
    lines = [line async for line in response.body_iterator]
    assert _contents(lines) == ["prefix "]
    assert not any('"error"' in line for line in lines)
    assert lines.count("data: [DONE]\n\n") == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["permanent", "prompt_feedback", "missing_terminal"])
async def test_failed_terminal_without_output_does_not_leak_prefill(kind):
    event = {"permanent": _chunk("MALFORMED_FUNCTION_CALL"),
             "prompt_feedback": _chunk(prompt_feedback=types.GenerateContentResponsePromptFeedback(
                 block_reason="OTHER")),
             "missing_terminal": _chunk()}[kind]
    client = _FakeClient([[event]])
    response = await _execute(client)
    lines = [line async for line in response.body_iterator]
    assert _contents(lines) == []
    assert any('"error"' in line for line in lines)
    assert lines.count("data: [DONE]\n\n") == 1
    assert client.aio.models.calls == 1
