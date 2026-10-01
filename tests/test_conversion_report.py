import asyncio
import json
from types import SimpleNamespace

import pytest
from google.genai import types

import api_helpers as helpers
from conversion_report import ConversionReport
from models import OpenAIRequest
from protocol_conversion import create_generation_config
from runtime_state import app_state
from upstreams.express_sdk import _build_thinking_config


def test_parameter_report_and_privacy(monkeypatch):
    monkeypatch.setattr(app_state, "get_effective_settings", lambda _: {"sampling_policy": "allowed"})
    request = OpenAIRequest(model="gemini-3.8-flash", messages=[{"role": "user", "content": "SECRET"}], temperature=0.5, thinking_budget=0, labels={"secret": "SECRET"})
    report = ConversionReport("vertex")
    report.inspect_request(request)
    create_generation_config(request, reporter=report)
    _build_thinking_config(request.model, request, False, reporter=report)
    entries = report.to_list()
    assert {entry["path"] for entry in entries} == {"labels", "generation_config.temperature", "thinking_budget"}
    assert "SECRET" not in json.dumps(entries)
    assert all(entry["channel"] == "vertex" for entry in entries)


def test_response_dedup_and_reset():
    report = ConversionReport("express")
    report.record("labels", "unsupported", "not_mapped_to_upstream")
    baseline = report.checkpoint()
    response = SimpleNamespace(candidates=[SimpleNamespace(content=SimpleNamespace(parts=[SimpleNamespace(audio_transcription={"text": "SECRET"}, inline_data=SimpleNamespace(display_name="SECRET"))]))])
    for _ in range(100):
        report.inspect_response(response)
    assert len(report.to_list()) == 3
    assert "SECRET" not in json.dumps(report.to_list())
    report.reset(baseline)
    assert len(report.to_list()) == 1


class FakeModels:
    def __init__(self, reason="STOP"):
        self.reason = reason

    def response(self):
        tool = types.Part(function_call=types.FunctionCall(name="lookup", args={"city": "上海"}), thought_signature=b"signature")
        return SimpleNamespace(candidates=[SimpleNamespace(content=SimpleNamespace(role="model", parts=[types.Part(text="ok"), tool, SimpleNamespace(audio_transcription={"text": "SECRET"})]), finish_reason=self.reason)], usage_metadata=None, prompt_feedback=None)

    async def generate_content(self, **kwargs):
        return self.response()

    async def generate_content_stream(self, **kwargs):
        async def chunks():
            yield self.response()
        return chunks()


@pytest.mark.parametrize("channel", ["express", "vertex"])
@pytest.mark.parametrize("mode", ["json", "stream", "fake"])
@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("reason", ["STOP", "PUP_LIMITED_DISABLED"])
async def test_output_contract(channel, mode, enabled, reason, monkeypatch, capsys):
    monkeypatch.setattr(helpers, "get_retry_settings", lambda _: (0, 0))
    request = OpenAIRequest(model="gemini-3.8-flash", messages=[], stream=mode != "json", compatibility_report=enabled, labels={"secret": "SECRET"}, stream_options={"include_usage": True})
    response = await helpers.execute_gemini_call(SimpleNamespace(aio=SimpleNamespace(models=FakeModels(reason))), request.model, lambda _: [], {}, request, channel_name=channel, force_fake_streaming=mode == "fake")
    if mode == "json":
        payload = json.loads(response.body)
        target = payload["error"] if "error" in payload else payload["choices"][0]["message"]
        assert ("vertex2openai" in target.get("extra_content", {})) == enabled
        if reason == "STOP":
            assert target["tool_calls"][0]["extra_content"]["google"]["thought_signature"] == "c2lnbmF0dXJl"
        if enabled:
            assert len(target["extra_content"]["vertex2openai"]["conversion_report"]) == 2
    else:
        lines = [line async for line in response.body_iterator]
        assert lines.count("data: [DONE]\n\n") == 1
        events = [json.loads(line[6:]) for line in lines if line.startswith("data: {")]
        reported = [event for event in events if "conversion_report" in json.dumps(event)]
        assert len(reported) == (1 if enabled else 0)
        if reason == "STOP":
            calls = [call for event in events for choice in event.get("choices", [])
                     for call in choice.get("delta", {}).get("tool_calls", [])]
            assert any(call.get("extra_content", {}).get("google", {}).get("thought_signature") == "c2lnbmF0dXJl" for call in calls)
            ends = [i for i, event in enumerate(events) if any(c.get("finish_reason") for c in event.get("choices", []))]
            usage = [i for i, event in enumerate(events) if not event.get("choices") and event.get("usage")]
            assert len(ends) == 1
            assert usage and ends[0] < usage[0]
    logs = capsys.readouterr().out
    assert logs.count("[转换诊断]") == 1
    assert "SECRET" not in logs


async def test_concurrent_reports_are_isolated(monkeypatch):
    monkeypatch.setattr(helpers, "get_retry_settings", lambda _: (0, 0))

    async def call(channel, field):
        request = OpenAIRequest(model="gemini-3.8-flash", messages=[], compatibility_report=True, **{field: {}})
        response = await helpers.execute_gemini_call(SimpleNamespace(aio=SimpleNamespace(models=FakeModels())), request.model, lambda _: [], {}, request, channel_name=channel)
        return json.loads(response.body)["choices"][0]["message"]["extra_content"]["vertex2openai"]["conversion_report"]

    a, b = await asyncio.gather(call("express", "labels"), call("vertex", "speech_config"))
    assert {e["channel"] for e in a} == {"express"}
    assert {e["channel"] for e in b} == {"vertex"}
    assert "speech_config" not in {e["path"] for e in a}
    assert "labels" not in {e["path"] for e in b}


async def test_retry_discards_previous_response_metadata(monkeypatch):
    monkeypatch.setattr(helpers, "get_retry_settings", lambda _: (1, 0))
    monkeypatch.setattr(helpers, "is_retryable_exception", lambda _: True)
    models = FakeModels()
    report = ConversionReport("express")
    attempt = 0

    async def stream(**kwargs):
        nonlocal attempt
        attempt += 1

        async def chunks():
            if attempt == 1:
                report.record("response.parts.speech_metadata", "unsupported", "not_representable_in_openai")
                raise ValueError("503 synthetic upstream failure")
            yield models.response()
        return chunks()

    request = OpenAIRequest(model="gemini-3.8-flash", messages=[], stream=True, compatibility_report=True)
    response = await helpers.execute_gemini_call(SimpleNamespace(aio=SimpleNamespace(models=SimpleNamespace(generate_content_stream=stream))), request.model, lambda _: [], {}, request, conversion_report=report)
    lines = [line async for line in response.body_iterator]
    assert attempt == 2
    reported = [json.loads(line[6:]) for line in lines if "conversion_report" in line]
    assert len(reported) == 1
    entries = reported[0]["choices"][0]["delta"]["extra_content"]["vertex2openai"]["conversion_report"]
    assert {e["path"] for e in entries} == {"response.parts.audio_transcription"}


async def test_nonstream_exception_report_survives_route(monkeypatch):
    from routes import chat_api

    monkeypatch.setattr(helpers, "get_retry_settings", lambda _: (0, 0))
    monkeypatch.setattr(chat_api.breaker, "is_cooling", lambda _: False)
    monkeypatch.setattr(chat_api.breaker, "report_failure", lambda *args, **kwargs: None)

    async def failing(**kwargs):
        raise ValueError("synthetic upstream failure")

    async def call(request, fastapi_request, failover_mode=False):
        return await helpers.execute_gemini_call(SimpleNamespace(aio=SimpleNamespace(models=SimpleNamespace(generate_content=failing))), request.model, lambda _: [], {}, request, channel_name="express")

    monkeypatch.setitem(chat_api.CHANNELS, "express", SimpleNamespace(chat_completions=call))
    request = OpenAIRequest(model="gemini-3.8-flash", messages=[], labels={}, compatibility_report=True)
    response = await chat_api._dispatch(["express"], request, None, False)
    entries = json.loads(response.body)["error"]["extra_content"]["vertex2openai"]["conversion_report"]
    assert entries[0]["path"] == "labels"


async def test_hybrid_final_stream_error_keeps_report(monkeypatch):
    from routes import chat_api

    monkeypatch.setattr(helpers, "get_retry_settings", lambda _: (0, 0))
    monkeypatch.setattr(chat_api.breaker, "report_failure", lambda *args, **kwargs: None)

    async def failing(**kwargs):
        raise ValueError("503 synthetic upstream failure")

    request = OpenAIRequest(model="gemini-3.8-flash", messages=[], stream=True, compatibility_report=True, labels={})
    response = await helpers.execute_gemini_call(SimpleNamespace(aio=SimpleNamespace(models=SimpleNamespace(generate_content_stream=failing))), request.model, lambda _: [], {}, request, channel_name="express", failover_mode=True)
    lines = [line async for line in chat_api._stream_with_failover(response, [], request, None, True, primary_channel="express")]
    assert lines.count("data: [DONE]\n\n") == 1
    error = next(json.loads(line[6:])["error"] for line in lines if '"error"' in line)
    assert error["extra_content"]["vertex2openai"]["conversion_report"][0]["path"] == "labels"
