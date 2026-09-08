"""消费真实 SSE 生成器，验证尾块、明细、重试及统计；全部上游使用 mock。"""
import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from google.genai import types

import api_helpers as helpers
from failover import UpstreamUnstartedError
from models import OpenAIRequest
from upstreams import cookie_proxy as cookie
from usage_mapping import map_usage, with_usage_null
from test_usage_mapping import META, EXPECTED, sdk_meta


MODEL = "gemini-3.6-flash"


def request(options=True, stream=True, model=MODEL, **kwargs):
    opts = {} if options is None else {"stream_options": {"include_usage": options}}
    return OpenAIRequest(model=model, messages=[{"role": "user", "content": "hi"}],
                         stream=stream, **opts, **kwargs)


def response(parts=None, usage=None, n=1):
    return types.GenerateContentResponse(candidates=[types.Candidate(
        index=i, content=types.Content(role="model", parts=parts or [types.Part(text="hello")]),
        finish_reason=types.FinishReason.STOP) for i in range(n)], usage_metadata=usage)


def client_for(chunks=(), full=None):
    async def stream(**kwargs):
        async def gen():
            for chunk in chunks:
                if isinstance(chunk, BaseException):
                    raise chunk
                yield chunk
        return gen()
    models = SimpleNamespace(generate_content_stream=AsyncMock(side_effect=stream),
                             generate_content=AsyncMock(return_value=full))
    return SimpleNamespace(aio=SimpleNamespace(models=models))


async def consume(resp):
    lines = [line async for line in resp.body_iterator]
    payloads = [json.loads(line[6:]) for line in lines
                if line.startswith("data: ") and line.strip() != "data: [DONE]"]
    return lines, payloads


def assert_usage_contract(lines, payloads, enabled, expected=EXPECTED):
    assert lines[-1] == "data: [DONE]\n\n"
    assert lines.count("data: [DONE]\n\n") == 1
    tails = [p for p in payloads if p.get("choices") == []]
    ordinary = [p for p in payloads if p.get("choices")]
    if enabled:
        assert len(tails) == 1
        assert payloads[-1] == tails[0]
        assert tails[0]["usage"] == expected
        assert all("usage" in p and p["usage"] is None for p in ordinary)
        assert all(p["id"] == tails[0]["id"] and p["model"] == tails[0]["model"] for p in ordinary)
        assert any(c.get("finish_reason") for c in ordinary[-1]["choices"])
    else:
        assert not tails
        assert all("usage" not in p for p in ordinary)


@pytest.fixture(autouse=True)
def isolate_stats_and_settings(monkeypatch):
    stats = Mock()
    monkeypatch.setattr(helpers, "stats", stats)
    monkeypatch.setattr(cookie, "stats", stats)
    monkeypatch.setattr(helpers, "get_retry_settings", lambda *args: (0, 0))
    monkeypatch.setattr(cookie, "get_retry_settings", lambda *args: (0, 0))
    monkeypatch.setattr(helpers.app_state, "get_effective_settings", lambda *args: {})
    monkeypatch.setattr(helpers.app_state, "get_setting", lambda key, default=None: default)
    return stats


@pytest.mark.parametrize("channel", ["express", "vertex"])
@pytest.mark.parametrize("fake", [False, True])
@pytest.mark.parametrize("options", [True, False, None])
async def test_sdk_stream_contract(channel, fake, options, isolate_stats_and_settings):
    full = response(usage=sdk_meta(META))
    chunks = [response(), types.GenerateContentResponse(usage_metadata=sdk_meta(META))]
    client = client_for(chunks, full)
    resp = await helpers.execute_gemini_call(client, MODEL, lambda _: [], {}, request(options),
                                           channel_name=channel, force_fake_streaming=fake,
                                           prefill_text="prefix ")
    lines, payloads = await consume(resp)
    assert_usage_contract(lines, payloads, options is True)
    isolate_stats_and_settings.add_tokens.assert_called_once()
    assert isolate_stats_and_settings.add_tokens.call_args.args == (100, 50)
    assert isolate_stats_and_settings.add_tokens.call_args.kwargs["cached"] == 40


@pytest.mark.parametrize("channel", ["express", "vertex"])
async def test_sdk_nonstream_usage(channel):
    resp = await helpers.execute_gemini_call(client_for(full=response(usage=sdk_meta(META))),
                                           MODEL, lambda _: [], {}, request(stream=False), channel_name=channel)
    assert json.loads(resp.body)["usage"] == EXPECTED


@pytest.mark.parametrize("fake", [False, True])
async def test_multiple_choices_share_one_usage(fake):
    full = response(usage=sdk_meta(META), n=2)
    resp = await helpers.execute_gemini_call(client_for([full], full), MODEL, lambda _: [], {},
                                           request(n=2), force_fake_streaming=fake)
    lines, payloads = await consume(resp)
    assert_usage_contract(lines, payloads, True)
    assert {c["index"] for p in payloads for c in p["choices"]} == {0, 1}


@pytest.mark.parametrize("fake", [False, True])
@pytest.mark.parametrize("synthetic", [False, True])
async def test_tool_output_preserves_usage(fake, synthetic):
    name = "deliver_response" if synthetic else "weather"
    args = {"content": "hello"} if synthetic else {"city": "上海"}
    full = response([types.Part(function_call=types.FunctionCall(name=name, args=args))], sdk_meta(META))
    resp = await helpers.execute_gemini_call(client_for([full], full), MODEL, lambda _: [], {},
                                           request(), force_fake_streaming=fake,
                                           synthetic_tool_name=name if synthetic else None)
    lines, payloads = await consume(resp)
    assert_usage_contract(lines, payloads, True)
    deltas = [c["delta"] for p in payloads for c in p["choices"]]
    if synthetic:
        assert "".join(d.get("content") or "" for d in deltas) == "hello"
    else:
        assert any(d.get("tool_calls") for d in deltas)


async def test_cumulative_usage_is_not_summed(isolate_stats_and_settings):
    only_usage = types.GenerateContentResponse(usage_metadata=sdk_meta(META))
    client = client_for([response(usage=sdk_meta(META)), only_usage, only_usage])
    resp = await helpers.execute_gemini_call(client, MODEL, lambda _: [], {}, request())
    lines, payloads = await consume(resp)
    assert_usage_contract(lines, payloads, True)
    assert len([p for p in payloads if p["choices"]]) == 1
    isolate_stats_and_settings.add_tokens.assert_called_once()
    assert isolate_stats_and_settings.add_tokens.call_args.args == (100, 50)


async def test_usage_only_before_retry_does_not_commit_or_leak(monkeypatch, isolate_stats_and_settings):
    monkeypatch.setattr(helpers, "get_retry_settings", lambda *args: (1, 0))
    first = client_for([types.GenerateContentResponse(usage_metadata=sdk_meta(META)), ValueError("429")])
    second = client_for([response()])
    calls = [first.aio.models.generate_content_stream, second.aio.models.generate_content_stream]
    async def stream(**kwargs):
        return await calls.pop(0)(**kwargs)
    client = SimpleNamespace(aio=SimpleNamespace(models=SimpleNamespace(generate_content_stream=stream)))
    resp = await helpers.execute_gemini_call(client, MODEL, lambda _: [], {}, request())
    lines, payloads = await consume(resp)
    assert_usage_contract(lines, payloads, True, map_usage())
    assert not calls
    isolate_stats_and_settings.add_tokens.assert_not_called()


@pytest.mark.parametrize("fake", [False, True])
async def test_failure_has_no_success_usage(fake):
    client = client_for([ValueError("upstream failed")])
    client.aio.models.generate_content.side_effect = ValueError("upstream failed")
    resp = await helpers.execute_gemini_call(client, MODEL, lambda _: [], {}, request(),
                                           force_fake_streaming=fake)
    lines, payloads = await consume(resp)
    assert lines[-1] == "data: [DONE]\n\n"
    assert not any(p.get("choices") == [] for p in payloads)


async def test_usage_only_still_allows_failover():
    client = client_for([types.GenerateContentResponse(usage_metadata=sdk_meta(META)), ValueError("429")])
    resp = await helpers.execute_gemini_call(client, MODEL, lambda _: [], {}, request(), failover_mode=True)
    with pytest.raises(UpstreamUnstartedError):
        await consume(resp)


async def test_cancel_does_not_emit_usage(isolate_stats_and_settings):
    client = client_for([response(), asyncio.CancelledError()])
    resp = await helpers.execute_gemini_call(client, MODEL, lambda _: [], {}, request())
    lines = []
    with pytest.raises(asyncio.CancelledError):
        async for line in resp.body_iterator:
            lines.append(line)
    assert not any('"choices": []' in line for line in lines)
    assert "data: [DONE]\n\n" not in lines
    isolate_stats_and_settings.add_tokens.assert_not_called()


@pytest.fixture
def cookie_upstream(monkeypatch):
    monkeypatch.setattr(cookie, "_get_cookie_string", lambda: "test-cookie")
    monkeypatch.setattr(cookie, "_get_project_id", lambda: "test-project")
    monkeypatch.setattr(cookie, "build_headers", lambda _: {"test": "test"})
    monkeypatch.setattr(cookie, "build_batch_graphql_body_async", AsyncMock(return_value={"variables": {}}))
    full = {"kind": "ok", "full_text": "hello", "finish_reason": "stop", "usage_meta": META}
    monkeypatch.setattr(cookie, "_collect_full_response", AsyncMock(return_value=full))
    async def events(*args, **kwargs):
        for event in [("text", "hello"), ("usage", META), ("finish", "STOP")]:
            yield event
    monkeypatch.setattr(cookie, "_execute_stream_request_generator", events)
    # 不创建实际 HTTP 客户端，所有 Cookie 事件均来自本地生成器。
    client = AsyncMock()
    client.__aenter__.return_value.post.return_value = SimpleNamespace(
        status_code=200, text=json.dumps({"results": [{"data": {
            "candidates": [{"content": {"parts": [{"text": "hello"}]}, "finishReason": "STOP"}],
            "usageMetadata": META,
        }}]}))
    monkeypatch.setattr(cookie.httpx, "AsyncClient", Mock(return_value=client))
    return cookie.CookieProxyUpstream(), SimpleNamespace(is_disconnected=AsyncMock(return_value=False))


@pytest.mark.parametrize("image", [False, True])
@pytest.mark.parametrize("options", [True, False, None])
async def test_cookie_stream_contract(cookie_upstream, image, options):
    upstream, fastapi_request = cookie_upstream
    model = "gemini-3-pro-image-preview" if image else MODEL
    resp = await upstream.chat_completions(request(options, model=model), fastapi_request)
    lines, payloads = await consume(resp)
    assert_usage_contract(lines, payloads, options is True)


async def test_cookie_nonstream_usage(cookie_upstream):
    upstream, fastapi_request = cookie_upstream
    resp = await upstream.chat_completions(request(stream=False), fastapi_request)
    assert json.loads(resp.body)["usage"] == EXPECTED


async def test_cookie_missing_usage_is_zero(cookie_upstream, monkeypatch):
    async def events(*args, **kwargs):
        yield "text", "hello"
        yield "finish", "STOP"
    monkeypatch.setattr(cookie, "_execute_stream_request_generator", events)
    upstream, fastapi_request = cookie_upstream
    resp = await upstream.chat_completions(request(), fastapi_request)
    lines, payloads = await consume(resp)
    assert_usage_contract(lines, payloads, True, map_usage())


@pytest.mark.parametrize("relay", [False, True])
async def test_buffered_text_precedes_finish_and_usage(relay):
    text = "hello<rela" if relay else "hello"
    client = client_for([response([types.Part(text=text)], sdk_meta(META))])
    resp = await helpers.execute_gemini_call(
        client, MODEL, lambda _: [], {}, request(),
        synthetic_tool_name=None if relay else "deliver_response",
        input_relay_strip_tag="relay" if relay else None)
    lines, payloads = await consume(resp)
    assert_usage_contract(lines, payloads, True)
    assert "".join(c["delta"].get("content") or "" for p in payloads for c in p["choices"]) == text
    finish_at = next(i for i, p in enumerate(payloads)
                     if any(c.get("finish_reason") for c in p["choices"]))
    assert not any(c["delta"].get("content") for p in payloads[finish_at + 1:] for c in p["choices"])


async def test_usage_with_prompt_block_is_not_silently_skipped(monkeypatch):
    converter = Mock(wraps=helpers.convert_chunk_to_openai)
    monkeypatch.setattr(helpers, "convert_chunk_to_openai", converter)
    blocked = types.GenerateContentResponse(
        usage_metadata=sdk_meta(META),
        prompt_feedback=types.GenerateContentResponsePromptFeedback(block_reason="SAFETY"))
    resp = await helpers.execute_gemini_call(client_for([blocked]), MODEL, lambda _: [], {}, request())
    await consume(resp)
    converter.assert_called_once()
    assert converter.call_args.args[0] is blocked


async def test_usage_null_wrapper_preserves_large_content_and_closes_source():
    text = 'data:image/png;base64,' + 'x' * 1_000_000 + ' "choices": [] "usage": {}'
    closed = []
    async def source():
        try:
            yield ": keep-alive\n\n"
            yield cookie._make_openai_chunk("id", MODEL, content=text)
            yield cookie._make_usage_chunk("id", MODEL, EXPECTED)
            yield 'data: {"error": {"message": "test"}}\n\n'
            yield "data: [DONE]\n\n"
        finally:
            closed.append(True)
    lines = [line async for line in with_usage_null(source(), True)]
    assert lines[0] == ": keep-alive\n\n"
    content = json.loads(lines[1][6:])
    assert content["usage"] is None
    assert content["choices"][0]["delta"]["content"] == text
    assert json.loads(lines[2][6:])["usage"] == EXPECTED
    assert "usage" not in json.loads(lines[3][6:])
    assert closed == [True]
