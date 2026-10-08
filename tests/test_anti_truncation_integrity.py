"""Issue #2: restoration must not imply a successful generation ending."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))

from anti_truncation import strip_synthetic_from_openai_dict
from outcome import GenerationEndError
from google.genai import types
from anti_truncation import StreamPartialState, transform_stream_chunk


def stream_chunk(fc=None, text=None, finish=None):
    parts = []
    if fc is not None:
        parts.append(types.Part(function_call=types.FunctionCall(**fc)))
    if text is not None:
        parts.append(types.Part(text=text))
    return types.GenerateContentResponse(candidates=[types.Candidate(
        content=types.Content(role="model", parts=parts), finish_reason=finish)])


def partial(path, value, continuing=True):
    return {"json_path": path, "string_value": value, "will_continue": continuing}


async def test_ocr_named_synthetic_partial_close():
    _, records = await stream_output([
        stream_chunk({"name": TOOL, "partial_args": [partial("$.content", "complete", False)], "will_continue": False}),
        stream_chunk(finish="STOP")])
    assert not any("error" in r for r in records)
    assert any(c.get("finish_reason") == "stop" for r in records for c in r.get("choices", []))


@pytest.mark.parametrize("named", [True, False])
async def test_ocr_real_last_partial_closes_call(named):
    last = {"partial_args": [partial("$.city", "Paris", False)], "will_continue": False}
    start = []
    if named:
        last.update(name="lookup", id="fixture-real")
    else:
        start = [stream_chunk({"name": "lookup", "id": "fixture-real", "will_continue": True})]
    last_chunk = stream_chunk(last)
    signature_chunk = last_chunk if named else start[0]
    signature_chunk.candidates[0].content.parts[0].thought_signature = b"fixture-signature"
    _, records = await stream_output([*start, last_chunk, stream_chunk(finish="STOP")])
    assert not any("error" in r for r in records)
    calls = [tc for r in records for c in r.get("choices", []) for tc in c.get("delta", {}).get("tool_calls", [])]
    assert len(calls) == 1
    assert json.loads(calls[0]["function"]["arguments"]) == {"city": "Paris"}
    assert calls[0]["id"].startswith("fixture-real")
    import base64
    assert calls[0]["extra_content"]["google"]["thought_signature"] == base64.b64encode(b"fixture-signature").decode()
    assert any(c.get("finish_reason") == "tool_calls" for r in records for c in r.get("choices", []))


@pytest.mark.parametrize("filter_kind", ["relay", "prefill"])
async def test_ocr_filtered_plain_does_not_lock_channel(filter_kind, monkeypatch):
    from runtime_state import app_state
    original = app_state.get_setting
    monkeypatch.setattr(app_state, "get_setting", lambda key, default=None:
                        0 if key == "anti_truncation_side_buffer_bytes" else original(key, default))
    kwargs = {"input_relay_strip_tag": "tag"} if filter_kind == "relay" else {"prefill_text": "prefix"}
    plain = "<tag>hidden</tag>" if filter_kind == "relay" else "prefix"
    _, records = await stream_output([
        stream_chunk(text=plain), stream_chunk({"name": TOOL, "args": {"content": "synthetic"}}),
        stream_chunk(finish="STOP")], **kwargs)
    text = "".join(c.get("delta", {}).get("content") or "" for r in records for c in r.get("choices", []))
    assert text == ("synthetic" if filter_kind == "relay" else "prefixsynthetic")


async def test_ocr_candidate_identity_is_stable_across_batches():
    def indexed(index, text=None, fc=None, finish=None):
        chunk = stream_chunk(fc=fc, text=text, finish=finish)
        chunk.candidates[0].index = index
        return chunk
    first = indexed(0, text="zero", finish="STOP")
    first.candidates.extend(indexed(1, fc={"name": "lookup", "args": {"city": "Paris"}}).candidates)
    _, records = await stream_output([first, indexed(1, text="one"), indexed(1, finish="STOP")],
                                    synthetic_tool_name=None, n=2)
    assert not any("error" in r for r in records)
    choices = [c for r in records for c in r.get("choices", [])]
    assert [(c["index"], c["finish_reason"]) for c in choices if c.get("finish_reason")] == [(0, "stop"), (1, "tool_calls")]
    assert [(c["index"], c["delta"]["content"]) for c in choices if c.get("delta", {}).get("content")] == [(0, "zero"), (1, "one")]


async def test_ocr_complete_args_still_require_call_close():
    _, records = await stream_output([
        stream_chunk({"name": TOOL, "args": {"content": "answer"}, "will_continue": True}),
        stream_chunk(finish="STOP")])
    assert any("error" in r for r in records)
    assert not any(c.get("finish_reason") for r in records for c in r.get("choices", []))


async def test_ocr_synthetic_foreign_id_is_rejected():
    _, records = await stream_output([
        stream_chunk({"name": TOOL, "id": "one", "will_continue": True}),
        stream_chunk({"id": "two", "partial_args": [partial("$.content", "foreign", False)], "will_continue": False}),
        stream_chunk(finish="STOP")])
    assert any("error" in r for r in records)
    assert not any(c.get("delta", {}).get("content") == "foreign" for r in records for c in r.get("choices", []))


async def test_ocr_first_empty_real_string_is_valid():
    _, records = await stream_output([
        stream_chunk({"name": "lookup", "will_continue": True}),
        stream_chunk({"partial_args": [partial("$.query", "", False)]}),
        stream_chunk({}), stream_chunk(finish="STOP")])
    assert not any("error" in r for r in records)
    calls = [tc for r in records for c in r.get("choices", []) for tc in c.get("delta", {}).get("tool_calls", [])]
    assert json.loads(calls[0]["function"]["arguments"]) == {"query": ""}


async def test_ocr_closed_synthetic_call_rejects_later_parameters():
    _, records = await stream_output([
        stream_chunk({"name": TOOL, "will_continue": True}),
        stream_chunk({"partial_args": [partial("$.content", "a")], "will_continue": False}),
        stream_chunk({"partial_args": [partial("$.content", "b", False)]}), stream_chunk(finish="STOP")])
    assert any("error" in r for r in records)
    text = "".join(c.get("delta", {}).get("content") or "" for r in records for c in r.get("choices", []))
    assert text == "a"


def test_ocr_mixed_invalid_real_arguments_are_not_executable():
    value = completion()
    value["choices"][0]["message"]["tool_calls"][1]["function"]["arguments"] = '{"city":'
    with pytest.raises(GenerationEndError):
        strip_synthetic_from_openai_dict(value, TOOL)


async def test_ocr_anonymous_synthetic_invalid_args_are_rejected():
    _, records = await stream_output([
        stream_chunk({"name": TOOL, "partial_args": [partial("$.content", "answer", False)], "will_continue": True}),
        stream_chunk({"args": {"content": 123}}), stream_chunk(finish="STOP")])
    assert any("error" in r for r in records)


def test_ocr_reverse_nested_real_path_does_not_discard_parameter():
    state = StreamPartialState(TOOL)
    transform_stream_chunk(stream_chunk({"name": "lookup", "will_continue": True}), 0, state)
    transform_stream_chunk(stream_chunk({"partial_args": [partial("$.city.name", "Paris", False)]}), 0, state)
    with pytest.raises(GenerationEndError):
        transform_stream_chunk(stream_chunk({"partial_args": [partial("$.city", "other", False)]}), 0, state)


def test_damaged_real_path_never_produces_empty_executable_call():
    state = StreamPartialState(TOOL)
    transform_stream_chunk(stream_chunk({"name": "lookup", "will_continue": True}), 0, state)
    with pytest.raises(GenerationEndError):
        transform_stream_chunk(stream_chunk({"partial_args": [partial("$.[", "broken")]}), 0, state)
        transform_stream_chunk(stream_chunk({}), 0, state)


def test_unclosed_real_call_is_not_flushed_as_executable():
    state = StreamPartialState(TOOL)
    transform_stream_chunk(stream_chunk({"name": "lookup", "will_continue": True}), 0, state)
    transform_stream_chunk(stream_chunk({"partial_args": [partial("$.city", "Paris")]}), 0, state)
    with pytest.raises(GenerationEndError):
        state.flush_pending_real()

TOOL = "v2o_emit_integrity_test"


def completion(arguments='{"content":"answer"}', finish="stop", real=True):
    calls = [{"id": "synthetic", "type": "function", "function": {
        "name": TOOL, "arguments": arguments}}]
    if real:
        calls.append({"id": "real", "type": "function", "function": {
            "name": "lookup", "arguments": "{}"}})
    return {"choices": [{"index": 0, "message": {
        "role": "assistant", "content": None, "tool_calls": calls},
        "finish_reason": finish}]}


@pytest.mark.parametrize("finish", ["length", "content_filter", "error", None])
def test_mixed_tools_preserve_non_success_finish(finish):
    result = strip_synthetic_from_openai_dict(completion(finish=finish), TOOL)
    assert result["choices"][0]["finish_reason"] == finish
    assert result["choices"][0]["message"]["tool_calls"][0]["id"] == "real"


@pytest.mark.parametrize("arguments", [
    '{"content":"partial', '{"content":"a","content":"b"}',
    '{"content":123}', '{"content":"bad\\x"}', '{"content":"ok"} trailing',
])
def test_normal_end_rejects_invalid_synthetic_arguments(arguments):
    with pytest.raises(GenerationEndError):
        strip_synthetic_from_openai_dict(completion(arguments, real=False), TOOL)


def test_length_salvages_partial_without_normalizing_finish():
    result = strip_synthetic_from_openai_dict(
        completion('{"content":"partial', finish="length", real=False), TOOL)
    assert result["choices"][0]["message"]["content"] == "partial"
    assert result["choices"][0]["finish_reason"] == "length"


def test_unknown_synthetic_path_is_not_user_text():
    state = StreamPartialState(TOOL)
    transform_stream_chunk(stream_chunk({"name": TOOL, "will_continue": True}), 0, state)
    with pytest.raises(GenerationEndError):
        transform_stream_chunk(stream_chunk({"partial_args": [partial("$.answer", "private")]}), 0, state)


async def stream_output(chunks, *, synthetic_tool_name=TOOL, n=1, **kwargs):
    from api_helpers import execute_gemini_call
    from models import OpenAIRequest
    from types import SimpleNamespace

    async def generate(**kwargs):
        async def source():
            for chunk in chunks:
                yield chunk
        return source()

    client = SimpleNamespace(aio=SimpleNamespace(models=SimpleNamespace(generate_content_stream=generate)))
    request = OpenAIRequest(model="gemini-test", messages=[{"role": "user", "content": "fixture"}], stream=True, n=n)
    response = await execute_gemini_call(client, request.model, lambda _: [], {}, request,
                                         synthetic_tool_name=synthetic_tool_name, **kwargs)
    wire = "".join([item async for item in response.body_iterator])
    records = [json.loads(line[6:]) for line in wire.splitlines() if line.startswith("data: {")]
    return wire, records


async def test_stop_with_unclosed_synthetic_stream_is_error_not_success():
    wire, records = await stream_output([
        stream_chunk({"name": TOOL, "will_continue": True}),
        stream_chunk({"partial_args": [partial("$.content", "partial")]}),
        stream_chunk(finish="STOP"),
    ])
    assert any("error" in r for r in records)
    assert not any(c.get("finish_reason") == "stop" for r in records for c in r.get("choices", []))
    assert wire.count("data: [DONE]") == 1


@pytest.mark.parametrize("synthetic_first", [True, False])
async def test_first_committed_text_channel_is_irreversible(synthetic_first):
    plain = stream_chunk(text="plain")
    synthetic = stream_chunk({"name": TOOL, "args": {"content": "synthetic"}})
    _, records = await stream_output([
        *([synthetic, plain] if synthetic_first else [plain, synthetic]), stream_chunk(finish="STOP")])
    text = "".join(c.get("delta", {}).get("content") or "" for r in records for c in r.get("choices", []))
    assert text == ("synthetic" if synthetic_first else "plain")


@pytest.mark.parametrize("path", ["$.a[10001]", "$.a..b", "garbage", "$." + ".".join(["a"] * 80)])
def test_invalid_real_paths_are_bounded_protocol_errors(path):
    state = StreamPartialState(TOOL)
    transform_stream_chunk(stream_chunk({"name": "lookup", "will_continue": True}), 0, state)
    with pytest.raises(GenerationEndError):
        transform_stream_chunk(stream_chunk({"partial_args": [partial(path, "x")]}), 0, state)


async def test_duplicate_finish_is_error_and_no_success_finish():
    wire, records = await stream_output([
        stream_chunk(text="text"), stream_chunk(finish="STOP"), stream_chunk(finish="STOP")])
    assert any("error" in r for r in records)
    assert not any(c.get("finish_reason") for r in records for c in r.get("choices", []))
    assert wire.count("data: [DONE]") == 1


async def test_utf8_threshold_commits_on_same_chunk_before_synthetic(monkeypatch):
    from runtime_state import app_state
    original = app_state.get_setting
    monkeypatch.setattr(app_state, "get_setting", lambda key, default=None:
                        7 if key == "anti_truncation_side_buffer_bytes" else original(key, default))
    _, records = await stream_output([
        stream_chunk(text="中😀"),
        stream_chunk({"name": TOOL, "args": {"content": "synthetic"}}),
        stream_chunk(finish="STOP")])
    text = "".join(c.get("delta", {}).get("content") or "" for r in records for c in r.get("choices", []))
    assert text == "中😀"


def test_interleaved_real_calls_are_rejected_not_cross_assigned():
    state = StreamPartialState(TOOL)
    transform_stream_chunk(stream_chunk({"name": "first", "id": "one", "will_continue": True}), 0, state)
    with pytest.raises(GenerationEndError):
        transform_stream_chunk(stream_chunk({"name": "second", "id": "two", "will_continue": True}), 0, state)


def test_synthetic_start_does_not_discard_pending_real_call():
    state = StreamPartialState(TOOL)
    transform_stream_chunk(stream_chunk({"name": "lookup", "will_continue": True}), 0, state)
    with pytest.raises(GenerationEndError):
        transform_stream_chunk(stream_chunk({"name": TOOL, "args": {"content": "text"}}), 0, state)


def test_partial_real_resource_limit(monkeypatch):
    import anti_truncation
    monkeypatch.setattr(anti_truncation, "MAX_ARGS_BYTES", 16)
    state = StreamPartialState(TOOL)
    transform_stream_chunk(stream_chunk({"name": "lookup", "will_continue": True}), 0, state)
    with pytest.raises(GenerationEndError):
        transform_stream_chunk(stream_chunk({"partial_args": [partial("$.city", "中" * 10)]}), 0, state)


async def test_split_surrogate_pair_is_emitted_as_valid_unicode():
    _, records = await stream_output([
        stream_chunk({"name": TOOL, "will_continue": True}),
        stream_chunk({"partial_args": [partial("$.content", "\ud83d")]}),
        stream_chunk({"partial_args": [partial("$.content", "\ude00")]}),
        stream_chunk({"partial_args": [partial("$.content", "", None)]}),
        stream_chunk({}), stream_chunk(finish="STOP")])
    text = "".join(c.get("delta", {}).get("content") or "" for r in records for c in r.get("choices", []))
    assert text == "😀"
    assert not any("error" in r for r in records)


async def test_multiple_candidates_rejected_before_synthetic_leaks():
    first = stream_chunk({"name": TOOL, "args": {"content": "one"}})
    first.candidates.append(types.Candidate(content=types.Content(parts=[
        types.Part(function_call=types.FunctionCall(name=TOOL, args={"content": "two"}))])))
    wire, records = await stream_output([first, stream_chunk(finish="STOP")])
    assert any("error" in r for r in records)
    assert TOOL not in wire


def test_second_closed_synthetic_call_is_not_concatenated():
    state = StreamPartialState(TOOL)
    transform_stream_chunk(stream_chunk({"name": TOOL, "args": {"content": "one"}}), 0, state)
    with pytest.raises(GenerationEndError):
        transform_stream_chunk(stream_chunk({"name": TOOL, "args": {"content": "two"}}), 0, state)


def test_nonstream_multiple_synthetic_calls_fail_instead_of_concatenate():
    value = completion(real=False)
    value["choices"][0]["message"]["tool_calls"] *= 2
    with pytest.raises(GenerationEndError):
        strip_synthetic_from_openai_dict(value, TOOL)


def test_dictionary_synthetic_arguments_obey_byte_limit(monkeypatch):
    import anti_truncation
    monkeypatch.setattr(anti_truncation, "MAX_ARGS_BYTES", 16)
    with pytest.raises(GenerationEndError):
        strip_synthetic_from_openai_dict(completion({"content": "x" * 17}, real=False), TOOL)


def test_sparse_array_expansion_obeys_node_budget():
    state = StreamPartialState(TOOL)
    transform_stream_chunk(stream_chunk({"name": "lookup", "will_continue": True}), 0, state)
    with pytest.raises(GenerationEndError):
        for index in range(80):
            transform_stream_chunk(stream_chunk({"partial_args": [
                partial(f"$.a{index}[1024].b[1024]", "x", False)]}), 0, state)


def test_anonymous_orphan_fragment_is_protocol_error():
    state = StreamPartialState(TOOL)
    with pytest.raises(GenerationEndError):
        transform_stream_chunk(stream_chunk({"partial_args": [partial("$.content", "orphan")]}), 0, state)


def test_conflicting_real_nested_path_does_not_overwrite_parameter():
    state = StreamPartialState(TOOL)
    transform_stream_chunk(stream_chunk({"name": "lookup", "will_continue": True}), 0, state)
    transform_stream_chunk(stream_chunk({"partial_args": [partial("$.city", "Paris", False)]}), 0, state)
    with pytest.raises(GenerationEndError):
        transform_stream_chunk(stream_chunk({"partial_args": [partial("$.city.name", "other")]}), 0, state)


def test_foreign_id_fragment_is_not_assigned_to_current_call():
    state = StreamPartialState(TOOL)
    transform_stream_chunk(stream_chunk({"name": "lookup", "id": "first", "will_continue": True}), 0, state)
    with pytest.raises(GenerationEndError):
        transform_stream_chunk(stream_chunk({"id": "second", "partial_args": [partial("$.city", "Paris")]}), 0, state)


async def test_explicit_function_close_on_last_partial_is_accepted():
    _, records = await stream_output([
        stream_chunk({"name": TOOL, "will_continue": True}),
        stream_chunk({"partial_args": [partial("$.content", "complete", False)], "will_continue": False}),
        stream_chunk(finish="STOP")])
    assert not any("error" in r for r in records)
    assert [c["finish_reason"] for r in records for c in r.get("choices", []) if c.get("finish_reason")] == ["stop"]


@pytest.mark.parametrize("finish, expected", [("MAX_TOKENS", "length"), ("SAFETY", "content_filter")])
async def test_partial_synthetic_preserves_explicit_cut_short_end(finish, expected):
    wire, records = await stream_output([
        stream_chunk({"name": TOOL, "will_continue": True}),
        stream_chunk({"partial_args": [partial("$.content", "partial")]}),
        stream_chunk(finish=finish)])
    assert [c["finish_reason"] for r in records for c in r.get("choices", []) if c.get("finish_reason")] == [expected]
    assert wire.count("data: [DONE]") == 1


@pytest.mark.parametrize("fake", [False, True])
async def test_buffered_invalid_synthetic_is_protocol_failure(fake):
    from api_helpers import execute_gemini_call
    from models import OpenAIRequest
    from types import SimpleNamespace
    async def generate(**kwargs):
        return stream_chunk({"name": TOOL, "args": {"content": 123}}, finish="STOP")
    client = SimpleNamespace(aio=SimpleNamespace(models=SimpleNamespace(generate_content=generate)))
    request = OpenAIRequest(model="gemini-test", messages=[{"role": "user", "content": "fixture"}], stream=fake)
    response = await execute_gemini_call(client, request.model, lambda _: [], {}, request,
                                         synthetic_tool_name=TOOL, force_fake_streaming=fake)
    if fake:
        wire = "".join([item async for item in response.body_iterator])
        assert '"error"' in wire
        assert '"finish_reason": "stop"' not in wire
        assert wire.count("data: [DONE]") == 1
    else:
        assert response.status_code == 502
        assert json.loads(response.body)["error"]["category"] == "empty_or_protocol"


@pytest.mark.parametrize("mode", ["buffered", "fake_stream", "stream"])
@pytest.mark.parametrize("shape,status,reason", [
    ("complete", "restored", "STOP"), ("partial", "partial", "MAX_TOKENS"),
    ("plain", "not_called", "STOP"), ("invalid", "integrity_failed", "STOP"),
])
async def test_p2_public_sdk_fixed_integrity_status(mode, shape, status, reason, monkeypatch):
    import request_log
    from api_helpers import execute_gemini_call
    from models import OpenAIRequest
    from types import SimpleNamespace
    events = []
    original_emit = request_log.RequestLogContext.emit
    def emit(self, event, *args, **fields):
        if event == "anti_truncation_integrity":
            events.append(fields)
        return original_emit(self, event, *args, **fields)
    monkeypatch.setattr(request_log.RequestLogContext, "emit", emit)
    request_log.begin_request("gemini-test", "fixture")
    fc = None if shape == "plain" else {"name": TOOL, "args": {"content": 123 if shape == "invalid" else "private fixture"}}
    chunk = stream_chunk(fc, text="plain" if fc is None else None, finish=reason)
    async def generate(**kwargs):
        return chunk
    async def generate_stream(**kwargs):
        async def source():
            yield chunk
        return source()
    client = SimpleNamespace(aio=SimpleNamespace(models=SimpleNamespace(
        generate_content=generate, generate_content_stream=generate_stream)))
    request = OpenAIRequest(model="gemini-test", messages=[{"role": "user", "content": "fixture"}], stream=mode != "buffered")
    response = await execute_gemini_call(client, request.model, lambda _: [], {}, request,
                                         synthetic_tool_name=TOOL, force_fake_streaming=mode == "fake_stream")
    if mode != "buffered":
        _ = [item async for item in response.body_iterator]
    assert any(e.get("status") == status for e in events)
    assert any(e.get("native_finish_reason") == reason for e in events)
    assert all(e.get("transport") in ("buffered", "fake_stream", "stream") for e in events)
    assert TOOL not in json.dumps(events)
    assert "private fixture" not in json.dumps(events)


@pytest.mark.parametrize("channel", ["express", "vertex"])
@pytest.mark.parametrize("enabled,image,status", [(False, False, "disabled"), (True, True, "skipped")])
async def test_p2_injection_disabled_and_skipped(channel, enabled, image, status, monkeypatch):
    import request_log
    import upstreams.express_sdk as sdk
    from upstreams.service_account import ServiceAccountUpstream
    from models import OpenAIRequest
    from types import SimpleNamespace
    events = []
    original_emit = request_log.RequestLogContext.emit
    def emit(self, event, *args, **fields):
        if event == "anti_truncation_integrity":
            events.append(fields)
        return original_emit(self, event, *args, **fields)
    monkeypatch.setattr(request_log.RequestLogContext, "emit", emit)
    request_log.begin_request("gemini-test", "fixture")
    monkeypatch.setattr(sdk, "is_enabled_for_request", lambda *args: enabled)
    async def generate(**kwargs):
        return stream_chunk(text="ok", finish="STOP")
    client = SimpleNamespace(aio=SimpleNamespace(models=SimpleNamespace(generate_content=generate)))
    upstream = sdk.ExpressSDKUpstream() if channel == "express" else ServiceAccountUpstream()
    monkeypatch.setattr(upstream, "_resolve_client", lambda *args: {"client": client, "model_to_call": "gemini-3.8-flash", "priority_paygo": False})
    request = OpenAIRequest(model="gemini-3-pro-image-preview" if image else "gemini-3.8-flash",
                            messages=[{"role": "user", "content": "fixture"}])
    response = await upstream.chat_completions(request, None)
    assert response.status_code == 200
    assert any(e.get("status") == status and e.get("transport") == "buffered" for e in events)


async def test_seeded_retry_resets_synthetic_and_real_first_fragment(monkeypatch):
    import random
    import api_helpers as helpers
    from models import OpenAIRequest
    from types import SimpleNamespace
    from unittest.mock import Mock
    monkeypatch.setattr(helpers, "get_retry_settings", lambda *_: (2, 0))
    monkeypatch.setattr(helpers, "stats", Mock())
    rng = random.Random(20261009)
    text = "unique deterministic answer"
    cuts = sorted(rng.sample(range(1, len(text)), 5))
    fragments = [text[left:right] for left, right in zip([0, *cuts], [*cuts, len(text)])]
    calls = []
    closed = []
    async def generate(**kwargs):
        attempt = len(calls)
        calls.append(attempt)
        async def source():
            try:
                if attempt == 0:
                    yield stream_chunk({"name": TOOL, "id": "failed-synthetic", "will_continue": True})
                    raise RuntimeError("429 fixture")
                if attempt == 1:
                    yield stream_chunk({"name": "lookup", "id": "failed-real", "will_continue": True})
                    raise RuntimeError("429 fixture")
                yield stream_chunk({"name": TOOL, "id": "successful", "will_continue": True})
                for index, fragment in enumerate(fragments):
                    last = index == len(fragments) - 1
                    yield stream_chunk({"id": "successful", "partial_args": [partial("$.content", fragment, not last)],
                                        "will_continue": not last})
                yield stream_chunk(finish="STOP")
            finally:
                closed.append(attempt)
        return source()
    client = SimpleNamespace(aio=SimpleNamespace(models=SimpleNamespace(generate_content_stream=generate)))
    request = OpenAIRequest(model="gemini-test", messages=[{"role": "user", "content": "fixture"}], stream=True)
    response = await helpers.execute_gemini_call(client, request.model, lambda _: [], {}, request, synthetic_tool_name=TOOL)
    wire = "".join([item async for item in response.body_iterator])
    records = [json.loads(line[6:]) for line in wire.splitlines() if line.startswith("data: {")]
    choices = [c for r in records for c in r.get("choices", [])]
    assert calls == [0, 1, 2]
    assert closed == [0, 1, 2]
    assert "".join(c.get("delta", {}).get("content") or "" for c in choices) == text
    assert not any("error" in r for r in records)
    assert not any(c.get("delta", {}).get("tool_calls") for c in choices)
    assert [c["finish_reason"] for c in choices if c.get("finish_reason")] == ["stop"]
    assert wire.count("data: [DONE]") == 1
    assert "failed-synthetic" not in wire and "failed-real" not in wire


def test_p2_logging_failure_does_not_affect_restoration(monkeypatch):
    import request_log
    from types import SimpleNamespace
    def emit(*args, **kwargs):
        raise RuntimeError("fixture logging failure")
    monkeypatch.setattr(request_log, "current", lambda: SimpleNamespace(emit=emit))
    value = strip_synthetic_from_openai_dict(completion(real=False), TOOL)
    assert value["choices"][0]["message"]["content"] == "answer"


@pytest.mark.parametrize("native_reason", ["private fixture text", "PRIVATE_FIXTURE_TOKEN_123"])
def test_p2_native_reason_never_logs_arbitrary_text(monkeypatch, native_reason):
    import request_log
    from anti_truncation import log_integrity
    from types import SimpleNamespace
    events = []
    monkeypatch.setattr(request_log, "current", lambda: SimpleNamespace(emit=lambda *args, **fields: events.append(fields)))
    log_integrity(status="restored", transport="buffered", native_finish_reason=native_reason, arguments="private")
    assert "native_finish_reason" not in events[0]
    assert "arguments" not in events[0]
    assert "private" not in json.dumps(events)


@pytest.mark.parametrize("reason", [
    "STOP", "MAX_TOKENS", "SAFETY", "RECITATION", "BLOCKLIST",
    "PROHIBITED_CONTENT", "SPII", "IMAGE_SAFETY", "IMAGE_PROHIBITED_CONTENT",
    "IMAGE_RECITATION", "ESCALATION", "PUP_LIMITED_DISABLED",
    "MALFORMED_FUNCTION_CALL", "UNEXPECTED_TOOL_CALL", "TOO_MANY_TOOL_CALLS",
    "MISSING_THOUGHT_SIGNATURE", "MALFORMED_RESPONSE", "IMAGE_OTHER", "NO_IMAGE",
    "OTHER", "UNSPECIFIED", "FINISH_REASON_UNSPECIFIED",
])
def test_p2_known_native_finish_reasons_are_preserved(reason, monkeypatch):
    import request_log
    from anti_truncation import log_integrity
    from types import SimpleNamespace
    events = []
    monkeypatch.setattr(request_log, "current", lambda: SimpleNamespace(emit=lambda *args, **fields: events.append(fields)))
    log_integrity(status="restored", transport="buffered", native_finish_reason=reason)
    assert events[0]["native_finish_reason"] == reason


def test_integrity_logging_contains_no_content_or_arguments(monkeypatch):
    import request_log
    from types import SimpleNamespace
    events = []
    monkeypatch.setattr(request_log, "current", lambda: SimpleNamespace(
        emit=lambda *args, **fields: events.append(fields)))
    strip_synthetic_from_openai_dict(completion('{"content":"private fixture text"}', real=False), TOOL)
    assert events[0]["structurally_complete"] is True
    assert "private fixture text" not in json.dumps(events)
    assert TOOL not in json.dumps(events)


@pytest.mark.parametrize("fake", [False, True])
@pytest.mark.parametrize("cancel", [False, True])
async def test_p2_cancelled_integrity_diagnostic(fake, cancel, monkeypatch):
    import asyncio
    import request_log
    from api_helpers import execute_gemini_call
    from models import OpenAIRequest
    from types import SimpleNamespace
    events = []
    original_emit = request_log.RequestLogContext.emit
    def emit(self, event, *args, **fields):
        if event == "anti_truncation_integrity":
            events.append(fields)
        return original_emit(self, event, *args, **fields)
    monkeypatch.setattr(request_log.RequestLogContext, "emit", emit)
    request_log.begin_request("gemini-test", "fixture")
    waiting = asyncio.Event()
    closed = []
    async def generate(**kwargs):
        try:
            waiting.set()
            await asyncio.Event().wait()
        finally:
            closed.append(True)
    async def source():
        try:
            yield stream_chunk(text="first")
            waiting.set()
            await asyncio.Event().wait()
        finally:
            closed.append(True)
    async def generate_stream(**kwargs):
        return source()
    client = SimpleNamespace(aio=SimpleNamespace(models=SimpleNamespace(
        generate_content=generate, generate_content_stream=generate_stream)))
    request = OpenAIRequest(model="gemini-test", messages=[{"role": "user", "content": "fixture"}], stream=True)
    response = await execute_gemini_call(client, request.model, lambda _: [], {}, request,
                                         synthetic_tool_name=TOOL, force_fake_streaming=fake)
    iterator = response.body_iterator
    await anext(iterator)
    if cancel:
        async def consume():
            async for _ in iterator:
                pass
        task = asyncio.create_task(consume())
        await asyncio.wait_for(waiting.wait(), 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    else:
        if not fake:
            while '"first"' not in await anext(iterator):
                pass
        await iterator.aclose()
    assert any(e.get("status") == "cancelled" and e.get("failure_category") == "client_closed"
               and e.get("stream_done") is False for e in events)
    if not fake or cancel:
        assert closed == [True]


async def test_client_close_closes_upstream_without_restarting():
    from api_helpers import execute_gemini_call
    from models import OpenAIRequest
    from types import SimpleNamespace
    closed = []
    calls = []
    async def source():
        try:
            yield stream_chunk(text="first")
            yield stream_chunk(text="second")
        finally:
            closed.append(True)
    async def generate(**kwargs):
        calls.append(True)
        return source()
    client = SimpleNamespace(aio=SimpleNamespace(models=SimpleNamespace(generate_content_stream=generate)))
    request = OpenAIRequest(model="gemini-test", messages=[{"role": "user", "content": "fixture"}], stream=True)
    response = await execute_gemini_call(client, request.model, lambda _: [], {}, request, synthetic_tool_name=TOOL)
    iterator = response.body_iterator
    while True:
        item = await anext(iterator)
        if '"first"' in item:
            break
    await iterator.aclose()
    assert closed == [True]
    assert calls == [True]


@pytest.mark.parametrize("message", ["429 fixture", "503 fixture", "network disconnected"])
async def test_exception_after_output_is_explicit_error_without_retry(message):
    from api_helpers import execute_gemini_call
    from models import OpenAIRequest
    from types import SimpleNamespace
    calls = []
    async def source():
        yield stream_chunk(text="partial")
        raise RuntimeError(message)
    async def generate(**kwargs):
        calls.append(True)
        return source()
    client = SimpleNamespace(aio=SimpleNamespace(models=SimpleNamespace(generate_content_stream=generate)))
    request = OpenAIRequest(model="gemini-test", messages=[{"role": "user", "content": "fixture"}], stream=True)
    response = await execute_gemini_call(client, request.model, lambda _: [], {}, request, synthetic_tool_name=TOOL)
    wire = "".join([item async for item in response.body_iterator])
    assert '"error"' in wire
    assert wire.count("data: [DONE]") == 1
    assert calls == [True]


def test_mixed_tool_topology_keeps_original_real_order_and_signature():
    value = completion()
    message = value["choices"][0]["message"]
    message["tool_calls"][1]["extra_content"] = {"google": {"thought_signature": "fixture-signature"}}
    message["extra_content"] = {"google": {"part_order": [
        {"type": "tool_call", "index": 0}, {"type": "tool_call", "index": 1}]}}
    result = strip_synthetic_from_openai_dict(value, TOOL)
    message = result["choices"][0]["message"]
    assert message["extra_content"]["google"]["part_order"] == [{"type": "tool_call", "index": 0}]
    assert message["tool_calls"][0]["extra_content"]["google"]["thought_signature"] == "fixture-signature"


async def test_synthetic_text_arrives_before_upstream_terminal():
    from api_helpers import execute_gemini_call
    from models import OpenAIRequest
    from types import SimpleNamespace
    import asyncio
    release = asyncio.Event()
    async def source():
        yield stream_chunk({"name": TOOL, "will_continue": True})
        yield stream_chunk({"partial_args": [partial("$.content", "first") ]})
        await release.wait()
        yield stream_chunk({"partial_args": [partial("$.content", "", None)]})
        yield stream_chunk({})
        yield stream_chunk(finish="STOP")
    async def generate(**kwargs):
        return source()
    client = SimpleNamespace(aio=SimpleNamespace(models=SimpleNamespace(generate_content_stream=generate)))
    request = OpenAIRequest(model="gemini-test", messages=[{"role": "user", "content": "fixture"}], stream=True)
    response = await execute_gemini_call(client, request.model, lambda _: [], {}, request, synthetic_tool_name=TOOL)
    iterator = response.body_iterator
    async def first_text():
        while True:
            line = await anext(iterator)
            if '"first"' in line:
                return line
    try:
        line = await asyncio.wait_for(first_text(), 2)
        assert '"finish_reason": null' in line
        assert not release.is_set()
        release.set()
        rest = "".join([line async for line in iterator])
        assert '"finish_reason": "stop"' in rest
        assert rest.count("data: [DONE]") == 1
    finally:
        await iterator.aclose()
