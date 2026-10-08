"""假流式生命周期诊断日志 + 逻辑请求关联（事件契约 v1）。

全部使用 mock 上游；等待节奏靠把间隔 monkeypatch 成几十毫秒或注入时钟/调度器，
绝不真实等待 30/60 秒。事件通过 monkeypatch rt_logger.push 收集。
"""
import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.testclient import TestClient

import api_helpers as helpers
import config as app_config
import logger as logger_mod
import request_log
from failover import ChannelBreaker, UpstreamUnstartedError
from routes import chat_api
from runtime_state import app_state
from test_stream_usage import client_for, consume, response
from test_stream_usage import request as make_request
from test_usage_mapping import META, sdk_meta

MODEL = "gemini-3.6-flash"


def is_heartbeat(p):
    return p.get("choices") == [{"delta": {"content": ""}, "index": 0, "finish_reason": None}]


@pytest.fixture(autouse=True)
def isolate(monkeypatch):
    stats = Mock()
    monkeypatch.setattr(helpers, "stats", stats)
    monkeypatch.setattr(helpers, "get_retry_settings", lambda *args: (0, 0))
    monkeypatch.setattr(helpers.app_state, "get_effective_settings", lambda *args: {})
    monkeypatch.setattr(helpers.app_state, "get_setting",
                        lambda key, default=None: 0.01 if key == "fake_streaming_interval" else default)
    logger_mod.REQUEST_CTX.set(None)
    yield stats
    logger_mod.REQUEST_CTX.set(None)


@pytest.fixture
def events(monkeypatch):
    got = []
    monkeypatch.setattr(logger_mod.rt_logger, "push", lambda record: got.append(record))
    return got


def of_type(events, event_type, request_id=None):
    return [e for e in events if e["event_type"] == event_type
            and (request_id is None or e.get("request_id") == request_id)]


def fast_timers(monkeypatch, interval=0.02, slow=0.07, keepalive=0.01):
    monkeypatch.setattr(request_log, "WAIT_PROGRESS_INTERVAL_SECONDS", interval)
    monkeypatch.setattr(request_log, "SLOW_REQUEST_WARN_SECONDS", slow)
    monkeypatch.setattr(helpers.app_state, "get_setting",
                        lambda key, default=None: keepalive if key == "fake_streaming_interval" else default)


async def start_fake(client, *, include_usage=True, failover_mode=False, channel="express"):
    ctx = request_log.begin_request(MODEL, "fake_stream", "express")
    resp = await helpers.execute_gemini_call(
        client, MODEL, lambda _: [], {}, make_request(include_usage), channel_name=channel,
        force_fake_streaming=True, failover_mode=failover_mode)
    return ctx, request_log.track_response(ctx, resp)


async def pump(resp, lines):
    async for line in resp.body_iterator:
        lines.append(line)


def pending_tasks():
    return [t for t in asyncio.all_tasks() if t is not asyncio.current_task() and not t.done()]


# ------------------------------------------------------------------ 验收 1：一直 pending

async def test_pending_upstream_reports_waiting_then_slow_once(monkeypatch, events):
    fast_timers(monkeypatch)
    release = asyncio.Event()

    async def slow(**kwargs):
        await release.wait()
        return response(usage=sdk_meta(META))
    client = client_for()
    client.aio.models.generate_content.side_effect = slow
    ctx, resp = await start_fake(client)
    lines = []
    task = asyncio.create_task(pump(resp, lines))
    await asyncio.sleep(0.35)
    before_release = list(lines)
    release.set()
    await task

    rid = ctx.request_id
    kinds = [e["event_type"] for e in events if e.get("request_id") == rid]
    waiting = of_type(events, "upstream_waiting", rid)
    slow_events = of_type(events, "upstream_slow", rid)
    assert waiting and len(slow_events) == 1, "慢告警只发一次，不重复"
    assert slow_events[0]["level"] == "WARN" and slow_events[0]["slow"] is True
    assert all(e["level"] == "INFO" for e in waiting)
    assert kinds.index("upstream_waiting") < kinds.index("upstream_slow")
    assert "等待上游完整响应；心跳不代表生成进度" in waiting[0]["message"]
    assert "等待上游完整响应；心跳不代表生成进度" in slow_events[0]["message"]
    # 等待期间只有原样的空 content 心跳，没有任何正文
    payloads = [json.loads(line[6:]) for line in before_release if line.startswith("data: ")]
    assert payloads and all(is_heartbeat(p) for p in payloads)
    end = of_type(events, "request_end", rid)
    assert len(end) == 1 and end[0]["status"] == "success" and end[0]["slow"] is True


# ------------------------------------------------------------------ 验收 2：重试后成功

async def test_retry_then_success_keeps_request_id_and_single_end(monkeypatch, events):
    monkeypatch.setattr(helpers, "get_retry_settings", lambda *args: (1, 0.01))
    monkeypatch.setattr(helpers, "is_retryable_exception", lambda e: True)
    fast_timers(monkeypatch, interval=5, slow=50)
    client = client_for()
    client.aio.models.generate_content.side_effect = [ValueError("429"), response(usage=sdk_meta(META))]
    lines, end_snapshots = [], []

    def push(record):
        events.append(record)
        if record["event_type"] == "request_end":
            end_snapshots.append(list(lines))
    monkeypatch.setattr(logger_mod.rt_logger, "push", push)

    ctx, resp = await start_fake(client)
    await pump(resp, lines)

    rid = ctx.request_id
    mine = [e for e in events if e.get("request_id") == rid]
    attempts = of_type(events, "attempt_start", rid)
    assert [a["attempt"] for a in attempts] == [1, 2]
    assert all(a["max_attempts"] == 2 for a in attempts)
    failed = of_type(events, "attempt_failed", rid)
    assert len(failed) == 1 and failed[0]["retryable"] is True and failed[0]["next_attempt"] == 2
    assert failed[0]["level"] == "WARN" and failed[0]["error_type"] == "ValueError"
    elapsed = [e["elapsed_ms"] for e in mine if "elapsed_ms" in e]
    assert elapsed == sorted(elapsed), "累计耗时单调，不会在重试时清零"
    end = of_type(events, "request_end", rid)
    assert len(end) == 1 and end[0]["status"] == "success" and end[0]["retries"] == 1
    # 成功只在最后一个 SSE 块（[DONE]）已被消费之后才出现
    assert end_snapshots[0][-1] == "data: [DONE]\n\n"
    assert end_snapshots[0] == lines
    assert end[0] is mine[-1]
    assert end[0]["usage"]["total_tokens"] == 150
    received = of_type(events, "upstream_received", rid)
    assert len(received) == 1 and received[0]["attempt"] == 2
    assert received[0]["usage"]["prompt_tokens"] == 100


async def test_received_event_usage_unknown_without_usage_metadata(events):
    client = client_for(full=response(usage=None))
    ctx, resp = await start_fake(client)
    await consume(resp)
    received = of_type(events, "upstream_received", ctx.request_id)
    assert received[0]["usage"] == "unknown"
    assert of_type(events, "request_end", ctx.request_id)[0]["usage"] == "unknown"


async def test_failed_retries_exhausted_end_failed(monkeypatch, events):
    client = client_for()
    client.aio.models.generate_content.side_effect = ValueError("boom key=AIzaSyFAKEFAKEFAKEFAKEFAKEFAKEFAKEFAKE1")
    ctx, resp = await start_fake(client)
    lines, _ = await consume(resp)
    assert lines[-1] == "data: [DONE]\n\n"
    failed = of_type(events, "attempt_failed", ctx.request_id)
    assert len(failed) == 1 and failed[0]["level"] == "ERROR" and failed[0]["retryable"] is False
    end = of_type(events, "request_end", ctx.request_id)
    assert len(end) == 1 and end[0]["status"] == "failed" and end[0]["level"] == "ERROR"
    assert end[0]["error_type"] == "ValueError" and end[0]["phase"] == "waiting_upstream"
    assert "AIzaSy" not in json.dumps(end[0]) and "AIzaSy" not in json.dumps(failed[0])


# ------------------------------------------------------------------ 验收 3：取消

def hanging_client():
    state = {"cancelled": False, "started": asyncio.Event()}

    async def hang(**kwargs):
        state["started"].set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            state["cancelled"] = True
            raise
    client = client_for()
    client.aio.models.generate_content.side_effect = hang
    return client, state


async def settle():
    for _ in range(5):
        await asyncio.sleep(0)


def assert_cancelled_while_waiting(events, ctx, state):
    end = of_type(events, "request_end", ctx.request_id)
    assert len(end) == 1
    assert end[0]["status"] == "cancelled" and end[0]["level"] == "WARN"
    assert end[0]["phase"] == "waiting_upstream"
    assert end[0]["upstream_task_done"] is False and end[0]["cancel_requested"] is True
    assert "连接断开或任务取消" in end[0]["message"]
    assert state["cancelled"], "mock 上游协程必须被取消"
    assert pending_tasks() == []
    assert not of_type(events, "request_end", ctx.request_id)[0]["status"] == "success"


async def test_task_cancel_while_waiting_cancels_upstream(monkeypatch, events):
    fast_timers(monkeypatch, interval=5, slow=50)
    client, state = hanging_client()
    ctx, resp = await start_fake(client)
    task = asyncio.create_task(pump(resp, []))
    await state["started"].wait()
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await settle()
    assert_cancelled_while_waiting(events, ctx, state)


async def test_aclose_while_waiting_cancels_upstream(monkeypatch, events):
    fast_timers(monkeypatch, interval=5, slow=50)
    client, state = hanging_client()
    ctx, resp = await start_fake(client)
    iterator = resp.body_iterator
    first = await iterator.__anext__()
    assert is_heartbeat(json.loads(first[6:]))
    await state["started"].wait()
    await iterator.aclose()
    await settle()
    assert_cancelled_while_waiting(events, ctx, state)


async def test_aclose_while_sending_records_sending_phase(monkeypatch, events):
    fast_timers(monkeypatch, interval=5, slow=50)
    client = client_for(full=response(usage=sdk_meta(META)))
    ctx, resp = await start_fake(client)
    iterator = resp.body_iterator
    while True:
        line = await iterator.__anext__()
        if not is_heartbeat(json.loads(line[6:])):
            break
    await iterator.aclose()
    await settle()
    end = of_type(events, "request_end", ctx.request_id)
    assert len(end) == 1 and end[0]["status"] == "cancelled"
    assert end[0]["phase"] == "sending_downstream"
    assert end[0]["upstream_task_done"] is True and end[0]["cancel_requested"] is False
    assert not [e for e in events if e.get("status") == "success"]
    assert pending_tasks() == []


async def test_client_disconnect_check_marks_cancelled(monkeypatch, events):
    fast_timers(monkeypatch, interval=5, slow=50)
    client, state = hanging_client()
    disconnected = {"v": False}

    async def is_disconnected():
        return disconnected["v"]
    fastapi_request = SimpleNamespace(is_disconnected=is_disconnected)
    ctx = request_log.begin_request(MODEL, "fake_stream", "express")
    resp = await helpers.execute_gemini_call(
        client, MODEL, lambda _: [], {}, make_request(True), channel_name="express",
        force_fake_streaming=True, fastapi_request=fastapi_request)
    resp = request_log.track_response(ctx, resp)
    iterator = resp.body_iterator
    await iterator.__anext__()
    disconnected["v"] = True
    with pytest.raises(StopAsyncIteration):
        while True:
            await iterator.__anext__()
    await settle()
    assert_cancelled_while_waiting(events, ctx, state)


# ------------------------------------------------------------------ 验收 4：hybrid 故障转移

async def test_hybrid_failover_keeps_one_request_id_and_one_end(monkeypatch, events):
    monkeypatch.setattr(chat_api, "breaker", ChannelBreaker())
    monkeypatch.setattr(chat_api, "_available_channels", lambda order: order)
    monkeypatch.setattr(chat_api, "_channel_order", lambda strategy: ["express", "vertex"])

    class Upstream:
        def __init__(self, client, channel):
            self.client, self.channel, self.calls = client, channel, 0

        async def chat_completions(self, request, fastapi_request, failover_mode=False):
            self.calls += 1
            return await helpers.execute_gemini_call(
                self.client, MODEL, lambda _: [], {}, request, channel_name=self.channel,
                force_fake_streaming=True, failover_mode=failover_mode)

    bad = client_for()
    bad.aio.models.generate_content.side_effect = ValueError("503 overloaded Bearer sk-secret")
    good = client_for(full=response(usage=sdk_meta(META)))
    first, second = Upstream(bad, "express"), Upstream(good, "vertex")
    monkeypatch.setattr(chat_api, "CHANNELS", {"express": first, "vertex": second})

    resp = await chat_api._chat_completions_with_strategy(
        None, make_request(True), "hybrid")
    lines, payloads = await consume(resp)

    assert first.calls == 1 and second.calls == 1
    assert lines[-1] == "data: [DONE]\n\n"
    assert "hello" in json.dumps(payloads, ensure_ascii=False)
    starts = of_type(events, "request_start")
    assert len(starts) == 1
    rid = starts[0]["request_id"]
    switches = of_type(events, "channel_switch", rid)
    assert len(switches) == 1
    assert switches[0]["source_channel"] == "express" and switches[0]["target_channel"] == "vertex"
    assert switches[0]["level"] == "WARN"
    assert "Bearer sk-secret" not in json.dumps(switches[0])
    assert "故障转移" in switches[0]["message"] and "切换至" in switches[0]["message"]
    ends = of_type(events, "request_end")
    assert len(ends) == 1 and ends[0]["request_id"] == rid
    assert ends[0]["status"] == "success" and ends[0]["channel"] == "vertex"
    assert ends[0]["switches"] == 1
    # 第一通道的失败只是 attempt_failed，没有自己的终态
    assert of_type(events, "attempt_failed", rid)
    assert {e["request_id"] for e in events if e.get("request_id")} == {rid}


async def test_hybrid_failover_final_failure_is_failed(monkeypatch, events):
    monkeypatch.setattr(chat_api, "breaker", ChannelBreaker())
    monkeypatch.setattr(chat_api, "_available_channels", lambda order: order)
    monkeypatch.setattr(chat_api, "_channel_order", lambda strategy: ["express", "vertex"])

    class Upstream:
        def __init__(self, channel):
            self.channel = channel
            self.client = client_for()
            self.client.aio.models.generate_content.side_effect = ValueError("503 overloaded")

        async def chat_completions(self, request, fastapi_request, failover_mode=False):
            return await helpers.execute_gemini_call(
                self.client, MODEL, lambda _: [], {}, request, channel_name=self.channel,
                force_fake_streaming=True, failover_mode=failover_mode)

    monkeypatch.setattr(chat_api, "CHANNELS", {"express": Upstream("express"), "vertex": Upstream("vertex")})
    resp = await chat_api._chat_completions_with_strategy(None, make_request(True), "hybrid")
    lines, _ = await consume(resp)
    assert lines[-1] == "data: [DONE]\n\n"
    ends = of_type(events, "request_end")
    assert len(ends) == 1 and ends[0]["status"] == "failed" and ends[0]["switches"] == 1


# ------------------------------------------------------------------ 其它终态路径

async def test_json_response_end_states(events):
    ok = request_log.begin_request(MODEL, "non_stream", "express")
    request_log.track_response(ok, JSONResponse(content={"ok": 1}))
    bad = request_log.begin_request(MODEL, "non_stream", "express")
    request_log.track_response(bad, JSONResponse(
        status_code=502, content={"error": {"message": "bad Bearer sk-secret\nsecond line"}}))
    request_log.finish(bad, "success")      # 幂等：不会再发第二个终态
    e_ok = of_type(events, "request_end", ok.request_id)
    e_bad = of_type(events, "request_end", bad.request_id)
    assert len(e_ok) == 1 and e_ok[0]["status"] == "success" and e_ok[0]["http_status"] == 200
    assert len(e_bad) == 1 and e_bad[0]["status"] == "failed" and e_bad[0]["http_status"] == 502
    assert "sk-secret" not in e_bad[0]["error_summary"] and "\n" not in e_bad[0]["error_summary"]


async def test_stream_without_diagnosis_ends_unknown(events):
    ctx = request_log.begin_request(MODEL, "stream", "cookie")

    async def plain():
        yield "data: x\n\n"
    resp = request_log.track_response(ctx, StreamingResponse(plain()))
    assert [line async for line in resp.body_iterator] == ["data: x\n\n"]
    end = of_type(events, "request_end", ctx.request_id)
    assert len(end) == 1 and end[0]["status"] == "unknown" and end[0]["level"] == "INFO"


async def test_stream_exception_ends_failed_and_propagates(events):
    ctx = request_log.begin_request(MODEL, "stream", "cookie")

    async def boom():
        yield "data: x\n\n"
        raise RuntimeError("kaput")
    resp = request_log.track_response(ctx, StreamingResponse(boom()))
    with pytest.raises(RuntimeError):
        async for _ in resp.body_iterator:
            pass
    end = of_type(events, "request_end", ctx.request_id)
    assert len(end) == 1 and end[0]["status"] == "failed" and end[0]["error_type"] == "RuntimeError"


# ------------------------------------------------------------------ 验收 5：并发与脱敏

async def test_concurrent_requests_do_not_mix_request_ids(events):
    secret_key = "AIzaSyFAKEFAKEFAKEFAKEFAKEFAKEFAKEFAKE1"

    async def one(tag, fail):
        ctx = request_log.begin_request(MODEL, "fake_stream", "express")
        print(f"普通日志 {tag} 开始")
        client = client_for(full=response(
            [helpers.types.Part(function_call=helpers.types.FunctionCall(
                name="weather", args={"city": "TOOLARG_SHANGHAI"}))], sdk_meta(META)))
        if fail:
            client.aio.models.generate_content.side_effect = ValueError(
                f"boom {secret_key} Authorization: Bearer sk-secret")
        else:
            client = client_for(full=response([helpers.types.Part(text="GENERATED_BODY_TEXT")],
                                              sdk_meta(META)))
        req = make_request(True)
        req.messages[0].content = "PROMPT_SECRET_TEXT"
        resp = await helpers.execute_gemini_call(
            client, MODEL, lambda _: [], {}, req, channel_name="express", force_fake_streaming=True)
        resp = request_log.track_response(ctx, resp)
        lines = []
        async for line in resp.body_iterator:
            lines.append(line)
            await asyncio.sleep(0)
        print(f"普通日志 {tag} 结束")
        return ctx, lines

    (ctx_a, lines_a), (ctx_b, lines_b) = await asyncio.gather(one("A", False), one("B", True))
    assert ctx_a.request_id != ctx_b.request_id
    assert "GENERATED_BODY_TEXT" in "".join(lines_a)

    for tag, ctx in (("A", ctx_a), ("B", ctx_b)):
        logs = [e for e in events if e["event_type"] == "log" and f"普通日志 {tag}" in e["message"]]
        assert len(logs) == 2 and all(e["request_id"] == ctx.request_id for e in logs)
        assert len(of_type(events, "request_end", ctx.request_id)) == 1
        assert len(of_type(events, "fake_stream_start", ctx.request_id)) == 1
    assert of_type(events, "request_end", ctx_a.request_id)[0]["status"] == "success"
    assert of_type(events, "request_end", ctx_b.request_id)[0]["status"] == "failed"

    structured = [e for e in events if e["event_type"] != "log"]
    dump = json.dumps(structured, ensure_ascii=False)
    for forbidden in (secret_key, "Bearer sk-secret", "PROMPT_SECRET_TEXT",
                      "GENERATED_BODY_TEXT", "TOOLARG_SHANGHAI"):
        assert forbidden not in dump
    # 没有 request_id 的普通 print 保持未分组
    print("未分组普通日志")
    assert [e for e in events if "未分组普通日志" in e["message"]][0].get("request_id") is None


def test_redact_patterns_and_shape():
    raw = ('Bearer abc.def key=AIzaSyFAKEFAKEFAKEFAKEFAKEFAKEFAKEFAKE1 ya29.A0-bcd SAPISID=xyz; '
           '__Secure-1PSID=qq "private_key": "-----BEGIN-----\\nabc"\nnext line ' + "x" * 300)
    out = request_log.redact(raw)
    for secret in ("abc.def", "AIzaSy", "ya29.A0", "SAPISID=xyz", "__Secure-1PSID=qq", "BEGIN"):
        assert secret not in out
    assert "\n" not in out and len(out) <= 201 and out.endswith("…")


# ------------------------------------------------------------------ 传输说明

@pytest.mark.parametrize("headers,tier", [
    ({}, "off"),
    ({"X-Vertex-AI-LLM-Request-Type": "shared"}, "standard"),
    ({"X-Vertex-AI-LLM-Request-Type": "shared", "X-Vertex-AI-LLM-Shared-Request-Type": "flex"}, "flex"),
    ({"X-Vertex-AI-LLM-Request-Type": "shared", "X-Vertex-AI-LLM-Shared-Request-Type": "priority"}, "priority"),
])
def test_note_transport_records_paygo_header_tier(headers, tier):
    ctx = request_log.begin_request(MODEL, "fake_stream", "express")
    request_log.note_transport(headers, 1800 if tier == "flex" else None)
    assert ctx.paygo_tier == tier
    assert ctx.transport_timeout_note


def test_note_transport_without_context_is_silent():
    logger_mod.REQUEST_CTX.set(None)
    request_log.note_transport({"X": "y"}, 5)


def test_describe_transport_timeout_branches(monkeypatch):
    import http_options
    assert "未覆盖" in http_options.describe_transport_timeout(None)
    for proxy in ("", "http://127.0.0.1:7890"):
        monkeypatch.setattr(app_config, "PROXY_URL", proxy)
        monkeypatch.setattr(app_config, "SSL_CERT_FILE", "")
        text = http_options.describe_transport_timeout(1800)
        assert "1800 秒" in text and "1800000 毫秒" in text


async def test_fake_stream_start_event_describes_attempts_and_transport(events):
    ctx = request_log.begin_request(MODEL, "fake_stream", "express")
    request_log.note_transport({"X-Vertex-AI-LLM-Request-Type": "shared",
                                "X-Vertex-AI-LLM-Shared-Request-Type": "priority"}, None)
    resp = await helpers.execute_gemini_call(
        client_for(full=response()), MODEL, lambda _: [], {}, make_request(True),
        channel_name="vertex", force_fake_streaming=True)
    await consume(resp)
    start = of_type(events, "fake_stream_start", ctx.request_id)[0]
    assert start["message"].startswith("🌊 [假流式] 已开始通过")
    assert "应用层最多 1 次尝试（不等于 SDK 内部 HTTP 次数）" in start["message"]
    assert start["max_attempts"] == 1 and start["paygo_tier"] == "priority"
    assert start["phase"] == "preparing" and start["mode"] == "fake_stream"
    assert "keepalive_interval_seconds" in start and "transport_timeout_note" in start


async def test_generator_called_directly_owns_and_finishes_its_request(events):
    resp = await helpers.execute_gemini_call(
        client_for(full=response()), MODEL, lambda _: [], {}, make_request(True),
        channel_name="express", force_fake_streaming=True)
    await consume(resp)
    assert len(of_type(events, "request_start")) == 1
    end = of_type(events, "request_end")
    assert len(end) == 1 and end[0]["status"] == "success"
    assert request_log.current() is None


# ------------------------------------------------------------------ 验收 7：观察器单元

class FakeHandle:
    def __init__(self, delay, callback):
        self.delay, self.callback, self.cancelled = delay, callback, False

    def cancel(self):
        self.cancelled = True


class FakeScheduler:
    def __init__(self):
        self.handles = []

    def __call__(self, delay, callback):
        handle = FakeHandle(delay, callback)
        self.handles.append(handle)
        return handle

    def fire_last(self):
        handle = self.handles[-1]
        assert not handle.cancelled
        handle.callback()


def make_observer(monkeypatch, events):
    clock = {"t": 0.0}
    monkeypatch.setattr(request_log, "_monotonic", lambda: clock["t"])
    ctx = request_log.begin_request(MODEL, "fake_stream", "express")
    loop = asyncio.get_running_loop()
    task = loop.create_future()
    scheduler = FakeScheduler()
    observer = request_log.UpstreamWaitObserver(
        ctx, task, 1, 2, clock=lambda: clock["t"], scheduler=scheduler).start()
    return clock, ctx, task, scheduler, observer


async def test_observer_cadence_30_60_90(monkeypatch, events):
    clock, ctx, task, scheduler, observer = make_observer(monkeypatch, events)
    assert len(scheduler.handles) == 1 and scheduler.handles[0].delay == 30
    kinds = []
    for t in (30, 60, 90):
        clock["t"] = float(t)
        scheduler.fire_last()
        kinds.append((events[-1]["event_type"], events[-1]["level"], events[-1]["attempt_elapsed_ms"]))
    assert kinds == [("upstream_waiting", "INFO", 30000),
                     ("upstream_slow", "WARN", 60000),
                     ("upstream_waiting", "INFO", 90000)]
    assert ctx.slow is True
    assert len(scheduler.handles) == 4, "每次回调结束后注册下一次，不创建 Task"
    assert "累计 90.0s" in events[-1]["message"] and "本轮已等待 90.0s" in events[-1]["message"]
    observer.close()


async def test_observer_done_cancels_timer_and_emits_received_once(monkeypatch, events):
    clock, ctx, task, scheduler, observer = make_observer(monkeypatch, events)
    clock["t"] = 12.0
    task.set_result(response(usage=sdk_meta(META)))
    await asyncio.sleep(0)
    received = [e for e in events if e["event_type"] == "upstream_received"]
    assert len(received) == 1 and received[0]["attempt_elapsed_ms"] == 12000
    assert received[0]["finish_reason"] == "STOP" and received[0]["usage"]["total_tokens"] == 150
    assert scheduler.handles[0].cancelled
    before = len(events)
    scheduler.handles[0].callback()          # 定时回调即使被触发，任务已完成也不再汇报
    observer.close()
    assert len(events) == before
    assert observer.wait_ms() == 12000


async def test_observer_failure_only_records_completion_time(monkeypatch, events):
    clock, ctx, task, scheduler, observer = make_observer(monkeypatch, events)
    clock["t"] = 5.0
    task.set_exception(ValueError("x"))
    await asyncio.sleep(0)
    assert not [e for e in events if e["event_type"] == "upstream_received"]
    assert observer.wait_ms() == 5000
    task.exception()


async def test_observer_close_cancels_handles_and_callback(monkeypatch, events):
    clock, ctx, task, scheduler, observer = make_observer(monkeypatch, events)
    clock["t"] = 30.0
    scheduler.fire_last()
    observer.close()
    assert all(h.cancelled for h in scheduler.handles[-1:])
    task.set_result(response())
    await asyncio.sleep(0)
    assert not [e for e in events if e["event_type"] == "upstream_received"]
    observer.close()           # 幂等


async def test_observer_close_flushes_already_done_task(monkeypatch, events):
    clock, ctx, task, scheduler, observer = make_observer(monkeypatch, events)
    clock["t"] = 3.0
    task.set_result(response())
    observer.close()            # done 回调还没来得及运行：close 补发，且只发一次
    await asyncio.sleep(0)
    assert len([e for e in events if e["event_type"] == "upstream_received"]) == 1


# ------------------------------------------------------------------ 完整 HTTP 栈

@pytest.fixture
def http_client(monkeypatch):
    import main as app_main
    import runtime_state
    import tempfile
    import os

    tmp = tempfile.mkdtemp(prefix="vertex2openai_reqlog_test_")
    monkeypatch.setattr(runtime_state, "STATE_FILE", os.path.join(tmp, "web_state.json"))

    class Upstream:
        async def chat_completions(self, request, fastapi_request, failover_mode=False):
            return await helpers.execute_gemini_call(
                client_for(full=response(usage=sdk_meta(META))), MODEL, lambda _: [], {}, request,
                channel_name="express", force_fake_streaming=True)

    monkeypatch.setattr(chat_api, "CHANNELS", {"express": Upstream()})
    monkeypatch.setattr(app_state, "get_channel_strategy", lambda: "express")
    monkeypatch.setattr(app_state, "get_express_keys", lambda: ["k"])
    monkeypatch.setattr(app_config, "VERTEX_EXPRESS_API_KEY_VAL", ["k"])
    monkeypatch.setattr(chat_api, "breaker", ChannelBreaker())
    with TestClient(app_main.app) as c:
        yield c


def test_full_stack_stream_groups_logs_under_one_request(http_client, events):
    resp = http_client.post(
        "/v1/chat/completions", headers={"Authorization": f"Bearer {app_config.API_KEY}"},
        json={"model": "fake-" + MODEL, "stream": True,
              "messages": [{"role": "user", "content": "hi"}]})
    assert resp.status_code == 200 and resp.text.rstrip().endswith("data: [DONE]")
    starts = of_type(events, "request_start")
    assert len(starts) == 1 and starts[0]["mode"] == "fake_stream" and starts[0]["status"] == "in_progress"
    rid = starts[0]["request_id"]
    ends = of_type(events, "request_end")
    assert len(ends) == 1 and ends[0]["request_id"] == rid and ends[0]["status"] == "success"
    # 流式响应体内部的普通 print（🚀 上游请求）也带同一个 request_id
    inner = [e for e in events if e["event_type"] == "log" and "[上游请求]" in e["message"]]
    assert inner and all(e.get("request_id") == rid for e in inner)
    assert of_type(events, "fake_stream_start", rid)
