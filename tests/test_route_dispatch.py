"""路由层多通道分发 / 故障转移 / 熔断跳过 / 流式 failover 测试（3.1 节核心逻辑）。

用可编程假上游替换 CHANNELS，不触碰真实网络：
- 非流式：JSONResponse / 抛异常
- 流式：StreamingResponse + 自定义 body_iterator
"""
import json

import pytest
from fastapi.responses import JSONResponse, StreamingResponse

import config as app_config
from routes import chat_api
from failover import ChannelBreaker, UpstreamUnstartedError
from models import OpenAIRequest
from runtime_state import app_state


def _req():
    return OpenAIRequest(model="gemini-3.6-flash", messages=[{"role": "user", "content": "你好"}])


def _json(status, body=None):
    return JSONResponse(status_code=status, content=body or {
        "error": {"message": "test", "type": "upstream_error"}})


async def _ok_stream():
    yield 'data: {"choices":[{"delta":{"content":"你好"}}]}\n\n'
    yield "data: [DONE]\n\n"


async def _fail_stream():
    raise UpstreamUnstartedError("express 未出流失败")
    yield  # 不可达：让函数成为 async generator（未出流 = 不产出任何 chunk 直接抛）


class FakeUpstream:
    """按行为字典响应：response=JSONResponse / stream=SSE chunks / raise=异常。"""

    def __init__(self, behavior):
        self.behavior = behavior
        self.calls = []

    async def chat_completions(self, request, fastapi_request, failover_mode=False):
        self.calls.append(failover_mode)
        if "raise" in self.behavior:
            raise self.behavior["raise"]
        if "stream" in self.behavior:
            async def _gen():
                for chunk in self.behavior["stream"]:
                    yield chunk
            return StreamingResponse(_gen(), media_type="text/event-stream")
        return self.behavior["response"]


@pytest.fixture
def env(monkeypatch):
    """基础环境：Express Key 配齐，熔断器全新。"""
    monkeypatch.setattr(app_config, "VERTEX_EXPRESS_API_KEY_VAL", ["test-key"])
    monkeypatch.setattr(chat_api, "breaker", ChannelBreaker())
    return {}


def _install(monkeypatch, express_behavior, cookie_behavior):
    ex = FakeUpstream(express_behavior)
    ck = FakeUpstream(cookie_behavior)
    monkeypatch.setattr(chat_api, "CHANNELS", {"express": ex, "cookie": ck})
    return ex, ck


class TestChannelOrder:
    def test_express_only(self):
        assert chat_api._channel_order("express") == ["express"]

    def test_cookie_only(self):
        assert chat_api._channel_order("cookie") == ["cookie"]

    def test_hybrid_express_first(self):
        assert chat_api._channel_order("hybrid") == ["express", "cookie"]

    def test_unknown_strategy_defaults_express(self):
        assert chat_api._channel_order("banana") == ["express"]


class TestAvailableChannels:
    def test_express_removed_without_key(self, monkeypatch):
        monkeypatch.setattr(app_config, "VERTEX_EXPRESS_API_KEY_VAL", [])
        monkeypatch.setattr(app_state, "get_express_keys", lambda: None)
        monkeypatch.setattr(app_state, "get_cookie_accounts", lambda: [{"cookie": "c", "project_id": "p"}])
        assert chat_api._available_channels(["express", "cookie"]) == ["cookie"]

    def test_console_keys_also_available(self, monkeypatch):
        """控制台管理的 key 也应通过预检（与 ExpressKeyManager 来源一致）。"""
        monkeypatch.setattr(app_config, "VERTEX_EXPRESS_API_KEY_VAL", [])
        monkeypatch.setattr(app_state, "get_express_keys", lambda: ["console-key-1"])
        monkeypatch.setattr(app_state, "get_cookie_accounts", lambda: [])
        assert chat_api._available_channels(["express"]) == ["express"]

    def test_cookie_removed_without_credentials(self, monkeypatch):
        monkeypatch.setattr(app_config, "VERTEX_EXPRESS_API_KEY_VAL", ["k"])
        monkeypatch.setattr(app_state, "get_express_keys", lambda: None)
        monkeypatch.setattr(app_state, "get_cookie_accounts", lambda: [])
        assert chat_api._available_channels(["express", "cookie"]) == ["express"]

    def test_both_removed(self, monkeypatch):
        monkeypatch.setattr(app_config, "VERTEX_EXPRESS_API_KEY_VAL", [])
        monkeypatch.setattr(app_state, "get_express_keys", lambda: None)
        monkeypatch.setattr(app_state, "get_cookie_accounts", lambda: [])
        assert chat_api._available_channels(["express", "cookie"]) == []


class TestDispatch:
    async def test_first_channel_success_no_fallback(self, env, monkeypatch):
        ex, ck = _install(monkeypatch, {"response": _json(200)}, {"response": _json(200)})
        resp = await chat_api._dispatch(["express", "cookie"], _req(), None, failover_mode=True)
        assert resp.status_code == 200
        assert len(ex.calls) == 1
        assert len(ck.calls) == 0

    async def test_429_fails_over_to_cookie(self, env, monkeypatch):
        ex, ck = _install(monkeypatch, {"response": _json(429)}, {"response": _json(200)})
        resp = await chat_api._dispatch(["express", "cookie"], _req(), None, failover_mode=True)
        assert resp.status_code == 200
        assert len(ex.calls) == 1 and len(ck.calls) == 1

    async def test_500_fails_over(self, env, monkeypatch):
        ex, ck = _install(monkeypatch, {"response": _json(500)}, {"response": _json(200)})
        resp = await chat_api._dispatch(["express", "cookie"], _req(), None, failover_mode=True)
        assert resp.status_code == 200

    @pytest.mark.parametrize("code", [400, 401, 403])
    async def test_config_errors_not_switchable(self, env, monkeypatch, code):
        """400/401/403 不切换：如实报错（3.1 节白名单）。"""
        ex, ck = _install(monkeypatch, {"response": _json(code)}, {"response": _json(200)})
        resp = await chat_api._dispatch(["express", "cookie"], _req(), None, failover_mode=True)
        assert resp.status_code == code
        assert len(ck.calls) == 0

    async def test_all_fail_returns_error_response(self, env, monkeypatch):
        ex, ck = _install(monkeypatch, {"response": _json(429)}, {"response": _json(503)})
        resp = await chat_api._dispatch(["express", "cookie"], _req(), None, failover_mode=True)
        assert resp.status_code >= 400
        assert len(ex.calls) == 1 and len(ck.calls) == 1

    async def test_no_channels_returns_503(self, env, monkeypatch):
        resp = await chat_api._dispatch([], _req(), None, failover_mode=True)
        assert resp.status_code == 503

    async def test_cooling_channel_skipped(self, env, monkeypatch):
        ex, ck = _install(monkeypatch, {"response": _json(200)}, {"response": _json(200)})
        for _ in range(3):
            chat_api.breaker.report_failure("express")
        assert chat_api.breaker.is_cooling("express")
        resp = await chat_api._dispatch(["express", "cookie"], _req(), None, failover_mode=True)
        assert resp.status_code == 200
        assert len(ex.calls) == 0 and len(ck.calls) == 1

    async def test_upstream_unstarted_switches_channel(self, env, monkeypatch):
        ex, ck = _install(monkeypatch, {"raise": UpstreamUnstartedError("express 失败")},
                          {"response": _json(200)})
        resp = await chat_api._dispatch(["express", "cookie"], _req(), None, failover_mode=True)
        assert resp.status_code == 200
        assert len(ex.calls) == 1 and len(ck.calls) == 1

    async def test_upstream_unstarted_no_fallback_returns_error(self, env, monkeypatch):
        ex, ck = _install(monkeypatch, {"raise": UpstreamUnstartedError("express 失败")},
                          {"response": _json(200)})
        resp = await chat_api._dispatch(["express"], _req(), None, failover_mode=True)
        assert isinstance(resp, JSONResponse)
        assert resp.status_code >= 400

    async def test_non_hybrid_exception_passthrough(self, env, monkeypatch):
        """非 hybrid 零行为变化：异常透传（测试记录里明确的一条）。"""
        ex, ck = _install(monkeypatch, {"raise": UpstreamUnstartedError("x")},
                          {"response": _json(200)})
        with pytest.raises(UpstreamUnstartedError):
            await chat_api._dispatch(["express"], _req(), None, failover_mode=False)

    async def test_non_hybrid_no_wrapping(self, env, monkeypatch):
        ex, ck = _install(monkeypatch, {"response": _json(429)}, {"response": _json(200)})
        resp = await chat_api._dispatch(["express"], _req(), None, failover_mode=False)
        assert resp.status_code == 429   # 原样返回，不尝试切换


class TestStreamFailover:
    async def test_unstarted_stream_switches_to_second(self, env, monkeypatch):
        async def _failing():
            raise UpstreamUnstartedError("express 未出流失败")
            yield  # 不可达：让函数成为 async generator（未出流）
        ex, ck = _install(
            monkeypatch,
            {"response": StreamingResponse(_failing(), media_type="text/event-stream")},
            {"stream": ['data: {"choices":[{"delta":{"content":"你好"}}]}\n\n', "data: [DONE]\n\n"]},
        )
        resp = await chat_api._dispatch(["express", "cookie"], _req(), None, failover_mode=True)
        assert isinstance(resp, StreamingResponse)
        body = "".join([c async for c in resp.body_iterator])
        assert "你好" in body
        assert "[DONE]" in body
        assert len(ex.calls) == 1 and len(ck.calls) == 1

    async def test_unstarted_stream_no_fallback_error_stream(self, env, monkeypatch):
        async def _failing():
            raise UpstreamUnstartedError("express 未出流失败")
            yield  # 不可达：让函数成为 async generator（未出流）
        ex, ck = _install(
            monkeypatch,
            {"response": StreamingResponse(_failing(), media_type="text/event-stream")},
            {"response": _json(200)},
        )
        resp = await chat_api._dispatch(["express"], _req(), None, failover_mode=True)
        body = "".join([c async for c in resp.body_iterator])
        assert "error" in body
        assert "[DONE]" in body

    async def test_normal_stream_passthrough(self, env, monkeypatch):
        ex, ck = _install(
            monkeypatch,
            {"stream": ['data: {"choices":[{"delta":{"content":"直接出"}}]}\n\n', "data: [DONE]\n\n"]},
            {"response": _json(200)},
        )
        resp = await chat_api._dispatch(["express", "cookie"], _req(), None, failover_mode=True)
        assert isinstance(resp, StreamingResponse)
        body = "".join([c async for c in resp.body_iterator])
        assert "直接出" in body
        assert "[DONE]" in body
        assert len(ck.calls) == 0   # 主通道正常出流，不触发兜底

    async def test_non_hybrid_stream_untouched(self, env, monkeypatch):
        ex, ck = _install(
            monkeypatch,
            {"stream": ["data: keep-alive\n\n"]},
            {"response": _json(200)},
        )
        resp = await chat_api._dispatch(["express"], _req(), None, failover_mode=False)
        assert isinstance(resp, StreamingResponse)   # 零包装透传
        body = "".join([c async for c in resp.body_iterator])
        assert body == "data: keep-alive\n\n"


class TestSuccessReporting:
    async def test_success_clears_breaker(self, env, monkeypatch):
        ex, ck = _install(monkeypatch, {"response": _json(429)}, {"response": _json(200)})
        for _ in range(2):
            chat_api.breaker.report_failure("express")
        await chat_api._dispatch(["express", "cookie"], _req(), None, failover_mode=True)
        # 429 → 切换 → cookie 成功；express 累计 3 次失败但成功通道是 cookie
        assert chat_api.breaker.status().get("express", {}).get("failures") == 3

    async def test_stream_success_reports_success(self, env, monkeypatch):
        ex, ck = _install(
            monkeypatch,
            {"stream": ["data: x\n\n"]},
            {"response": _json(200)},
        )
        resp = await chat_api._dispatch(["express", "cookie"], _req(), None, failover_mode=True)
        _ = [c async for c in resp.body_iterator]
        assert chat_api.breaker.status() == {}   # 主通道流式成功 → 计数清零
