"""隐式缓存前缀稳定性：未启用的防截断 / 仅 fake 的顶部注入不得改写普通请求。

2026-10-14 SA 真机实测：防截断每次请求生成随机合成工具名、顶部注入随机方案，
都会让上游隐式缓存失效。这里锁住"关掉就真的不改写请求"的边界。
"""
import pytest
from fastapi.responses import JSONResponse

from models import OpenAIRequest
from runtime_state import app_state
from upstreams import cookie_proxy, express_sdk
from upstreams.express_sdk import ExpressSDKUpstream

_SETTING_KEYS = ("anti_truncation_field", "top_input_injection_mode", "top_input_injection_plans",
                 "top_input_injection_selected_plan_id", "top_input_injection_random",
                 "top_input_injection_fake_only")
_PLANS = [{"id": "p", "name": "甲", "role": "system", "content": "顶部方案"}]


@pytest.fixture(autouse=True)
def _restore_settings():
    before = app_state.get_settings()
    yield
    app_state.update_settings({k: before.get(k) for k in _SETTING_KEYS})


class _KeyMgr:
    def get_total_keys(self):
        return 1

    def get_express_api_key(self):
        return (0, "k")


class _State:
    express_key_manager = _KeyMgr()


class _App:
    state = _State


class _Request:
    app = _App()


async def _capture_express(monkeypatch, model, **extra):
    captured = {}

    async def fake_execute(client, model_to_call, prompt_func, gen_config, request_obj, **kwargs):
        captured["tools"] = [f.get("name") for t in (gen_config.get("tools") or []) if isinstance(t, dict)
                             for f in (t.get("function_declarations") or [])]
        captured["messages"] = [(m.role, m.content) for m in request_obj.messages]
        captured["synthetic"] = kwargs.get("synthetic_tool_name")
        return JSONResponse(status_code=200, content={"ok": True})

    monkeypatch.setattr(express_sdk, "execute_gemini_call", fake_execute)
    monkeypatch.setattr(express_sdk, "_log_resolved_endpoint", lambda *a, **k: None)
    req = OpenAIRequest(model=model, messages=[{"role": "system", "content": "原始系统"},
                                               {"role": "user", "content": "你好"}], **extra)
    await ExpressSDKUpstream().chat_completions(req, _Request())
    return captured


class TestAntiTruncationOff:
    """自定义启用字段 anti_qyh：只有 anti_qyh=true 才注入合成工具。"""

    @pytest.mark.parametrize("extra", [
        {},
        {"anti_truncation": True},
        {"anti_qyh": False},
        {"anti_qyh": "false"},
    ])
    async def test_not_enabled_keeps_request_untouched(self, monkeypatch, extra):
        app_state.update_settings({"anti_truncation_field": "anti_qyh"})
        captured = await _capture_express(monkeypatch, "gemini-3.7-flash", **extra)
        assert captured["synthetic"] is None
        assert captured["tools"] == []
        assert captured["messages"] == [("system", "原始系统"), ("user", "你好")]

    async def test_enabled_field_injects(self, monkeypatch):
        app_state.update_settings({"anti_truncation_field": "anti_qyh"})
        captured = await _capture_express(monkeypatch, "fake-gemini-3.7-flash", anti_qyh=True)
        assert captured["synthetic"] and captured["synthetic"].startswith("v2o_emit_")
        assert captured["tools"] == [captured["synthetic"]]


class TestTopInjectionFakeOnly:
    def _enable(self, fake_only):
        app_state.update_settings({
            "top_input_injection_mode": "always",
            "top_input_injection_plans": _PLANS,
            "top_input_injection_selected_plan_id": "p",
            "top_input_injection_random": "off",
            "top_input_injection_fake_only": fake_only,
        })

    async def test_fake_only_leaves_normal_request_untouched(self, monkeypatch):
        self._enable(True)
        captured = await _capture_express(monkeypatch, "gemini-3.7-flash")
        assert captured["messages"] == [("system", "原始系统"), ("user", "你好")]

    async def test_fake_only_injects_fake_request(self, monkeypatch):
        self._enable(True)
        captured = await _capture_express(monkeypatch, "fake-gemini-3.7-flash")
        assert captured["messages"][0] == ("system", "顶部方案\n原始系统")

    async def test_unchecked_injects_every_request(self, monkeypatch):
        self._enable(False)
        captured = await _capture_express(monkeypatch, "gemini-3.7-flash")
        assert captured["messages"][0] == ("system", "顶部方案\n原始系统")

    @pytest.mark.parametrize("model,expected", [("gemini-3.7-flash", False), ("fake-gemini-3.7-flash", True)])
    async def test_cookie_uses_request_fake_prefix(self, monkeypatch, model, expected):
        self._enable(True)
        calls = []

        class _Stop(Exception):
            pass

        def fake_apply(messages, config, channel=""):
            calls.append(channel)
            return messages, []

        def stop_after_top_injection(settings):
            # 顶部注入之后紧跟输入搬运配置读取：在这里截断，绝不发出真实 Cookie 请求
            raise _Stop

        monkeypatch.setattr(cookie_proxy, "_get_cookie_string", lambda: "SAPISID=x")
        monkeypatch.setattr(cookie_proxy, "_get_project_id", lambda: "p")
        monkeypatch.setattr(cookie_proxy, "build_headers", lambda cookie: {"Authorization": "x"})
        monkeypatch.setattr(cookie_proxy, "apply_top_input_injection", fake_apply)
        monkeypatch.setattr(cookie_proxy, "get_input_relay_config", stop_after_top_injection)
        req = OpenAIRequest(model=model, messages=[{"role": "user", "content": "你好"}])
        with pytest.raises(_Stop):
            await cookie_proxy.CookieProxyUpstream().chat_completions(req, None)
        assert calls == (["cookie"] if expected else [])
