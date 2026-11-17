"""raw 纯净入口测试：路由固定 SA / 跳过提示词改写 / 保留传输能力与额度保护。"""
import json

import pytest
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from routes import chat_api
from routes import models_api
from runtime_state import app_state
from upstreams.express_sdk import ExpressSDKUpstream, _normalize_model_name
from upstreams import express_sdk
from upstreams import service_account
import config as app_config
import request_log
import raw_mode


class FakeUpstream:
    def __init__(self, name, behavior=None):
        self.name = name
        self.behavior = behavior or {}
        self.calls = []

    async def chat_completions(self, request, fastapi_request, failover_mode=False, raw_mode=False):
        self.calls.append({"failover_mode": failover_mode, "raw_mode": raw_mode})
        if "raise" in self.behavior:
            raise self.behavior["raise"]
        return self.behavior["response"]


def _ok_response(model="gemini-3.6-flash"):
    return JSONResponse(status_code=200, content={
        "id": "chatcmpl-test", "object": "chat.completion",
        "created": 0, "model": model,
        "choices": [{"index": 0,
                     "message": {"role": "assistant", "content": "ok"},
                     "finish_reason": "stop"}]})


@pytest.fixture
def raw_env(monkeypatch):
    """完整 HTTP 栈：真实 main.app + 假上游；SA 预检已配置。"""
    import main as app_main
    import runtime_state
    import tempfile, os

    tmp = tempfile.mkdtemp(prefix="vertex2openai_raw_test_")
    monkeypatch.setattr(runtime_state, "STATE_FILE", os.path.join(tmp, "web_state.json"))

    ex = FakeUpstream("express", {"response": _ok_response()})
    ck = FakeUpstream("cookie", {"response": _ok_response()})
    sa = FakeUpstream("vertex", {"response": _ok_response()})
    monkeypatch.setattr(chat_api, "CHANNELS", {"express": ex, "cookie": ck, "vertex": sa})
    monkeypatch.setattr(app_state, "get_express_keys", lambda: ["k"])
    monkeypatch.setattr(app_state, "get_cookie_accounts", lambda: [{"cookie": "c", "project_id": "p"}])
    monkeypatch.setattr(app_state, "get_sa_accounts", lambda: [{"sa_json": "{}", "project_id": "p"}])
    monkeypatch.setattr(app_config, "VERTEX_EXPRESS_API_KEY_VAL", ["k"])

    from failover import ChannelBreaker
    monkeypatch.setattr(chat_api, "breaker", ChannelBreaker())

    with TestClient(app_main.app) as c:
        yield c, ex, ck, sa


def _raw_chat(c, model="gemini-3.6-flash"):
    return c.post(
        "/raw/v1/chat/completions",
        headers={"Authorization": f"Bearer {app_config.API_KEY}"},
        json={"model": model, "messages": [{"role": "user", "content": "hi"}]},
    )


class TestRawRoute:
    def test_raw_route_registered_before_channel_wildcard(self):
        chat_paths = [getattr(r, "path", "") for r in chat_api.router.routes]
        assert chat_paths.index("/raw/v1/chat/completions") < \
            chat_paths.index("/{channel}/v1/chat/completions")
        model_paths = [getattr(r, "path", "") for r in models_api.router.routes]
        assert model_paths.index("/raw/v1/models") < \
            model_paths.index("/{channel}/v1/models")

    def test_raw_forces_vertex_and_ignores_console_strategy(self, raw_env):
        c, ex, ck, sa = raw_env
        app_state.set_channel_strategy("express")
        try:
            resp = _raw_chat(c)
        finally:
            app_state.set_channel_strategy("express")
        assert resp.status_code == 200
        assert len(sa.calls) == 1 and sa.calls[0] == {"failover_mode": False, "raw_mode": True}
        assert ex.calls == [] and ck.calls == []

    def test_raw_without_sa_credentials_is_503(self, raw_env, monkeypatch):
        c, ex, ck, sa = raw_env
        monkeypatch.setattr(app_state, "get_sa_accounts", lambda: [])
        resp = _raw_chat(c)
        assert resp.status_code == 503
        assert "raw" in resp.json()["error"]["message"]
        assert sa.calls == [] and ex.calls == [] and ck.calls == []

    def test_raw_models_route(self, raw_env):
        c, *_ = raw_env
        resp = c.get("/raw/v1/models", headers={"Authorization": f"Bearer {app_config.API_KEY}"})
        assert resp.status_code == 200
        assert resp.json()["object"] == "list"

    def test_raw_skipped_field_in_request_start(self, raw_env, monkeypatch):
        c, *_ = raw_env
        records = []
        monkeypatch.setattr(request_log, "emit_record",
                            lambda message, **kw: records.append((message, kw)))
        monkeypatch.setattr(raw_mode, "detect_skipped_features",
                            lambda request, settings=None: ["top_injection", "input_relay"])
        assert _raw_chat(c).status_code == 200
        starts = [kw for _, kw in records if kw.get("event_type") == "request_start"]
        assert starts and starts[0]["fields"]["raw_skipped"] == ["top_injection", "input_relay"]


class TestRawSkipsPromptLayer:
    """直接驱动 ExpressSDKUpstream：断言各提示词改写函数在 raw 下不被调用。"""

    @pytest.fixture
    def wired(self, monkeypatch):
        request_settings = {
            "history_tool_mode": "all",
            "input_relay_mode": "always",
            "input_relay_tag": "relay",
            "input_relay_placeholder": "…",
            "top_input_injection_mode": "always",
            "top_input_injection_plans": [
                {"id": "p1", "name": "p", "role": "system", "content": "TOP"}],
            "top_input_injection_selected_plan_id": "p1",
            "inject_system_instruction": "INJECT",
            "inject_prefill": "PREFILL",
            "prefill_mode": "smart",
            "default_temperature": 0.7,
            "native_thinking_mode": "off",
            "thinking_g3_level": "low",
        }
        monkeypatch.setattr(app_state, "get_effective_settings", lambda model: dict(request_settings))
        monkeypatch.setattr(app_state, "get_setting", lambda key, default=None: (
            app_config.DEFAULT_SETTINGS.get(key, default)))
        monkeypatch.setattr(express_sdk, "history_tool_active", lambda settings, is_fake: True)
        calls = {"relay": 0, "console": 0, "prefill": 0, "synthetic": 0, "top": 0}

        monkeypatch.setattr(express_sdk, "apply_input_relay",
                            lambda messages, config: calls.__setitem__("relay", calls["relay"] + 1) or (messages, []))
        monkeypatch.setattr(express_sdk, "apply_console_injection",
                            lambda *a, **k: calls.__setitem__("console", calls["console"] + 1) or (a[0], []))
        monkeypatch.setattr(express_sdk, "apply_prefill_compat",
                            lambda *a, **k: calls.__setitem__("prefill", calls["prefill"] + 1) or (a[0], "", False))
        monkeypatch.setattr(express_sdk, "inject_request",
                            lambda request: calls.__setitem__("synthetic", calls["synthetic"] + 1) or (request, "synthetic_tool"))
        monkeypatch.setattr(express_sdk, "apply_top_input_injection",
                            lambda *a, **k: calls.__setitem__("top", calls["top"] + 1) or (a[0], []))

        captured = {}

        async def fake_execute(client, model_to_call, prompt_func, gen_config, request_obj, **kwargs):
            captured["gen_config"] = gen_config
            captured["kwargs"] = kwargs
            captured["model_to_call"] = model_to_call
            return _ok_response()

        monkeypatch.setattr(express_sdk, "execute_gemini_call", fake_execute)
        monkeypatch.setattr(express_sdk, "validate_request_schemas", lambda request: None)
        upstream = ExpressSDKUpstream()
        monkeypatch.setattr(upstream, "_resolve_client", lambda fastapi_request, base, settings: {
            "client": object(), "model_to_call": base,
            "priority_paygo": False, "fallback_model": None,
            "fallback_client_factory": None,
        })
        return upstream, calls, captured

    @pytest.mark.asyncio
    async def test_raw_skips_all_prompt_rewrites(self, wired):
        upstream, calls, captured = wired
        from models import OpenAIRequest
        request = OpenAIRequest(model="fake-gemini-3.8-flash-search", stream=True, messages=[
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "prefill"},
        ])
        resp = await upstream.chat_completions(request, None, raw_mode=True)
        assert resp.status_code == 200
        assert calls == {"relay": 0, "console": 0, "prefill": 0, "synthetic": 0, "top": 0}
        # fake- 与 -search 仍生效：模型名归一 + 假流式强制
        assert captured["model_to_call"] == "gemini-3.8-flash"
        assert captured["kwargs"]["force_fake_streaming"] is True
        assert captured["kwargs"]["prefill_text"] == ""
        assert captured["kwargs"]["synthetic_tool_name"] is None
        assert captured["kwargs"]["input_relay_strip_tag"] is None
        # 生成参数走内置默认：控制台的 default_temperature/native_thinking_mode 不生效
        assert captured["gen_config"].get("temperature") is None
        assert "thinking_config" not in captured["gen_config"] or \
            captured["gen_config"]["thinking_config"].get("thinking_level") != "low"

    @pytest.mark.asyncio
    async def test_non_raw_still_applies_features(self, wired):
        """对照组：同样配置在普通入口仍会调用各改写函数（证明上面的断言不是空转）。"""
        upstream, calls, _ = wired
        from models import OpenAIRequest
        request = OpenAIRequest(model="gemini-3.8-flash", messages=[
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "prefill"},
        ])
        await upstream.chat_completions(request, None, raw_mode=False)
        assert calls["relay"] == 1
        assert calls["console"] == 1
        assert calls["prefill"] == 1
        assert calls["synthetic"] == 1
        assert calls["top"] == 1


class TestRawKeepsTransport:
    def test_schema_validation_still_rejects(self):
        """raw 仍做 schema 校验：非法工具 parameters → 400。"""
        from models import OpenAIRequest
        request = OpenAIRequest(model="gemini-3.8-flash", messages=[{"role": "user", "content": "hi"}],
                                tools=[{"type": "function", "function": {
                                    "name": "f", "parameters": {"type": "not-a-type"}}}])
        upstream = ExpressSDKUpstream()
        import asyncio
        resp = asyncio.run(upstream.chat_completions(request, None, raw_mode=True))
        assert resp.status_code == 400

    def test_normalize_model_name_unchanged(self):
        assert _normalize_model_name("fake-gemini-3.8-flash-search") == ("gemini-3.8-flash", True, True, None)

    def test_raw_sa_budget_still_blocks(self, monkeypatch):
        """raw 仍走完整 SA 额度保护：账本未初始化 → 429，且不构造 Client。"""
        import sa_budget
        import asyncio
        fake_sa = json.dumps({"client_email": "sa@x.iam.gserviceaccount.com", "project_id": "p1"})
        monkeypatch.setattr(app_state, "get_current_sa_account",
                            lambda preferred_project=None: ("p1", "global", fake_sa))
        monkeypatch.setattr(sa_budget, "status",
                            lambda key: {"initialized": False, "blocked": True, "expired": False})
        upstream = service_account.ServiceAccountUpstream()
        from models import OpenAIRequest
        request = OpenAIRequest(model="gemini-3.8-flash", messages=[{"role": "user", "content": "hi"}])
        resp = asyncio.run(upstream.chat_completions(request, None, raw_mode=True))
        assert resp.status_code == 429

class TestRawOcrRegressions:
    def test_detect_skipped_handles_pay_fake_prefix(self, monkeypatch):
        """[PAY]fake- 组合前缀仍视为 fake（与 SA 上游归一一致）。"""
        from models import OpenAIRequest
        settings = {
            "top_input_injection_mode": "always",
            "top_input_injection_plans": [
                {"id": "p1", "name": "p", "role": "system", "content": "TOP"}],
            "top_input_injection_selected_plan_id": "p1",
            "top_input_injection_fake_only": True,
            "history_tool_mode": "fake_stream_only",
            "input_relay_mode": "off",
        }
        monkeypatch.setattr(app_state, "get_effective_settings", lambda model: dict(settings))
        monkeypatch.setattr(app_state, "get_model_overrides", lambda: {})
        request = OpenAIRequest(model="[PAY]fake-gemini-3.8-flash", stream=True,
                                messages=[{"role": "user", "content": "hi"}])
        keys = raw_mode.detect_skipped_features(request)
        assert "top_injection" in keys
        assert "history_tool" in keys

    def test_skipped_log_runs_after_begin_request(self, raw_env, monkeypatch):
        """逐项跳过日志必须带本次 request_id（begin_request 之后）。"""
        c, *_ = raw_env
        seen = []
        monkeypatch.setattr(raw_mode, "detect_skipped_features",
                            lambda request, settings=None: ["top_injection"])
        monkeypatch.setattr(raw_mode, "log_skipped_features",
                            lambda keys: seen.append((list(keys), request_log.current() is not None)))
        assert _raw_chat(c).status_code == 200
        assert seen == [(["top_injection"], True)]

    def test_non_raw_image_system_instruction_keeps_global_only(self, monkeypatch):
        """普通入口该键只读全局值（不被模型覆盖改变）——OCR 回归。"""
        from models import OpenAIRequest
        from protocol_conversion import create_generation_config
        monkeypatch.setattr(app_state, "get_setting",
                            lambda key, default=None: True if key == "image_system_instruction" else default)
        monkeypatch.setattr(app_state, "get_effective_settings", lambda model: {
            **app_config.DEFAULT_SETTINGS, "image_system_instruction": False})
        request = OpenAIRequest(model="gemini-3.1-flash-image", messages=[
            {"role": "system", "content": "sys"}, {"role": "user", "content": "draw"}])
        assert "system_instruction" in create_generation_config(request)

    def test_raw_image_system_instruction_uses_builtin_default(self):
        """raw 入口即使控制台打开该键，也按内置默认剥离生图 system_instruction。"""
        from models import OpenAIRequest
        from protocol_conversion import create_generation_config
        request = OpenAIRequest(model="gemini-3.1-flash-image", messages=[
            {"role": "system", "content": "sys"}, {"role": "user", "content": "draw"}])
        builtin = dict(app_config.DEFAULT_SETTINGS)
        assert builtin["image_system_instruction"] is False
        assert "system_instruction" not in create_generation_config(request, settings=builtin)

    def test_non_raw_settings_read_by_public_model_name(self, monkeypatch):
        """普通入口仍按 request.model（含 -search 等公开名）读取覆盖——OCR 回归。"""
        from models import OpenAIRequest
        from protocol_conversion import create_generation_config
        seen_models = []

        def effective(model):
            seen_models.append(model)
            data = dict(app_config.DEFAULT_SETTINGS)
            data["default_max_tokens"] = 777
            return data

        monkeypatch.setattr(app_state, "get_effective_settings", effective)
        request = OpenAIRequest(model="gemini-3.8-flash-search",
                                messages=[{"role": "user", "content": "hi"}])
        cfg = create_generation_config(request)
        assert "gemini-3.8-flash-search" in seen_models
        assert cfg.get("max_output_tokens") == 777
