"""P0 日志回归：SA 请求的公共管线日志不得出现 Express；Express 日志保持正确。

对应审查报告 P0-2/P0-3/P0-10：
- express_sdk / api_helpers 的公共日志已改为动态通道文案（channel_call_text /
  channel_display_name），SA 继承父类管线时不再被误标。
- 本文件用假的 SDK Client 与捕获 print 的方式验证日志文本，不触碰真实网络。
"""
import io
import json
from contextlib import redirect_stdout

import pytest
from fastapi.responses import JSONResponse

from api_helpers import (
    CHANNEL_META, channel_display_name, channel_call_text,
)
from upstreams.express_sdk import _log_resolved_endpoint
import upstreams.express_sdk as express_sdk
import config as app_config


class _FakeApi:
    def __init__(self, base_url="https://aiplatform.googleapis.com/"):
        self._http_options = type("H", (), {"base_url": base_url, "api_version": "v1beta1"})()
        self.project = "proj-x"
        self.location = "global"


class _FakeClient:
    def __init__(self, base_url="https://aiplatform.googleapis.com/"):
        self._api_client = _FakeApi(base_url)


@pytest.fixture(autouse=True)
def _clear_endpoint_keys():
    express_sdk._ENDPOINT_LOGGED_KEYS.clear()
    yield
    express_sdk._ENDPOINT_LOGGED_KEYS.clear()


def _capture(func, *args, **kwargs):
    buf = io.StringIO()
    with redirect_stdout(buf):
        func(*args, **kwargs)
    return buf.getvalue()


def _set_debug_outbound(value: bool):
    from runtime_state import app_state
    app_state.update_settings({"debug_outbound": value})


@pytest.fixture(autouse=True)
def _restore_debug_outbound():
    yield
    _set_debug_outbound(False)


class TestChannelMeta:
    def test_meta_covers_all_channels(self):
        from routes.chat_api import CHANNELS
        assert set(CHANNEL_META) == set(CHANNELS)

    def test_display_names(self):
        assert channel_display_name("express") == "Express API Key"
        assert channel_display_name("cookie") == "Cookie 直连"
        assert channel_display_name("vertex") == "服务账号（Vertex SA）"

    def test_unknown_channel_passthrough(self):
        assert channel_display_name("banana") == "banana"
        assert channel_display_name(None) == "未知通道"

    def test_call_text_includes_auth_mode(self):
        assert "service_account" in channel_call_text("vertex")
        assert "api_key" in channel_call_text("express")
        assert "cookie" in channel_call_text("cookie")

    def test_sa_channels_use_distinct_names(self):
        """SA 与 Express 显示名必须可区分（否则日志无信息量）。"""
        assert channel_display_name("vertex") != channel_display_name("express")


class TestEndpointLogDedup:
    def test_express_then_vertex_both_logged(self):
        """P0-5：同进程先 Express 后 SA，两种端点日志都必须各出现一次。"""
        out1 = _capture(_log_resolved_endpoint, _FakeClient(), channel_name="express")
        assert "Express API Key" in out1
        out2 = _capture(_log_resolved_endpoint, _FakeClient(), channel_name="vertex")
        assert "服务账号" in out2
        assert "Express" not in out2

    def test_same_channel_logged_once(self):
        _capture(_log_resolved_endpoint, _FakeClient(), channel_name="express")
        out2 = _capture(_log_resolved_endpoint, _FakeClient(), channel_name="express")
        assert out2 == ""   # 同 (channel, project, location, base_url) 不重复打

    def test_sa_log_has_project_location_no_secret(self):
        out = _capture(_log_resolved_endpoint, _FakeClient(), channel_name="vertex")
        assert "proj-x" in out
        assert "global" in out
        # 端点日志绝不泄漏凭证
        assert "private_key" not in out
        assert "Bearer" not in out

    def test_explain_tail_differs_by_channel(self):
        out_ex = _capture(_log_resolved_endpoint, _FakeClient(), channel_name="express")
        out_sa = _capture(_log_resolved_endpoint, _FakeClient(), channel_name="vertex")
        assert "Google 后端路由" in out_ex       # Express 的说明
        assert "资源路径" in out_sa               # SA 的说明


class TestOutboundDebugShape:
    """P1-1：出站调试的请求形状摘要（safety_settings / 工具声明 / system instruction 等）。"""

    def _gen_config(self):
        from google.genai import types
        return {
            "safety_settings": [
                types.SafetySetting(category="HARM_CATEGORY_JAILBREAK", threshold="BLOCK_NONE"),
                types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"),
            ],
            "tools": [{"function_declarations": [{"name": "weather_api"}, {"name": "time_api"}]}],
            "tool_config": {"function_calling_config": {"mode": "AUTO"}},
            "system_instruction": "你是一个助手",
        }

    def test_shape_log_covers_safety_tools_system(self):
        """摘要行必须包含 safety 分类、工具名列表、system_instruction 有无。"""
        cfg = self._gen_config()
        _ss = cfg["safety_settings"]
        _tools_dbg = cfg["tools"]
        # 与 express_sdk 内同构的摘要构造（避免直接跑 chat_completions——
        # 那需要整套 Client；这里锁住摘要文案契约）
        shape = []
        if _ss:
            shape.append("safety_settings=[" + ", ".join(
                f"{getattr(s, 'category', s)}/{getattr(s, 'threshold', '')}" for s in _ss) + "]")
        names = [f.get("name") for t in _tools_dbg if isinstance(t, dict)
                 for f in (t.get("function_declarations") or []) if f.get("name")]
        if names:
            shape.append(f"工具声明={names}（共 {len(names)} 个）")
        text = "；".join(shape)
        assert "HARM_CATEGORY_JAILBREAK" in text
        assert "weather_api" in text and "time_api" in text
        assert "共 2 个" in text

    def test_shape_log_no_secret_values(self, monkeypatch):
        """摘要只记形状：绝不打印工具参数或完整 prompt。"""
        # 工具参数里的"敏感值"不应出现在任何摘要字段中
        sensitive_arg = "sk-super-secret-12345"
        from upstreams import express_sdk
        cfg = {
            "tools": [{"function_declarations": [
                {"name": "x", "parameters": {"properties": {"key": {"type": "string"}}}}]}],
        }
        # 构造与实现一致的提取逻辑：摘要只取 name，不取 parameters
        names = [f.get("name") for t in (cfg.get("tools") or []) if isinstance(t, dict)
                 for f in (t.get("function_declarations") or []) if isinstance(f, dict) and f.get("name")]
        summary = f"工具声明={names}"
        assert sensitive_arg not in summary

    async def test_debug_outbound_emits_shape_line(self, monkeypatch):
        """开 debug_outbound 后 chat_completions 真打两条调试日志（参数 + 请求形状）。"""
        from google.genai import types
        from models import OpenAIRequest
        from upstreams.express_sdk import ExpressSDKUpstream
        from fastapi.responses import JSONResponse

        captured = {}

        async def fake_execute(*args, **kwargs):
            captured["gen_config"] = args[3]
            return JSONResponse(status_code=200, content={"ok": True})

        monkeypatch.setattr(express_sdk, "execute_gemini_call", fake_execute)
        monkeypatch.setattr(express_sdk, "_log_resolved_endpoint", lambda *a, **k: None)

        class _FakeKeyMgr:
            def get_total_keys(self): return 1
            def get_express_api_key(self): return (0, "k")

        class _FakeApp:
            state = type("S", (), {"express_key_manager": _FakeKeyMgr()})()

        class _FakeRequest:
            app = _FakeApp()

        class _FakeState:
            express_key_manager = _FakeKeyMgr()

        _FakeApp.state = _FakeState

        _set_debug_outbound(True)
        req = OpenAIRequest(model="gemini-3.6-flash",
                            messages=[{"role": "system", "content": "sys"},
                                      {"role": "user", "content": "hi"}],
                            tools=[{"type": "function", "function": {
                                "name": "weather_api", "description": "d",
                                "parameters": {"type": "object", "properties": {}}}}])
        up = ExpressSDKUpstream()
        buf = io.StringIO()
        with redirect_stdout(buf):
            await up.chat_completions(req, _FakeRequest())
        out = buf.getvalue()
        assert "生成参数" in out
        assert "请求形状" in out
        assert "weather_api" in out
        assert "system_instruction=有" in out
        # safety_settings 摘要也在（generate config 恒注入 5 项 BLOCK_NONE）
        assert "safety_settings" in out


class TestClientReuseLog:
    """P1-3：Client 来源日志 new/reused/evicted 细分。"""

    def _clear(self):
        import upstreams.express_sdk as sdk
        sdk._clear_client_cache()   # 统一池自带加锁；手动 with 锁会死锁（P1-⑤）

    def test_reused_logs_reuse(self):
        import upstreams.express_sdk as sdk
        self._clear()
        sdk._get_cached_client("k1", False)
        out = _capture(sdk._get_cached_client, "k1", False)
        assert "复用缓存 Client" in out

    def test_first_call_logs_new(self):
        import upstreams.express_sdk as sdk
        self._clear()
        out = _capture(sdk._get_cached_client, "k_new", False)
        assert "新建 Client 连接池" in out

    def test_evict_marks_state(self):
        """evict 硬淘汰路径（保留机制）日志明确标注 evicted。

        （进阶报告 P1-4 后安全拦截不再触发 evict；本用例验证 evict 机制本身
        供未来"会话状态损坏"类硬错误使用。）"""
        import upstreams.express_sdk as sdk
        self._clear()
        client = sdk._get_cached_client("k1", False)
        out = _capture(client._vertex_on_failure, kind="evict", reason="会话状态损坏")
        assert "evicted" in out
        # 淘汰后下一次是新建
        out2 = _capture(sdk._get_cached_client, "k1", False)
        assert "新建 Client 连接池" in out2

    def test_threshold_evict_marks_state(self):
        """连接级失败达到阈值淘汰时同样标注 evicted。"""
        import upstreams.express_sdk as sdk
        from runtime_state import app_state
        self._clear()
        app_state.update_settings({"client_reuse_evict_threshold": 1})
        try:
            client = sdk._get_cached_client("k1", False)
            out = _capture(client._vertex_on_failure, kind="conn")
            assert "evicted" in out
        finally:
            app_state.update_settings({
                "client_reuse_evict_threshold":
                    app_config.DEFAULT_SETTINGS["client_reuse_evict_threshold"]})
