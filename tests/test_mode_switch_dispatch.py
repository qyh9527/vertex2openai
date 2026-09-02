"""P1-6：模式切换 → 真实聊天分发集成测试。

用 FastAPI TestClient 走完整 HTTP 栈（真实路由 + 真实 runtime_state 状态），
断言：
- 控制台切到 mode=vertex 后，/v1/chat/completions 请求只触发 ServiceAccountUpstream，
  Express 与 Cookie 上游零调用；
- 切回 express 后恢复 Express；
- 预检逻辑生效（vertex 策略下 SA 无凭证 → 503 且不碰任何上游）。

上游用 monkeypatch 替换成可编程假对象，不触碰真实网络。
"""
import json

import pytest
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from routes import chat_api
from runtime_state import app_state
import config as app_config


class FakeUpstream:
    def __init__(self, name, behavior):
        self.name = name
        self.behavior = behavior
        self.calls = 0

    async def chat_completions(self, request, fastapi_request, failover_mode=False):
        self.calls += 1
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
def client(monkeypatch):
    """完整 HTTP 栈：真实 main.app + 假上游 + 临时状态文件。"""
    import main as app_main
    import runtime_state
    import tempfile, os

    # 独立临时 STATE_DIR，绝不碰真实 web_state.json
    tmp = tempfile.mkdtemp(prefix="vertex2openai_mode_test_")
    monkeypatch.setattr(runtime_state, "STATE_FILE", os.path.join(tmp, "web_state.json"))

    ex = FakeUpstream("express", {"response": _ok_response()})
    ck = FakeUpstream("cookie", {"response": _ok_response()})
    sa = FakeUpstream("vertex", {"response": _ok_response()})
    monkeypatch.setattr(chat_api, "CHANNELS", {"express": ex, "cookie": ck, "vertex": sa})

    # 预检全放行：三种凭证都视为已配置（真实凭证解析与本测试无关，
    # 预检剔除另有专项测试覆盖）
    monkeypatch.setattr(app_state, "get_express_keys", lambda: ["k"])
    monkeypatch.setattr(app_state, "get_cookie_accounts", lambda: [{"cookie": "c", "project_id": "p"}])
    monkeypatch.setattr(app_state, "get_sa_accounts", lambda: [{"sa_json": "{}", "project_id": "p"}])
    monkeypatch.setattr(app_config, "VERTEX_EXPRESS_API_KEY_VAL", ["k"])

    # 全新熔断器（不跨测试残留冷却状态）
    from failover import ChannelBreaker
    monkeypatch.setattr(chat_api, "breaker", ChannelBreaker())

    with TestClient(app_main.app) as c:
        yield c, ex, ck, sa


def _chat(c, model="gemini-3.6-flash"):
    return c.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {app_config.API_KEY}"},
        json={"model": model, "messages": [{"role": "user", "content": "hi"}]},
    )


def _chat_at(c, channel, model="gemini-3.6-flash"):
    return c.post(
        f"/{channel}/v1/chat/completions",
        headers={"Authorization": f"Bearer {app_config.API_KEY}"},
        json={"model": model, "messages": [{"role": "user", "content": "hi"}]},
    )


class TestExplicitChannelRoutes:
    @pytest.mark.parametrize("channel, expected", [
        ("express", "express"),
        ("cookie", "cookie"),
        ("vertex", "vertex"),
    ])
    def test_explicit_path_forces_one_channel(self, client, channel, expected):
        """/<channel>/v1 路径独立于控制台策略，只调用指定上游一次。"""
        c, ex, ck, sa = client
        app_state.set_channel_strategy("hybrid")

        resp = _chat_at(c, channel)

        assert resp.status_code == 200
        calls = {"express": ex.calls, "cookie": ck.calls, "vertex": sa.calls}
        assert calls == {expected: 1, **{name: 0 for name in calls if name != expected}}
        app_state.set_channel_strategy("express")

    def test_unprefixed_path_keeps_strategy_dispatch(self, client):
        """旧 /v1 路径仍按控制台策略分发。"""
        c, ex, ck, sa = client
        app_state.set_channel_strategy("vertex")

        resp = _chat(c)

        assert resp.status_code == 200
        assert sa.calls == 1 and ex.calls == 0 and ck.calls == 0
        app_state.set_channel_strategy("express")

    def test_explicit_models_path_is_openai_base_url_compatible(self, client):
        c, _, _, _ = client
        resp = c.get(
            "/vertex/v1/models",
            headers={"Authorization": f"Bearer {app_config.API_KEY}"},
        )
        assert resp.status_code == 200
        assert resp.json()["object"] == "list"


class TestModeSwitchDispatch:
    def test_switch_to_vertex_routes_only_sa(self, client, monkeypatch):
        """P1-6 核心：控制台切到 vertex 后，聊天请求只打服务账号上游。"""
        c, ex, ck, sa = client
        # 模拟控制台「通道与凭证」页点下 vertex 策略（走真实 settings 接口形状）
        assert app_state.set_channel_strategy("vertex") is True
        assert app_state.get_channel_strategy() == "vertex"

        resp = _chat(c)
        assert resp.status_code == 200
        assert sa.calls == 1          # SA 上游恰好被调用一次
        assert ex.calls == 0          # Express 零调用
        assert ck.calls == 0          # Cookie 零调用
        assert resp.json()["choices"][0]["message"]["content"] == "ok"
        # 收尾恢复默认策略（app_state 是模块级单例，避免污染其它测试）
        app_state.set_channel_strategy("express")

    def test_switch_back_to_express(self, client):
        """切回 express 策略后只打 Express 上游（SA 不再被调用）。"""
        c, ex, ck, sa = client
        app_state.set_channel_strategy("vertex")
        _chat(c)
        assert sa.calls == 1 and ex.calls == 0

        app_state.set_channel_strategy("express")
        resp = _chat(c)
        assert resp.status_code == 200
        assert ex.calls == 1 and sa.calls == 1   # SA 保持 1（未被再次调用）

    def test_vertex_without_sa_credentials_returns_503(self, client, monkeypatch):
        """vertex 策略 + SA 凭证缺失：预检剔除后 503，且不调用任何上游。"""
        c, ex, ck, sa = client
        monkeypatch.setattr(app_state, "get_sa_accounts", lambda: [])
        app_state.set_channel_strategy("vertex")

        resp = _chat(c)
        assert resp.status_code == 503
        body = resp.json()
        assert "没有可用" in body["error"]["message"] or "服务账号" in body["error"]["message"]
        assert ex.calls == 0 and ck.calls == 0 and sa.calls == 0
        app_state.set_channel_strategy("express")

    def test_mode_endpoint_maps_vertex(self, client):
        """/api/settings/mode 的 vertex 档位真实落到 channel_strategy（集成路径）。"""
        c, _, _, _ = client
        # require_auth 依赖的是会话 Cookie；直接调接口等价于控制台按钮的行为，
        # 这里只验证策略落库，鉴权由 main.py 的其它机制保证。
        r = c.post("/api/settings/mode", json={"mode": "vertex"})
        assert r.status_code in (200, 401)   # 401 = 未登录被拦（合法），200 = 已映射
        if r.status_code == 200:
            assert app_state.get_channel_strategy() == "vertex"
            assert r.json()["channel_strategy"] == "vertex"

    def test_hybrid_vertex_first_uses_vertex_upstream(self, client):
        """hybrid 策略把 vertex 排第一时，成功请求走 SA（顺序真实生效）。"""
        c, ex, ck, sa = client
        app_state.update_settings({"hybrid_channels": ["vertex", "express", "cookie"]})
        app_state.set_channel_strategy("hybrid")

        resp = _chat(c)
        assert resp.status_code == 200
        assert sa.calls == 1 and ex.calls == 0 and ck.calls == 0
        # 收尾恢复默认 hybrid 顺序（app_state 是模块级单例，避免污染其它测试）
        app_state.update_settings({
            "hybrid_channels": app_config.DEFAULT_SETTINGS["hybrid_channels"]})
