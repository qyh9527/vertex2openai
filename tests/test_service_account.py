"""服务账号（第三通道）测试：凭证校验 / 环境变量兜底 / Client 缓存 / 请求级快照 / 通道 client 解析。

不触碰真实网络：只验证构造与路由逻辑（genai.Client 构造是惰性的，不发起请求）。
"""
import json
import threading

import pytest

from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization

from upstreams.service_account import (
    validate_sa_credentials, build_sa_credentials, ServiceAccountUpstream,
    _get_cached_sa_client, _SA_CLIENT_CACHE, _SA_CLIENT_CACHE_LOCK,
)
from runtime_state import app_state
import config as app_config


def _gen_sa_json(project="proj-test", email=None) -> str:
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    ).decode()
    return json.dumps({
        "type": "service_account",
        "project_id": project,
        "private_key_id": "k1",
        "private_key": pem,
        "client_email": email or f"sa-{project}@proj.iam.gserviceaccount.com",
        "client_id": "1",
        "token_uri": "https://oauth2.googleapis.com/token",
    })


# 生成一次供全文件复用（2048 位 RSA，约几百 ms）
_SA_A = _gen_sa_json("proj-a")
_SA_B = _gen_sa_json("proj-b")


class TestValidateCredentials:
    def test_valid_json(self):
        r = validate_sa_credentials(_SA_A)
        assert r["valid"] is True
        assert r["project_id"] == "proj-a"

    def test_bad_json(self):
        r = validate_sa_credentials("{ not json")
        assert r["valid"] is False
        assert "JSON" in r["message"]

    def test_not_object(self):
        r = validate_sa_credentials("[1,2]")
        assert r["valid"] is False

    def test_missing_private_key(self):
        info = json.loads(_SA_A)
        del info["private_key"]
        r = validate_sa_credentials(json.dumps(info))
        assert r["valid"] is False
        assert "private_key" in r["message"]

    def test_wrong_type(self):
        info = json.loads(_SA_A)
        info["type"] = "user"
        r = validate_sa_credentials(json.dumps(info))
        assert r["valid"] is False

    def test_credentials_constructible(self):
        creds = build_sa_credentials(_SA_A)
        assert creds.service_account_email == f"sa-proj-a@proj.iam.gserviceaccount.com"
        assert creds.project_id == "proj-a"
        # 服务账号凭证天然带 refresh（OAuth2 JWT-bearer 换 token 的官方实现）
        assert hasattr(creds, "refresh")


class TestEnvFallback:
    def test_inline_json(self, monkeypatch):
        import runtime_state
        monkeypatch.setattr(app_config, "VERTEX_SA_JSON", _SA_A)
        monkeypatch.setattr(app_config, "VERTEX_SA_FILE", None)
        accs = runtime_state._sa_env_fallback()
        assert len(accs) == 1
        assert accs[0]["project_id"] == "proj-a"
        assert accs[0]["location"] == "global"

    def test_file_path(self, monkeypatch, tmp_path):
        import runtime_state
        f = tmp_path / "sa.json"
        f.write_text(_SA_A, encoding="utf-8")
        monkeypatch.setattr(app_config, "VERTEX_SA_JSON", None)
        monkeypatch.setattr(app_config, "VERTEX_SA_FILE", str(f))
        accs = runtime_state._sa_env_fallback()
        assert len(accs) == 1
        assert accs[0]["project_id"] == "proj-a"

    def test_missing_file_returns_empty(self, monkeypatch, tmp_path):
        import runtime_state
        monkeypatch.setattr(app_config, "VERTEX_SA_JSON", None)
        monkeypatch.setattr(app_config, "VERTEX_SA_FILE", str(tmp_path / "nope.json"))
        assert runtime_state._sa_env_fallback() == []

    def test_invalid_json_returns_empty(self, monkeypatch):
        import runtime_state
        monkeypatch.setattr(app_config, "VERTEX_SA_JSON", "xxx")
        monkeypatch.setattr(app_config, "VERTEX_SA_FILE", None)
        assert runtime_state._sa_env_fallback() == []


class TestSaState:
    """runtime_state 的服务账号持久化 + 请求级快照（不串号）。"""

    @pytest.fixture
    def fresh(self, tmp_path, monkeypatch):
        import runtime_state
        monkeypatch.setattr(runtime_state, "STATE_FILE", str(tmp_path / "web_state.json"))
        return runtime_state.AppState()

    def test_set_and_get(self, fresh):
        fresh.set_sa_accounts([{"project_id": "p1", "location": "us-central1", "sa_json": _SA_A}])
        got = fresh.get_sa_accounts()
        assert len(got) == 1 and got[0]["location"] == "us-central1"
        # 控制台列表存在时环境变量兜底不生效
        assert fresh.get_sa_accounts_console() == got

    def test_clear_falls_back_to_env(self, fresh, monkeypatch):
        monkeypatch.setattr(app_config, "VERTEX_SA_JSON", _SA_A)
        fresh.set_sa_accounts([{"project_id": "p1", "sa_json": _SA_B}])
        assert fresh.get_sa_accounts()[0]["sa_json"] == _SA_B
        fresh.set_sa_accounts([])   # 清空 → 回落环境变量
        assert fresh.get_sa_accounts()[0]["sa_json"] == _SA_A

    def test_snapshot_same_account_within_request(self, fresh):
        """同一请求内重复读取恒返回同一账号（contextvar 快照，重试/流式不串号）。"""
        fresh.set_sa_accounts([
            {"project_id": "p1", "sa_json": _SA_A},
            {"project_id": "p2", "sa_json": _SA_B},
        ])
        first = fresh.get_current_sa_account()
        second = fresh.get_current_sa_account()
        assert first == second
        assert first[2] in (_SA_A, _SA_B)

    def test_roundrobin_rotates_across_requests(self, fresh):
        import runtime_state
        fresh.set_sa_accounts([
            {"project_id": "p1", "sa_json": _SA_A},
            {"project_id": "p2", "sa_json": _SA_B},
        ])
        fresh.update_settings({"roundrobin": True})
        got = set()
        for _ in range(4):
            runtime_state._current_sa_account.set(None)   # 模拟新请求
            got.add(fresh.get_current_sa_account()[0])
        assert got == {"p1", "p2"}

    def test_no_account_returns_none(self, fresh):
        assert fresh.get_current_sa_account() == (None, None, None)

    def test_vertex_strategy_valid(self, fresh):
        assert fresh.set_channel_strategy("vertex") is True
        assert fresh.get_channel_strategy() == "vertex"

    def test_hybrid_channels_default_and_sanitize(self, fresh):
        assert fresh.get_hybrid_channels() == ["express", "cookie"]
        fresh.update_settings({"hybrid_channels": ["vertex", "express"]})
        assert fresh.get_hybrid_channels() == ["vertex", "express"]
        fresh.update_settings({"hybrid_channels": ["bogus", "express"]})
        assert fresh.get_hybrid_channels() == ["express"]   # 非法键被剔除

    def test_channel_retry_override(self, fresh):
        assert fresh.get_channel_retry("express") is None
        fresh.update_settings({"channel_retry_overrides": {"express": 3, "cookie": None}})
        assert fresh.get_channel_retry("express") == 3
        assert fresh.get_channel_retry("cookie") is None
        assert fresh.get_channel_retry("vertex") is None


class TestSaClientCache:
    def _clear(self):
        with _SA_CLIENT_CACHE_LOCK:
            _SA_CLIENT_CACHE.clear()

    def test_same_account_reuses_same_client(self):
        self._clear()
        a = _get_cached_sa_client(_SA_A, "proj-a", "global", None)
        b = _get_cached_sa_client(_SA_A, "proj-a", "global", None)
        assert a is b

    def test_different_sa_different_client(self):
        self._clear()
        a = _get_cached_sa_client(_SA_A, "proj-a", "global", None)
        b = _get_cached_sa_client(_SA_B, "proj-b", "global", None)
        assert b is not a

    def test_location_in_cache_key(self):
        self._clear()
        a = _get_cached_sa_client(_SA_A, "proj-a", "global", None)
        b = _get_cached_sa_client(_SA_A, "proj-a", "us-central1", None)
        assert b is not a

    def test_paygo_headers_in_cache_key(self):
        self._clear()
        headers = {"X-Vertex-AI-LLM-Request-Type": "shared",
                   "X-Vertex-AI-LLM-Shared-Request-Type": "priority"}
        a = _get_cached_sa_client(_SA_A, "proj-a", "global", None)
        b = _get_cached_sa_client(_SA_A, "proj-a", "global", headers)
        assert b is not a


class TestResolveClient:
    def test_channel_name_is_vertex(self):
        """每通道独立重试靠 channel_name 区分：SA 通道必须是 vertex，不是继承的 express。"""
        up = ServiceAccountUpstream()
        assert up.channel_name == "vertex"
        from upstreams.express_sdk import ExpressSDKUpstream
        assert ExpressSDKUpstream().channel_name == "express"

    def test_no_account_returns_error(self, monkeypatch):
        monkeypatch.setattr(app_state, "get_current_sa_account", lambda: (None, None, None))
        up = ServiceAccountUpstream()
        resolved = up._resolve_client(object(), "gemini-3.6-flash", {})
        assert "error" in resolved
        assert resolved["error"].status_code == 401

    def test_with_account_returns_client_and_bare_model(self, monkeypatch):
        monkeypatch.setattr(app_state, "get_current_sa_account",
                            lambda: ("proj-a", "global", _SA_A))
        up = ServiceAccountUpstream()
        resolved = up._resolve_client(object(), "gemini-3.6-flash", {"paygo_tier": "off"})
        assert "error" not in resolved
        assert resolved["model_to_call"] == "gemini-3.6-flash"   # 裸名，路径由 SDK 拼
        assert resolved["priority_paygo"] is False
        assert resolved["fallback_model"] is None

    def test_global_priority_headers_applied(self, monkeypatch):
        monkeypatch.setattr(app_state, "get_current_sa_account",
                            lambda: ("proj-a", "global", _SA_A))
        up = ServiceAccountUpstream()
        resolved = up._resolve_client(object(), "gemini-3.6-flash", {"paygo_tier": "priority"})
        assert resolved["priority_paygo"] is True
        # 客户端应带上 Priority 层级头
        headers = resolved["client"]._api_client._http_options.headers or {}
        assert headers.get("X-Vertex-AI-LLM-Shared-Request-Type") == "priority"

    async def test_pay_prefix_stripped_before_chat(self, monkeypatch):
        """[PAY] 旧前缀被剥掉后再走本通道（B4 兼容入口）。"""
        from models import OpenAIRequest
        from upstreams.express_sdk import ExpressSDKUpstream
        captured = {}

        async def fake_super(self, request_obj, fastapi_request, failover_mode=False):
            captured["model"] = request_obj.model
            return "ok"

        monkeypatch.setattr(ExpressSDKUpstream, "chat_completions", fake_super)
        up = ServiceAccountUpstream()
        req = OpenAIRequest(model="[PAY] gemini-3.6-flash",
                            messages=[{"role": "user", "content": "hi"}])
        result = await up.chat_completions(req, object())
        assert result == "ok"
        assert captured["model"] == "gemini-3.6-flash"

        # 不带 [PAY] 前缀的模型名原样透传
        req2 = OpenAIRequest(model="gemini-3.6-flash",
                             messages=[{"role": "user", "content": "hi"}])
        await up.chat_completions(req2, object())
        assert captured["model"] == "gemini-3.6-flash"
