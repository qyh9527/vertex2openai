"""PayGo 流量等级头矩阵测试（ST-Vertex-PayGo 方案融合，TODO D 阶段）。"""
import pytest

import config as app_config

from http_options import (
    resolve_paygo_headers, paygo_timeout, resolve_paygo_bundle, is_flex_supported,
    FLEX_TIMEOUT_SECONDS, PRIORITY_PAYGO_HEADERS,
)


class TestResolvePaygoHeaders:
    def test_off_always_empty(self):
        assert resolve_paygo_headers("off", False, True) == ({}, [])
        assert resolve_paygo_headers("off", True, True) == ({}, [])

    def test_auto_global_priority(self):
        h, w = resolve_paygo_headers("auto", False, True)
        assert w == []
        assert h == PRIORITY_PAYGO_HEADERS

    def test_auto_non_global_empty(self):
        assert resolve_paygo_headers("auto", False, False) == ({}, [])

    def test_standard_paygo_only_shared(self):
        h, _ = resolve_paygo_headers("standard", True, True)
        assert h == {"X-Vertex-AI-LLM-Request-Type": "shared"}

    def test_standard_without_paygo_only_empty(self):
        assert resolve_paygo_headers("standard", False, True) == ({}, [])

    def test_flex_global(self):
        h, w = resolve_paygo_headers("flex", False, True)
        assert w == []
        assert h["X-Vertex-AI-LLM-Request-Type"] == "shared"
        assert h["X-Vertex-AI-LLM-Shared-Request-Type"] == "flex"
        assert h["X-Server-Timeout"] == str(FLEX_TIMEOUT_SECONDS)

    def test_priority_global(self):
        h, _ = resolve_paygo_headers("priority", False, True)
        assert h["X-Vertex-AI-LLM-Shared-Request-Type"] == "priority"

    def test_flex_non_global_downgrades_with_warning(self):
        h, w = resolve_paygo_headers("flex", False, False)
        assert h == {}
        assert len(w) == 1 and "仅对 global" in w[0]

    def test_priority_non_global_downgrades(self):
        h, w = resolve_paygo_headers("priority", False, False)
        assert h == {} and len(w) == 1


class TestPaygoTimeout:
    def test_flex_timeout(self):
        assert paygo_timeout("flex") == FLEX_TIMEOUT_SECONDS
        assert paygo_timeout("priority") is None
        assert paygo_timeout("off") is None
        assert paygo_timeout("auto") is None


class TestResolvePaygoBundle:
    def test_flex_bundle(self):
        headers, timeout, warnings = resolve_paygo_bundle(True, {"paygo_tier": "flex"})
        assert timeout == FLEX_TIMEOUT_SECONDS
        assert headers["X-Vertex-AI-LLM-Shared-Request-Type"] == "flex"
        assert warnings == []

    def test_priority_bundle_no_timeout(self):
        headers, timeout, _ = resolve_paygo_bundle(True, {"paygo_tier": "priority"})
        assert timeout is None
        assert headers["X-Vertex-AI-LLM-Shared-Request-Type"] == "priority"

    def test_off_bundle_empty(self):
        headers, timeout, warnings = resolve_paygo_bundle(True, {"paygo_tier": "off"})
        assert headers == {} and timeout is None and warnings == []

    def test_non_global_flex_downgrade(self):
        headers, timeout, warnings = resolve_paygo_bundle(False, {"paygo_tier": "flex"})
        assert headers == {} and timeout is None and warnings


class TestFlexModelBlacklist:
    """Flex 层级对 gemini-2.x 不支持（真机 400），自动化黑名单降级。"""

    def test_is_flex_supported(self):
        assert is_flex_supported("gemini-3.6-flash") is True
        assert is_flex_supported("gemini-3.7-flash") is True
        assert is_flex_supported("gemini-4.0-flash") is True      # 未来模型前向放行
        assert is_flex_supported("gemini-2.5-flash") is False      # 真机确认不支持
        assert is_flex_supported("gemini-2.5-pro") is False
        assert is_flex_supported("Gemini-2.0-flash") is False      # 大小写不敏感
        assert is_flex_supported("gemini-3-pro-image") is True     # 生图 3.x 放行

    def test_flex_downgraded_for_2x_model(self):
        headers, timeout, warnings = resolve_paygo_bundle(
            True, {"paygo_tier": "flex"}, model_name="gemini-2.5-flash")
        assert headers == {} and timeout is None
        assert len(warnings) == 1 and "不支持 Flex" in warnings[0]

    def test_flex_kept_for_3x_model(self):
        headers, timeout, warnings = resolve_paygo_bundle(
            True, {"paygo_tier": "flex"}, model_name="gemini-3.6-flash")
        assert headers.get("X-Vertex-AI-LLM-Shared-Request-Type") == "flex"
        assert timeout == FLEX_TIMEOUT_SECONDS
        assert warnings == []

    def test_priority_not_affected_by_model(self):
        headers, _, _ = resolve_paygo_bundle(
            True, {"paygo_tier": "priority"}, model_name="gemini-2.5-flash")
        assert headers.get("X-Vertex-AI-LLM-Shared-Request-Type") == "priority"


class TestGetHttpOptionsProxy:
    """PROXY_URL/SSL_CERT_FILE 走预构建 httpx client（genai client_args['proxy'] 的
    mTLS SSLContext 走代理隧道会 ConnectTimeout，见 http_options.py 注释与实测）。"""

    def test_proxy_uses_prebuilt_httpx_client(self, monkeypatch):
        import httpx
        from http_options import get_http_options
        monkeypatch.setattr(app_config, "PROXY_URL", "http://127.0.0.1:7897")
        monkeypatch.setattr(app_config, "SSL_CERT_FILE", None)
        opts = get_http_options()
        assert opts is not None
        assert isinstance(opts.httpx_client, httpx.Client)
        assert isinstance(opts.httpx_async_client, httpx.AsyncClient)
        # 不再用 client_args（genai 会注入 mTLS SSLContext 导致代理隧道超时）
        assert opts.client_args is None and opts.async_client_args is None

    def test_no_proxy_keeps_plain(self, monkeypatch):
        from http_options import get_http_options
        monkeypatch.setattr(app_config, "PROXY_URL", None)
        monkeypatch.setattr(app_config, "SSL_CERT_FILE", None)
        opts = get_http_options()
        assert opts is None or (opts.httpx_client is None and opts.httpx_async_client is None)

    def test_flex_timeout_propagated_to_client(self, monkeypatch):
        import httpx
        from http_options import get_http_options, FLEX_TIMEOUT_SECONDS
        monkeypatch.setattr(app_config, "PROXY_URL", "http://127.0.0.1:7897")
        monkeypatch.setattr(app_config, "SSL_CERT_FILE", None)
        opts = get_http_options(timeout=FLEX_TIMEOUT_SECONDS)
        assert isinstance(opts.httpx_async_client, httpx.AsyncClient)
        assert opts.timeout is None   # 超时已进预构建 client，不再重复设 options.timeout
