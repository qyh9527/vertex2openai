"""Express 通道模型名解析与 location 钉定测试（PROJECT_SUMMARY 3.5 / README location 章节）。"""
import pytest

import config as app_config
from upstreams.express_sdk import _normalize_model_name, resolve_express_model_path
from runtime_state import app_state


class TestNormalizeModelName:
    def test_plain_model(self):
        assert _normalize_model_name("gemini-3.6-flash") == ("gemini-3.6-flash", False, False, None)

    def test_fake_prefix(self):
        assert _normalize_model_name("fake-gemini-3.6-flash") == ("gemini-3.6-flash", False, True, None)

    def test_search_suffix(self):
        assert _normalize_model_name("gemini-3.6-flash-search") == ("gemini-3.6-flash", True, False, None)

    def test_fake_and_search_combined(self):
        assert _normalize_model_name("fake-gemini-3.6-flash-search") == ("gemini-3.6-flash", True, True, None)

    def test_fake_after_legacy_prefix(self):
        assert _normalize_model_name("[EXPRESS] fake-gemini-x") == ("gemini-x", False, True, None)

    def test_legacy_prefix_after_fake(self):
        assert _normalize_model_name("fake-[EXPRESS] gemini-x") == ("gemini-x", False, True, None)

    def test_double_fake_still_flagged(self):
        assert _normalize_model_name("fake-fake-gemini-x") == ("gemini-x", False, True, None)

    def test_legacy_pay_rejected(self):
        result = _normalize_model_name("[PAY] gemini-x")
        assert result[1] is False and result[2] is False
        assert "移除" in result[3]

    def test_openai_direct_rejected(self):
        result = _normalize_model_name("gemini-x-openai")
        assert "移除" in result[3]

    def test_unchanged_model_passthrough(self):
        assert _normalize_model_name("gemini-2.5-pro") == ("gemini-2.5-pro", False, False, None)


class TestResolveModelPath:
    def test_no_location_returns_bare(self):
        assert resolve_express_model_path("gemini-x", {}) == "gemini-x"

    def test_empty_location_returns_bare(self):
        assert resolve_express_model_path("gemini-x", {"express_location": ""}) == "gemini-x"

    def test_location_without_project_returns_bare(self, monkeypatch):
        monkeypatch.setattr(app_config, "GOOGLE_PROJECT_ID", None)
        monkeypatch.setattr(app_state, "get_project_id", lambda: "")
        assert resolve_express_model_path("gemini-x", {"express_location": "global"}) == "gemini-x"

    def test_location_with_env_project(self, monkeypatch):
        monkeypatch.setattr(app_config, "GOOGLE_PROJECT_ID", "proj-env")
        assert resolve_express_model_path("gemini-x", {"express_location": "global"}) == (
            "projects/proj-env/locations/global/publishers/google/models/gemini-x")

    def test_location_with_state_project(self, monkeypatch):
        monkeypatch.setattr(app_config, "GOOGLE_PROJECT_ID", None)
        monkeypatch.setattr(app_state, "get_project_id", lambda: "proj-state")
        assert resolve_express_model_path("gemini-x", {"express_location": "us-central1"}) == (
            "projects/proj-state/locations/us-central1/publishers/google/models/gemini-x")

    def test_client_provided_full_path_passthrough(self, monkeypatch):
        monkeypatch.setattr(app_config, "GOOGLE_PROJECT_ID", "proj-env")
        full = "projects/abc/locations/global/publishers/google/models/gemini-x"
        assert resolve_express_model_path(full, {"express_location": "global"}) == full

    def test_fake_prefix_stripped_before_resolve(self, monkeypatch):
        monkeypatch.setattr(app_config, "GOOGLE_PROJECT_ID", "proj-1")
        base, _, is_fake, _ = _normalize_model_name("fake-gemini-x")
        assert is_fake is True
        assert resolve_express_model_path(base, {"express_location": "global"}) == (
            "projects/proj-1/locations/global/publishers/google/models/gemini-x")


class TestClientCache:
    """_get_cached_client 按 (key, base_url, priority_paygo) 复用（3.6 节设计约束）。"""

    def _clear_cache(self):
        import upstreams.express_sdk as sdk
        with sdk._CLIENT_CACHE_LOCK:
            sdk._CLIENT_CACHE.clear()

    def test_same_key_reuses_same_client(self):
        self._clear_cache()
        import upstreams.express_sdk as sdk
        a = sdk._get_cached_client("key1", False)
        b = sdk._get_cached_client("key1", False)
        assert a is b

    def test_different_key_different_client(self):
        self._clear_cache()
        import upstreams.express_sdk as sdk
        a = sdk._get_cached_client("key1", False)
        d = sdk._get_cached_client("key2", False)
        assert d is not a

    def test_priority_paygo_never_shared_with_plain(self):
        """缓存键必须含 priority_paygo（3.6 节第一条设计约束）。"""
        self._clear_cache()
        import upstreams.express_sdk as sdk
        plain = sdk._get_cached_client("key1", False)
        paygo = sdk._get_cached_client("key1", True)
        assert paygo is not plain
