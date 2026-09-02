"""runtime_state 持久化与迁移逻辑测试。

覆盖 PROJECT_SUMMARY 3.1 / 测试记录里的核心行为：
旧 use_web_proxy 布尔迁移、旧布尔接口兼容、非法策略拒绝、深拷贝隔离、原子写。
"""
import json

import pytest

import runtime_state


@pytest.fixture
def state(tmp_path, monkeypatch):
    """每个测试独立 STATE_FILE + 全新 AppState 实例，互不污染。"""
    monkeypatch.setattr(runtime_state, "STATE_FILE", str(tmp_path / "web_state.json"))
    return runtime_state.AppState()


def _seed(tmp_path, data: dict):
    f = tmp_path / "web_state.json"
    f.write_text(json.dumps(data), encoding="utf-8")
    return f


class TestUseWebProxyMigration:
    def test_true_migrates_to_cookie(self, tmp_path, monkeypatch):
        _seed(tmp_path, {"use_web_proxy": True})
        monkeypatch.setattr(runtime_state, "STATE_FILE", str(tmp_path / "web_state.json"))
        st = runtime_state.AppState()
        assert st.get_channel_strategy() == "cookie"

    def test_false_migrates_to_express(self, tmp_path, monkeypatch):
        _seed(tmp_path, {"use_web_proxy": False})
        monkeypatch.setattr(runtime_state, "STATE_FILE", str(tmp_path / "web_state.json"))
        st = runtime_state.AppState()
        assert st.get_channel_strategy() == "express"

    def test_existing_strategy_not_touched(self, tmp_path, monkeypatch):
        """已存在 channel_strategy 时不做迁移，旧键原样保留。"""
        _seed(tmp_path, {"channel_strategy": "hybrid", "use_web_proxy": False})
        monkeypatch.setattr(runtime_state, "STATE_FILE", str(tmp_path / "web_state.json"))
        st = runtime_state.AppState()
        assert st.get_channel_strategy() == "hybrid"
        data = json.loads((tmp_path / "web_state.json").read_text(encoding="utf-8"))
        assert "use_web_proxy" in data  # 只读启动不改盘，下次 _save 才持久化迁移

    def test_migration_persisted_on_next_save(self, tmp_path, monkeypatch):
        """迁移在内存生效，落盘发生在下次写操作时（迁移结果被持久化）。"""
        _seed(tmp_path, {"use_web_proxy": True})
        monkeypatch.setattr(runtime_state, "STATE_FILE", str(tmp_path / "web_state.json"))
        st = runtime_state.AppState()
        st.set_google_cookie("cook=1")   # 触发一次落盘
        data = json.loads((tmp_path / "web_state.json").read_text(encoding="utf-8"))
        assert data["channel_strategy"] == "cookie"
        assert "use_web_proxy" not in data


class TestLegacyBooleanInterfaces:
    def test_enable_web_proxy_maps_to_cookie(self, state):
        assert state.is_web_proxy_enabled() is False
        state.enable_web_proxy(True)
        assert state.get_channel_strategy() == "cookie"
        assert state.is_web_proxy_enabled() is True

    def test_disable_web_proxy_maps_to_express(self, state):
        state.enable_web_proxy(True)
        state.enable_web_proxy(False)
        assert state.get_channel_strategy() == "express"
        assert state.is_web_proxy_enabled() is False


class TestChannelStrategy:
    def test_invalid_strategy_rejected(self, state):
        assert state.set_channel_strategy("banana") is False
        assert state.get_channel_strategy() == "express"

    def test_case_and_whitespace_normalized(self, state):
        assert state.set_channel_strategy(" Hybrid ") is True
        assert state.get_channel_strategy() == "hybrid"

    def test_all_valid_strategies(self, state):
        for s in runtime_state.AppState.CHANNEL_STRATEGIES:
            assert state.set_channel_strategy(s) is True
            assert state.get_channel_strategy() == s

    def test_persisted_to_disk(self, state, tmp_path):
        state.set_channel_strategy("hybrid")
        data = json.loads((tmp_path / "web_state.json").read_text(encoding="utf-8"))
        assert data["channel_strategy"] == "hybrid"


class TestSettingsIsolation:
    def test_get_settings_returns_copy(self, state):
        s1 = state.get_settings()
        s1["fake_streaming"] = True
        s1["model_overrides"]["x"] = {"thinking_g3_level": "high"}
        assert state.get_settings()["fake_streaming"] is False
        assert "x" not in state.get_settings()["model_overrides"]

    def test_model_overrides_are_copied(self, state):
        state.set_model_override("gemini-3.6-flash", {"thinking_g3_level": "low"})
        ov = state.get_model_overrides()
        ov["gemini-3.6-flash"]["thinking_g3_level"] = "high"
        assert state.get_model_overrides()["gemini-3.6-flash"]["thinking_g3_level"] == "low"


class TestModelOverrides:
    def test_unknown_keys_filtered(self, state):
        state.set_model_override("gemini-x", {"thinking_g3_level": "high", "bogus_key": 1})
        ov = state.get_model_overrides()
        assert ov["gemini-x"] == {"thinking_g3_level": "high"}

    def test_effective_settings_merges_override_over_global(self, state):
        state.set_model_override("gemini-x", {"thinking_g3_level": "low"})
        eff = state.get_effective_settings("gemini-x")
        assert eff["thinking_g3_level"] == "low"
        assert eff["retry_max"] == 10          # 全局默认仍在
        # 返回的是深拷贝：改 eff 不影响内部状态
        eff["thinking_g3_level"] = "high"
        assert state.get_effective_settings("gemini-x")["thinking_g3_level"] == "low"

    def test_effective_settings_other_model_untouched(self, state):
        state.set_model_override("gemini-x", {"thinking_g3_level": "low"})
        eff = state.get_effective_settings("gemini-y")
        assert eff["thinking_g3_level"] == ""   # 全局默认

    def test_clear_override(self, state):
        state.set_model_override("gemini-x", {"thinking_g3_level": "high"})
        assert state.clear_model_override("gemini-x") is True
        assert state.clear_model_override("gemini-x") is False  # 已清除
        assert state.get_model_overrides() == {}


class TestUpdateSettings:
    def test_known_keys_only(self, state):
        state.update_settings({"retry_max": 5, "bogus_key": 1})
        assert state.get_setting("retry_max") == 5
        assert state.get_setting("bogus_key", "default") == "default"

    def test_model_overrides_not_updated_via_update_settings(self, state):
        state.update_settings({"model_overrides": {"gemini-x": {"thinking_g3_level": "high"}}})
        assert state.get_model_overrides() == {}

    def test_hybrid_dispatch_mode_defaults_to_priority(self, state):
        assert state.get_hybrid_dispatch_mode() == "priority"

    def test_hybrid_dispatch_mode_roundtrips_and_invalid_falls_back(self, state):
        state.update_settings({"hybrid_dispatch_mode": "random"})
        assert state.get_hybrid_dispatch_mode() == "random"
        state.update_settings({"hybrid_dispatch_mode": "unexpected"})
        assert state.get_hybrid_dispatch_mode() == "priority"


class TestPersistence:
    def test_atomic_save_no_temp_left(self, state, tmp_path):
        state.set_channel_strategy("hybrid")
        assert (tmp_path / "web_state.json").exists()
        assert list(tmp_path.glob(".web_state-*")) == []   # 临时文件已清理
        data = json.loads((tmp_path / "web_state.json").read_text(encoding="utf-8"))
        assert data["channel_strategy"] == "hybrid"

    def test_corrupt_file_falls_back_to_memory(self, tmp_path, monkeypatch):
        (tmp_path / "web_state.json").write_text("{ not valid json", encoding="utf-8")
        monkeypatch.setattr(runtime_state, "STATE_FILE", str(tmp_path / "web_state.json"))
        st = runtime_state.AppState()   # 不应抛异常
        assert st.get_channel_strategy() == "express"

    def test_roundtrip_reload(self, tmp_path, monkeypatch):
        monkeypatch.setattr(runtime_state, "STATE_FILE", str(tmp_path / "web_state.json"))
        st = runtime_state.AppState()
        st.set_google_cookie("PSID=abc")
        st.set_project_id("proj-1")
        st2 = runtime_state.AppState()   # 模拟重启：新实例从磁盘读回
        assert st2.get_google_cookie() == "PSID=abc"
        assert st2.get_project_id() == "proj-1"
