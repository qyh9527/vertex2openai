"""远程模型获取改造测试：不再自动获取 / 磁盘缓存 / 手动刷新 / 自定义模型合并持久化。"""
import pytest

import model_loader as ml
import runtime_state


@pytest.fixture
def clear_cache():
    ml._model_cache = None
    yield
    ml._model_cache = None


class TestDiskCache:
    def test_save_and_load_roundtrip(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ml, "_MODELS_DISK_FILE", str(tmp_path / "models.json"))
        ml._save_disk_cache({"models": ["gemini-a", "gemini-b"]})
        got = ml._load_disk_cache()
        assert got == {"models": ["gemini-a", "gemini-b"]}

    def test_load_missing_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ml, "_MODELS_DISK_FILE", str(tmp_path / "nope.json"))
        assert ml._load_disk_cache() is None


class TestGetModelsConfig:
    async def test_no_auto_fetch_uses_disk(self, tmp_path, monkeypatch, clear_cache):
        """get_models_config 不再远程获取：磁盘缓存优先，fetch 不应被调用。"""
        monkeypatch.setattr(ml, "_MODELS_DISK_FILE", str(tmp_path / "models.json"))
        ml._save_disk_cache({"models": ["gemini-disk"]})
        called = {"n": 0}

        async def _should_not_call():
            called["n"] += 1
            return {"models": []}

        monkeypatch.setattr(ml, "fetch_and_parse_models_config", _should_not_call)
        cfg = await ml.get_models_config()
        assert cfg == {"models": ["gemini-disk"]}
        assert called["n"] == 0

    async def test_no_cache_falls_back_local(self, tmp_path, monkeypatch, clear_cache):
        monkeypatch.setattr(ml, "_MODELS_DISK_FILE", str(tmp_path / "nope.json"))
        monkeypatch.setattr(ml, "_load_local_models_config", lambda: {"models": ["gemini-local"]})
        cfg = await ml.get_models_config()
        assert cfg == {"models": ["gemini-local"]}

    async def test_builtin_fallback_never_empty(self, tmp_path, monkeypatch, clear_cache):
        """磁盘与本地都空时使用内置默认列表，/v1/models 绝不返回空。"""
        monkeypatch.setattr(ml, "_MODELS_DISK_FILE", str(tmp_path / "nope.json"))
        monkeypatch.setattr(ml, "_load_local_models_config", lambda: {"models": []})
        cfg = await ml.get_models_config()
        assert len(cfg["models"]) > 0
        assert "gemini-3.7-flash" in cfg["models"]


class TestListModelsRoute:
    async def test_v1_models_returns_list_without_credentials(self, tmp_path, monkeypatch,
                                                              clear_cache):
        """/v1/models 无凭证也返回模型列表（回归：曾因移除自动刷新时误删 current_time
        导致 NameError 500，客户端以为接口被删）。"""
        from routes.models_api import list_models
        result = await list_models(fastapi_request=object(), api_key="x")
        assert result["object"] == "list"
        assert len(result["data"]) > 0
        ids = [m["id"] for m in result["data"]]
        assert "gemini-3.7-flash" in ids
        assert any(m["created"] > 0 for m in result["data"])   # current_time 必须可用


class TestRefresh:
    async def test_refresh_writes_disk(self, tmp_path, monkeypatch, clear_cache):
        """控制台「获取远程模型」：拉取并持久化到磁盘，更新内存缓存。"""
        monkeypatch.setattr(ml, "_MODELS_DISK_FILE", str(tmp_path / "models.json"))

        async def fake_fetch():
            return {"models": ["gemini-new"]}

        monkeypatch.setattr(ml, "fetch_and_parse_models_config", fake_fetch)
        ok = await ml.refresh_models_config_cache()
        assert ok is True
        assert ml._model_cache == {"models": ["gemini-new"]}
        assert ml._load_disk_cache() == {"models": ["gemini-new"]}

    async def test_refresh_failure_returns_false(self, tmp_path, monkeypatch, clear_cache):
        monkeypatch.setattr(ml, "_MODELS_DISK_FILE", str(tmp_path / "models.json"))

        async def fake_fetch():
            return None

        monkeypatch.setattr(ml, "fetch_and_parse_models_config", fake_fetch)
        assert await ml.refresh_models_config_cache() is False


class TestMergeCustom:
    async def test_get_express_models_merges_custom(self, tmp_path, monkeypatch, clear_cache):
        """get_express_models = 配置模型 + 自定义模型（去重保序）。"""
        monkeypatch.setattr(ml, "_MODELS_DISK_FILE", str(tmp_path / "models.json"))
        ml._save_disk_cache({"models": ["gemini-a", "gemini-b"]})
        monkeypatch.setattr(runtime_state, "STATE_FILE", str(tmp_path / "web_state.json"))
        st = runtime_state.AppState()
        st.set_custom_models(["gemini-b", "gemini-custom"])
        monkeypatch.setattr(runtime_state, "app_state", st)
        models = await ml.get_express_models()
        assert models == ["gemini-a", "gemini-b", "gemini-custom"]

    async def test_custom_persists_across_reload(self, tmp_path, monkeypatch):
        """自定义模型列表落盘：新实例（模拟容器重启）读回。"""
        monkeypatch.setattr(runtime_state, "STATE_FILE", str(tmp_path / "web_state.json"))
        st = runtime_state.AppState()
        st.set_custom_models(["gemini-x", "  ", "gemini-y", "gemini-x"])
        assert st.get_custom_models() == ["gemini-x", "gemini-y"]
        st2 = runtime_state.AppState()
        assert st2.get_custom_models() == ["gemini-x", "gemini-y"]
