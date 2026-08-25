"""多账号凭证管理测试：Express Key 列表 + 多 Cookie 账号（选择/迁移/持久化）。"""
import pytest

import runtime_state
import config as app_config


@pytest.fixture
def state(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime_state, "STATE_FILE", str(tmp_path / "web_state.json"))
    return runtime_state.AppState()


def _reset_account_selection():
    """清掉请求级账号快照，让下一次 get_current_cookie_account 重新选号。"""
    runtime_state._current_cookie_account.set(None)


class TestExpressKeys:
    def test_unset_returns_none(self, state):
        assert state.get_express_keys() is None

    def test_set_and_get(self, state):
        state.set_express_keys(["k1", "k2"])
        assert state.get_express_keys() == ["k1", "k2"]

    def test_filters_blanks_and_dedupes(self, state):
        state.set_express_keys(["k1", "", "  k2 ", "k1", None])
        assert state.get_express_keys() == ["k1", "k2"]

    def test_empty_list_clears(self, state):
        state.set_express_keys(["k1"])
        assert state.get_express_keys() == ["k1"]
        state.set_express_keys([])
        assert state.get_express_keys() == []   # 显式清空（区别于从未保存的 None）

    def test_persisted_to_disk(self, state, tmp_path):
        state.set_express_keys(["k1"])
        st2 = runtime_state.AppState()
        assert st2.get_express_keys() == ["k1"]


class TestCookieAccounts:
    def test_none_returns_empty(self, state):
        assert state.get_cookie_accounts() == []

    def test_legacy_single_account_migrates(self, state):
        state.set_google_cookie("SAPISID=a; SID=b")
        state.set_project_id("proj-1")
        accounts = state.get_cookie_accounts()
        assert accounts == [{"cookie": "SAPISID=a; SID=b", "project_id": "proj-1"}]

    def test_set_accounts_and_legacy_sync(self, state):
        state.set_cookie_accounts([
            {"cookie": "SAPISID=c1; SID=d1", "project_id": "proj-a"},
            {"cookie": "SAPISID=c2; SID=d2", "project_id": "proj-b"},
        ])
        accounts = state.get_cookie_accounts()
        assert len(accounts) == 2
        # 旧读取接口同步为第一个账号（location 钉定等仍有效）
        assert state.get_google_cookie() == "SAPISID=c1; SID=d1"
        assert state.get_project_id() == "proj-a"

    def test_set_filters_invalid_entries(self, state):
        state.set_cookie_accounts([
            {"cookie": "SAPISID=c1; SID=d1", "project_id": "proj-a"},
            {"cookie": "   ", "project_id": "proj-b"},   # 空 cookie 被剔除
            {"cookie": "SAPISID=c3; SID=d3", "project_id": "  proj-c  "},
        ])
        accounts = state.get_cookie_accounts()
        assert len(accounts) == 2
        assert accounts[1]["project_id"] == "proj-c"   # project 去空白

    def test_clear_accounts(self, state):
        state.set_cookie_accounts([{"cookie": "SAPISID=c1; SID=d1", "project_id": "p"}])
        state.set_cookie_accounts([])
        assert state.get_cookie_accounts() == []
        assert state.get_google_cookie() == ""   # 旧字段同步清空
        assert state.get_project_id() == ""

    def test_persisted_to_disk(self, state, tmp_path):
        state.set_cookie_accounts([{"cookie": "SAPISID=c1; SID=d1", "project_id": "p1"}])
        st2 = runtime_state.AppState()
        assert st2.get_cookie_accounts() == [{"cookie": "SAPISID=c1; SID=d1", "project_id": "p1"}]


class TestCurrentCookieAccount:
    def test_no_accounts_returns_none(self, state):
        _reset_account_selection()
        assert state.get_current_cookie_account() == (None, None)

    def test_single_account_always_returned(self, state):
        state.set_cookie_accounts([{"cookie": "c1", "project_id": "p1"}])
        _reset_account_selection()
        assert state.get_current_cookie_account() == ("c1", "p1")

    def test_roundrobin_cycles(self, state, monkeypatch):
        monkeypatch.setattr(state, "get_setting", lambda key, default=None: True)
        state.set_cookie_accounts([
            {"cookie": "c1", "project_id": "p1"},
            {"cookie": "c2", "project_id": "p2"},
            {"cookie": "c3", "project_id": "p3"},
        ])
        picked = []
        for _ in range(5):
            _reset_account_selection()
            picked.append(state.get_current_cookie_account())
        assert picked == [("c1", "p1"), ("c2", "p2"), ("c3", "p3"), ("c1", "p1"), ("c2", "p2")]

    def test_random_covers_all(self, state, monkeypatch):
        monkeypatch.setattr(state, "get_setting", lambda key, default=None: False)
        state.set_cookie_accounts([
            {"cookie": "c1", "project_id": "p1"},
            {"cookie": "c2", "project_id": "p2"},
            {"cookie": "c3", "project_id": "p3"},
        ])
        picked = set()
        for _ in range(100):
            _reset_account_selection()
            picked.add(state.get_current_cookie_account())
        assert picked == {("c1", "p1"), ("c2", "p2"), ("c3", "p3")}

    def test_snapshot_stable_within_request(self, state):
        """同一请求内多次读取必须返回同一账号（重试/流式路径防串号）。"""
        state.set_cookie_accounts([
            {"cookie": "c1", "project_id": "p1"},
            {"cookie": "c2", "project_id": "p2"},
        ])
        _reset_account_selection()
        first = state.get_current_cookie_account()
        for _ in range(10):
            assert state.get_current_cookie_account() == first   # 快照复用，不轮换


class TestKeyManagerIntegration:
    def test_controlled_keys_priority(self, monkeypatch):
        """控制台列表优先于环境变量。"""
        monkeypatch.setattr(runtime_state.app_state, "get_express_keys", lambda: ["ctrl-1", "ctrl-2"])
        monkeypatch.setattr(app_config, "VERTEX_EXPRESS_API_KEY_VAL", ["env-1"])
        from express_key_manager import ExpressKeyManager
        m = ExpressKeyManager()
        assert m.express_keys == ["ctrl-1", "ctrl-2"]

    def test_env_keys_fallback(self, monkeypatch):
        """未配置控制台列表时回落环境变量。"""
        monkeypatch.setattr(runtime_state.app_state, "get_express_keys", lambda: None)
        monkeypatch.setattr(app_config, "VERTEX_EXPRESS_API_KEY_VAL", ["env-1", "env-2"])
        from express_key_manager import ExpressKeyManager
        m = ExpressKeyManager()
        assert m.express_keys == ["env-1", "env-2"]

    def test_refresh_picks_up_console_change(self, monkeypatch):
        console_keys: list = []
        monkeypatch.setattr(runtime_state.app_state, "get_express_keys",
                            lambda: console_keys or None)
        monkeypatch.setattr(app_config, "VERTEX_EXPRESS_API_KEY_VAL", ["env-1"])
        from express_key_manager import ExpressKeyManager
        m = ExpressKeyManager()
        assert m.express_keys == ["env-1"]
        console_keys[:] = ["ctrl-1"]          # 控制台保存后
        m.refresh_keys()                      # 热生效
        assert m.express_keys == ["ctrl-1"]
        console_keys.clear()                  # 清空控制台列表
        m.refresh_keys()
        assert m.express_keys == ["env-1"]    # 回落环境变量


class TestCookieProxyIntegration:
    def test_get_cookie_string_uses_current_account(self, monkeypatch):
        from upstreams.cookie_proxy import _get_cookie_string, _get_project_id
        monkeypatch.setattr(app_config, "GOOGLE_COOKIE", None)
        monkeypatch.setattr(app_config, "GOOGLE_PROJECT_ID", None)
        monkeypatch.setattr(runtime_state.app_state, "get_current_cookie_account",
                            lambda: ("SAPISID=x; SID=y", "proj-9"))
        assert _get_cookie_string() == "SAPISID=x; SID=y"
        assert _get_project_id() == "proj-9"

    def test_env_vars_fallback_when_no_accounts(self, monkeypatch):
        from upstreams.cookie_proxy import _get_cookie_string, _get_project_id
        monkeypatch.setattr(app_config, "GOOGLE_COOKIE", "ENV_COOKIE")
        monkeypatch.setattr(app_config, "GOOGLE_PROJECT_ID", "env-proj")
        monkeypatch.setattr(runtime_state.app_state, "get_current_cookie_account",
                            lambda: (None, None))
        assert _get_cookie_string() == "ENV_COOKIE"
        assert _get_project_id() == "env-proj"
