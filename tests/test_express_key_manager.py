"""ExpressKeyManager 多 Key 选择逻辑测试（随机 / 轮询 / 刷新）。"""
import pytest

import config as app_config
from express_key_manager import ExpressKeyManager


def _manager(keys):
    m = ExpressKeyManager()
    m.express_keys = list(keys)
    m.round_robin_index = 0
    return m


class TestRoundRobin:
    def test_cycles_through_keys(self):
        m = _manager(["k1", "k2", "k3"])
        picks = [m.get_roundrobin_express_key()[0] for _ in range(5)]
        assert picks == [0, 1, 2, 0, 1]

    def test_returns_key_and_index(self):
        m = _manager(["k1", "k2"])
        idx, key = m.get_roundrobin_express_key()
        assert (idx, key) == (0, "k1")

    def test_no_keys_returns_none(self):
        m = _manager([])
        assert m.get_roundrobin_express_key() is None


class TestRandom:
    def test_covers_all_keys(self):
        m = _manager(["k1", "k2", "k3"])
        seen = {m.get_random_express_key()[0] for _ in range(100)}
        assert seen == {0, 1, 2}

    def test_no_keys_returns_none(self):
        m = _manager([])
        assert m.get_random_express_key() is None


class TestDispatch:
    def test_respects_roundrobin_setting(self, monkeypatch):
        from runtime_state import app_state
        monkeypatch.setattr(app_state, "get_setting", lambda key, default=None: True)
        m = _manager(["k1", "k2", "k3"])
        picks = [m.get_express_api_key()[0] for _ in range(4)]
        assert picks == [0, 1, 2, 0]

    def test_defaults_to_random(self, monkeypatch):
        from runtime_state import app_state
        monkeypatch.setattr(app_state, "get_setting", lambda key, default=None: False)
        m = _manager(["k1", "k2", "k3"])
        seen = {m.get_express_api_key()[0] for _ in range(100)}
        assert seen == {0, 1, 2}


class TestRefresh:
    def test_reloads_from_config_and_resets_index(self, monkeypatch):
        m = _manager(["k1"])
        m.round_robin_index = 5
        monkeypatch.setattr(app_config, "VERTEX_EXPRESS_API_KEY_VAL", ["a", "b", "c"])
        m.refresh_keys()
        assert m.express_keys == ["a", "b", "c"]
        assert m.round_robin_index == 0

    def test_get_total_keys(self):
        m = _manager(["k1", "k2"])
        assert m.get_total_keys() == 2
