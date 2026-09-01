"""统一 ClientPool 测试（进阶报告 P1-⑤）。

Express 与 SA 原先各自维护缓存字典 + 失败计数 + 锁，已合并为
upstreams.client_pool 的全进程唯一池。本测试锁住池的独立语义与
Express/SA 两适配层与池的协作行为。
"""
import pytest

from runtime_state import app_state
from upstreams.client_pool import ClientPool


class TestPoolSemantics:
    def _pool(self):
        return ClientPool()

    def test_get_or_create_reuses(self):
        p = self._pool()
        a, r1 = p.get_or_create("k", lambda key: object())
        b, r2 = p.get_or_create("k", lambda key: object())
        assert a is b
        assert r1 is False and r2 is True   # 首次新建 / 二次命中

    def test_different_keys_isolated(self):
        p = self._pool()
        a, _ = p.get_or_create("k1", lambda key: object())
        b, _ = p.get_or_create("k2", lambda key: object())
        assert a is not b

    def test_clear_resets(self):
        p = self._pool()
        p.get_or_create("k", lambda key: object())
        p.clear()
        assert p.stats()["cached_clients"] == 0

    def test_conn_failure_threshold_evicts(self, monkeypatch):
        p = self._pool()
        monkeypatch.setattr(app_state, "get_setting",
                            lambda key, default=None: 2
                            if key == "client_reuse_evict_threshold" else default)
        obj, _ = p.get_or_create("k", lambda key: object())
        p.on_failure("k", kind="conn")
        fresh, _ = p.get_or_create("k", lambda key: object())   # 计数 1 < 2，仍复用
        assert fresh is obj
        p.on_failure("k", kind="conn")   # 计数 2 达阈值
        after, _ = p.get_or_create("k", lambda key: object())
        assert after is not obj

    def test_evict_kind_immediate(self):
        p = self._pool()
        obj, _ = p.get_or_create("k", lambda key: object())
        p.on_failure("k", kind="evict", reason="会话状态损坏")
        after, _ = p.get_or_create("k", lambda key: object())
        assert after is not obj

    def test_client_reuse_disabled_skips_pool(self, monkeypatch):
        """client_reuse=False：factory 直接新建、不入池（key=None 不挂回调语义在适配层）。"""
        p = self._pool()
        monkeypatch.setattr(app_state, "get_setting",
                            lambda key, default=None: False
                            if key == "client_reuse" else default)
        c, reused = p.get_or_create("k", lambda key: object())
        assert reused is False
        assert p.stats()["cached_clients"] == 0
        # factory 收到 key=None（适配层据此不挂失败回调）
        seen = []
        p.get_or_create("k", lambda key: seen.append(key) or object())
        assert seen == [None]

    def test_threshold_zero_never_evicts(self, monkeypatch):
        p = self._pool()
        monkeypatch.setattr(app_state, "get_setting",
                            lambda key, default=None: 0
                            if key == "client_reuse_evict_threshold" else default)
        obj, _ = p.get_or_create("k", lambda key: object())
        for _ in range(5):
            p.on_failure("k", kind="conn")
        after, _ = p.get_or_create("k", lambda key: object())
        assert after is obj


class TestAdapterPoolUnification:
    """Express 与 SA 适配层共用同一个池实例（合并的核心事实）。"""

    def test_same_pool_instance(self):
        import upstreams.express_sdk as sdk
        import upstreams.service_account as sa
        import upstreams.client_pool as cp
        assert sdk._POOL is cp.client_pool
        assert sa._POOL is cp.client_pool

    def test_express_and_sa_keys_coexist(self):
        """两条通道的 key 在同一池中互不冲突（Express 的 key 含 api_key 字符串，
        SA 的 key 含 sa_json 字符串，元组内容天然不同）。

        SA 凭证构造需要真实可解析的 RSA 私钥——复用 test_service_account 的
        生成器（同款桩，避免每个文件独立生成 RSA 的几百 ms 开销，genai.Client
        构造是惰性的不会发起网络请求）。"""
        from test_service_account import _gen_sa_json
        import upstreams.express_sdk as sdk
        import upstreams.service_account as sa
        sdk._clear_client_cache()
        try:
            ex = sdk._get_cached_client("key-express", False)   # 适配层返回 Client 单对象
            sa_client = sa._get_cached_sa_client(_gen_sa_json("proj-x"), "proj-x", "global", None)
            assert sa_client is not ex
            assert ex is sdk._get_cached_client("key-express", False)
        finally:
            sdk._clear_client_cache()
