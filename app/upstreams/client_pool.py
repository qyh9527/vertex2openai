"""统一 Client/Transport Pool（进阶报告 P1-⑤）。

Express 与服务账号（SA）两条通道原先各自维护一份"缓存字典 + 失败计数 +
锁"的同构代码（express_sdk._CLIENT_CACHE / service_account._SA_CLIENT_CACHE）；
本模块把两者合并成一套池，语义保持不变：

  - key 由调用方构造（仍是通道各自的 (secret, base_url/project/location,
    priority_paygo, headers) 元组——P2 candidate planner 接管后再收成
    credential fingerprint + transport profile，本次不动 key 形状避免行为漂移）；
  - client_reuse=False 时每请求新建，不入池；
  - 连接级失败计数达 client_reuse_evict_threshold 自动淘汰；
  - kind="evict" 立即淘汰（安全拦截已不再走此路径，P1-4）。

既有引用零改动：express_sdk / service_account 模块级名
（_CLIENT_CACHE、_SA_CLIENT_CACHE 等）继续指向本池的别名，
既有测试（直接操作这些字典 clear）不受影响。
"""

import threading

import config as app_config
from runtime_state import app_state


class ClientPool:
    """google-genai Client 复用池（GIL 下 dict get/set 原子，异步协程并发安全）。

    注意使用 threading.Lock（不可重入）：get_or_create / on_failure 自带加锁，
    调用方绝不能在持有 _lock 时再调它们（Express 适配层曾因此死锁过一次）。
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._cache: dict = {}      # key -> Client
        self._failures: dict = {}   # key -> 连续连接级失败次数

    # ---------- 兼容层：既有代码把池当 dict 用 ----------

    @property
    def data(self) -> dict:
        return self._cache

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
            self._failures.clear()

    # ---------- 核心接口（自带加锁，调用方不得嵌套持锁）----------

    def get_or_create(self, key, factory):
        """取（必要时创建并缓存）Client；client_reuse 关闭时直接新建不入池。

        返回 (client, reused)：reused=True 表示命中既有缓存（供来源日志区分
        new/reused）。factory(key) 在锁内调用：首个请求创建，后续命中复用。
        """
        if not app_state.get_setting("client_reuse", True):
            return factory(None), False
        with self._lock:
            client = self._cache.get(key)
            reused = client is not None
            if client is None:
                client = factory(key)
                self._cache[key] = client
            return client, reused

    def on_failure(self, key, kind: str = "conn", reason: str = "",
                   log_prefix: str = "[Client 复用]") -> None:
        """复用 Client 的一次失败上报（由 Client 上挂的 _vertex_on_failure 回调触发）。

        kind="evict"：立即淘汰（真正的连接/会话状态损坏类硬错误）；
        kind="conn"：连接级失败计数，达到 client_reuse_evict_threshold 才淘汰。
        """
        if key is None:
            return   # 非复用 Client（client_reuse=False 新建）无缓存可淘汰
        if kind == "evict":
            with self._lock:
                self._cache.pop(key, None)
                self._failures.pop(key, None)
            print(f"⚠️ {log_prefix} 已立即舍弃缓存 Client（{reason or '硬错误'}），"
                  f"下次请求将重建连接池（状态=evicted）。")
            return
        try:
            threshold = int(app_state.get_setting(
                "client_reuse_evict_threshold",
                app_config.DEFAULT_SETTINGS["client_reuse_evict_threshold"]))
        except (TypeError, ValueError):
            threshold = app_config.DEFAULT_SETTINGS["client_reuse_evict_threshold"]
        if threshold <= 0:
            return
        with self._lock:
            cnt = self._failures.get(key, 0) + 1
            if cnt >= threshold:
                self._cache.pop(key, None)
                self._failures.pop(key, None)
                print(f"⚠️ {log_prefix} 缓存 Client 连续 {cnt} 次连接级失败，已淘汰"
                      f"（状态=evicted），下次请求将重建连接池。")
            else:
                self._failures[key] = cnt

    def stats(self) -> dict:
        """池观测指标（进阶报告 §4.3：命中/新建/淘汰可从这里扩展）。"""
        with self._lock:
            return {"cached_clients": len(self._cache),
                    "tracked_keys": len(self._failures)}


# 全进程唯一池：Express 与 SA 通道共用（key 各自构造，互不冲突）
client_pool = ClientPool()
