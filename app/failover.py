import threading
import time

import config as app_config
import outcome
from runtime_state import app_state


def _chan(channel: str) -> str:
    """熔断日志的通道显示名（与路由层共用同一显示来源，P0-4）。"""
    try:
        from api_helpers import channel_display_name
        return channel_display_name(channel)
    except Exception:
        return channel


def _cred(credential_id: str) -> str:
    """熔断/冷却日志的凭证标识（脱敏：只显示末 4 位）。"""
    cid = str(credential_id or "")
    return f"（凭证 …{cid[-4:]}）" if len(cid) > 4 else (f"（凭证 {cid}）" if cid else "")


class UpstreamUnstartedError(Exception):
    """未向客户端发出任何数据时的上游失败信号（hybrid 模式下可切换通道）。

    两条通道的流式 generator 内部已做过各自的重试；重试耗尽后若仍
    未出流（未 yield 过任何有效数据），就抛这个异常交给路由层做故障转移。
    一旦出流（已向客户端吐过 chunk），upstream 只会发错误 chunk 收尾，
    绝不允许抛此异常 —— SSE 流一旦开始就不能中途切换上游。
    """


class ChannelBreaker:
    """通道+凭证熔断器（内存版，进阶报告 P0-2：从 channel 粒度升级候选粒度）。

    hybrid 模式下用于"限流风暴"保护：某通道/凭证连续失败超过阈值后进入冷却，
    冷却期间路由层直接跳过，避免反复撞墙；冷却结束自动恢复探测。

    key 粒度分两层：
      - 通道级 key（"express"）：与改造前完全一致的既有行为——单凭证部署时
        语义不变；旧调用 report_failure(channel) / is_cooling(channel) 继续有效。
      - 候选级 key（("express", credential_id)）：凭证在请求内被选中后追加记录，
        一个坏账号/坏 Key 的连续失败不再把整条通道拖进冷却；但通道级仍在计数，
        「整条通道全坏」（所有凭证都失败）时依旧熔断保护。
    """

    def __init__(self, failure_threshold: int = 3, cooldown_seconds: float = 60.0):
        self._lock = threading.Lock()
        self._state: dict = {}  # key -> {"failures": int, "cooldown_until": float}
        self.failure_threshold = failure_threshold
        self.cooldown_seconds = cooldown_seconds

    @staticmethod
    def _candidate_key(channel: str, credential_id) -> tuple:
        return (channel, str(credential_id))

    def _threshold(self) -> int:
        try:
            return max(1, int(app_state.get_setting(
                "failover_threshold", self.failure_threshold)))
        except (TypeError, ValueError):
            return self.failure_threshold

    def _cooldown(self) -> float:
        try:
            return max(0.0, float(app_state.get_setting(
                "failover_cooldown_seconds", self.cooldown_seconds)))
        except (TypeError, ValueError):
            return self.cooldown_seconds

    # ---------- 冷却窗口（429 Retry-After 精确化，进阶报告 §15.2）----------

    def set_cooldown(self, key, seconds: float, reason: str = "") -> None:
        """把某个 key（通道或候选）显式置入冷却（如 429 带 Retry-After 的精确窗口）。"""
        if seconds <= 0:
            return
        now = time.time()
        with self._lock:
            rec = self._state.get(key, {"failures": 0, "cooldown_until": 0.0})
            rec["cooldown_until"] = max(rec.get("cooldown_until", 0.0), now + seconds)
            if rec["failures"] < self._threshold():
                rec["failures"] = self._threshold()   # 冷却期必须被 is_cooling 命中
            self._state[key] = rec

    def report_rate_limited(self, channel: str, credential_id=None,
                            message: str = "") -> None:
        """429/RESOURCE_EXHAUSTED 专用上报：能解析出 Retry-After/quota reset 就按精确
        时间冷却该候选（解析不出走 report_failure 的通用阈值逻辑）。"""
        seconds = outcome.parse_retry_after(message)
        if seconds is None:
            self.report_failure(channel, credential_id)
            return
        key = self._candidate_key(channel, credential_id) if credential_id \
            else channel
        self.set_cooldown(key, seconds, reason="429 Retry-After")
        print(f"⚠️ [熔断器] {_chan(channel)}{_cred(credential_id or '')} "
              f"收到限流且带精确冷却窗口（{seconds:.0f}s），已按 Retry-After 冷却。")

    # ---------- 既有接口（通道级，行为与改造前一致）----------

    def is_cooling(self, key) -> bool:
        """该 key（通道名或 (channel, credential_id) 元组）是否处于冷却期。

        语义变化（唯一一处）：候选 key 的通道若在通道级冷却中，候选也算冷却
        ——否则通道熔断会被候选粒度绕空。单独查询通道名不受影响。
        """
        with self._lock:
            if isinstance(key, tuple) and len(key) == 2:
                channel, _ = key
                rec = self._state.get(channel)
                if rec and rec["failures"] >= self._threshold() \
                        and rec["cooldown_until"] > time.time():
                    return True
            rec = self._state.get(key)
            if not rec:
                return False
            if rec["failures"] < self._threshold():
                return False
            return rec["cooldown_until"] > time.time()

    def report_success(self, key, credential_id=None) -> None:
        """成功：清零连续失败计数，立即解除冷却。

        通道级调用（既有行为）：清通道计数，同时把该通道下所有候选计数一并清零
        ——通道恢复意味着通道内所有凭证恢复了可用性。
        候选级调用（credential_id 非空）：只清该候选；通道计数不动。
        """
        with self._lock:
            if isinstance(key, tuple) and len(key) == 2:
                if self._state.pop(key, None):
                    print(f"✅ [熔断器] {_chan(key[0])}{_cred(key[1])} 候选恢复，"
                          "连续失败计数已清零。")
                return
            if self._state.pop(key, None):
                print(f"✅ [熔断器] {_chan(key)} 通道恢复，连续失败计数已清零。")
            # 通道成功 ⇒ 清除该通道全部候选记录
            cand_keys = [k for k in self._state if isinstance(k, tuple)
                         and len(k) == 2 and k[0] == key]
            for k in cand_keys:
                self._state.pop(k, None)

    def report_failure(self, key, credential_id=None) -> None:
        """失败：连续失败计数 +1，达到阈值即进入冷却。

        credential_id 非空时同时给候选 key 计数（候选粒度）；
        通道 key 始终计数（保持既有通道级熔断行为）。
        """
        now = time.time()
        targets = [key]
        if credential_id:
            targets.append(self._candidate_key(key, credential_id))
        with self._lock:
            for target in targets:
                rec = self._state.get(target, {"failures": 0, "cooldown_until": 0.0})
                rec["failures"] += 1
                if rec["failures"] >= self._threshold():
                    rec["cooldown_until"] = now + self._cooldown()
                    if isinstance(target, tuple):
                        print(f"⚠️ [熔断器] {_chan(target[0])}{_cred(target[1])} "
                              f"连续失败 {rec['failures']} 次，进入 {self._cooldown():.0f}s "
                              f"冷却，期间优先使用其它凭证/通道。")
                    else:
                        print(f"⚠️ [熔断器] {_chan(target)} 通道连续失败 {rec['failures']} 次，"
                              f"进入 {self._cooldown():.0f}s 冷却，期间自动切换其它通道。")
                self._state[target] = rec

    def status(self) -> dict:
        """控制台用：各通道健康状态（cooling / failures / cooldown_remaining）。

        只输出通道级条目（与改造前形状一致）；候选级条目折叠为通道下的
        credential_cooldowns 计数字段，不改变既有字段。
        """
        now = time.time()
        with self._lock:
            out = {}
            for key, rec in self._state.items():
                if isinstance(key, tuple):
                    continue
                remain = max(0.0, rec["cooldown_until"] - now)
                entry = out.setdefault(key, {
                    "cooling": False, "failures": 0, "cooldown_remaining": 0.0})
                entry["cooling"] = entry["cooling"] or (
                    remain > 0 and rec["failures"] >= self._threshold())
                entry["failures"] = max(entry["failures"], rec["failures"])
                entry["cooldown_remaining"] = max(
                    entry["cooldown_remaining"], round(remain, 1))
            # 候选级：每通道汇总冷却中的凭证数
            cand_cooling: dict = {}
            for key, rec in self._state.items():
                if not isinstance(key, tuple):
                    continue
                if rec["failures"] >= self._threshold() and rec["cooldown_until"] > now:
                    cand_cooling[key[0]] = cand_cooling.get(key[0], 0) + 1
            for ch, n in cand_cooling.items():
                out.setdefault(ch, {
                    "cooling": False, "failures": 0, "cooldown_remaining": 0.0})
                out[ch]["credential_cooldowns"] = n
            return out


# 单例：全进程共享一份熔断状态
breaker = ChannelBreaker()
