import threading
import time

import config as app_config
from runtime_state import app_state


class UpstreamUnstartedError(Exception):
    """未向客户端发出任何数据时的上游失败信号（hybrid 模式下可切换通道）。

    两条通道的流式 generator 内部已做过各自的重试；重试耗尽后若仍
    未出流（未 yield 过任何有效数据），就抛这个异常交给路由层做故障转移。
    一旦出流（已向客户端吐过 chunk），upstream 只会发错误 chunk 收尾，
    绝不允许抛此异常 —— SSE 流一旦开始就不能中途切换上游。
    """


class ChannelBreaker:
    """通道熔断器（内存版）。

    hybrid 模式下用于"限流风暴"保护：某通道连续失败超过阈值后进入冷却，
    冷却期间路由层直接跳过该通道，避免反复撞墙；冷却结束自动恢复探测。
    """

    def __init__(self, failure_threshold: int = 3, cooldown_seconds: float = 60.0):
        self._lock = threading.Lock()
        self._state: dict = {}  # channel -> {"failures": int, "cooldown_until": float}
        self.failure_threshold = failure_threshold
        self.cooldown_seconds = cooldown_seconds

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

    def is_cooling(self, channel: str) -> bool:
        """该通道是否处于冷却期（冷却期直接跳过，不再发起请求）。"""
        with self._lock:
            rec = self._state.get(channel)
            if not rec:
                return False
            if rec["failures"] < self._threshold():
                return False
            return rec["cooldown_until"] > time.time()

    def report_success(self, channel: str) -> None:
        """通道成功：清零连续失败计数，立即解除冷却。"""
        with self._lock:
            if self._state.pop(channel, None):
                print(f"✅ [熔断器] {channel} 通道恢复，连续失败计数已清零。")

    def report_failure(self, channel: str) -> None:
        """通道失败：连续失败计数 +1，达到阈值即进入冷却。"""
        now = time.time()
        with self._lock:
            rec = self._state.get(channel, {"failures": 0, "cooldown_until": 0.0})
            rec["failures"] += 1
            if rec["failures"] >= self._threshold():
                rec["cooldown_until"] = now + self._cooldown()
                print(f"⚠️ [熔断器] {channel} 通道连续失败 {rec['failures']} 次，"
                      f"进入 {self._cooldown():.0f}s 冷却，期间自动切换其它通道。")
            self._state[channel] = rec

    def status(self) -> dict:
        """控制台用：各通道健康状态（cooling / failures / cooldown_remaining）。"""
        now = time.time()
        with self._lock:
            out = {}
            for channel, rec in self._state.items():
                remain = max(0.0, rec["cooldown_until"] - now)
                out[channel] = {
                    "cooling": remain > 0 and rec["failures"] >= self._threshold(),
                    "failures": rec["failures"],
                    "cooldown_remaining": round(remain, 1),
                }
            return out


# 单例：全进程共享一份熔断状态
breaker = ChannelBreaker()
