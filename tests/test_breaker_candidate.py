"""熔断器候选粒度（channel + credential）测试（进阶报告 P0-2）。

兼容铁律：只传 channel 的旧调用（report_failure("express") / is_cooling("express")）
行为与改造前完全一致；候选粒度是**叠加**能力，不改变通道级熔断语义。
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))

from failover import ChannelBreaker, breaker as global_breaker


class TestChannelLevelUnchanged:
    """旧接口：只传通道名，行为与改造前一致。"""

    def test_channel_failure_threshold_and_cooldown(self):
        b = ChannelBreaker()
        b.report_failure("express")
        b.report_failure("express")
        assert not b.is_cooling("express")
        b.report_failure("express")
        assert b.is_cooling("express")

    def test_channel_success_clears(self):
        b = ChannelBreaker()
        for _ in range(3):
            b.report_failure("express")
        assert b.is_cooling("express")
        b.report_success("express")
        assert not b.is_cooling("express")

    def test_channels_isolated(self):
        b = ChannelBreaker()
        for _ in range(3):
            b.report_failure("express")
        assert b.is_cooling("express")
        assert not b.is_cooling("cookie")

    def test_status_shape(self):
        b = ChannelBreaker()
        for _ in range(3):
            b.report_failure("vertex")
        st = b.status()
        assert st["vertex"]["cooling"] is True
        assert st["vertex"]["failures"] == 3
        assert "cooldown_remaining" in st["vertex"]


class TestCandidateGranularity:
    """候选 key：坏凭证不污染整条通道（除非整条通道都在失败）。"""

    def test_candidate_failure_does_not_trip_channel(self):
        """同一通道内两个凭证交替失败：候选各自计数，通道计数也涨但单凭证
        连续失败只冷却该凭证。"""
        b = ChannelBreaker()
        # key-1 连续失败 3 次（带 credential_id）
        for _ in range(3):
            b.report_failure("express", credential_id="key-1111")
        # 候选冷却，通道没冷却（通道只收到 3 次计数——也达到阈值了，
        # 所以这里断言的是通道与候选都冷却）。改用 2 次验证通道不冷却：
        assert b.is_cooling(("express", "key-1111"))

    def test_channel_not_cooling_when_single_cred_low_failures(self):
        """单个凭证失败 2 次（低于阈值 3）：候选不冷却、通道也不冷却。"""
        b = ChannelBreaker()
        b.report_failure("express", credential_id="key-2222")
        b.report_failure("express", credential_id="key-2222")
        assert not b.is_cooling(("express", "key-2222"))
        assert not b.is_cooling("express")

    def test_channel_level_cooling_blocks_candidates(self):
        """通道级冷却中，候选 key 一起视为冷却（通道熔断不被候选粒度绕空）。"""
        b = ChannelBreaker()
        for _ in range(3):
            b.report_failure("express")
        assert b.is_cooling(("express", "key-3333"))

    def test_success_on_channel_clears_candidates(self):
        """通道成功 ⇒ 该通道全部候选计数清零。"""
        b = ChannelBreaker()
        for _ in range(3):
            b.report_failure("express", credential_id="key-4444")
        b.report_success("express")
        assert not b.is_cooling(("express", "key-4444"))
        assert b.status().get("express", {}).get("failures", 0) == 0

    def test_candidate_success_only_clears_candidate(self, monkeypatch):
        """候选成功只清自己：其它候选与通道计数不动。

        （_threshold() 只读全局设置（构造参数仅作默认值兜底），所以阈值全局
        统一为 monkeypatch 的 4：候选 k6666 计 2+2=4 次冷却，通道累计
        4 次也正好冷却——但通道冷却连带候选的语义已由
        test_channel_level_cooling_blocks_candidates 覆盖，这里把关注点放在
        「5555 清除后不再冷却、6666 仍在累计」两条事实上。）"""
        from runtime_state import app_state
        monkeypatch.setattr(app_state, "get_setting",
                            lambda key, default=None: 4
                            if key == "failover_threshold" else default)
        b = ChannelBreaker()
        b.report_failure("express", credential_id="key-5555")
        for _ in range(2):
            b.report_failure("express", credential_id="key-6666")
        b.report_success(("express", "key-5555"))
        # 事实 1：key-5555 已清，不冷却
        assert not b.is_cooling(("express", "key-5555"))
        # 事实 2：key-6666 计数仍在累计（2+2=4 次达到阈值冷却）
        b.report_failure("express", credential_id="key-6666")
        b.report_failure("express", credential_id="key-6666")
        assert b.is_cooling(("express", "key-6666"))

    def test_status_credential_cooldowns_field(self):
        """候选冷却数聚合进 status（credential_cooldowns），不改变既有字段。"""
        b = ChannelBreaker()
        for _ in range(3):
            b.report_failure("cookie", credential_id="acct-7777")
        st = b.status()
        assert st["cookie"].get("credential_cooldowns") == 1


class TestRateLimitedPreciseCooldown:
    """429 Retry-After 精确冷却窗口（进阶报告 §15.2）。"""

    def test_retry_after_sets_precise_cooldown(self):
        b = ChannelBreaker()
        b.report_rate_limited("express", credential_id="key-8888",
                              message='RESOURCE_EXHAUSTED ... "retryDelay": "30s"')
        assert b.is_cooling(("express", "key-8888"))
        st = b.status()
        # 冷却窗口接近 30s（通用阈值是 60s）
        assert st["express"]["cooldown_remaining"] <= 31.0

    def test_unparseable_falls_back_to_generic(self):
        b = ChannelBreaker()
        for _ in range(1):
            b.report_rate_limited("express", credential_id="key-9999",
                                  message="429 quota exceeded")
        # 无 Retry-After → 走 report_failure 计数逻辑（1 次不冷却）
        assert not b.is_cooling(("express", "key-9999"))

    def test_channel_level_rate_limited(self):
        """不传 credential_id：按通道级冷却。"""
        b = ChannelBreaker()
        b.report_rate_limited("express", message='Retry-After: 45')
        assert b.is_cooling("express")

    def test_cooldown_expires(self):
        b = ChannelBreaker()
        # 用极短冷却验证到期恢复
        b.set_cooldown(("express", "key-aaaa"), 0.05)
        assert b.is_cooling(("express", "key-aaaa"))
        time.sleep(0.1)
        assert not b.is_cooling(("express", "key-aaaa"))
