"""ChannelBreaker 熔断器逻辑测试（PROJECT_SUMMARY 3.1 核心行为）。"""
import time as time_module

import pytest

import failover


@pytest.fixture
def breaker():
    return failover.ChannelBreaker()


class TestFailureThreshold:
    def test_below_threshold_not_cooling(self, breaker):
        breaker.report_failure("express")
        breaker.report_failure("express")
        assert not breaker.is_cooling("express")

    def test_threshold_enters_cooldown(self, breaker):
        for _ in range(3):
            breaker.report_failure("express")
        assert breaker.is_cooling("express")

    def test_extra_failures_keep_cooling(self, breaker):
        for _ in range(5):
            breaker.report_failure("express")
        assert breaker.is_cooling("express")


class TestCooldown:
    def test_cooldown_expires(self, breaker, monkeypatch):
        clock = {"now": 1000.0}
        monkeypatch.setattr(time_module, "time", lambda: clock["now"])
        for _ in range(3):
            breaker.report_failure("express")
        assert breaker.is_cooling("express")
        clock["now"] += 61   # 冷却 60s 后恢复
        assert not breaker.is_cooling("express")

    def test_success_clears_immediately(self, breaker):
        for _ in range(2):
            breaker.report_failure("express")
        breaker.report_success("express")
        assert not breaker.is_cooling("express")

    def test_success_on_cooling_channel_clears(self, breaker):
        for _ in range(3):
            breaker.report_failure("express")
        assert breaker.is_cooling("express")
        breaker.report_success("express")
        assert not breaker.is_cooling("express")


class TestChannelIndependence:
    def test_channels_isolated(self, breaker):
        for _ in range(3):
            breaker.report_failure("express")
        assert breaker.is_cooling("express")
        assert not breaker.is_cooling("cookie")

    def test_success_on_untracked_channel_is_noop(self, breaker):
        breaker.report_success("cookie")   # 无记录通道成功：不报错、不影响其它通道
        assert not breaker.is_cooling("cookie")


class TestStatus:
    def test_status_shape(self, breaker):
        for _ in range(3):
            breaker.report_failure("express")
        st = breaker.status()
        assert st["express"]["cooling"] is True
        assert st["express"]["failures"] == 3
        assert st["express"]["cooldown_remaining"] > 0

    def test_status_empty_initially(self, breaker):
        assert breaker.status() == {}

    def test_status_after_success_empty(self, breaker):
        for _ in range(3):
            breaker.report_failure("express")
        breaker.report_success("express")
        assert breaker.status() == {}


class TestSettingsDrivenThreshold:
    def test_threshold_from_app_state(self, monkeypatch):
        from runtime_state import app_state

        def _fake_get_setting(key, default=None):
            return 1 if key == "failover_threshold" else default

        monkeypatch.setattr(app_state, "get_setting", _fake_get_setting)
        b = failover.ChannelBreaker()
        b.report_failure("express")
        assert b.is_cooling("express")

    def test_invalid_threshold_falls_back(self, monkeypatch):
        from runtime_state import app_state

        def _fake_get_setting(key, default=None):
            return "not-a-number"

        monkeypatch.setattr(app_state, "get_setting", _fake_get_setting)
        b = failover.ChannelBreaker()
        for _ in range(3):
            b.report_failure("express")
        assert b.is_cooling("express")   # 退回内置默认 3
