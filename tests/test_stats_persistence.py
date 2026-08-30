"""用量统计持久化测试：落盘 / 重启恢复 / 按天记录 / 损坏降级 / 日志历史读取。

不触碰真实网络；用独立的临时 STATE_DIR / stats.json 隔离。
"""
import os

import pytest

import logger as logger_mod


class TestStatsPersistence:
    def _fresh(self, tmp_path, monkeypatch):
        """隔离的 stats 实例 + 独立 stats.json 路径。"""
        monkeypatch.setattr(logger_mod, "STATS_FILE", str(tmp_path / "stats.json"))
        return logger_mod.ProxyStats(), tmp_path

    def test_persist_and_reload(self, tmp_path, monkeypatch):
        """重建容器（新实例）也能看到历史统计。"""
        st, _ = self._fresh(tmp_path, monkeypatch)
        st.add_tokens(100, 50)
        st.add_tokens(200, 100)
        st.increment_total()
        st.increment_total()
        st.add_error()
        st.add_retry()
        st._flush()
        st2 = logger_mod.ProxyStats()   # 模拟重启：从磁盘恢复
        assert st2.prompt_tokens == 300
        assert st2.completion_tokens == 150
        assert st2.total_requests == 2
        assert st2.success_requests == 2
        assert st2.error_requests == 1
        assert st2.retry_counts == 1

    def test_daily_recorded(self, tmp_path, monkeypatch):
        st, _ = self._fresh(tmp_path, monkeypatch)
        st.add_tokens(100, 50)
        st.add_success()
        st._flush()
        d = st.get_json_stats()
        assert len(d["daily"]) == 1
        day = d["daily"][0]
        assert day["prompt_tokens"] == 100
        assert day["completion_tokens"] == 50
        assert day["success"] == 2

    def test_cached_and_cost_persisted(self, tmp_path, monkeypatch):
        """缓存命中 token 与美刀成本也落盘并恢复（模拟重启）。"""
        st, _ = self._fresh(tmp_path, monkeypatch)
        st.add_tokens(1_000_000, 1_000_000, cached=500_000, model="gemini-3.6-flash")
        st._flush()
        st2 = logger_mod.ProxyStats()
        assert st2.cached_prompt_tokens == 500_000
        # 0.375(未命中输入) + 0.0375(命中输入 10%) + 3.75(输出)
        assert abs(st2.cost - (0.375 + 0.0375 + 3.75)) < 1e-6
        day = st2.get_json_stats()["daily"][0]
        assert day["cached_prompt_tokens"] == 500_000
        assert day["cost"] > 0

    def test_unknown_model_no_cost(self, tmp_path, monkeypatch):
        """未知模型不计费（cost 保持 0，不误报）。"""
        st, _ = self._fresh(tmp_path, monkeypatch)
        st.add_tokens(1000, 500, model="gemini-9.9-flash")
        assert st.cost == 0.0

    def test_corrupt_file_fallback(self, tmp_path, monkeypatch):
        monkeypatch.setattr(logger_mod, "STATS_FILE", str(tmp_path / "stats.json"))
        (tmp_path / "stats.json").write_text("{ not json", encoding="utf-8")
        st = logger_mod.ProxyStats()   # 损坏文件降级为空，不影响运行
        assert st.prompt_tokens == 0
        assert st.get_json_stats()["daily"] == []

    def test_read_recent_log_lines(self, tmp_path, monkeypatch):
        monkeypatch.setenv("STATE_DIR", str(tmp_path))
        log_file = tmp_path / "vertex2openai.log"
        lines = [f"2026-08-31 12:00:{i:02d} line {i}" for i in range(10)]
        log_file.write_text("\n".join(lines), encoding="utf-8")
        got = logger_mod.read_recent_log_lines(5)
        assert len(got) == 5
        assert got[-1] == "2026-08-31 12:00:09 line 9"

    def test_read_recent_log_lines_missing(self, tmp_path, monkeypatch):
        monkeypatch.setenv("STATE_DIR", str(tmp_path))
        assert logger_mod.read_recent_log_lines(5) == []
