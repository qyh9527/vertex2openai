"""模型定价测试：官方按量价匹配（前缀/前缀剥离/未知）与美刀成本估算（含缓存折扣）。"""
from model_pricing import get_model_price, estimate_cost, MODEL_PRICING


class TestGetModelPrice:
    def test_exact_and_prefix_match(self):
        assert get_model_price("gemini-3.6-flash") == (0.75, 3.75)
        assert get_model_price("gemini-3.5-flash") == (1.50, 9.00)
        assert get_model_price("gemini-2.5-pro") == (1.25, 10.00)

    def test_lite_priority_over_base(self):
        """3.5-flash-lite 必须命中 lite 价，而不是被 3.5-flash 前缀吃掉。"""
        assert get_model_price("gemini-3.5-flash-lite") == (0.30, 2.50)
        assert get_model_price("gemini-3.1-flash-lite") == (0.25, 1.50)

    def test_proxy_prefix_stripped(self):
        assert get_model_price("fake-gemini-3.6-flash") == (0.75, 3.75)
        assert get_model_price("gemini-3.6-flash-search") == (0.75, 3.75)
        assert get_model_price("fake-gemini-3.6-flash-search") == (0.75, 3.75)

    def test_unknown_returns_none(self):
        assert get_model_price("gemini-9.9-flash") is None
        assert get_model_price("") is None
        assert get_model_price(None) is None


class TestEstimateCost:
    def test_plain_input_output(self):
        # 1M prompt + 1M completion on 3.6-flash (0.75 / 3.75)
        c = estimate_cost("gemini-3.6-flash", 1_000_000, 1_000_000, 0)
        assert abs(c - (0.75 + 3.75)) < 1e-6

    def test_cached_input_discount(self):
        # 全部命中缓存：输入只按 10% 计（90% 折扣）
        c = estimate_cost("gemini-3.6-flash", 1_000_000, 0, 1_000_000)
        assert abs(c - 0.75 * 0.1) < 1e-6
        # 部分命中
        c2 = estimate_cost("gemini-3.6-flash", 1_000_000, 1_000_000, 500_000)
        assert abs(c2 - (0.375 + 0.0375 + 3.75)) < 1e-6

    def test_unknown_or_zero_returns_none(self):
        assert estimate_cost("gemini-9.9-flash", 100, 100) is None
        assert estimate_cost("gemini-3.6-flash", 0, 0) is None

    def test_cached_clamped_to_prompt(self):
        c = estimate_cost("gemini-3.6-flash", 1_000_000, 0, 2_000_000)
        assert abs(c - 0.075) < 1e-6


class TestTierFactors:
    """官方三档：Priority = Standard × 1.8，Flex = Standard × 0.5；auto 语义 = Priority。"""

    def test_priority_factor(self):
        std = estimate_cost("gemini-3.6-flash", 1_000_000, 1_000_000, 0, tier="standard")
        pri = estimate_cost("gemini-3.6-flash", 1_000_000, 1_000_000, 0, tier="priority")
        assert abs(pri - std * 1.8) < 1e-9

    def test_auto_uses_priority(self):
        std = estimate_cost("gemini-3.6-flash", 1_000_000, 0, 0, tier="standard")
        auto = estimate_cost("gemini-3.6-flash", 1_000_000, 0, 0, tier="auto")
        assert abs(auto - std * 1.8) < 1e-9

    def test_flex_factor(self):
        std = estimate_cost("gemini-3.6-flash", 1_000_000, 1_000_000, 0, tier="standard")
        flex = estimate_cost("gemini-3.6-flash", 1_000_000, 1_000_000, 0, tier="flex")
        assert abs(flex - std * 0.5) < 1e-9

    def test_flex_not_for_2x_falls_back_to_standard(self):
        """2.x 模型不支持 flex（请求侧自动降级），计费同样降级为 standard。"""
        std = estimate_cost("gemini-2.5-flash", 1_000_000, 1_000_000, 0, tier="standard")
        flex = estimate_cost("gemini-2.5-flash", 1_000_000, 1_000_000, 0, tier="flex")
        assert abs(flex - std) < 1e-9

    def test_off_uses_standard(self):
        std = estimate_cost("gemini-2.5-flash", 1_000_000, 0, 0, tier="standard")
        off = estimate_cost("gemini-2.5-flash", 1_000_000, 0, 0, tier="off")
        assert abs(off - std) < 1e-9

    def test_priority_cached_discount_still_applies(self):
        pri = estimate_cost("gemini-3.6-flash", 1_000_000, 0, 1_000_000, tier="priority")
        # Priority input $1.35，命中按 10% → 0.135
        assert abs(pri - 0.135) < 1e-9
