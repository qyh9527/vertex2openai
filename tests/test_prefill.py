"""预填充去重测试（流式与非流式）。

README「预填充智能兼容」章节：预填充会拼回输出开头，对模型复述的重叠部分自动去重。
"""
import pytest

from message_processing import strip_prefill_overlap, PrefillDeduper


class TestStripPrefillOverlap:
    def test_full_repeat_removed(self):
        prefill = "你好，这是一段预填充内容"
        assert strip_prefill_overlap(prefill, prefill + "正文来了") == "正文来了"

    def test_tail_overlap_cut(self):
        prefill = "0123456789ABCDEF"
        assert strip_prefill_overlap(prefill, "89ABCDEF正文") == "正文"

    def test_short_overlap_below_minimum_untouched(self):
        # "先思考再回答" 结尾 "再回答" 与输出开头重叠仅 3 字符 < min_overlap=8，不裁
        assert strip_prefill_overlap("先思考再回答", "再回答具体内容") == "再回答具体内容"

    def test_no_overlap_untouched(self):
        assert strip_prefill_overlap("预填充AAA", "完全不同") == "完全不同"

    def test_overlap_below_minimum_untouched(self):
        # 3 字符重叠 < min_overlap=8，不裁（避免误伤正常文本）
        assert strip_prefill_overlap("预填充ABCDEFG", "EFG其它") == "EFG其它"

    def test_empty_inputs(self):
        assert strip_prefill_overlap("", "x") == "x"
        assert strip_prefill_overlap("x", "") == ""
        assert strip_prefill_overlap("", "") == ""


class TestPrefillDeduper:
    def test_full_repeat_streamed(self):
        prefill = "甲乙丙丁戊己庚辛壬癸"
        d = PrefillDeduper(prefill)
        out = "".join(x for x in (d.feed(c) for c in prefill + "正文内容") if x)
        assert out == "正文内容"

    def test_early_release_when_not_ambiguous(self):
        """P2-4：缓冲已不可能成为预填充子串时提前放行，不加首 token 延迟。"""
        d = PrefillDeduper("甲乙丙丁戊")
        out = d.feed("完全不同完全不同")
        assert out == "完全不同完全不同"

    def test_buffered_until_ambiguity_resolved(self):
        d = PrefillDeduper("甲乙丙丁戊")
        assert d.feed("甲") == ""       # 仍可能是预填充开头，继续攒
        assert d.feed("乙丙丁戊正文") == "正文"

    def test_flush_returns_buffered_tail(self):
        d = PrefillDeduper("甲乙丙丁戊")
        assert d.feed("甲") == ""
        assert d.flush() == "甲"
        assert d.flush() == ""          # 已 resolve，不再返回

    def test_no_prefill_passthrough(self):
        d = PrefillDeduper("")
        assert d.feed("直接出") == "直接出"
        assert d.flush() == ""

    def test_short_buffer_released_without_overlap(self):
        d = PrefillDeduper("甲乙丙丁戊")
        assert d.feed("别") == "别"     # "别" 不是预填充子串 → 立即 resolve 放行
        assert d.flush() == ""
