"""顶部注入的纯逻辑测试。"""

import top_input_injection as top_input
from models import OpenAIMessage
from top_input_injection import (
    MODE_ALWAYS,
    MODE_NON_VERTEX_ONLY,
    TopInputInjectionConfig,
    TopInputPlan,
    apply_top_input_injection,
    get_top_input_injection_config,
    top_input_injection_active_for_channel,
)


PLANS = (
    TopInputPlan("p-system", "系统方案", "system", "顶部规则：固定方案"),
    TopInputPlan("p-ai", "AI 方案", "assistant", "AI 固定前缀"),
)


def _messages(text="用户输入"):
    return [
        OpenAIMessage(role="system", content="原始系统提示"),
        OpenAIMessage(role="assistant", content="既有助手内容"),
        OpenAIMessage(role="user", content=text),
    ]


class TestTopInputInjectionConfig:
    def test_no_plan_is_a_safe_noop(self):
        config, note = get_top_input_injection_config({
            "top_input_injection_mode": "always",
            "top_input_injection_plans": [],
        })
        assert config is None
        assert "没有有效方案" in note

    def test_normalizes_ai_role_and_discards_invalid_plans(self):
        config, note = get_top_input_injection_config({
            "top_input_injection_mode": "always",
            "top_input_injection_plans": [
                {"id": "a", "name": "AI", "role": "ai", "content": "内容"},
                {"id": "bad", "name": "坏", "role": "other", "content": "内容"},
                {"id": "a", "name": "重复", "role": "user", "content": "内容"},
            ],
        })
        assert note is None
        assert len(config.plans) == 1
        assert config.plans[0].role == "assistant"

    def test_legacy_fake_stream_mode_maps_to_non_vertex_route_mode(self):
        config, note = get_top_input_injection_config({
            "top_input_injection_mode": "fake_stream_only",
            "top_input_injection_plans": [
                {"id": "p", "role": "system", "content": "内容"},
            ],
        })
        assert note is None
        assert config.mode == MODE_ALWAYS
        assert config.random_mode == MODE_NON_VERTEX_ONLY

    def test_channel_gating_uses_actual_candidate_route(self):
        config = TopInputInjectionConfig(MODE_ALWAYS, PLANS, "p-system", MODE_NON_VERTEX_ONLY)
        for channel in ("express", "cookie", "vertex"):
            assert top_input_injection_active_for_channel(config, channel)
        assert top_input.random_active_for_channel(config, "express")
        assert top_input.random_active_for_channel(config, "cookie")
        assert not top_input.random_active_for_channel(config, "vertex")


class TestApplyTopInputInjection:
    def test_merges_with_same_role_first_message_without_replacing_user(self):
        config = TopInputInjectionConfig(MODE_ALWAYS, PLANS, "p-system", False)
        messages = _messages("真实输入")

        updated, notes = apply_top_input_injection(messages, config)

        assert len(updated) == len(messages)
        assert updated[0].role == "system"
        assert updated[0].content == "顶部规则：固定方案\n原始系统提示"
        assert updated[1:] == messages[1:]
        assert "系统方案" in notes[0]
        assert "并入" in notes[0]

    def test_merges_model_alias_as_ai_role(self):
        config = TopInputInjectionConfig(MODE_ALWAYS, PLANS, "p-ai", False)
        messages = [
            OpenAIMessage(role="model", content="原始 AI 前缀"),
            OpenAIMessage(role="user", content="真实输入"),
        ]

        updated, _ = apply_top_input_injection(messages, config)

        assert len(updated) == 2
        assert updated[0].role == "model"
        assert updated[0].content == "AI 固定前缀\n原始 AI 前缀"

    def test_injects_exact_plan_content_without_reading_user_input(self):
        config = TopInputInjectionConfig(
            MODE_ALWAYS,
            (TopInputPlan("literal", "原样方案", "assistant", "方案 {{input}} 原样保留"),),
            "literal",
            False,
        )

        updated, _ = apply_top_input_injection(_messages("真实输入"), config)

        assert updated[0].role == "assistant"
        assert updated[0].content == "方案 {{input}} 原样保留"

    def test_injects_even_when_user_message_is_multimodal(self):
        config = TopInputInjectionConfig(MODE_ALWAYS, PLANS, "p-system", False)
        messages = _messages()
        messages[-1] = OpenAIMessage(role="user", content=[{"type": "text", "text": "多段"}])

        updated, _ = apply_top_input_injection(messages, config)

        assert updated[0].content == "顶部规则：固定方案\n原始系统提示"
        assert updated[1:] == messages[1:]

    def test_injects_when_request_has_no_user_message(self):
        config = TopInputInjectionConfig(MODE_ALWAYS, PLANS, "p-ai", False)
        messages = [OpenAIMessage(role="system", content="原始系统提示")]

        updated, _ = apply_top_input_injection(messages, config)

        assert [(message.role, message.content) for message in updated] == [
            ("assistant", "AI 固定前缀"),
            ("system", "原始系统提示"),
        ]

    def test_random_mode_selects_a_persisted_plan(self, monkeypatch):
        config = TopInputInjectionConfig(MODE_ALWAYS, PLANS, "p-system", True)
        monkeypatch.setattr(top_input.secrets, "choice", lambda options: options[1])

        updated, notes = apply_top_input_injection(_messages("真实输入"), config)

        assert updated[0].role == "assistant"
        assert "AI 方案" in notes[0]
        assert "随机" in notes[0]
