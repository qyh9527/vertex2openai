"""顶部输入注入的纯逻辑测试。"""

import top_input_injection as top_input
from input_relay import MODE_ALWAYS, MODE_FAKE_STREAM_ONLY
from models import OpenAIMessage
from top_input_injection import (
    INPUT_PLACEHOLDER,
    TopInputInjectionConfig,
    TopInputPlan,
    apply_top_input_injection,
    get_top_input_injection_config,
    top_input_injection_active_for_stream,
)


PLANS = (
    TopInputPlan("p-system", "系统方案", "system", "顶部规则：" + INPUT_PLACEHOLDER),
    TopInputPlan("p-ai", "AI 方案", "assistant", "AI 看到的原始输入"),
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

    def test_same_three_state_gating(self):
        fake_only = TopInputInjectionConfig(MODE_FAKE_STREAM_ONLY, PLANS, "p-system", False)
        always = TopInputInjectionConfig(MODE_ALWAYS, PLANS, "p-system", False)
        assert not top_input_injection_active_for_stream(fake_only, False)
        assert top_input_injection_active_for_stream(fake_only, True)
        assert top_input_injection_active_for_stream(always, False)


class TestApplyTopInputInjection:
    def test_inserts_selected_template_at_absolute_front_without_replacing_user(self):
        config = TopInputInjectionConfig(MODE_ALWAYS, PLANS, "p-system", False)
        messages = _messages("真实输入")

        updated, notes = apply_top_input_injection(messages, config)

        assert updated[0].role == "system"
        assert updated[0].content == "顶部规则：真实输入"
        assert updated[1:] == messages
        assert "系统方案" in notes[0]

    def test_appends_input_when_template_has_no_placeholder(self):
        config = TopInputInjectionConfig(MODE_ALWAYS, PLANS, "p-ai", False)

        updated, _ = apply_top_input_injection(_messages("真实输入"), config)

        assert updated[0].role == "assistant"
        assert updated[0].content == "AI 看到的原始输入\n\n真实输入"

    def test_random_mode_selects_a_persisted_plan(self, monkeypatch):
        config = TopInputInjectionConfig(MODE_ALWAYS, PLANS, "p-system", True)
        monkeypatch.setattr(top_input.secrets, "choice", lambda options: options[1])

        updated, notes = apply_top_input_injection(_messages("真实输入"), config)

        assert updated[0].role == "assistant"
        assert "AI 方案" in notes[0]
        assert "随机" in notes[0]

    def test_multimodal_latest_user_is_not_rewritten(self):
        config = TopInputInjectionConfig(MODE_ALWAYS, PLANS, "p-system", False)
        messages = _messages()
        messages[-1] = OpenAIMessage(role="user", content=[{"type": "text", "text": "多段"}])

        updated, notes = apply_top_input_injection(messages, config)

        assert updated is messages
        assert "不是非空纯文本" in notes[0]
