"""可配置输入搬运的纯逻辑测试。"""

from input_relay import (
    MODE_ALWAYS,
    MODE_FAKE_STREAM_ONLY,
    InputRelayConfig,
    RelayBlockStreamStripper,
    apply_input_relay,
    get_input_relay_config,
    input_relay_active_for_stream,
    strip_generated_relay_blocks,
)
from models import OpenAIMessage


CONFIG = InputRelayConfig(
    tag="relay_input",
    placeholder="请基于已附加的输入继续。",
    mode=MODE_ALWAYS,
    strip_generated=True,
)


def _messages(user_text: str):
    return [
        OpenAIMessage(role="system", content="系统规则"),
        OpenAIMessage(role="assistant", content="上一轮助手回复。"),
        OpenAIMessage(role="user", content=user_text),
    ]


class TestInputRelayConfig:
    def test_no_defaults_means_inactive(self):
        config, note = get_input_relay_config({"input_relay_mode": "always"})
        assert config is None
        assert "为空" in note

    def test_tag_must_be_a_safe_element_name(self):
        config, note = get_input_relay_config({
            "input_relay_mode": "always",
            "input_relay_tag": "<not-a-name>",
            "input_relay_placeholder": "继续",
        })
        assert config is None
        assert "无效" in note

    def test_complete_config_has_no_implicit_values(self):
        config, note = get_input_relay_config({
            "input_relay_mode": "always",
            "input_relay_tag": "client_input",
            "input_relay_placeholder": "下一步",
            "input_relay_strip_generated": True,
        })
        assert note is None
        assert config == InputRelayConfig("client_input", "下一步", MODE_ALWAYS, True)

    def test_three_modes_gate_the_actual_stream_type(self):
        fake_only = InputRelayConfig("client_input", "下一步", MODE_FAKE_STREAM_ONLY)
        always = InputRelayConfig("client_input", "下一步", MODE_ALWAYS)
        assert not input_relay_active_for_stream(fake_only, False)
        assert input_relay_active_for_stream(fake_only, True)
        assert input_relay_active_for_stream(fake_only, False, treat_fake_only_as_always=True)
        assert input_relay_active_for_stream(always, False)
        assert input_relay_active_for_stream(always, True)


class TestApplyInputRelay:
    def test_moves_payload_to_previous_assistant_tail(self):
        messages = _messages("\n <RELAY_input>\n用户真实输入\n</relay_INPUT> \n")

        updated, notes = apply_input_relay(messages, CONFIG)

        assert updated[1].content == "上一轮助手回复。\n\n<relay_input>\n用户真实输入\n</relay_input>"
        assert updated[2].content == "请基于已附加的输入继续。"
        assert "已从最新 user 消息提取" in notes[0]
        assert messages[1].content == "上一轮助手回复。"  # 不原地污染故障转移重试用对象

    def test_requires_tag_to_wrap_entire_latest_user_message(self):
        messages = _messages("前缀 <relay_input>真实输入</relay_input>")

        updated, notes = apply_input_relay(messages, CONFIG)

        assert updated is messages
        assert updated[-1].content == messages[-1].content
        assert "必须包住整条" in notes[0]

    def test_uses_latest_user_even_if_client_appends_assistant_prefill(self):
        messages = _messages("<relay_input>真实输入</relay_input>")
        messages.append(OpenAIMessage(role="assistant", content="客户端预填充"))

        updated, _ = apply_input_relay(messages, CONFIG)

        assert updated[1].content.endswith("</relay_input>")
        assert updated[2].content == CONFIG.placeholder
        assert updated[3].content == "客户端预填充"

    def test_skips_when_no_ordinary_assistant_exists(self):
        messages = [OpenAIMessage(role="user", content="<relay_input>真实输入</relay_input>")]

        updated, notes = apply_input_relay(messages, CONFIG)

        assert updated is messages
        assert "未找到可追加" in notes[0]

    def test_skips_multimodal_user_to_avoid_losing_parts(self):
        messages = _messages("<relay_input>旧输入</relay_input>")
        messages[-1] = OpenAIMessage(role="user", content=[{"type": "text", "text": "输入"}])

        updated, notes = apply_input_relay(messages, CONFIG)

        assert updated is messages
        assert "不是纯文本" in notes[0]


class TestStripGeneratedRelayBlocks:
    def test_non_stream_removes_only_closed_configured_blocks(self):
        text = "正文前<relay_input>应移除</relay_input>正文后<other>保留</other>"
        assert strip_generated_relay_blocks(text, "relay_input") == "正文前正文后<other>保留</other>"

    def test_stream_handles_tags_split_across_deltas(self):
        stripper = RelayBlockStreamStripper("relay_input")
        chunks = ["正文<rel", "ay_input>要移", "除</relay_in", "put>尾巴"]

        output = "".join(stripper.feed(chunk) for chunk in chunks) + stripper.flush()

        assert output == "正文尾巴"

    def test_stream_fails_open_for_unclosed_block(self):
        stripper = RelayBlockStreamStripper("relay_input")
        output = stripper.feed("正文<relay_input>半截") + stripper.flush()
        assert output == "正文<relay_input>半截"

    def test_stream_keeps_similarly_named_tag(self):
        stripper = RelayBlockStreamStripper("relay_input")
        output = stripper.feed("<relay_input_extra>保留</relay_input_extra>") + stripper.flush()
        assert output == "<relay_input_extra>保留</relay_input_extra>"
