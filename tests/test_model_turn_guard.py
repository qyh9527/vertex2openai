"""3.x 轮次兜底：转换后以 model 轮结尾的请求必须补 user 推动语。

背景（真机日志）：末尾 assistant 带 tool_calls / 仅 reasoning_content /
仅思考签名 / role=model 等形状会漏过 apply_prefill_compat（它只处理末尾
assistant 纯文本），但 create_gemini_prompt / _convert_messages_to_contents
转换后仍以 model 轮收尾，Gemini 3.x 直接 400 "Requests ending with a model
turn are not supported"。兜底在两个转换出口处按模型能力补一句极短 user。
"""

from message_processing import create_gemini_prompt, MODEL_TURN_GUARD_NUDGE
from models import OpenAIMessage


def _text_of(content) -> str:
    parts = getattr(content, "parts", []) or []
    for p in parts:
        t = getattr(p, "text", None)
        if isinstance(t, str) and t:
            return t
    return ""


# ---- create_gemini_prompt（Express / SA 共用管线）----

def test_g3_model_turn_tail_gets_user_nudge():
    """role=model 结尾（3.x）：出口补一句 user 推动语，末轮不再是 model。"""
    msgs = [
        OpenAIMessage(role="user", content="你好"),
        OpenAIMessage(role="model", content="这是我上一句"),
    ]
    out = create_gemini_prompt(msgs, model_name="gemini-3.6-flash")
    assert out, "转换结果不能为空"
    assert out[-1].role != "model", "3.x 上绝不能以 model 轮收尾"
    assert _text_of(out[-1]) == MODEL_TURN_GUARD_NUDGE


def test_g3_dangling_tool_calls_tail_gets_user_nudge():
    """末尾 assistant 带悬空 tool_calls（预填充兼容有意放过）：同样兜底。"""
    msgs = [
        OpenAIMessage(role="user", content="查一下天气"),
        OpenAIMessage(role="assistant", content=None, tool_calls=[{
            "id": "call_1", "type": "function",
            "function": {"name": "get_weather", "arguments": "{}"},
        }]),
    ]
    out = create_gemini_prompt(msgs, model_name="gemini-3.6-flash")
    assert out[-1].role == "user"
    assert _text_of(out[-1]) == MODEL_TURN_GUARD_NUDGE


def test_g3_reasoning_only_tail_gets_user_nudge():
    """末尾 assistant 只有 reasoning_content、无文本（_is_empty_message 视为空）：同样兜底。"""
    msgs = [
        OpenAIMessage(role="user", content="继续"),
        OpenAIMessage(role="assistant", content=None, reasoning_content="思考中"),
    ]
    out = create_gemini_prompt(msgs, model_name="gemini-3.6-flash")
    assert out[-1].role == "user"


def test_user_tail_untouched():
    """正常 user 结尾：兜底必须零操作（不追加任何消息）。"""
    msgs = [
        OpenAIMessage(role="user", content="你好"),
        OpenAIMessage(role="assistant", content="你好呀"),
        OpenAIMessage(role="user", content="讲个笑话"),
    ]
    out = create_gemini_prompt(msgs, model_name="gemini-3.6-flash")
    assert out[-1].role == "user"
    assert _text_of(out[-1]) == "讲个笑话"


def test_g25_model_tail_left_alone():
    """2.5 及更早允许 model 结尾（原生预填充续写）：兜底不掺和。"""
    msgs = [
        OpenAIMessage(role="user", content="你好"),
        OpenAIMessage(role="assistant", content="接下去写"),
    ]
    out = create_gemini_prompt(msgs, model_name="gemini-2.5-pro")
    # 2.5 允许 model 结尾：不再追加，末轮保持 model（原生透传续写）
    assert out[-1].role == "model"


def test_keep_turn_prefill_still_works():
    """keep_turn 处理过的请求已经以 user 结尾，兜底不再叠第二条推动语。"""
    msgs = [
        OpenAIMessage(role="user", content="你好"),
        OpenAIMessage(role="assistant", content="我写到一半"),
        OpenAIMessage(role="user", content="[继续] 接着写"),
    ]
    out = create_gemini_prompt(msgs, model_name="gemini-3.6-flash")
    assert out[-1].role == "user"
    assert _text_of(out[-1]) == "[继续] 接着写"
    model_tails = [m for m in out if m.role == "model"]
    assert model_tails, "中间的 model 轮应保留"
