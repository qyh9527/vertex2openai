"""防截断合成传输协议（Anti-Truncation，可选单请求启用）

灵感/参考：github.com/Xeltra233/Antigravity-anti-truncation-gateway（Go 反截断网关）。

**为什么需要它**：重提示词场景（SillyTavern 复杂预设/角色卡、超长历史）下，模型回答
经常被 max_output_tokens 提前截断，丢尾巴、破格式。参考网关的思路是——给请求注入一个
请求级唯一的高熵随机名「合成传输工具」（`v2o_emit_<96bit hex>`），指示模型把最终可见
回答放进该工具调用的 `content` 参数输出（Function Call 通道），从而绕开普通文本生成
通道的截断；代理收到响应后解构合成工具调用、还原为标准 `assistant.content`，对下游
完全透明。真实工具调用原样保留、不被吞。

**与本项目管线的对接方式**：
- 注入在 OpenAI 请求层（`express_sdk.ExpressSDKUpstream.chat_completions`）：
  `tools` 追加合成工具声明 + `messages` 末尾追加一条 user 控制消息（末尾 user 恰好也
  满足 Gemini 3.x 拒绝以 assistant 结尾的约束）。
- 解构在 SDK 响应层（`api_helpers.execute_gemini_call`）三条路径：
  非流式/假流式——转换出 OpenAI dict 后剥离合成 tool_call、内容还原；
  真流式——逐 chunk 剥离合成 functionCall part，把 `content` 作为正文 delta 输出。
- 默认不启用 Gemini 的 `stream_function_call_arguments`（流式参数增量），functionCall
  一次带完整参数 dict，解构直白可靠。

**启用方式**：下游请求体带控制台可配置字段（默认 `"anti_truncation": true`）即对本
请求启用；字段名可在控制台自定义。生图/非文本模型自动排除。
"""

import json
import secrets
from typing import Any, Optional

from models import OpenAIMessage

# 合成工具名统一前缀（v2o = vertex2openai；后接 24 hex = 96-bit 随机 nonce）
TOOL_PREFIX = "v2o_emit_"
# 控制台设置键：启用字段名（下游请求体里置 true 即启用）
SETTING_FIELD = "anti_truncation_field"
DEFAULT_FIELD = "anti_truncation"


def generate_synthetic_tool_name(existing_names: Optional[list] = None) -> str:
    """生成请求级唯一合成工具名：96-bit 随机 nonce，天然不与真实工具名冲突。"""
    existing = set(existing_names or [])
    while True:
        name = TOOL_PREFIX + secrets.token_hex(12)  # 12 bytes = 96 bits = 24 hex
        if name not in existing:
            return name


def build_synthetic_tool(tool_name: str) -> dict:
    """OpenAI 格式的合成传输工具声明（参数仅一个 content 字符串）。"""
    return {
        "type": "function",
        "function": {
            "name": tool_name,
            "description": (
                "Use this transport tool exactly once to output the final user-visible "
                "answer. Put the complete answer in `content`. Never wrap genuine tool "
                "calls in it."
            ),
            "parameters": {
                "type": "object",
                "properties": {"content": {"type": "string"}},
                "required": ["content"],
            },
        },
    }


def build_control_message(tool_name: str) -> dict:
    """控制消息：末尾 user 轮次，指示模型走合成工具输出最终回答。"""
    return {
        "role": "user",
        "content": (
            f"Always use tool `{tool_name}` to output your final reply in its `content` "
            f"argument. Do not output anything outside this tool call."
        ),
    }


def get_enabled_field(settings: Optional[dict] = None) -> str:
    """启用字段名（控制台可自定义，默认 anti_truncation）。"""
    if settings is None:
        try:
            from runtime_state import app_state
            import config as app_config
            settings = {"value": app_state.get_setting(
                SETTING_FIELD, app_config.DEFAULT_SETTINGS.get(SETTING_FIELD, DEFAULT_FIELD))}
        except Exception:
            return DEFAULT_FIELD
    val = settings.get(SETTING_FIELD) or settings.get("value")
    return str(val).strip() or DEFAULT_FIELD


def is_enabled_for_request(request_obj: Any, settings: Optional[dict] = None) -> bool:
    """读下游请求体扩展字段：值为 true / "true" 即启用（字段名可自定义）。"""
    field = get_enabled_field(settings)
    extra = getattr(request_obj, "__pydantic_extra__", None) or {}
    val = extra.get(field)
    if val is None:
        val = getattr(request_obj, field, None)  # 兼容未来显式字段
    return val is True or str(val).strip().lower() == "true"


def inject_request(request_obj: Any) -> tuple[Any, str]:
    """给请求注入合成传输工具 + 控制消息。

    返回 (新请求对象, 合成工具名)。调用方需在响应解构时携带该工具名。
    """
    existing_names = []
    for tool in (request_obj.tools or []):
        if isinstance(tool, dict):
            fn = tool.get("function") or {}
            if isinstance(fn, dict) and fn.get("name"):
                existing_names.append(fn["name"])
    tool_name = generate_synthetic_tool_name(existing_names)

    new_tools = list(request_obj.tools or []) + [build_synthetic_tool(tool_name)]
    new_messages = list(request_obj.messages) + [OpenAIMessage(**build_control_message(tool_name))]

    # tool_choice=none 时合成工具在 tool_config=NONE 下不会被调用，防截断失效。
    # 下游显式请求防截断（字段=true）时以 auto 覆盖；具体函数名的强制选择保留原样
    # （若模型因此不调合成工具，本次防截断自然不生效，如实透传，不破坏下游意图）。
    tool_choice = request_obj.tool_choice
    if tool_choice == "none":
        tool_choice = "auto"

    return request_obj.model_copy(update={
        "tools": new_tools,
        "messages": new_messages,
        "tool_choice": tool_choice,
    }), tool_name


def extract_content_from_args(args: Any) -> Optional[str]:
    """从合成工具参数（dict 或 JSON 字符串）提取 content 正文。"""
    if args is None:
        return None
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except Exception:
            return None
    if isinstance(args, dict):
        content = args.get("content")
        if isinstance(content, str) and content.strip():
            return content
    return None


def is_synthetic_part(part: Any, tool_name: str) -> bool:
    """判定 Gemini Part 是否是合成工具调用。"""
    fc = getattr(part, "function_call", None)
    return fc is not None and getattr(fc, "name", None) == tool_name


def strip_synthetic_from_openai_dict(openai_dict: dict, tool_name: str) -> dict:
    """非流式/假流式：从 OpenAI 响应 dict 解构合成工具调用，还原为标准 assistant.content。

    - 合成内容作为最终正文（绝不与非合成 content 拼接，杜绝双来源拼接错误）；
    - 真实工具调用保留并重排 index / part_order；
    - 仅剩合成调用时清空 tool_calls 并把 finish_reason 修正为 stop。
    """
    if not tool_name or not isinstance(openai_dict, dict):
        return openai_dict
    for choice in openai_dict.get("choices", []):
        if not isinstance(choice, dict):
            continue
        message = choice.get("message")
        if not isinstance(message, dict):
            continue
        tool_calls = message.get("tool_calls")
        if not isinstance(tool_calls, list) or not tool_calls:
            continue

        real_calls = []
        synthetic_contents = []
        synthetic_found = False
        for tc in tool_calls:
            if not isinstance(tc, dict):
                continue
            fn = tc.get("function") or {}
            if not isinstance(fn, dict):
                continue
            if fn.get("name") == tool_name:
                synthetic_found = True
                content = extract_content_from_args(fn.get("arguments"))
                if content is not None:
                    synthetic_contents.append(content)
            else:
                real_calls.append(tc)
        if not synthetic_found:
            continue  # 本 choice 没有合成调用，原样

        # 合成内容为最终正文（取代普通 content，防止双来源拼接）；
        # 模型调用合成工具却给出空 content 时，回退普通 content 兜底。
        message["content"] = "".join(synthetic_contents) or message.get("content")

        if real_calls:
            for i, tc in enumerate(real_calls):
                tc["index"] = i
            message["tool_calls"] = real_calls
            choice["finish_reason"] = "tool_calls"
            _rebuild_part_order(message, tool_name)
        else:
            message.pop("tool_calls", None)
            if choice.get("finish_reason") == "tool_calls":
                choice["finish_reason"] = "stop"
            _drop_tool_call_entries(message, tool_name)
    return openai_dict


def _rebuild_part_order(message: dict, tool_name: str) -> None:
    """解构后按真实工具调用重排 part_order（移除合成条目，index 连续）。"""
    google = (message.get("extra_content") or {}).get("google")
    if not isinstance(google, dict):
        return
    part_order = google.get("part_order")
    if not isinstance(part_order, list):
        return
    tool_calls = message.get("tool_calls") or []
    # 原始 index → 新 index（合成移除，真实保序）
    mapping = {}
    next_index = 0
    for i, tc in enumerate(tool_calls):
        mapping[i] = next_index
        tc["index"] = next_index
        next_index += 1
    new_order = []
    for item in part_order:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "tool_call":
            old_idx = item.get("index")
            if isinstance(old_idx, int) and old_idx < len(tool_calls):
                new_order.append({"type": "tool_call", "index": mapping[old_idx]})
        else:
            new_order.append(item)
    google["part_order"] = new_order


def _drop_tool_call_entries(message: dict, tool_name: str) -> None:
    """只剩合成调用时：从 part_order 移除全部 tool_call 条目。"""
    google = (message.get("extra_content") or {}).get("google")
    if not isinstance(google, dict):
        return
    part_order = google.get("part_order")
    if not isinstance(part_order, list):
        return
    google["part_order"] = [item for item in part_order
                            if not (isinstance(item, dict) and item.get("type") == "tool_call")]


def strip_synthetic_from_stream_chunk(chunk: Any, candidate_index: int, tool_name: str):
    """真流式：从流式 chunk 剥离合成工具 part。

    返回 (剥离后 chunk 或 None, 合成 content 列表)：
      - 无合成 part → 原 chunk 原样返回；
      - 剥离后仍有真实 part → 返回深拷贝副本（供 convert_chunk_to_openai 处理）；
      - 只剩合成 part → 返回 None（合成 content 由调用方作为正文 delta 输出）。
    """
    if not tool_name:
        return chunk, []
    candidates = getattr(chunk, "candidates", None) or []
    if candidate_index >= len(candidates):
        return chunk, []
    candidate = candidates[candidate_index]
    content_obj = getattr(candidate, "content", None)
    parts = list(getattr(content_obj, "parts", None) or [])
    if not parts:
        return chunk, []

    synthetic_contents = []
    kept = []
    for part in parts:
        if is_synthetic_part(part, tool_name):
            content = extract_content_from_args(
                getattr(getattr(part, "function_call", None), "args", None))
            if content is not None:
                synthetic_contents.append(content)
        else:
            kept.append(part)

    if not synthetic_contents and len(kept) == len(parts):
        return chunk, []
    if not kept:
        return None, synthetic_contents
    try:
        new_chunk = chunk.model_copy(deep=True)
        new_candidates = list(new_chunk.candidates)
        nc = new_candidates[candidate_index]
        nc.content = content_obj.model_copy(deep=True)
        nc.content.parts = kept
        new_candidates[candidate_index] = nc
        new_chunk.candidates = new_candidates
        return new_chunk, synthetic_contents
    except Exception as e:
        print(f"⚠️ [防截断] 流式 chunk 剥离失败，回退原样处理：{e}")
        return chunk, []
