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
- **真流式增量下发**（3.33，对齐 Antigravity-gateway 1.0.9）：真流式请求会给
  `tool_config.function_calling_config` 加 `stream_function_call_arguments=true`，
  上游于是把参数按 `FunctionCall.partial_args`（`json_path` + `string_value`）**分片**下发，
  本模块的 `StreamPartialState` 边收边解、边把 `$.content` 的增量片段作为正文 delta 下发——
  否则 functionCall 参数一次给全，开了防截断的真流式会退化成"憋到最后一次性刷全文"。
  上游不支持该字段时（老模型/区域）自动降级并退回整段下发，不影响正确性。

**启用方式**：下游请求体带控制台可配置字段（默认 `"anti_truncation": true`）即对本
请求启用；字段名可在控制台自定义。生图/非文本模型自动排除。
"""

import json
import secrets
import re
from dataclasses import dataclass
from outcome import GenerationEndError, classify_generation_end
from typing import Any, Optional

from models import OpenAIMessage

# 合成工具名统一前缀（v2o = vertex2openai；后接 24 hex = 96-bit 随机 nonce）
TOOL_PREFIX = "v2o_emit_"
# 控制台设置键：启用字段名（下游请求体里置 true 即启用）
SETTING_FIELD = "anti_truncation_field"
DEFAULT_FIELD = "anti_truncation"
# 控制台设置键：真流式 side-buffer 字节阈值（0 = 直通不缓冲，见 api_helpers 真流式段落）
SETTING_SIDE_BUFFER = "anti_truncation_side_buffer_bytes"
# 控制台设置键：真流式增量参数下发（stream_function_call_arguments）总开关
SETTING_PARTIAL_ARGS = "anti_truncation_partial_args"

# 正文键名候选（对齐 Antigravity-gateway repair.go 的 scanAndExtractContent）：
# 模型偶尔不按声明的 `content` 写，而是用 text/answer/response 等同义键。
CONTENT_KEYS = ("content", "text", "answer", "response", "result", "output", "message")
# 参数解析体积上限：畸形/超大参数直接放弃提取，防止吃满内存
MAX_ARGS_BYTES = 8 * 1024 * 1024
# 递归兜底的最大下钻深度
MAX_CONTENT_DEPTH = 4

# ---------------------------------------------------------------------------
# 增量参数下发（stream_function_call_arguments）能力位
# ---------------------------------------------------------------------------
# 该字段仅 Vertex（Agent Platform）支持，本项目 Express 与 SA 两通道都是 vertexai=True。
# 一旦上游回 "不支持该字段" 类错误，本进程内永久降级（重启恢复），后续请求不再下发，
# 退回 functionCall 参数一次给全的旧行为——正确性不受影响，只是少了逐字流式。
_PARTIAL_ARGS_SUPPORTED = True
_PARTIAL_ARGS_UNSUPPORTED_HINTS = (
    "stream_function_call_arguments",
    "streamfunctioncallarguments",
    "partial_args",
    "partialargs",
)


def partial_args_supported() -> bool:
    """本进程内上游是否仍接受 stream_function_call_arguments（降级后为 False）。"""
    return _PARTIAL_ARGS_SUPPORTED


def note_partial_args_unsupported(error_text: Any) -> bool:
    """上游报错疑似"不认识增量参数字段"时置降级位。

    返回是否由本次调用触发降级（True = 刚降级，调用方可打一次显眼告警）。
    """
    global _PARTIAL_ARGS_SUPPORTED
    low = str(error_text or "").lower()
    if not any(hint in low for hint in _PARTIAL_ARGS_UNSUPPORTED_HINTS):
        return False
    if not _PARTIAL_ARGS_SUPPORTED:
        return False
    _PARTIAL_ARGS_SUPPORTED = False
    print("⚠️ [防截断] 上游不接受 stream_function_call_arguments（增量参数下发），"
          "本进程已自动降级为整段下发（功能不受影响，仅真流式首字延迟变差）。"
          "重启服务可恢复重试。")
    return True


def partial_args_enabled(settings: Optional[dict] = None) -> bool:
    """增量参数下发是否可用：控制台开关未关 + 未降级。"""
    if not partial_args_supported():
        return False
    if settings is None:
        try:
            import config as app_config
            from runtime_state import app_state
            settings = {"value": app_state.get_setting(
                SETTING_PARTIAL_ARGS,
                app_config.DEFAULT_SETTINGS.get(SETTING_PARTIAL_ARGS, True))}
        except Exception:
            return True
    raw = settings.get(SETTING_PARTIAL_ARGS, settings.get("value", True))
    if isinstance(raw, str):
        return raw.strip().lower() not in ("0", "false", "off", "no", "")
    return bool(raw)


def enable_stream_partial_args(gen_config_dict: dict) -> bool:
    """给生成配置打开真流式增量参数下发（幂等；不覆盖已有 mode/allowed_function_names）。"""
    if not isinstance(gen_config_dict, dict):
        return False
    tool_config = gen_config_dict.get("tool_config")
    if not isinstance(tool_config, dict):
        tool_config = {}
        gen_config_dict["tool_config"] = tool_config
    fcc = tool_config.get("function_calling_config")
    if not isinstance(fcc, dict):
        # 下游没给 tool_choice 时没有 tool_config：补一个 AUTO，让合成工具可被调用
        fcc = {"mode": "AUTO"}
        tool_config["function_calling_config"] = fcc
    fcc["stream_function_call_arguments"] = True
    return True



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
                # strict 约束：禁止模型在参数里加额外字段（SDK Schema 支持该键，
                # 真机已验证 FunctionDeclaration 接受 additionalProperties=false）
                "additionalProperties": False,
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


def _find_content_field(obj: Any, depth: int = 0) -> Optional[str]:
    """递归在参数对象里找正文（兜底：模型把正文塞进嵌套对象或换用同义键时）。

    有界递归（depth ≤ MAX_CONTENT_DEPTH）防深层嵌套；命中第一个非空字符串即返回。
    """
    if depth > MAX_CONTENT_DEPTH or not isinstance(obj, dict):
        return None
    for key in CONTENT_KEYS:
        value = obj.get(key)
        if isinstance(value, str) and value.strip():
            return value
    for value in obj.values():
        if isinstance(value, dict):
            found = _find_content_field(value, depth + 1)
            if found is not None:
                return found
    return None


def _strip_code_fences(text: str) -> str:
    """剥 Markdown 代码围栏（模型爱把参数包进 ```json ... ``` 里）。"""
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    body_lines = lines[1:-1] if stripped.rstrip().endswith("```") else lines[1:]
    return "\n".join(body_lines).strip()


def repair_json_arguments(text: str) -> str:
    """结构性修复畸形 JSON（对齐 Antigravity-gateway repair.go）：

    截到第一个 `{` → 闭合未结束的字符串 → 转义字符串内的裸控制字符 →
    去掉收尾逗号 → 按开括号/方括号计数补齐。纯字符串处理，不解析不抛异常。
    """
    s = text.strip()
    open_pos = s.find("{")
    if open_pos == -1:
        return s
    s = s[open_pos:]

    out = []
    in_string = False
    escaped = False
    braces = 0
    brackets = 0
    for ch in s:
        if in_string:
            if escaped:
                escaped = False
                out.append(ch)
            elif ch == "\\":
                escaped = True
                out.append(ch)
            elif ch == '"':
                in_string = False
                out.append(ch)
            elif ch in ("\n", "\r", "\t"):
                # JSON 字符串里不允许裸控制字符：转义掉
                out.append({"\n": "\\n", "\r": "\\r", "\t": "\\t"}[ch])
            else:
                out.append(ch)
        else:
            if ch == '"':
                in_string = True
                out.append(ch)
            elif ch == "{":
                braces += 1
                out.append(ch)
            elif ch == "}":
                if braces > 0:
                    braces -= 1
                    out.append(ch)
            elif ch == "[":
                brackets += 1
                out.append(ch)
            elif ch == "]":
                if brackets > 0:
                    brackets -= 1
                    out.append(ch)
            else:
                out.append(ch)
    if in_string:
        out.append('"')
    res = "".join(out).rstrip(" \t\r\n,")
    res += "]" * brackets
    res += "}" * braces
    return res


def scan_content_string(text: str) -> Optional[str]:
    r"""有界扫描兜底：直接找 `"content"\s*:\s*"` 之类的键并按 JSON 规则反转义取值。

    用于 JSON 整体解析失败的场景（结构修复也救不回来）。扫到文件尾仍未闭合时，
    已收集到的内容照常返回——宁可多吐也不能静默丢正文。
    """
    for key in CONTENT_KEYS:
        marker = f'"{key}"'
        key_idx = text.find(marker)
        if key_idx == -1:
            continue
        sub = text[key_idx + len(marker):]
        colon_idx = sub.find(":")
        if colon_idx == -1:
            continue
        sub = sub[colon_idx + 1:].lstrip()
        if not sub.startswith('"'):
            continue
        buf = []
        escaped = False
        started = False
        i = 0
        while i < len(sub):
            ch = sub[i]
            if not started:
                if ch == '"':
                    started = True
                i += 1
                continue
            if escaped:
                escaped = False
                if ch in ('"', "\\", "/"):
                    buf.append(ch)
                elif ch == "n":
                    buf.append("\n")
                elif ch == "t":
                    buf.append("\t")
                elif ch == "r":
                    buf.append("\r")
                elif ch == "b":
                    buf.append("\b")
                elif ch == "f":
                    buf.append("\f")
                elif ch == "u":
                    hex_str = sub[i + 1:i + 5]
                    try:
                        buf.append(chr(int(hex_str, 16)))
                        i += 4
                    except ValueError:
                        buf.append("\\u")
                else:
                    buf.append(ch)
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                # 真收尾引号：后面只允许空白 + } 或 ,（否则是正文里的字面引号）
                rest = sub[i + 1:].lstrip(" \t\r\n")
                if rest == "" or rest.startswith("}") or rest.startswith(","):
                    return "".join(buf)
                buf.append('"')
            else:
                buf.append(ch)
            i += 1
        if buf:
            return "".join(buf)
    return None


def extract_content_from_args(args: Any) -> Optional[str]:
    """从合成工具参数（dict 或 JSON 字符串）提取正文。

    提取管线（对齐 Antigravity-gateway repair.go，适配 SDK 场景）：
    1. 标准提取：顶层 content（及其同义键）字符串字段（SDK 正常场景，一次到位）；
    2. JSON 字符串解析失败时先剥 Markdown 代码围栏再试；
    3. 仍失败 → 结构性修复（未闭合字符串/尾逗号/缺括号）后再解析；
    4. 仍失败 → 有界扫描兜底（直接扫 `"content": "..."` 并反转义）；
    5. 递归兜底：正文被塞进嵌套对象（shape 写错）时逐层找；
    6. 全失败返回 None（调用方回退普通输出，不静默丢正文）。
    """
    if args is None:
        return None
    if isinstance(args, dict):
        for key in CONTENT_KEYS:
            content = args.get(key)
            if isinstance(content, str) and content.strip():
                return content
        # 顶层没有：递归找嵌套正文（模型 shape 写错的兜底）
        return _find_content_field(args)
    if not isinstance(args, str):
        return None
    if len(args) > MAX_ARGS_BYTES:
        print(f"⚠️ [防截断] 合成工具参数过大（{len(args)} 字节 > {MAX_ARGS_BYTES}），已放弃提取。")
        return None

    candidates = [args.strip(), _strip_code_fences(args)]
    for text in candidates:
        if not text:
            continue
        try:
            parsed = json.loads(text)
        except Exception:
            continue
        if isinstance(parsed, dict):
            for key in CONTENT_KEYS:
                content = parsed.get(key)
                if isinstance(content, str) and content.strip():
                    return content
            found = _find_content_field(parsed)
            if found is not None:
                return found

    # 结构修复后再试一次（模型输出的 JSON 半截/多逗号）
    for text in candidates:
        if not text:
            continue
        try:
            parsed = json.loads(repair_json_arguments(text))
        except Exception:
            continue
        if isinstance(parsed, dict):
            found = _find_content_field(parsed)
            if found is None:
                for key in CONTENT_KEYS:
                    value = parsed.get(key)
                    if isinstance(value, str) and value.strip():
                        found = value
                        break
            if found is not None:
                return found

    # 最后兜底：有界扫描
    for text in candidates:
        if not text:
            continue
        scanned = scan_content_string(text)
        if scanned is not None and scanned.strip():
            return scanned
    return None


@dataclass(frozen=True)
class SyntheticRestoreResult:
    text: Optional[str]
    structurally_complete: bool
    recovered_partial: bool
    failure_reason: Optional[str]


def log_integrity(**fields) -> None:
    """Only fixed protocol metadata belongs in request-scoped diagnostics."""
    allowed = {"status", "transport", "synthetic_seen", "restored", "structurally_complete",
               "recovered_partial", "finish_reason", "native_finish_reason", "failure_category",
               "selected_text_channel", "stream_done"}
    fields = {key: value for key, value in fields.items() if key in allowed}
    fields.setdefault("transport", "buffered")
    fields.setdefault("status", "integrity_failed" if fields.get("failure_category") else
                      "partial" if fields.get("recovered_partial") or fields.get("finish_reason") in ("length", "content_filter") else
                      "restored" if fields.get("synthetic_seen") else "not_called")
    native = fields.get("native_finish_reason")
    if native is not None:
        from outcome import normalize_finish_reason
        native = normalize_finish_reason(native)
        # Match outcome.normalize_finish_reason's known protocol reasons, not
        # arbitrary uppercase text supplied by an upstream response.
        known = {
            "STOP", "MAX_TOKENS", "SAFETY", "RECITATION", "BLOCKLIST",
            "PROHIBITED_CONTENT", "SPII", "IMAGE_SAFETY", "IMAGE_PROHIBITED_CONTENT",
            "IMAGE_RECITATION", "ESCALATION", "PUP_LIMITED_DISABLED",
            "MALFORMED_FUNCTION_CALL", "UNEXPECTED_TOOL_CALL", "TOO_MANY_TOOL_CALLS",
            "MISSING_THOUGHT_SIGNATURE", "MALFORMED_RESPONSE", "IMAGE_OTHER", "NO_IMAGE",
            "OTHER", "UNSPECIFIED", "FINISH_REASON_UNSPECIFIED",
        }
        if native in known:
            fields["native_finish_reason"] = native
        else:
            fields.pop("native_finish_reason", None)
    try:
        import request_log
        ctx = request_log.current()
        if ctx is not None:
            ctx.emit("anti_truncation_integrity", "防截断完整性诊断", **fields)
    except Exception:
        pass


def integrity_error() -> GenerationEndError:
    log_integrity(structurally_complete=False, failure_category="empty_or_protocol")
    return GenerationEndError(classify_generation_end("MALFORMED_FUNCTION_CALL", is_final=True))


def restore_synthetic_args(args: Any) -> SyntheticRestoreResult:
    """Strict integrity proof is independent of the existing salvage parser."""
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate_field")
            result[key] = value
        return result

    try:
        if isinstance(args, str):
            if len(args.encode("utf-8")) > MAX_ARGS_BYTES:
                return SyntheticRestoreResult(None, False, False, "resource_limit")
            parsed = json.loads(args, object_pairs_hook=unique_object)
        else:
            parsed = args
            if len(json.dumps(args, ensure_ascii=False).encode("utf-8")) > MAX_ARGS_BYTES:
                return SyntheticRestoreResult(None, False, False, "resource_limit")
        complete = (isinstance(parsed, dict) and set(parsed) == {"content"}
                    and isinstance(parsed["content"], str))
        if complete:
            parsed["content"].encode("utf-8")
            return SyntheticRestoreResult(parsed["content"], True, False, None)
    except (ValueError, TypeError, UnicodeError, RecursionError):
        pass
    text = extract_content_from_args(args)
    return SyntheticRestoreResult(text, False, text is not None, "invalid_arguments")


def is_synthetic_part(part: Any, tool_name: str) -> bool:
    """判定 Gemini Part 是否是合成工具调用。"""
    fc = getattr(part, "function_call", None)
    return fc is not None and getattr(fc, "name", None) == tool_name


def has_synthetic_tool_call(openai_dict: dict, tool_name: str) -> bool:
    """OpenAI 响应 dict 中是否出现合成工具调用（判断防截断是否实际生效）。

    需在解构（strip_synthetic_from_openai_dict）**之前**调用：解构会把合成调用移除，
    之后无法再区分"模型没调合成工具"与"已剥离"。
    """
    if not tool_name or not isinstance(openai_dict, dict):
        return False
    for choice in openai_dict.get("choices", []):
        if not isinstance(choice, dict):
            continue
        for tc in (choice.get("message") or {}).get("tool_calls") or []:
            if not isinstance(tc, dict):
                continue
            fn = tc.get("function") or {}
            if isinstance(fn, dict) and fn.get("name") == tool_name:
                return True
    return False


def strip_synthetic_from_openai_dict(openai_dict: dict, tool_name: str, *,
                                     transport: str = "buffered", native_finish_reasons=None) -> dict:
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
                if synthetic_found:
                    raise integrity_error()
                synthetic_found = True
                restored = restore_synthetic_args(fn.get("arguments"))
                content = restored.text
                log_integrity(transport=transport, synthetic_seen=True,
                              status="integrity_failed" if not restored.structurally_complete and choice.get("finish_reason") in ("stop", "tool_calls") else
                              "partial" if restored.recovered_partial or choice.get("finish_reason") in ("length", "content_filter") else "restored",
                              native_finish_reason=(native_finish_reasons or {}).get(choice.get("index", 0)),
                              restored=content is not None,
                              structurally_complete=restored.structurally_complete,
                              recovered_partial=restored.recovered_partial,
                              finish_reason=choice.get("finish_reason")
                              if choice.get("finish_reason") in ("stop", "tool_calls", "length", "content_filter") else None)
                if (not restored.structurally_complete
                        and choice.get("finish_reason") in ("stop", "tool_calls")):
                    raise integrity_error()
                if content is not None:
                    synthetic_contents.append(content)
            else:
                real_calls.append(tc)
        if not synthetic_found:
            continue  # 本 choice 没有合成调用，原样

        # 合成内容为最终正文（取代普通 content，防止双来源拼接）；
        # 模型调用合成工具却给出空 content 时，回退普通 content 兜底（防合成工具泄漏），
        # 并留一行日志便于排查"开了防截断却输出为空"。
        if synthetic_contents:
            message["content"] = "".join(synthetic_contents)
        else:
            message["content"] = message.get("content")
            print(f"⚠️ [防截断] 模型调用了合成工具 {tool_name} 但 content 为空，已回退普通输出。")

        if real_calls:
            if choice.get("finish_reason") in ("stop", "tool_calls"):
                for tc in real_calls:
                    arguments = (tc.get("function") or {}).get("arguments")
                    try:
                        parsed = json.loads(arguments) if isinstance(arguments, str) else arguments
                        if not isinstance(parsed, dict):
                            raise ValueError("invalid_arguments")
                    except (ValueError, TypeError, RecursionError):
                        raise integrity_error() from None
            for i, tc in enumerate(real_calls):
                tc["index"] = i
            message["tool_calls"] = real_calls
            if choice.get("finish_reason") in ("stop", "tool_calls"):
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


def _partial_arg_value(partial_arg: Any) -> Any:
    """从 PartialArg 取出本片携带的值（string/number/bool/null）。"""
    for attr in ("string_value", "number_value", "bool_value"):
        value = getattr(partial_arg, attr, None)
        if value is not None:
            return value
    if getattr(partial_arg, "null_value", None) is not None:
        return None
    return None


def _is_partial_end_marker(partial_arg: Any) -> bool:
    """空串 + 不续传 = 该 json_path 的结束标记（上游用它收尾，不是内容）。"""
    value = getattr(partial_arg, "string_value", None)
    return value == "" and not getattr(partial_arg, "will_continue", None)


def _json_path_tail(path: Any) -> str:
    """取 json_path 的最后一段（$.content → content；$.a.b[0] → 0）。"""
    if not isinstance(path, str) or not path:
        return ""
    text = path.strip().rstrip()
    if text.endswith("]"):
        text = text[:text.rfind("[")]
    for sep in (".", "/"):
        if sep in text:
            text = text.rsplit(sep, 1)[1]
    return text.strip("'\"")


def _json_path_set(target: dict, path: str, value: Any, budget: Optional[list] = None) -> bool:
    """把 partial_args 的值写进累积 dict（支持 $.a.b[0].c 与 /a/b 两种写法）。

    返回是否写入成功；解析不了的路径返回 False（调用方降级处理，不能静默丢参）。
    """
    if not isinstance(path, str) or not path:
        return False
    text = path.strip()
    if len(text) > 1024 or not (
            re.fullmatch(r"\$(?:\.[A-Za-z_][A-Za-z_0-9]*|\[\d+\])+", text)
            or re.fullmatch(r"(?:/[^/~]+(?:~[01][^/~]*)*)+", text)):
        return False
    if text.startswith("$"):
        text = text[1:]
    tokens: list = []
    buf = ""
    i = 0
    while i < len(text):
        ch = text[i]
        if ch in (".", "/"):
            if buf:
                tokens.append(buf)
                buf = ""
            i += 1
            continue
        if ch == "[":
            if buf:
                tokens.append(buf)
                buf = ""
            end = text.find("]", i)
            if end == -1:
                return False
            inner = text[i + 1:end].strip().strip("'\"")
            if not inner:
                return False
            tokens.append(int(inner) if inner.isdigit() else inner)
            i = end + 1
            continue
        buf += ch
        i += 1
    if buf:
        tokens.append(buf)
    if not tokens or len(tokens) > 64 or any(isinstance(t, int) and t > 1024 for t in tokens):
        return False

    if budget is not None:
        # Reserve before expansion; conservative overcount keeps memory bounded.
        cost = sum(t + 1 if isinstance(t, int) else 1 for t in tokens)
        if cost > budget[0]:
            return False
        budget[0] -= cost
    cursor: Any = target
    for idx, token in enumerate(tokens):
        last = idx == len(tokens) - 1
        nxt = None if last else tokens[idx + 1]
        if isinstance(token, int):
            if not isinstance(cursor, list):
                return False
            while len(cursor) <= token:
                cursor.append(None)
            if last:
                previous = cursor[token] if isinstance(cursor, list) else cursor.get(token)
                if isinstance(previous, (dict, list)) and previous != value:
                    return False
                cursor[token] = value
            else:
                if cursor[token] is not None and not isinstance(cursor[token], (dict, list)):
                    return False
                if cursor[token] is None:
                    cursor[token] = [] if isinstance(nxt, int) else {}
                cursor = cursor[token]
        else:
            if not isinstance(cursor, dict):
                return False
            if last:
                previous = cursor[token] if isinstance(cursor, list) else cursor.get(token)
                if isinstance(previous, (dict, list)) and previous != value:
                    return False
                cursor[token] = value
            else:
                if token in cursor and not isinstance(cursor[token], (dict, list)):
                    return False
                if token not in cursor:
                    cursor[token] = [] if isinstance(nxt, int) else {}
                cursor = cursor[token]
    return True


class _CandidatePartialState:
    """单个候选（candidate）的增量参数状态。"""

    def __init__(self) -> None:
        self.mode: Optional[str] = None      # None | "synthetic" | "real"
        self.synthetic_seen = False
        self.synthetic_id = None
        self.synthetic_content_seen = False  # 是否已经吐出过合成正文（防重复输出）
        self.content_closed = False
        self.call_closed = False
        self.synthetic_complete = False
        self.synthetic_bytes = 0
        self.unicode_pending = ""
        self.real_bytes = 0
        self.real_part: Any = None           # 真实调用首片（带 name / id / 思考签名）
        self.real_args: Optional[dict] = None
        self.real_ok = True
        self.real_open_paths = set()
        self.real_values = {}
        self.real_degraded_warned = False


class StreamPartialState:
    """一次真流式请求内的增量参数状态机（重试必须新建实例归零）。

    上游开启 `stream_function_call_arguments` 后，functionCall 参数按片下发，实测形状：
      首片   —— name/id/思考签名齐全，无参数（will_continue=True）
      中间片 —— name 为空，partial_args=[(json_path, string_value, will_continue)]
      结束片 —— partial_args=[(json_path, "", None)]（空串即该路径收尾）
      收尾片 —— 空的 functionCall part，随后才是 finish_reason

    - 合成工具的 `$.content` 片段：**边收边吐**（真·逐字流式）
    - 真实工具调用：分片累积，收尾时把完整参数写回**首片原对象**再交给既有转换管线——
      这样 name/id/思考签名（首片自带）逐字节保留，OpenAI 侧仍是"一次完整 arguments"，
      不改变任何既有工具调用语义。
    """

    # 无名 functionCall 分片可能属于合成工具，也可能属于真实调用；两种模式互斥切换。
    def __init__(self, synthetic_tool_name: Optional[str] = None,
                 allow_real_buffering: bool = True) -> None:
        self.synthetic_tool_name = synthetic_tool_name
        self.allow_real_buffering = allow_real_buffering
        self.partial_args_seen = False        # 本次流是否真的出现了分片参数
        self._non_content_warned = False
        self._candidates: dict = {}
        self._node_budget = [65536]

    # ---------- 内部 ----------
    def _cand(self, candidate_index: int) -> _CandidatePartialState:
        if candidate_index not in self._candidates:
            self._candidates[candidate_index] = _CandidatePartialState()
        return self._candidates[candidate_index]

    def _feed_synthetic(self, cs: _CandidatePartialState, partials: list) -> str:
        """累积合成工具分片，返回本片新增正文。"""
        chunks = []
        for pa in partials:
            value = getattr(pa, "string_value", None)
            if getattr(pa, "json_path", None) != "$.content" or not isinstance(value, str):
                raise integrity_error()
            if cs.content_closed or cs.call_closed:
                raise integrity_error()
            value = cs.unicode_pending + value
            cs.unicode_pending = ""
            if value and 0xD800 <= ord(value[-1]) <= 0xDBFF:
                cs.unicode_pending = value[-1]
                value = value[:-1]
            try:
                value = value.encode("utf-16-le", "surrogatepass").decode("utf-16-le")
            except UnicodeError:
                raise integrity_error() from None
            cs.synthetic_bytes += len(value.encode("utf-8"))
            if cs.synthetic_bytes > MAX_ARGS_BYTES:
                raise integrity_error()
            chunks.append(value)
            if getattr(pa, "will_continue", None) is not True:
                if cs.unicode_pending:
                    raise integrity_error()
                cs.content_closed = True
        text = "".join(chunks)
        if text:
            cs.synthetic_content_seen = True
            self.partial_args_seen = True
        return text

    def _feed_real(self, cs: _CandidatePartialState, partials: list) -> None:
        """累积真实调用分片到 cs.real_args。"""
        for pa in partials:
            path = getattr(pa, "json_path", None)
            if not isinstance(path, str) or not path:
                raise integrity_error()
            if _is_partial_end_marker(pa) and path in cs.real_open_paths:
                cs.real_open_paths.discard(path)
                continue
            value = _partial_arg_value(pa)
            cs.real_bytes += len(json.dumps(value, ensure_ascii=True).encode("utf-8")) + len(path.encode("utf-8"))
            if cs.real_bytes > MAX_ARGS_BYTES or len(cs.real_values) >= 1024:
                raise integrity_error()
            if path in cs.real_values:
                if path not in cs.real_open_paths or not isinstance(value, str):
                    raise integrity_error()
                value = cs.real_values[path] + value
            cs.real_values[path] = value
            if getattr(pa, "will_continue", None) is True:
                cs.real_open_paths.add(path)
            else:
                cs.real_open_paths.discard(path)
            if not _json_path_set(cs.real_args, path, value, self._node_budget):
                cs.real_ok = False
                raise integrity_error()

    def _flush_real(self, cs: _CandidatePartialState) -> Any:
        """把累积参数写回首片并返回（未缓冲任何调用时返回 None）。"""
        src = cs.real_part
        cs.real_part = None
        args = cs.real_args
        cs.real_args = None
        if src is None:
            return None
        if not cs.real_ok or cs.real_open_paths:
            raise integrity_error()
        try:
            part = src.model_copy(deep=True)
            fc = part.function_call
            fc.args = args
            fc.partial_args = None
            fc.will_continue = None
            return part
        except Exception:
            raise integrity_error() from None

    # ---------- 对外 ----------
    @property
    def synthetic_seen(self) -> bool:
        """本次流是否出现过合成工具调用（含只出现首片、还没吐正文的情况）。"""
        return any(c.synthetic_seen for c in self._candidates.values())

    @property
    def synthetic_content_seen(self) -> bool:
        """本次流是否已经解出过合成正文（流末用于"调了工具却空正文"的告警判定）。"""
        return any(c.synthetic_content_seen for c in self._candidates.values())

    def validate_terminal(self, candidate_index: int, finish_reason: Optional[str],
                          native_finish_reason=None) -> None:
        cs = self._cand(candidate_index)
        complete = (cs.synthetic_complete or cs.content_closed) and cs.call_closed
        log_integrity(transport="stream", synthetic_seen=cs.synthetic_seen,
                      status="integrity_failed" if cs.real_part is not None or (cs.synthetic_seen and not complete and finish_reason in ("stop", "tool_calls")) else
                      "not_called" if not cs.synthetic_seen else
                      "partial" if not complete or finish_reason in ("length", "content_filter") else "restored",
                      native_finish_reason=native_finish_reason,
                      restored=cs.synthetic_content_seen, structurally_complete=complete,
                      recovered_partial=cs.synthetic_content_seen and not complete,
                      finish_reason=finish_reason)
        if cs.real_part is not None:
            raise integrity_error()
        if (cs.synthetic_seen and finish_reason in ("stop", "tool_calls")
                and not complete):
            raise integrity_error()

    def flush_pending_real(self) -> list:
        """流结束时冲洗未收尾的真实调用，返回 [(candidate_index, part), ...]。"""
        out = []
        for idx, cs in self._candidates.items():
            if cs.mode == "real" and cs.real_part is not None:
                raise integrity_error()
        return out


def transform_stream_chunk(chunk: Any, candidate_index: int,
                           state: StreamPartialState) -> tuple[Any, list]:
    """真流式 chunk 的增量参数/合成工具处理（3.33 新增的唯一入口）。

    返回 (chunk 或 None, 合成正文片段列表)：
      - 无任何改动 → 原 chunk 原样返回（调用方按对象同一性判断"本块无合成调用"）；
      - 全部 part 被剥离/扣住 → None；
      - 其余 → 深拷贝副本。
    """
    if state is None:
        return chunk, []
    candidates = getattr(chunk, "candidates", None) or []
    if candidate_index >= len(candidates):
        return chunk, []
    candidate = candidates[candidate_index]
    content_obj = getattr(candidate, "content", None)
    parts = list(getattr(content_obj, "parts", None) or [])
    if not parts:
        return chunk, []

    cs = state._cand(candidate_index)
    tool_name = state.synthetic_tool_name
    synthetic_texts: list = []
    kept: list = []
    changed = False

    for part in parts:
        fc = getattr(part, "function_call", None)
        if fc is None:
            kept.append(part)
            continue
        name = getattr(fc, "name", None)
        partials = getattr(fc, "partial_args", None)
        args = getattr(fc, "args", None)

        # ① 合成工具（首片带名字；后续无名分片见 ③）
        if tool_name and is_synthetic_part(part, tool_name):
            if cs.real_part is not None or cs.call_closed:
                raise integrity_error()
            if cs.synthetic_seen and getattr(fc, "id", None) and fc.id != cs.synthetic_id:
                raise integrity_error()
            if not cs.synthetic_seen:
                cs.synthetic_id = getattr(fc, "id", None)
            cs.synthetic_seen = True
            cs.mode = "synthetic"
            changed = True
            if partials:
                text = state._feed_synthetic(cs, partials)
                if text:
                    synthetic_texts.append(text)
            elif args is not None and not cs.synthetic_content_seen:
                # 上游没走增量（老模型/降级）：沿用整段提取。
                # 已经吐过分片正文时忽略整段参数，避免同一份正文输出两次。
                restored = restore_synthetic_args(args)
                cs.synthetic_complete = restored.structurally_complete
                content = restored.text
                if content:
                    cs.synthetic_content_seen = True
                    synthetic_texts.append(content)
            elif args is not None:
                cs.synthetic_complete = restore_synthetic_args(args).structurally_complete
            if (getattr(fc, "will_continue", None) is False
                    or (getattr(fc, "will_continue", None) is None and args is not None)):
                cs.call_closed = True
            continue

        # ② 真实调用的首片（带名字）
        if name:
            if cs.real_part is not None:
                raise integrity_error()
            done = state._flush_real(cs)
            if done is not None:
                kept.append(done)
                changed = True
            if not state.allow_real_buffering:
                kept.append(part)
                cs.mode = "real"
                continue
            will_continue = getattr(fc, "will_continue", None)
            if partials is None and args is not None and not will_continue:
                # 参数一次给全：原样交给既有转换管线
                kept.append(part)
                cs.mode = "real"
                continue
            cs.mode = "real"
            cs.real_part = part
            cs.real_args = {}
            cs.real_ok = True
            cs.real_open_paths = set()
            cs.real_values = {}
            cs.real_bytes = 0
            changed = True
            if partials:
                state.partial_args_seen = True
                state._feed_real(cs, partials)
                if will_continue is False:
                    kept.append(state._flush_real(cs))
                    cs.mode = None
            continue

        # ③ 无名分片：归属当前模式；带 ID 时必须匹配首片。
        if (cs.real_part is not None and getattr(fc, "id", None)
                and getattr(fc, "id", None) != getattr(cs.real_part.function_call, "id", None)):
            raise integrity_error()
        if cs.mode == "synthetic":
            if cs.call_closed or (getattr(fc, "id", None) and fc.id != cs.synthetic_id):
                raise integrity_error()
            changed = True
            if args is not None:
                restored = restore_synthetic_args(args)
                if not restored.structurally_complete:
                    raise integrity_error()
                cs.synthetic_complete = True
                if not cs.synthetic_content_seen and restored.text:
                    cs.synthetic_content_seen = True
                    synthetic_texts.append(restored.text)
            if partials:
                text = state._feed_synthetic(cs, partials)
                if text:
                    synthetic_texts.append(text)
            if (getattr(fc, "will_continue", None) is False
                    or (getattr(fc, "will_continue", None) is None and not partials)):
                cs.call_closed = True
            continue
        if cs.mode == "real":
            if partials:
                state.partial_args_seen = True
                if state.allow_real_buffering:
                    changed = True
                    state._feed_real(cs, partials)
                    if getattr(fc, "will_continue", None) is False:
                        kept.append(state._flush_real(cs))
                        cs.mode = None
                else:
                    kept.append(part)
                continue
            if state.allow_real_buffering:
                done = state._flush_real(cs)
                cs.mode = None
                changed = True
                if done is not None:
                    kept.append(done)
            else:
                kept.append(part)
                cs.mode = None
            continue
        if partials:
            raise integrity_error()
        kept.append(part)

    if not changed:
        return chunk, []
    if not kept:
        return None, synthetic_texts
    try:
        new_chunk = chunk.model_copy(deep=True)
        new_candidates = list(new_chunk.candidates)
        nc = new_candidates[candidate_index]
        nc.content = content_obj.model_copy(deep=True)
        nc.content.parts = kept
        new_candidates[candidate_index] = nc
        new_chunk.candidates = new_candidates
        return new_chunk, synthetic_texts
    except Exception as e:
        print(f"⚠️ [防截断] 流式 chunk 改写失败，回退原样处理：{e}")
        return chunk, []


def strip_synthetic_from_stream_chunk(chunk: Any, candidate_index: int, tool_name: str):
    """真流式：从流式 chunk 剥离合成工具 part（向后兼容入口，无状态）。

    返回 (剥离后 chunk 或 None, 合成 content 列表)：
      - 无合成 part → 原 chunk 原样返回；
      - 剥离后仍有真实 part → 返回深拷贝副本（供 convert_chunk_to_openai 处理）；
      - 只剩合成 part → 返回 None（合成 content 由调用方作为正文 delta 输出）。

    需要增量参数（partial_args）能力时请改用 `StreamPartialState` + `transform_stream_chunk`。
    """
    state = StreamPartialState(tool_name, allow_real_buffering=False)
    if not tool_name:
        return chunk, []
    return transform_stream_chunk(chunk, candidate_index, state)
