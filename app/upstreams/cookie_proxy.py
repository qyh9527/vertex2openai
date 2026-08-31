"""
batchGraphql 直连代理上游通道

基于 Agent Platform Studio Express Mode 的 batchGraphql 协议实现。
无需任何浏览器，直接通过 Cookie + SAPISIDHASH 鉴权调用 batchGraphql 端点。

支持真正的实时流式响应（防 60s 超时）、429 智能退避重试，
并在客户端断开连接时立即停止上游调用。
"""

import re
import json
import time
import uuid
import base64
import hashlib
import asyncio
import httpx
import traceback
from typing import Any, Optional, List, Dict, AsyncGenerator
from fastapi import Request
from fastapi.responses import StreamingResponse, JSONResponse

from models import OpenAIRequest, normalize_content_part
from upstreams.base import BaseUpstream
from runtime_state import app_state
import config as app_config
import model_capabilities as mc
from message_processing import (
    DEFAULT_IMAGE_PREFILL_NUDGE,
    _create_safety_ratings_html,
    apply_console_injection,
    apply_prefill_compat,
    strip_prefill_overlap,
    PrefillDeduper,
    openai_content_to_wire_parts,
    _coerce_tool_response,
    resolve_tool_call_signature,
    signature_from_extra,
    thought_signature_extra,
    ordinary_part_metadata,
    _requires_signature,
)
from logger import stats
from api_helpers import get_retry_settings, FAKE_PREFIX
from anti_truncation import is_enabled_for_request, get_enabled_field
from failover import UpstreamUnstartedError
from signature_store import SignatureRecord, SignatureState, signature_store
from schema_validation import SchemaValidationError, validate_request_schemas

from cookie_auth import (
    build_headers,
    BATCH_GRAPHQL_URL,
    STREAM_GENERATE_QUERY_SIGNATURE,
    STREAM_GENERATE_OPERATION_NAME,
)

# ========== 重试关键词 ==========
# 可重试的错误关键词（429 限流类）
RETRYABLE_KEYWORDS = [
    "resource exhausted",
    "try again later",
    "429",
    "quota",
    "rate limit",
    "overloaded",
    "temporarily unavailable",
    "internal error",
]

# Cookie 过期/权限失效的错误关键词（不可重试，需要刷新 Cookie）
COOKIE_EXPIRED_KEYWORDS = [
    "permission",
    "denied",
    "aiplatform.endpoints.predict",
    "not authorized",
    "unauthenticated",
    "login required",
    "session expired",
    "invalid credentials",
]

# **项目级**问题的关键词：错的不是 Cookie，是 Project ID / 计费 / 项目权限。
# 必须比 COOKIE_EXPIRED_KEYWORDS 先判，否则 "Permission ... denied on resource
# //aiplatform.googleapis.com/projects/xxx" 会被当成"Cookie 过期"，
# 让人反复重取 Cookie 却永远好不了（实测踩过）。
PROJECT_ERROR_KEYWORDS = [
    "requires billing",
    "billing to be enabled",
    "billing account",
    "has not been used in project",
    "is not found and cannot be used",
    "project not found",
    "invalid argument",
]

PROJECT_ERROR_HINT = (
    "\n\n💡 这看起来是**项目层面**的问题，不是 Cookie 失效，重取 Cookie 无用。请依次检查：\n"
    "1) 控制台里的 Project ID 是否填对（要用你能在 Agent Platform Studio 里正常出文的那个项目）；\n"
    "2) 该项目是否已**开启计费**（Google 对这条接口要求计费账号；未开启会报 requires billing）；\n"
    "3) 当前登录的 Google 账号对该项目是否有权限（换项目或换账号试试）。"
)


def _is_project_error(error_msg: str) -> bool:
    """是否为项目/计费/权限类错误（与 Cookie 失效区分开）。"""
    lower = (error_msg or "").lower()
    if any(kw in lower for kw in PROJECT_ERROR_KEYWORDS):
        return True
    # "Permission 'aiplatform.endpoints.predict' denied on resource ...projects/xxx"
    # 这种同时命中 cookie 关键词，但既可能是 Cookie 没登录、也可能是项目没权限，
    # 只要错误里点名了具体项目资源，就按项目问题给出更有用的指引。
    return ("projects/" in lower or "project #" in lower) and "denied" in lower

COOKIE_REFRESH_HINT = (
    "\n\n💡 Cookie 通常较为持久（只要不退出登录/改密码/被 Google 主动失效，可维持数周甚至更久）；"
    "仅当确实出现权限错误时才需更新。"
    "重新获取：电脑浏览器打开 console.cloud.google.com，F12 → Network，"
    "复制任意请求的 Cookie 头（或用 Cookie-Editor 导出），到大盘粘贴保存。"
)

# “只有思考没有正文”的可操作提示：多为原生思考在高强度下跑飞/被截断（尤其酒馆预设 + 前端恒发 xhigh）。
_THINKING_RUNAWAY_HINT = (
    "这通常是原生思考在高强度下跑飞或被截断（前端如 SillyTavern 常发 reasoning_effort=xhigh，"
    "会覆盖控制台档位）。解决：控制台“模型参数→思考强度→原生思考控制”选“关闭原生思考”"
    "（可用“保存为该模型专属”只对本模型生效），即忽略前端强度、压到 minimal 并剥离原生思考。"
)
_NO_BODY_HINT = (
    "可能被安全策略拦截或接口行为变化；原始响应样本已写入运行日志，可到大盘“运行日志”页查看。"
)


def _safety_html_if_enabled(ratings: Any) -> str:
    """开关打开时把 batchGraphql 的 safetyRatings 渲染成附加块，否则空串。"""
    if not ratings:
        return ""
    if not app_state.get_setting("safety_score", app_config.SAFETY_SCORE):
        return ""
    return _create_safety_ratings_html(ratings)


def _prefill_tpl(user_template: str, is_image_model: bool) -> str:
    """选用续写指令：生图模型在用户未自定义时改用要图片的那句。

    通用模板说的是"从断点处无缝往下写"，生图模型会老老实实继续写**文本**，
    结果吐出一段字符画而不是图片（实测复现）。用户填了自定义模板则以用户的为准。
    """
    tpl = (user_template or "").strip()
    if tpl:
        return tpl
    return DEFAULT_IMAGE_PREFILL_NUDGE if is_image_model else ""


def _prefill_log(mode: str, prefill_text: str) -> str:
    """按模式说明预填充被怎么处理了——三种模式的差别对使用者影响很大。"""
    n = len(prefill_text)
    if mode == "keep_turn":
        return (f"🩹 [预填充兼容] 预填充（{n} 字）保留为 model 轮次，其后补一句续写推动语；"
                "输出会把它拼回开头。")
    if mode == "minimal":
        return ("🩹 [预填充兼容] 仅补占位 user 保证不报错；预填充**不会**拼回输出开头"
                "（预设的思考开标签可能因此缺失，酒馆正则会抓不到）。")
    return (f"🩹 [预填充兼容] 预填充（{n} 字）已并入末尾 user 消息作为续写指令，并将拼回输出开头。"
            "若预填充停在半截词/半截标签且模型接不上，可试试「保留模型轮次」。")


def _is_retryable_error(error_msg: str) -> bool:
    """判断错误是否可重试（429 限流类）"""
    lower = error_msg.lower()
    return any(kw in lower for kw in RETRYABLE_KEYWORDS)


def _is_cookie_expired_error(error_msg: str) -> bool:
    """判断是否为 Cookie 过期/权限失效错误"""
    lower = error_msg.lower()
    return any(kw in lower for kw in COOKIE_EXPIRED_KEYWORDS)


# ========== requestContext 模板 ==========

def _get_experiment_flags() -> str:
    """获取 experimentFlagsBinary（Express Mode 权限的关键标识，可选，从配置读取）"""
    return app_config.EXPERIMENT_FLAGS or ""


def _build_request_context(project_id: str) -> dict:
    """
    构建 batchGraphql 的 requestContext
    包含 experimentFlagsBinary，这是 Express Mode 权限的关键标识。
    """
    return {
        "clientVersion": "boq_cloud-boq-clientweb-vertexaistudio_20260609.06_p0",
        "pagePath": "/agent-platform/studio/multimodal",
        "pageViewId": int(time.time() * 1000) % (10**15),
        "trackingId": str(int(time.time() * 1000000) % (10**17)),
        "backendOverrides": {},
        "clientSessionId": str(uuid.uuid4()).upper(),
        "projectId": project_id,
        "selectedPurview": {"projectId": project_id},
        "jurisdiction": "global",
        "experimentFlagsBinary": _get_experiment_flags(),
        "localizationData": {"locale": "zh_CN", "timezone": "Asia/Hong_Kong"}
    }


# ========== 思考配置（委托中心能力模块，转 batchGraphql 的 camelCase） ==========

def _build_thinking_config(model_name: str, request: OpenAIRequest,
                           prefill_active: bool = False) -> Optional[dict]:
    """
    按模型能力档案 + 控制台设置 + 单次请求构建 thinkingConfig：
    - Gemini 3 及以上：thinkingLevel（MINIMAL/LOW/MEDIUM/HIGH）
    - Gemini 2.5：thinkingBudget（-1 动态；flash 可 0 关闭）
    - 其它/生图：None
    prefill_active=True 时按控制台开关压制思考（见 mc.resolve_thinking）。
    """
    settings = app_state.get_effective_settings(model_name)
    t = mc.resolve_thinking(model_name, request, settings, prefill_active=prefill_active)
    if t.get("mode") == "level":
        return {"thinkingLevel": t["level"].upper(), "includeThoughts": t.get("include_thoughts", True)}
    if t.get("mode") == "budget":
        return {"thinkingBudget": t["budget"], "includeThoughts": t.get("include_thoughts", True)}
    return None


# ========== OpenAI → batchGraphql 消息格式转换 ==========

# 可映射到 Studio 内建搜索的工具名（前端把“联网搜索”做成函数声明时用的常见命名）
_BUILTIN_SEARCH_NAMES = {
    "google_search", "googlesearch", "google_search_retrieval",
    "web_search", "websearch", "search_web", "browse", "search",
}


def classify_tool_traffic(messages: list, tools: Any = None) -> dict:
    """区分「只是声明了工具」和「历史里真有函数调用往返」。

    这个区分是必需的：RikkaHub 这类前端只要在模型卡里勾了工具能力，**每一条**
    请求都会带上 `tools` 声明，哪怕本轮完全没有调用需求。旧实现只看 `tools`
    是否为空就整单拒绝（400），等于让 Studio 通道在这些前端下彻底不可用。

    - declared：本轮带了工具声明（可安全丢弃 → 正常对话）
    - history：历史里有 assistant.tool_calls 或 role="tool"（真调用往返，
      Cookie 通道无法用 functionCall/functionResponse 如实表达，只能降级成文本）
    - builtin_search：声明里含可映射到 Studio 内建 googleSearch 的工具
    - custom_names：其余自定义函数名（仅用于日志）
    """
    declared = bool(tools)
    builtin_search = False
    custom_names: list = []
    for t in (tools or []):
        if not isinstance(t, dict):
            continue
        fn = t.get("function") if isinstance(t.get("function"), dict) else {}
        name = str(fn.get("name") or t.get("name") or t.get("type") or "").strip()
        if name.lower().replace("-", "_") in _BUILTIN_SEARCH_NAMES:
            builtin_search = True
        elif name:
            custom_names.append(name)

    history = False
    for m in messages or []:
        if getattr(m, "role", None) == "tool" or getattr(m, "tool_calls", None):
            history = True
            break

    return {"declared": declared, "history": history,
            "builtin_search": builtin_search, "custom_names": custom_names}


def has_tool_traffic(messages: list, tools: Any = None) -> bool:
    """兼容旧签名：请求里是否含任何函数调用相关内容。"""
    tt = classify_tool_traffic(messages, tools)
    return bool(tt["declared"] or tt["history"])


_SCHEMA_TYPES = {
    "object": "OBJECT", "array": "ARRAY", "string": "STRING",
    "integer": "INTEGER", "number": "NUMBER", "boolean": "BOOLEAN",
    "null": "NULL",
}


def _cookie_ui_schema(schema: Any) -> Any:
    """OpenAI JSON Schema -> Studio private UI Schema.

    The private batchGraphql input uses upper-case scalar names and encodes an
    object's property map as an ordered ``[{key, value}]`` list.  Recursion is
    required for both nested objects and array item schemas.
    """
    if not isinstance(schema, dict):
        return schema

    out = {}
    raw_type = schema.get("type")
    if isinstance(raw_type, list):
        non_null = [t for t in raw_type if str(t).lower() != "null"]
        raw_type = non_null[0] if non_null else "null"
        if len(non_null) != len(schema.get("type") or []):
            out["nullable"] = True
    if not raw_type:
        if isinstance(schema.get("properties"), dict):
            raw_type = "object"
        elif "items" in schema:
            raw_type = "array"
    if raw_type:
        out["type"] = _SCHEMA_TYPES.get(str(raw_type).lower(), str(raw_type).upper())

    for key, value in schema.items():
        if key == "type":
            continue
        if key == "properties" and isinstance(value, dict):
            out[key] = [{"key": name, "value": _cookie_ui_schema(child)}
                        for name, child in value.items()]
        elif key in {"items", "additionalProperties", "not"} and isinstance(value, dict):
            out[key] = _cookie_ui_schema(value)
        elif key in {"oneOf", "anyOf", "allOf", "prefixItems"} and isinstance(value, list):
            out[key] = [_cookie_ui_schema(item) for item in value]
        else:
            out[key] = value
    return out


def _tool_function(tool: Any) -> dict:
    return tool.get("function") if isinstance(tool, dict) and isinstance(tool.get("function"), dict) else {}


def _normalized_tool_name(tool: Any) -> str:
    fn = _tool_function(tool)
    return str(fn.get("name") or (tool.get("name") if isinstance(tool, dict) else "") or "").strip()


def _is_builtin_search_name(name: str) -> bool:
    return name.lower().replace("-", "_") in _BUILTIN_SEARCH_NAMES


def _build_function_declarations(tools: Any) -> tuple[list[dict], bool]:
    declarations = []
    builtin_search = False
    for tool in tools or []:
        if not isinstance(tool, dict):
            continue
        name = _normalized_tool_name(tool)
        if not name:
            continue
        if _is_builtin_search_name(name):
            builtin_search = True
            continue
        fn = _tool_function(tool)
        declaration = {"name": name}
        if fn.get("description") is not None:
            declaration["description"] = fn["description"]
        declaration["parameters"] = _cookie_ui_schema(fn.get("parameters") or {"type": "object"})
        declarations.append(declaration)
    return declarations, builtin_search


def _build_tool_config(tool_choice: Any, function_names: list[str]) -> Optional[dict]:
    """Map OpenAI selection to Studio without naming built-in search as a function."""
    if tool_choice is None:
        mode, allowed = "AUTO", None
    elif isinstance(tool_choice, str):
        choice = tool_choice.lower()
        if choice == "none":
            mode, allowed = "NONE", None
        elif choice == "required":
            mode, allowed = "ANY", function_names or None
        else:  # auto and permissive handling of unknown OpenAI-compatible values
            mode, allowed = "AUTO", None
    elif isinstance(tool_choice, dict):
        name = _normalized_tool_name(tool_choice)
        mode = "ANY"
        # googleSearch is a built-in Tool, not a FunctionDeclaration. Putting its
        # OpenAI alias in allowedFunctionNames makes the wire config self-invalid.
        allowed = None if (name and _is_builtin_search_name(name)) else (
            [name] if name else (function_names or None))
    else:
        mode, allowed = "AUTO", None

    config = {"mode": mode}
    if allowed:
        config["allowedFunctionNames"] = allowed
    return {"functionCallingConfig": config}


def _native_tool_error(error_msg: str) -> bool:
    """Narrow detector for schema/model rejection eligible for one safe degrade."""
    lower = (error_msg or "").lower()
    tool_terms = ("functiondeclaration", "function declaration", "functioncallingconfig",
                  "function calling", "toolconfig", "tool config", "variables.tools")
    reject_terms = ("unsupported", "not supported", "unknown field", "unrecognized",
                    "invalid argument", "invalid value", "cannot use", "not enabled")
    return any(term in lower for term in tool_terms) and any(term in lower for term in reject_terms)


def _native_tool_error_response(status_code: int, payload: str) -> bool:
    """Avoid mistaking ordinary model text about tools for a protocol error."""
    if status_code != 200:
        return _native_tool_error(payload)
    has_error_envelope = bool(re.search(r'"errors?"\s*:', payload or ""))
    return has_error_envelope and _native_tool_error(payload)


def _stable_cookie_tool_call_id(response_id: str, call_index: int, name: str, args: Any) -> str:
    """Generate deterministic OpenAI ids because batchGraphql functionCall has no id."""
    canonical = json.dumps(args or {}, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(f"{response_id}\0{call_index}\0{name}\0{canonical}".encode()).hexdigest()[:20]
    return f"call_{digest}"


def _signature_bytes_from_wire(value: Any) -> Optional[bytes]:
    if isinstance(value, bytes):
        return value or None
    if not isinstance(value, str) or not value:
        return None
    try:
        return base64.b64decode(value, validate=True) or None
    except Exception:
        return value.encode("utf-8")


def _openai_tool_call(response_id: str, call_index: int, call: dict) -> dict:
    name, args = call["name"], call.get("args") or {}
    call_id = _stable_cookie_tool_call_id(response_id, call_index, name, args)
    signature = _signature_bytes_from_wire(call.get("thought_signature"))
    if signature:
        signature_store.put_record(call_id, SignatureRecord(SignatureState.SIGNED, signature))
    elif call_index > 0:
        signature_store.put_unsigned_follower(call_id)
    else:
        signature_store.put_unknown(call_id)
    payload = {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)},
    }
    extra = thought_signature_extra(signature)
    if extra:
        payload["extra_content"] = extra
    return payload


def _plain_text_of(content: Any) -> str:
    """取消息的纯文本（用于把工具往返渲染成可读文本）。"""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    out = []
    for p in (content or []):
        p = normalize_content_part(p)
        if isinstance(p, dict) and p.get("type") == "text":
            out.append(p.get("text", ""))
    if out:
        return "\n".join(out)
    try:
        return json.dumps(content, ensure_ascii=False)
    except Exception:
        return str(content)


def _encode_wire_signature(signature: Optional[bytes]) -> Optional[str]:
    return base64.b64encode(signature).decode("ascii") if signature else None


def _parse_function_args(value: Any) -> dict:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def _ordinary_wire_parts_from_extra(message: Any) -> Optional[list[dict]]:
    google = ((getattr(message, "extra_content", None) or {}).get("google") or {})
    raw = google.get("ordinary_parts")
    if not isinstance(raw, list):
        return None
    parts = []
    for item in raw:
        if not isinstance(item, dict) or item.get("kind") not in {"text", "thought", "signature_only"}:
            return None
        text = item.get("text")
        if not isinstance(text, str) or (item["kind"] == "text" and "data:image/" in text):
            return None
        part = {"text": text}
        if item["kind"] == "thought":
            part["thought"] = True
        if item.get("thought_signature"):
            part["thoughtSignature"] = item["thought_signature"]
        parts.append(part)
    return parts


def _attach_message_signature(parts: list[dict], message: Any) -> None:
    signature, kind = signature_from_extra(message)
    encoded = _encode_wire_signature(signature)
    if not encoded:
        return
    target = None
    if kind == "thought":
        target = next((p for p in reversed(parts) if _is_thought_part(p)), None)
    elif kind == "text":
        target = next((p for p in reversed(parts)
                       if "text" in p and not _is_thought_part(p)), None)
    if target is None and kind != "signature_only":
        target = parts[-1] if parts else None
    if target is None:
        target = {"text": ""}
        parts.append(target)
    target["thoughtSignature"] = encoded


def _convert_messages_to_contents(messages: list, model_name: str = "", native_tools: bool = True) -> tuple:
    """OpenAI messages → batchGraphql contents.

    Native mode emits camelCase ``functionCall`` / ``functionResponse`` Parts.
    ``native_tools=False`` is the controlled compatibility fallback used only
    after a clear schema/model rejection from the upstream.
    """
    contents = []
    system_parts = []
    require_sig = _requires_signature(model_name)
    call_names = {}
    for message in messages or []:
        for tc in (getattr(message, "tool_calls", None) or []):
            tc_id = str((tc or {}).get("id") or "")
            fn_name = str(((tc or {}).get("function") or {}).get("name") or "")
            if tc_id and fn_name:
                call_names[tc_id] = fn_name

    for msg in messages:
        role = msg.role
        content = msg.content

        if role == "system":
            if isinstance(content, str):
                system_parts.append(content)
            else:
                for p in (content or []):
                    p = normalize_content_part(p)
                    if isinstance(p, dict) and p.get("type") == "text":
                        system_parts.append(p.get("text", ""))
            continue

        if role == "tool":
            name = getattr(msg, "name", None) or call_names.get(getattr(msg, "tool_call_id", "") or "")
            if native_tools and name:
                contents.append({"role": "user", "parts": [{
                    "functionResponse": {"name": name, "response": _coerce_tool_response(content)}
                }]})
            else:
                body = _plain_text_of(content)
                contents.append({"role": "user",
                                 "parts": [{"text": f"[工具执行结果 · {name or 'tool'}]\n{body}"}]})
            continue

        tool_calls = getattr(msg, "tool_calls", None)
        if role in ("assistant", "model") and tool_calls:
            if not native_tools:
                lines = []
                for tc in tool_calls:
                    fn = (tc or {}).get("function") or {}
                    lines.append(f"[请求调用工具 · {fn.get('name', 'unknown')}] 参数：{fn.get('arguments', '{}')}")
                said = _plain_text_of(content)
                if said:
                    lines.insert(0, said)
                contents.append({"role": "model", "parts": [{"text": "\n".join(lines)}]})
                continue

            call_parts = []
            for tool_index, tc in enumerate(tool_calls):
                fn = (tc or {}).get("function") or {}
                explicit_sig, _ = signature_from_extra(tc)
                _, signature = resolve_tool_call_signature(
                    str((tc or {}).get("id") or ""),
                    require_signature=require_sig and tool_index == 0,
                    explicit_signature=explicit_sig,
                    explicit_unsigned=tool_index > 0,
                )
                part = {"functionCall": {
                    "name": str(fn.get("name") or "unknown"),
                    "args": _parse_function_args(fn.get("arguments", {})),
                }}
                encoded = _encode_wire_signature(signature)
                if encoded:
                    part["thoughtSignature"] = encoded
                call_parts.append(part)

            ordinary_parts = _ordinary_wire_parts_from_extra(msg)
            if ordinary_parts is None:
                ordinary_parts = []
                reasoning = getattr(msg, "reasoning_content", None)
                if reasoning:
                    ordinary_parts.append({"text": reasoning, "thought": True})
                ordinary_parts.extend(openai_content_to_wire_parts(content))
                _attach_message_signature(ordinary_parts, msg)

            google = ((getattr(msg, "extra_content", None) or {}).get("google") or {})
            order = google.get("part_order")
            if isinstance(order, list):
                parts, used_calls, used_ordinary = [], set(), set()
                for item in order:
                    if not isinstance(item, dict):
                        continue
                    idx = item.get("index")
                    if item.get("type") == "tool_call" and isinstance(idx, int) and 0 <= idx < len(call_parts):
                        parts.append(call_parts[idx]); used_calls.add(idx)
                    elif item.get("type") == "ordinary" and isinstance(idx, int) and 0 <= idx < len(ordinary_parts):
                        parts.append(ordinary_parts[idx]); used_ordinary.add(idx)
                parts.extend(p for i, p in enumerate(call_parts) if i not in used_calls)
                parts.extend(p for i, p in enumerate(ordinary_parts) if i not in used_ordinary)
            else:
                parts = call_parts + ordinary_parts
            contents.append({"role": "model", "parts": parts})
            continue

        gemini_role = "user" if role == "user" else "model"
        parts = (_ordinary_wire_parts_from_extra(msg)
                 if role in ("assistant", "model") else None)
        if parts is None:
            parts = openai_content_to_wire_parts(content)
            if role in ("assistant", "model"):
                _attach_message_signature(parts, msg)
        if parts:
            contents.append({"role": gemini_role, "parts": parts})

    # Keep parallel FunctionResponses adjacent and unsigned: FC1, FC2 then FR1, FR2.
    merged: list = []
    for c in contents:
        if merged and merged[-1]["role"] == c["role"]:
            previous_fr = bool(merged[-1]["parts"]) and all(
                "functionResponse" in p for p in merged[-1]["parts"])
            current_fr = bool(c["parts"]) and all("functionResponse" in p for p in c["parts"])
            # Only adjacent FunctionResponse messages belong to the same parallel
            # result turn. Never merge a later fresh user message into that turn.
            if previous_fr != current_fr:
                merged.append(c)
                continue
            if not (previous_fr and current_fr):
                merged[-1]["parts"].append({"text": "\n\n"})
            merged[-1]["parts"].extend(c["parts"])
        else:
            merged.append(c)

    system_text = "\n".join(t for t in system_parts if t) if system_parts else None
    return merged, system_text


async def build_batch_graphql_body_async(project_id: str, model_name: str,
                                         request: OpenAIRequest,
                                         prefill_active: bool = False,
                                         force_search: bool = False,
                                         native_tools: bool = True) -> dict:
    """在线程里构建请求体：内部有远程图片下载与 PIL 压缩，不能阻塞事件循环（P1-2）。"""
    return await asyncio.to_thread(_build_batch_graphql_body, project_id, model_name,
                                   request, prefill_active, force_search, native_tools)


def _build_batch_graphql_body(
    project_id: str,
    model_name: str,
    request: OpenAIRequest,
    prefill_active: bool = False,
    force_search: bool = False,
    native_tools: bool = True,
) -> dict:
    contents, system_text = _convert_messages_to_contents(
        request.messages, model_name=model_name, native_tools=native_tools)
    model_path = f"projects/{project_id}/locations/global/publishers/google/models/{model_name}"

    settings = app_state.get_effective_settings(model_name)
    profile = mc.apply_sampling_policy(mc.get_profile(model_name), settings)
    allowed = profile["allowed_sampling"]

    gen_config = {}

    if profile["is_image"]:
        # 生图：设 responseModalities + imageConfig，不发采样/思考参数
        gen_config["responseModalities"] = ["TEXT", "IMAGE"]
        img_cfg = {}
        size = mc.resolve_image_size(model_name, request, settings)
        if size:
            img_cfg["imageSize"] = size
        ar = mc.resolve_aspect_ratio(model_name, request, settings)
        if ar:
            img_cfg["aspectRatio"] = ar
        if img_cfg:
            gen_config["imageConfig"] = img_cfg
    else:
        # 文本/多模态：仅注入该模型支持、且被显式设置（请求或控制台）的采样参数。
        # 与标准（Express）通道对齐：都未设置时**不发**该参数，交给模型默认——
        # 旧版会编造 temperature=1 / topP=0.95 / maxOutputTokens=65535 兜底值发出。
        if "temperature" in allowed:
            tv = request.temperature if request.temperature is not None else settings.get("default_temperature")
            if tv is not None:
                gen_config["temperature"] = tv
        if "top_p" in allowed:
            pv = request.top_p if request.top_p is not None else settings.get("default_top_p")
            if pv is not None:
                gen_config["topP"] = pv
        if "max_output_tokens" in allowed:
            mv = request.max_tokens if request.max_tokens is not None else getattr(request, "max_completion_tokens", None)
            if mv is None:
                mv = settings.get("default_max_tokens")
            if mv is not None:
                gen_config["maxOutputTokens"] = mv

        thinking_config = _build_thinking_config(model_name, request, prefill_active=prefill_active)
        if thinking_config:
            gen_config["thinkingConfig"] = thinking_config

    # OFF = 分类器整个关掉，上游**不会回传** safetyRatings；BLOCK_NONE = 照常评分但从不拦截。
    # 想看安全分就必须用后者，因此跟着「输出附加安全分」开关走（默认仍是 OFF，行为不变）。
    _want_scores = bool(app_state.get_setting("safety_score", app_config.SAFETY_SCORE))
    _threshold = "BLOCK_NONE" if _want_scores else "OFF"
    safety_settings = [
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": _threshold},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": _threshold},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": _threshold},
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": _threshold},
        # 与标准（Express）通道对齐：不发这个分类时，越狱式提示词（roleplay 预设常见）
        # 可能被默认越狱过滤拦掉“正文”而思考照常流出 → 表现为只有思考没有正文。
        # 已真机验证 batchGraphql 接受该分类（OFF / BLOCK_NONE 均可）。
        {"category": "HARM_CATEGORY_JAILBREAK", "threshold": _threshold},
    ]

    variables = {
        "contents": contents,
        "model": model_path,
        "generationConfig": gen_config,
        "safetySettings": safety_settings,
    }

    if system_text:
        variables["systemInstruction"] = {"parts": [{"text": system_text}]}

    if request.stop and "stop_sequences" in allowed:
        gen_config["stopSequences"] = request.stop if isinstance(request.stop, list) else [request.stop]

    # Live batchGraphql verification shows that custom functionDeclarations and
    # googleSearch are each supported, but mixing them in one request is rejected:
    # "Multiple tools are supported only when they are all search tools." Choose a
    # single upstream mode deterministically instead of sending a doomed payload.
    declarations, declared_search = _build_function_declarations(request.tools) if native_tools else ([], False)
    if profile["is_image"]:
        declarations = []

    choice_name = _normalized_tool_name(request.tool_choice) if isinstance(request.tool_choice, dict) else ""
    forced_search = bool(choice_name and _is_builtin_search_name(choice_name))
    tools_disabled = isinstance(request.tool_choice, str) and request.tool_choice.lower() == "none"
    _wants_search = (force_search or declared_search or forced_search
                     or (hasattr(request, 'model') and request.model.endswith("-search")))

    selected_declarations = []
    use_google_search = False
    if not tools_disabled:
        if forced_search:
            # Built-in search has no functionDeclaration name, so it cannot be
            # referenced by allowedFunctionNames. Send googleSearch alone.
            use_google_search = bool(profile["supports_search"])
        elif declarations:
            # Custom functions take precedence for mixed AUTO/required traffic.
            selected_declarations = declarations
        elif _wants_search and profile["supports_search"]:
            use_google_search = True

    if selected_declarations:
        variables["tools"] = [{"functionDeclarations": selected_declarations}]
        function_names = [item["name"] for item in selected_declarations]
        variables["toolConfig"] = _build_tool_config(request.tool_choice, function_names)
    elif use_google_search:
        variables["tools"] = [{"googleSearch": {}}]
    # With no upstream tools, omitting toolConfig is equivalent to NONE and avoids
    # applying custom functionCallingConfig to the built-in search protocol.

    return {
        "requestContext": _build_request_context(project_id),
        "querySignature": STREAM_GENERATE_QUERY_SIGNATURE,
        "operationName": STREAM_GENERATE_OPERATION_NAME,
        "variables": variables,
    }


# ========== batchGraphql 流式响应解析 ==========

async def _iter_json_objects(response) -> AsyncGenerator[dict, None]:
    buffer = ""
    async for chunk in response.aiter_text():
        if not chunk:
            continue
        buffer += chunk

        while True:
            start = buffer.find('{')
            if start == -1:
                buffer = ""
                break

            brace_count = 0
            in_string = False
            escape = False
            end = -1

            for i in range(start, len(buffer)):
                c = buffer[i]
                if escape:
                    escape = False
                    continue
                if c == '\\':
                    escape = True
                    continue
                if c == '"':
                    in_string = not in_string
                    continue
                if not in_string:
                    if c == '{':
                        brace_count += 1
                    elif c == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            end = i
                            break

            if end == -1:
                buffer = buffer[start:]
                break

            json_str = buffer[start:end + 1]
            buffer = buffer[end + 1:]

            try:
                yield json.loads(json_str)
            except json.JSONDecodeError:
                pass


# 会被安全/合规拦截的 finishReason（映射为 OpenAI content_filter）
_FINISH_FILTERED = {"SAFETY", "PROHIBITED_CONTENT", "RECITATION", "BLOCKLIST", "SPII", "IMAGE_SAFETY"}


def _map_finish_reason(raw: str | None) -> str:
    """batchGraphql finishReason → OpenAI finish_reason。"""
    r = (raw or "").upper()
    if r == "MAX_TOKENS":
        return "length"
    if r in _FINISH_FILTERED:
        return "content_filter"
    return "stop"


def _is_thought_part(part: dict) -> bool:
    """健壮的思考 part 判定：兼容 true / "true" / "True" 等表示，
    避免上游把布尔编码成字符串时（"false" 为真值）把正文误判成思考。"""
    t = part.get("thought")
    if isinstance(t, str):
        return t.strip().lower() == "true"
    return bool(t)


def _summ_obj(obj: dict, width: int = 700) -> str:
    """把一个响应对象压成截断的单行 JSON（诊断日志用，图片 base64 会掐掉）。"""
    try:
        s = json.dumps(obj, ensure_ascii=False)
    except Exception:
        s = str(obj)
    s = re.sub(r'"data":\s*"[A-Za-z0-9+/=]{64,}"', '"data": "<base64 已省略>"', s)
    return s[:width] + ("…" if len(s) > width else "")


class _RawSampler:
    """收集原始响应对象的首尾样本，供“无正文/空响应”时输出诊断日志。"""

    def __init__(self, keep: int = 2):
        self.keep = keep
        self.head: list = []
        self.tail: list = []
        self.count = 0

    def add(self, obj: dict):
        self.count += 1
        if len(self.head) < self.keep:
            self.head.append(_summ_obj(obj))
        else:
            self.tail.append(_summ_obj(obj))
            if len(self.tail) > self.keep:
                self.tail.pop(0)

    def dump(self) -> str:
        if self.count == 0:
            return "（上游未返回任何可解析的 JSON 对象）"
        lines = [f"共 {self.count} 个对象；首尾样本："]
        lines += [f"  ▸ {s}" for s in self.head]
        if self.tail:
            lines.append(f"  …")
            lines += [f"  ▸ {s}" for s in self.tail]
        return "\n".join(lines)


def _extract_from_results(obj: dict):
    if "error" in obj:
        yield ("error", obj["error"])
        return

    results = obj.get("results", [])
    for result in results:
        if "errors" in result:
            for err in result["errors"]:
                yield ("error", err)
            continue

        data = result.get("data")
        if not data:
            continue

        # 提示词级拦截（promptFeedback.blockReason）：无候选、直接被挡。
        # 注意：batchGraphql 每个流式块都会带 blockReason=BLOCKED_REASON_UNSPECIFIED
        # （proto 枚举默认值 = 未拦截），必须过滤掉，只透出真实拦截。
        pf = data.get("promptFeedback")
        if isinstance(pf, dict):
            br = str(pf.get("blockReason") or "")
            if br and not br.upper().endswith("UNSPECIFIED"):
                msg = br
                if pf.get("blockReasonMessage"):
                    msg += f"（{pf['blockReasonMessage']}）"
                yield ("blocked", msg)

        candidates = data.get("candidates", [])
        for candidate in candidates:
            content_obj = candidate.get("content") or {}
            parts = content_obj.get("parts") or []

            for part in parts:
                text = part.get("text", "")
                if text:
                    if _is_thought_part(part):
                        yield ("thought", text)
                    else:
                        yield ("text", text)

                # Proto JSON includes empty defaults for functionCall,
                # functionResponse and thoughtSignature on ordinary Parts.  Only
                # a named call and a non-empty signature are semantically real.
                function_call = part.get("functionCall")
                if isinstance(function_call, dict) and str(function_call.get("name") or "").strip():
                    args = function_call.get("args")
                    if not isinstance(args, dict):
                        args = {}
                    yield ("function_call", {
                        "name": str(function_call["name"]),
                        "args": args,
                        "thought_signature": part.get("thoughtSignature") or None,
                    })
                elif part.get("thoughtSignature"):
                    yield ("thought_signature", {
                        "value": part["thoughtSignature"],
                        "part_kind": "thought" if _is_thought_part(part) else ("text" if text else "signature_only"),
                    })

                inline_data = part.get("inlineData")
                if inline_data:
                    mime_type = inline_data.get("mimeType", "")
                    b64 = inline_data.get("data", "")
                    if mime_type and b64:
                        image_md = f"![Generated Image](data:{mime_type};base64,{b64})"
                        yield ("image", image_md)

            # finishReason 全量透出（含 SAFETY/PROHIBITED_CONTENT 等），由消费方统一映射；
            # 旧实现只认 STOP/MAX_TOKENS/SAFETY 白名单，其余被静默吞掉，导致“无正文”无从排查。
            # FINISH_REASON_UNSPECIFIED 是流式中间块的枚举默认值（非真实结束），需过滤。
            finish_reason = candidate.get("finishReason")
            if finish_reason and not str(finish_reason).upper().endswith("UNSPECIFIED"):
                # 安全分只在最后一块透出：每个流式块都带 safetyRatings，
                # 逐块附加会把同一份评分重复插进正文。
                ratings = candidate.get("safetyRatings")
                if ratings:
                    yield ("safety", ratings)
                yield ("finish", str(finish_reason))

        # 尽力解析 token 用量（私有接口通常不回传；保留解析以防未来补上）
        usage = data.get("usageMetadata")
        if isinstance(usage, dict) and usage:
            yield ("usage", usage)


# ========== token 用量映射 ==========

def _map_usage(usage_meta: dict | None) -> dict:
    """把 batchGraphql 的 usageMetadata 转成 OpenAI usage 字典。

    注意：私有 batchGraphql 接口通常不回传用量（恒为 0），因此 Cookie 通道
    不再打印 💰 统计行——大盘的 token 统计仅由标准（Express）通道计入；
    Cookie 通道的成功数改由 stats.add_success() 单独计入。
    """
    usage_meta = usage_meta or {}
    p = int(usage_meta.get("promptTokenCount", 0) or 0)
    c = int(usage_meta.get("candidatesTokenCount", 0) or 0)
    t = int(usage_meta.get("totalTokenCount", p + c) or (p + c))
    return {"prompt_tokens": p, "completion_tokens": c, "total_tokens": t}


# ========== OpenAI SSE 格式化 ==========

def _make_openai_chunk(
    response_id: str,
    model: str,
    content: str = None,
    reasoning_content: str = None,
    finish_reason: str = None,
    role: str = None,
    tool_calls: Optional[list[dict]] = None,
    extra_content: Optional[dict] = None,
) -> str:
    delta = {}
    if role:
        delta["role"] = role
    if content is not None:
        delta["content"] = content
    if reasoning_content is not None:
        delta["reasoning_content"] = reasoning_content
    if tool_calls is not None:
        delta["tool_calls"] = tool_calls
    if extra_content is not None:
        delta["extra_content"] = extra_content

    chunk = {
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "delta": delta,
            "finish_reason": finish_reason,
        }]
    }
    return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"


def _sse_heartbeat() -> str:
    """SSE 注释行心跳：符合 SSE 规范、被客户端忽略、不注入任何内容。
    用于 429 退避等待期间保活连接，避免前端因长时间无字节而超时中断。"""
    return ": keep-alive\n\n"


async def _sleep_with_heartbeat(total_sec: float, hb_interval: float = 3.0):
    """边等待边吐心跳的异步生成器：每 hb_interval 秒 yield 一次心跳，直到累计 total_sec。"""
    waited = 0.0
    step = max(0.5, min(hb_interval, total_sec)) if total_sec > 0 else 0
    while waited < total_sec:
        await asyncio.sleep(min(step, total_sec - waited))
        waited += step
        yield _sse_heartbeat()


def _make_usage_chunk(response_id: str, model: str, usage: dict) -> str:
    """OpenAI 风格的用量尾块（choices 为空，仅携带 usage）"""
    chunk = {
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [],
        "usage": usage,
    }
    return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"


# ========== 认证解析 ==========

def _get_cookie_string() -> str:
    # 多账号：按请求从当前账号取（见 runtime_state.get_current_cookie_account 注释，
    # 重试/流式路径会多次调用，必须固定同一份账号）。
    # 无账号列表时回落环境变量 GOOGLE_COOKIE。
    account_cookie, _ = app_state.get_current_cookie_account()
    return account_cookie or app_config.GOOGLE_COOKIE or ""

def _get_project_id() -> str:
    account_cookie, account_project = app_state.get_current_cookie_account()
    return account_project or app_config.GOOGLE_PROJECT_ID or ""


def _wants_usage(request_obj: OpenAIRequest) -> bool:
    opts = getattr(request_obj, "stream_options", None)
    if isinstance(opts, dict):
        return bool(opts.get("include_usage"))
    return False


# ========== 单次流式请求执行器（重构为真正的异步实时生成器） ==========

async def _execute_stream_request_generator_once(
    client: httpx.AsyncClient,
    headers: dict,
    body: dict,
    sampler: "_RawSampler | None" = None,
) -> AsyncGenerator[tuple[str, Any], None]:
    """
    异步生成器：真正实时地拉取数据，抛出**原始事件**（不做 SSE 格式化）：
    ("text"|"thought"|"image", str)、("finish", 原始finishReason)、("usage", dict)、
    ("blocked", str)、("api_error_text", str)、("cookie_error"|"retryable_error"|"fatal_error", str)。
    由消费方负责格式化成 OpenAI chunk（便于预填充去重与无正文诊断）。
    """
    has_content = False
    try:
        async with client.stream("POST", BATCH_GRAPHQL_URL, headers=headers, json=body) as response:

            # 1. 拦截 HTTP 状态码错误
            if response.status_code != 200:
                error_text = await response.aread()
                error_msg = error_text.decode('utf-8', errors='replace')[:1000]

                if response.status_code in (401, 403) or _is_cookie_expired_error(error_msg) \
                        or _is_project_error(error_msg):
                    hint = PROJECT_ERROR_HINT if _is_project_error(error_msg) else COOKIE_REFRESH_HINT
                    yield "cookie_error", error_msg + hint
                    return

                if _native_tool_error(error_msg):
                    yield "native_tool_unsupported", error_msg
                    return
                is_retryable = response.status_code in (429, 503, 500) or _is_retryable_error(error_msg)
                yield "retryable_error" if is_retryable else "fatal_error", error_msg
                return

            # 2. 实时遍历并抛出流式 JSON 事件块
            async for obj in _iter_json_objects(response):
                if sampler is not None:
                    sampler.add(obj)
                for event_type, data in _extract_from_results(obj):
                    if event_type in ("text", "thought", "image", "function_call", "thought_signature"):
                        has_content = True
                        yield event_type, data

                    elif event_type in ("finish", "usage", "blocked", "safety"):
                        yield event_type, data

                    elif event_type == "error":
                        err_msg = data.get("message", str(data)) if isinstance(data, dict) else str(data)

                        # 在还没发送任何有效数据前遇到错误，尝试走顶层重试
                        if _native_tool_error(err_msg) and not has_content:
                            yield "native_tool_unsupported", err_msg
                            return
                        if (_is_cookie_expired_error(err_msg) or _is_project_error(err_msg)) \
                                and not has_content:
                            hint = PROJECT_ERROR_HINT if _is_project_error(err_msg) else COOKIE_REFRESH_HINT
                            yield "cookie_error", err_msg + hint
                            return
                        if _is_retryable_error(err_msg) and not has_content:
                            yield "retryable_error", err_msg
                            return

                        # 如果流已经开始输出才发生错误，直接作为文本信息告知前端
                        yield "api_error_text", err_msg

    except Exception as e:
        err_msg = str(e)
        is_retryable = _is_retryable_error(err_msg) or "timeout" in err_msg.lower()
        if not has_content:
            yield "retryable_error" if is_retryable else "fatal_error", err_msg
        else:
            # 数据传输中途断开
            yield "api_error_text", f"连接中断: {err_msg}"


async def _execute_stream_request_generator(
    client: httpx.AsyncClient,
    headers: dict,
    body: dict,
    sampler: "_RawSampler | None" = None,
    fallback_body: Any = None,
    fallback_state: Optional[dict] = None,
) -> AsyncGenerator[tuple[str, Any], None]:
    """Execute native tools, lazily degrading after one clear protocol rejection."""
    async for status, data in _execute_stream_request_generator_once(client, headers, body, sampler):
        if status != "native_tool_unsupported":
            yield status, data
            continue
        if fallback_body is None:
            yield "fatal_error", data
            return
        if fallback_state is not None:
            fallback_state["latched"] = True
        resolved_fallback = fallback_body() if callable(fallback_body) else fallback_body
        if hasattr(resolved_fallback, "__await__"):
            resolved_fallback = await resolved_fallback
        print("⚠️ [Studio] 当前模型/协议拒绝原生函数 Schema；本次及后续重试固定降级为文本工具观测。")
        async for fallback_status, fallback_data in _execute_stream_request_generator_once(
                client, headers, resolved_fallback, sampler):
            yield fallback_status, fallback_data
        return


async def _collect_full_response(project_id, base_model_name, request_obj, headers, client_kwargs,
                                 retry_max, backoff_sec, fastapi_request) -> dict:
    """非流式地完整取回一次响应（供生图假流式复用）。返回 dict。"""
    for attempt in range(retry_max + 1):
        if await fastapi_request.is_disconnected():
            return {"kind": "error", "message": "客户端已断开连接。"}
        try:
            body = await build_batch_graphql_body_async(
                project_id, base_model_name, request_obj)
            req_headers = build_headers(_get_cookie_string()) or headers
            async with httpx.AsyncClient(**client_kwargs) as client:
                response = await client.post(BATCH_GRAPHQL_URL, headers=req_headers, json=body)

            if response.status_code in (429, 503, 500) and attempt < retry_max:
                stats.add_retry()
                print(f"⚠️ [Studio] HTTP {response.status_code}（生图，尝试 {attempt+1}），{backoff_sec}s 后重试...")
                await asyncio.sleep(backoff_sec)
                continue
            if response.status_code != 200:
                return {"kind": "error", "message": f"HTTP {response.status_code}: {response.text[:300]}"}

            full_text, reasoning_text, api_error, usage_meta = "", "", None, None
            finish_raw, blocked_msg = None, None
            safety_html = ""
            sampler = _RawSampler()

            class _F:
                def __init__(self, t): self._t = t
                async def aiter_text(self): yield self._t

            async for obj in _iter_json_objects(_F(response.text)):
                sampler.add(obj)
                for et, data in _extract_from_results(obj):
                    if et == "text":
                        full_text += data
                    elif et == "thought":
                        reasoning_text += data
                    elif et == "image":
                        full_text += data
                    elif et == "finish":
                        finish_raw = data
                    elif et == "safety":
                        safety_html = _safety_html_if_enabled(data)
                    elif et == "blocked":
                        blocked_msg = data
                    elif et == "usage":
                        usage_meta = data
                    elif et == "error":
                        em = data.get("message", str(data)) if isinstance(data, dict) else str(data)
                        if _is_retryable_error(em) and attempt < retry_max:
                            api_error = em
                            break
                        full_text += f"\n[错误] {em}"

            if api_error and attempt < retry_max:
                stats.add_retry()
                print(f"⚠️ [Studio] 生图 429/限流（尝试 {attempt+1}），{backoff_sec}s 后重试...")
                await asyncio.sleep(backoff_sec)
                continue

            if not full_text.strip():
                # 无正文：给出可见的诊断信息，而不是静默空回复
                detail = f"finishReason={finish_raw or '无'}"
                if blocked_msg:
                    detail += f"；promptFeedback 拦截：{blocked_msg}"
                print(f"🔎 [Studio 诊断] 生图无正文（{detail}）。原始响应样本：\n{sampler.dump()}")
                return {"kind": "error", "message": f"上游未返回图片/正文（{detail}）。已在日志记录原始响应样本。"}

            return {"kind": "ok", "full_text": full_text + safety_html, "reasoning_text": reasoning_text,
                    "finish_reason": _map_finish_reason(finish_raw), "usage_meta": usage_meta}
        except Exception as e:
            em = str(e)
            if (_is_retryable_error(em) or "timeout" in em.lower()) and attempt < retry_max:
                stats.add_retry()
                print(f"⚠️ [Studio] 生图异常（尝试 {attempt+1}）：{em[:80]}，{backoff_sec}s 后重试...")
                await asyncio.sleep(backoff_sec)
                continue
            return {"kind": "error", "message": f"batchGraphql proxy error: {em}"}
    return {"kind": "error", "message": "重试多次仍失败。"}


# ========== 主代理类 ==========

class CookieProxyUpstream(BaseUpstream):
    """
    batchGraphql 直连代理
    使用 Cookie + SAPISIDHASH 鉴权调用 batchGraphql 端点。
    """

    async def chat_completions(self, request_obj: OpenAIRequest, fastapi_request: Request,
                               failover_mode: bool = False):
        try:
            validate_request_schemas(request_obj)
        except SchemaValidationError as exc:
            return JSONResponse(status_code=400, content={
                "error": {"message": str(exc), "type": "invalid_request_error", "code": 400}
            })

        # ===== 1. 验证认证 =====
        cookie_str = _get_cookie_string()
        if not cookie_str:
            return JSONResponse(status_code=401, content={"error": {"message": (
                "未配置 Google Cookie。\n"
                "请在大盘控制台中粘贴 Cookie 和 Project ID，\n"
                "或设置环境变量 GOOGLE_COOKIE 和 GOOGLE_PROJECT_ID。"
            ), "type": "auth_error"}})

        project_id = _get_project_id()
        if not project_id:
            return JSONResponse(status_code=400, content={"error": {"message": (
                "未配置 Google Cloud Project ID。\n"
                "请在大盘中填写，或设置环境变量 GOOGLE_PROJECT_ID。\n"
                "可从 Studio URL 中获取：...?project=YOUR_PROJECT_ID"
            ), "type": "config_error"}})

        # ===== 2. 构建请求头 =====
        headers = build_headers(cookie_str)
        if not headers:
            return JSONResponse(status_code=401, content={"error": {"message": (
                "Cookie 中未找到 SAPISID，无法计算认证头。\n"
                "请确保 Cookie 来自已登录的 console.cloud.google.com 页面。"
            ), "type": "auth_error"}})

        # ===== 2.5 工具流量处理：原生函数调用 =====
        _tool_info = classify_tool_traffic(request_obj.messages, request_obj.tools)
        force_builtin_search = bool(_tool_info["builtin_search"])
        _choice_name = _normalized_tool_name(request_obj.tool_choice) if isinstance(request_obj.tool_choice, dict) else ""
        _forced_search = bool(_choice_name and _is_builtin_search_name(_choice_name))
        native_custom_tools = bool(_tool_info["custom_names"] or _tool_info["history"])
        if force_builtin_search and _tool_info["custom_names"]:
            if _forced_search:
                print("🔎 [Studio] batchGraphql 不支持内建搜索与自定义函数混用；本轮按强制选择仅启用 googleSearch。")
            else:
                print("⚠️ [Studio] batchGraphql 不支持内建搜索与自定义函数混用；本轮优先启用自定义函数，忽略搜索声明。")
        elif force_builtin_search:
            print("🔎 [Studio] 请求声明了搜索类工具，已映射为 Studio 内建 googleSearch。")
        if _tool_info["custom_names"] and not _forced_search:
            print(f"🛠️ [Studio] 原生函数调用：{'、'.join(_tool_info['custom_names'][:5])}"
                  f"{' 等' if len(_tool_info['custom_names']) > 5 else ''}。")

        # ===== 2.6 防截断状态提示 =====
        # Cookie 通道走 batchGraphql，无工具参数传输机制（内建搜索除外），不支持防截断协议。
        # 若下游请求体带了启用字段，明确提示已忽略，避免"开了防截断却静默失效"的困惑。
        if is_enabled_for_request(request_obj):
            print(f"⛔ [防截断] 本次调用下游已启用（字段「{get_enabled_field()}」=true），"
                  "但 Cookie 通道无工具参数传输机制、不支持防截断，已忽略启用字段（走普通通道）。")

        # ===== 3. 解析模型名 =====
        model_display = request_obj.model
        base_model_name = model_display
        # fake- 前缀最先剥：fake-gemini-x-search → gemini-x-search → gemini-x
        # （Cookie 通道无假流式实现，剥掉前缀当普通模型处理；model_display 保留原样回显）
        if base_model_name.startswith(FAKE_PREFIX):
            base_model_name = base_model_name[len(FAKE_PREFIX):]
        if base_model_name.endswith("-search"):
            base_model_name = base_model_name[:-len("-search")]

        # ===== 3.5 预填充智能兼容（按控制台模式 + 模型能力；新模型自动生效）=====
        _profile = mc.get_profile(base_model_name)
        prefill_text = ""
        prefill_active = False
        # 控制台注入（轻量前端用；两个字段都留空时是空操作）。
        # 原生工具往返不能插入 assistant 预填充，否则会破坏 FC/FR 拓扑。
        _inj_settings = app_state.get_effective_settings(base_model_name)
        _injected, _inj_notes = apply_console_injection(
            request_obj.messages,
            system_text=_inj_settings.get("inject_system_instruction", ""),
            prefill_text=_inj_settings.get("inject_prefill", ""),
            has_tools=bool(_tool_info["declared"] or _tool_info["history"]),
            is_image_model=_profile["is_image"],
            allow_image_prefill=bool(_inj_settings.get("inject_prefill_for_image", False)),
        )
        for _n in _inj_notes:
            print(_n)
        if _injected is not request_obj.messages:
            request_obj = request_obj.model_copy(update={"messages": _injected})

        _prefill_mode = app_state.get_setting("prefill_mode", app_config.DEFAULT_SETTINGS["prefill_mode"])
        if _prefill_mode != "off":
            _new_msgs, prefill_text, prefill_active = apply_prefill_compat(
                request_obj.messages, _prefill_mode,
                allow_model_last=not _profile["requires_user_last_turn"],
                instruction_template=_prefill_tpl(_inj_settings.get("prefill_instruction", ""), _profile["is_image"]),
                cot_guard=bool(_inj_settings.get("prefill_cot_guard", True)) and not _profile["is_image"],
            )
            if _new_msgs is not request_obj.messages:
                request_obj = request_obj.model_copy(update={"messages": _new_msgs})
                print(_prefill_log(_prefill_mode, prefill_text))
            elif prefill_text:
                print(f"🩹 [预填充兼容] 该模型支持 model 结尾，预填充原生透传（{len(prefill_text)} 字），模型将直接续写。")
            else:
                # 没检测到预填充也要说一声：很多人以为整个预设就是预填充，
                # 实际只有「最后一条 assistant 消息」才算。没有它，压制原生思考也不会触发。
                print("ℹ️ [预填充兼容] 未检测到预填充（请求最后一条不是 assistant 消息）。"
                      "预设里的思维链指令属于 system/user 条目，不是预填充；"
                      "若要用预设思维链顶掉原生思考，需在预设末尾放一条 assistant 条目（通常是思考块的开标签）。")
            if prefill_active and app_state.get_setting("prefill_suppress_thinking", True):
                print("🧠 [预填充兼容] 已按模型压制原生思考（可在控制台关闭），让预设思维链接管。")

        # ===== 4. HTTP 客户端配置 =====
        client_kwargs = {
            "timeout": httpx.Timeout(connect=30.0, read=180.0, write=30.0, pool=10.0),
            "follow_redirects": True,
        }
        if app_config.PROXY_URL:
            client_kwargs["proxy"] = app_config.PROXY_URL

        # 重试配置（控制台可调）；语义与 Express 通道统一：总尝试次数 = retry_max + 1
        # Cookie 通道的重试次数可独立覆盖（channel_retry_overrides["cookie"]）
        retry_max, backoff_sec = get_retry_settings("cookie")

        is_stream = request_obj.stream
        response_id = f"chatcmpl-studio-{int(time.time())}-{uuid.uuid4().hex[:8]}"
        start_time = time.time()
        want_usage = _wants_usage(request_obj)
        cookie_debug = bool(app_state.get_setting("debug_outbound", False)
                            or app_state.get_setting("cookie_debug", False))

        # batchGraphql(Studio) 会忽略 includeThoughts=false（真机验证），因此当解析出的思考配置
        # 要求不回传思考时，由本通道在响应侧主动剥离思考块。生图无思考，strip 恒 False。
        _eff_settings = app_state.get_effective_settings(base_model_name)
        _tk = mc.resolve_thinking(base_model_name, request_obj, _eff_settings, prefill_active=prefill_active)
        strip_thoughts = bool(_tk.get("mode") and not _tk.get("include_thoughts", True))
        if strip_thoughts and cookie_debug:
            print("🔎 [Studio 调试] 已启用响应侧思考剥离（batchGraphql 不支持 includeThoughts=false）。")

        # 打印请求日志
        msg_count = len(request_obj.messages)
        print(f"→ [Studio] {base_model_name} | {msg_count} 条消息 | {'流式' if is_stream else '非流式'}")

        is_image = _profile["is_image"]

        # ========== 生图 + 流式：强制假流式 ==========
        # 生图输出是超大 base64，若按流式分块传输会卡死前端解析器；
        # 因此先完整取回，再把整张图作为“单个 chunk”一次性发出（与官方 SDK 通道一致）。
        if is_stream and is_image:
            async def image_fake_stream():
                if await fastapi_request.is_disconnected():
                    return
                print(f"🖼️ [生图保护] 图片模型 {base_model_name} 已自动切换为假流式输出（Cookie 通道），避免分块 base64 卡死前端。")
                res = await _collect_full_response(
                    project_id, base_model_name, request_obj, headers, client_kwargs,
                    retry_max, backoff_sec, fastapi_request,
                )
                if res.get("kind") != "ok":
                    # hybrid 故障转移：生图失败且未出流（role chunk 在失败检查之后才发）→ 抛给路由层切兜底通道
                    if failover_mode:
                        raise UpstreamUnstartedError(res.get("message", "生图失败"))
                    yield _make_openai_chunk(response_id, model_display, role="assistant")
                    stats.add_error()
                    print(f"❌ [Studio] 生图失败 | {res.get('message', '')[:150]}")
                    yield _make_openai_chunk(response_id, model_display, content=f"[Studio 错误] {res.get('message', '生图失败')}")
                    yield _make_openai_chunk(response_id, model_display, finish_reason="stop")
                    yield "data: [DONE]\n\n"
                    return
                yield _make_openai_chunk(response_id, model_display, role="assistant")
                full_text = res.get("full_text") or ""
                if prefill_text:
                    full_text = prefill_text + strip_prefill_overlap(prefill_text, full_text)
                if res.get("reasoning_text"):
                    yield _make_openai_chunk(response_id, model_display, reasoning_content=res["reasoning_text"])
                # 关键：整张图作为单个 chunk 发出，绝不分块
                yield _make_openai_chunk(response_id, model_display, content=full_text or " ")
                stats.add_success()
                usage = _map_usage(res.get("usage_meta"))
                if want_usage:
                    yield _make_usage_chunk(response_id, model_display, usage)
                yield _make_openai_chunk(response_id, model_display, finish_reason=res.get("finish_reason", "stop"))
                yield "data: [DONE]\n\n"
                print(f"✅ [Studio] {base_model_name} | 生图假流式完成")
            return StreamingResponse(image_fake_stream(), media_type="text/event-stream")

        # ========== 流式处理（彻底解决 60s 超时的真·流式机制） ==========
        if is_stream:
            async def stream_generator():
                nonlocal start_time

                # 立即吐一个心跳，尽快建立连接（避免首个上游调用较慢/429 时前端久等无字节）
                yield _sse_heartbeat()
                native_fallback_state = {"latched": False}

                for attempt in range(retry_max + 1):
                    # 客户端已断开则立即停止，避免无谓的上游调用与重试
                    if await fastapi_request.is_disconnected():
                        print("ℹ️ [Studio] 客户端已断开连接，停止流式重试。")
                        return

                    use_native_tools = not native_fallback_state["latched"]
                    if use_native_tools:
                        body = await build_batch_graphql_body_async(
                            project_id, base_model_name, request_obj, prefill_active=prefill_active,
                            force_search=force_builtin_search)
                    else:
                        body = await build_batch_graphql_body_async(
                            project_id, base_model_name, request_obj, prefill_active=prefill_active,
                            force_search=force_builtin_search, native_tools=False)
                    fallback_body = None
                    if native_custom_tools and use_native_tools:
                        # Building can download/compress images, so keep it lazy
                        # until the native schema is actually rejected. Invoking
                        # this factory also latches degraded mode across retries.
                        async def fallback_body():
                            native_fallback_state["latched"] = True
                            return await build_batch_graphql_body_async(
                                project_id, base_model_name, request_obj,
                                prefill_active=prefill_active,
                                force_search=force_builtin_search, native_tools=False)
                    req_headers = build_headers(_get_cookie_string()) or headers
                    if cookie_debug:
                        print(f"🔎 [Studio 调试] 出站 generationConfig: "
                              f"{json.dumps(body['variables'].get('generationConfig', {}), ensure_ascii=False)}")

                    async with httpx.AsyncClient(**client_kwargs) as client:
                        has_yielded_to_client = False
                        got_text = False       # 是否收到过“正文”（text/image）
                        got_thought = False    # 是否收到过思考
                        got_tool_call = False
                        tool_call_index = 0
                        stream_ordinary_metadata = []
                        stream_part_order = []
                        should_retry = False
                        error_to_raise = None
                        usage_meta = None
                        finish_raw = None
                        blocked_msg = None
                        sampler = _RawSampler()
                        deduper = PrefillDeduper(prefill_text)

                        # 消费实时生成器（原始事件 → 此处格式化为 OpenAI chunk）
                        async for status, data in _execute_stream_request_generator(
                            client, req_headers, body, sampler=sampler,
                            fallback_body=fallback_body,
                        ):
                            # 思考剥离：batchGraphql 忽略 includeThoughts，这里按解析出的配置主动丢弃思考块
                            if status == "thought":
                                got_thought = True
                                if strip_thoughts:
                                    continue  # 不建连、不输出，等真正的正文

                            if status == "function_call":
                                if not has_yielded_to_client:
                                    print(f"⚡ [Studio] {base_model_name} | 连接建立，正在实时流式输出...")
                                    yield _make_openai_chunk(response_id, model_display, role="assistant")
                                    has_yielded_to_client = True
                                tool_payload = _openai_tool_call(response_id, tool_call_index, data)
                                tool_payload["index"] = tool_call_index
                                stream_part_order.append({"type": "tool_call", "index": tool_call_index})
                                yield _make_openai_chunk(response_id, model_display, tool_calls=[tool_payload])
                                tool_call_index += 1
                                got_tool_call = True

                            elif status == "thought_signature":
                                signature = _signature_bytes_from_wire(data.get("value"))
                                extra = thought_signature_extra(signature, data.get("part_kind"))
                                if extra:
                                    if stream_ordinary_metadata:
                                        stream_ordinary_metadata[-1]["thought_signature"] = _encode_wire_signature(signature)
                                    elif data.get("part_kind") == "signature_only":
                                        stream_part_order.append({"type": "ordinary", "index": 0})
                                        stream_ordinary_metadata.append(
                                            ordinary_part_metadata("signature_only", "", signature))
                                    google = extra.setdefault("google", {})
                                    if stream_ordinary_metadata:
                                        google["ordinary_parts"] = list(stream_ordinary_metadata)
                                    if got_tool_call:
                                        google["part_order"] = list(stream_part_order)
                                    if not has_yielded_to_client:
                                        yield _make_openai_chunk(response_id, model_display, role="assistant")
                                        has_yielded_to_client = True
                                    yield _make_openai_chunk(response_id, model_display, extra_content=extra)

                            elif status in ("text", "thought", "image", "api_error_text"):
                                if not has_yielded_to_client:
                                    print(f"⚡ [Studio] {base_model_name} | 连接建立，正在实时流式输出...")
                                    yield _make_openai_chunk(response_id, model_display, role="assistant")
                                    has_yielded_to_client = True
                                    # 预填充智能兼容：先把预填充文本作为回复开头发出
                                    if prefill_text:
                                        yield _make_openai_chunk(response_id, model_display, content=prefill_text)

                                if status == "thought":
                                    stream_part_order.append({"type": "ordinary", "index": len(stream_ordinary_metadata)})
                                    stream_ordinary_metadata.append(ordinary_part_metadata("thought", data, None))
                                    yield _make_openai_chunk(response_id, model_display, reasoning_content=data)
                                elif status == "api_error_text":
                                    yield _make_openai_chunk(response_id, model_display, content=f"\n[Studio API 错误] {data}")
                                elif status == "image":
                                    got_text = True
                                    stream_part_order.append({"type": "ordinary", "index": len(stream_ordinary_metadata)})
                                    stream_ordinary_metadata.append(ordinary_part_metadata("text", data, None))
                                    yield _make_openai_chunk(response_id, model_display, content=data)
                                else:  # text（经过预填充去重器）
                                    got_text = True
                                    out = deduper.feed(data)
                                    if out:
                                        stream_part_order.append({"type": "ordinary", "index": len(stream_ordinary_metadata)})
                                        stream_ordinary_metadata.append(ordinary_part_metadata("text", out, None))
                                        yield _make_openai_chunk(response_id, model_display, content=out)

                            elif status == "finish":
                                finish_raw = data
                            elif status == "safety":
                                # 最后一块才到，作为一个独立 chunk 追加在正文之后
                                _sh = _safety_html_if_enabled(data)
                                if _sh:
                                    yield _make_openai_chunk(response_id, model_display, content=_sh)
                            elif status == "usage":
                                usage_meta = data
                            elif status == "blocked":
                                blocked_msg = data

                            # 如果属于 Cookie 权限错误
                            elif status == "cookie_error":
                                # hybrid 故障转移：会话失效且未出流 → 抛给路由层切兜底通道。
                                # Cookie 挂了不代表 API Key 不可用，让请求成功、日志提示刷新 Cookie。
                                if failover_mode and not has_yielded_to_client:
                                    raise UpstreamUnstartedError(data)
                                if not has_yielded_to_client:
                                    yield _make_openai_chunk(response_id, model_display, role="assistant")
                                stats.add_error()
                                print(f"🔑 [Studio] 权限错误: {data[:150]}")
                                yield _make_openai_chunk(response_id, model_display, content=f"[Studio 权限错误] {data}")
                                yield _make_openai_chunk(response_id, model_display, finish_reason="stop")
                                yield "data: [DONE]\n\n"
                                return

                            # 如果属于其他网络故障或限流错误
                            elif status in ("retryable_error", "fatal_error"):
                                # 未发送任何有效数据前发生可重试错误 -> 触发安全退避重试
                                if not has_yielded_to_client and status == "retryable_error" and attempt < retry_max:
                                    should_retry = True
                                    error_to_raise = data
                                    break  # 跳出当前 async for，进入外部循环的 sleep 阶段
                                else:
                                    # hybrid 故障转移：重试耗尽（429 类限流）且未出流 → 抛给路由层切兜底通道。
                                    # fatal_error 是硬错误，不切换——切换也不会变好，直接如实报给前端。
                                    if failover_mode and not has_yielded_to_client and status == "retryable_error":
                                        raise UpstreamUnstartedError(data)
                                    # 如果已经开始了输出，或者错误不可重试，直接抛给前端结束
                                    if not has_yielded_to_client:
                                        yield _make_openai_chunk(response_id, model_display, role="assistant")

                                    err_prefix = "不可重试错误" if status == "fatal_error" else "重试耗尽"
                                    stats.add_error()
                                    print(f"❌ [Studio] {err_prefix} | {data[:150]}")
                                    yield _make_openai_chunk(response_id, model_display, content=f"\n[Studio 错误] {data}")
                                    yield _make_openai_chunk(response_id, model_display, finish_reason="stop")
                                    yield "data: [DONE]\n\n"
                                    return

                        # 如果标记了需要重试，在当前 attempt 结束时等待并开启下一次循环
                        if should_retry:
                            # 重试前再次确认客户端仍在
                            if await fastapi_request.is_disconnected():
                                print("ℹ️ [Studio] 客户端已断开连接，取消后续重试。")
                                return
                            wait_sec = backoff_sec
                            stats.add_retry()
                            print(f"⚠️ [Studio] 遇到可重试拥堵/限流: {error_to_raise[:80]}... {wait_sec}s 后进行第 {attempt+2} 次退避重试")
                            # 退避等待期间持续吐心跳，保活前端连接（issue4：3.1-pro 频繁 429 长等待易被前端断开）
                            async for _hb in _sleep_with_heartbeat(wait_sec):
                                if await fastapi_request.is_disconnected():
                                    print("ℹ️ [Studio] 客户端已断开连接，取消后续重试。")
                                    return
                                yield _hb
                            start_time = time.time()
                            continue

                        # ===== 正常收尾（修复旧版“空响应静默关流”，并显式暴露“只有思考没有正文”）=====
                        if not has_yielded_to_client:
                            yield _make_openai_chunk(response_id, model_display, role="assistant")
                            has_yielded_to_client = True
                            if prefill_text:
                                yield _make_openai_chunk(response_id, model_display, content=prefill_text)

                        # 预填充去重器里可能还攒着开头的一小段（短回复场景）
                        tail = deduper.flush()
                        if tail:
                            stream_part_order.append({"type": "ordinary", "index": len(stream_ordinary_metadata)})
                            stream_ordinary_metadata.append(ordinary_part_metadata("text", tail, None))
                            yield _make_openai_chunk(response_id, model_display, content=tail)

                        if got_text or got_tool_call:
                            stats.add_success()
                            if got_tool_call and stream_ordinary_metadata:
                                google = {
                                    "ordinary_parts": list(stream_ordinary_metadata),
                                    "part_order": list(stream_part_order),
                                }
                                last_signed = next((item for item in reversed(stream_ordinary_metadata)
                                                    if item.get("thought_signature")), None)
                                if last_signed:
                                    google["thought_signature"] = last_signed["thought_signature"]
                                    google["thought_signature_part"] = last_signed["kind"]
                                yield _make_openai_chunk(
                                    response_id, model_display, extra_content={"google": google})
                            yield _make_openai_chunk(
                                response_id, model_display,
                                finish_reason="tool_calls" if got_tool_call else _map_finish_reason(finish_raw))
                        else:
                            # 无正文：明确告知客户端 + 落诊断日志（旧版此处一个字节都不发就关流）
                            detail = f"finishReason={finish_raw or '无'}"
                            if blocked_msg:
                                detail += f"；promptFeedback 拦截：{blocked_msg}"
                            desc = "只返回了思考、未返回正文" if got_thought else "未返回任何内容"
                            hint = (_THINKING_RUNAWAY_HINT if (got_thought and not strip_thoughts) else _NO_BODY_HINT)
                            stats.add_error()
                            print(f"❌ [Studio] {base_model_name} | 上游{desc}（{detail}）")
                            print(f"🔎 [Studio 诊断] 原始响应样本：\n{sampler.dump()}")
                            yield _make_openai_chunk(
                                response_id, model_display,
                                content=f"\n[Studio 提示] 上游{desc}（{detail}）。{hint}")
                            filtered = bool(blocked_msg) or ((finish_raw or "").upper() in _FINISH_FILTERED)
                            yield _make_openai_chunk(response_id, model_display,
                                                     finish_reason="content_filter" if filtered else "stop")

                        usage = _map_usage(usage_meta)
                        if want_usage:
                            yield _make_usage_chunk(response_id, model_display, usage)

                        yield "data: [DONE]\n\n"

                        elapsed = time.time() - start_time
                        if got_text or got_tool_call:
                            print(f"✅ [Studio] {base_model_name} | 流式传输顺利完毕 | 耗时 {elapsed:.1f}s")
                        return

            return StreamingResponse(stream_generator(), media_type="text/event-stream")

        # ========== 非流式处理 ==========
        else:
            native_fallback_state = {"latched": False}
            for attempt in range(retry_max + 1):
                # 客户端已断开则立即停止，避免无谓的上游调用与重试
                if await fastapi_request.is_disconnected():
                    print("ℹ️ [Studio] 客户端已断开连接，停止非流式重试。")
                    return JSONResponse(status_code=499, content={
                        "error": {"message": "客户端已断开连接，请求已取消。", "type": "client_closed_request"}
                    })
                try:
                    use_native_tools = not native_fallback_state["latched"]
                    if use_native_tools:
                        body = await build_batch_graphql_body_async(
                            project_id, base_model_name, request_obj, prefill_active=prefill_active,
                            force_search=force_builtin_search)
                    else:
                        body = await build_batch_graphql_body_async(
                            project_id, base_model_name, request_obj, prefill_active=prefill_active,
                            force_search=force_builtin_search, native_tools=False)
                    fallback_body = None
                    if native_custom_tools and use_native_tools:
                        # Building can download/compress images, so keep it lazy
                        # until the native schema is actually rejected. Invoking
                        # this factory also latches degraded mode across retries.
                        async def fallback_body():
                            native_fallback_state["latched"] = True
                            return await build_batch_graphql_body_async(
                                project_id, base_model_name, request_obj,
                                prefill_active=prefill_active,
                                force_search=force_builtin_search, native_tools=False)
                    req_headers = build_headers(_get_cookie_string()) or headers
                    if cookie_debug:
                        print(f"🔎 [Studio 调试] 出站 generationConfig: "
                              f"{json.dumps(body['variables'].get('generationConfig', {}), ensure_ascii=False)}")

                    async with httpx.AsyncClient(**client_kwargs) as client:
                        response = await client.post(
                            BATCH_GRAPHQL_URL, headers=req_headers, json=body
                        )
                        if fallback_body is not None and _native_tool_error_response(
                                response.status_code, response.text):
                            native_fallback_state["latched"] = True
                            resolved_fallback = fallback_body()
                            if hasattr(resolved_fallback, "__await__"):
                                resolved_fallback = await resolved_fallback
                            print("⚠️ [Studio] 当前模型/协议拒绝原生函数 Schema；本次及后续重试固定降级为文本工具观测。")
                            response = await client.post(
                                BATCH_GRAPHQL_URL, headers=req_headers, json=resolved_fallback
                            )
                            fallback_body = None

                    if response.status_code in (429, 503, 500):
                        if attempt < retry_max:
                            wait_sec = backoff_sec
                            stats.add_retry()
                            print(f"⚠️ [Studio] HTTP {response.status_code} (尝试 {attempt+1}), {wait_sec}s 后重试...")
                            await asyncio.sleep(wait_sec)
                            continue

                    if response.status_code != 200:
                        elapsed = time.time() - start_time
                        print(f"❌ [Studio] {base_model_name} | HTTP {response.status_code} | {elapsed:.1f}s")
                        return JSONResponse(status_code=response.status_code, content={
                            "error": {"message": response.text[:500], "type": "upstream_error"}
                        })

                    full_text = ""
                    reasoning_text = ""
                    safety_html = ""
                    tool_calls = []
                    ordinary_metadata = []
                    part_order = []
                    message_extra = None
                    api_error = None
                    usage_meta = None
                    finish_raw = None
                    blocked_msg = None
                    sampler = _RawSampler()

                    class _FakeResponse:
                        def __init__(self, text):
                            self._text = text
                        async def aiter_text(self):
                            yield self._text

                    fake_resp = _FakeResponse(response.text)
                    async for obj in _iter_json_objects(fake_resp):
                        sampler.add(obj)
                        for event_type, data in _extract_from_results(obj):
                            if event_type == "text":
                                full_text += data
                                part_order.append({"type": "ordinary", "index": len(ordinary_metadata)})
                                ordinary_metadata.append(ordinary_part_metadata("text", data, None))
                            elif event_type == "thought":
                                reasoning_text += data   # 先收集（供诊断判断），输出时按 strip 决定是否附带
                                part_order.append({"type": "ordinary", "index": len(ordinary_metadata)})
                                ordinary_metadata.append(ordinary_part_metadata("thought", data, None))
                            elif event_type == "image":
                                full_text += data
                                part_order.append({"type": "ordinary", "index": len(ordinary_metadata)})
                                ordinary_metadata.append(ordinary_part_metadata("text", data, None))
                            elif event_type == "function_call":
                                part_order.append({"type": "tool_call", "index": len(tool_calls)})
                                tool_calls.append(_openai_tool_call(response_id, len(tool_calls), data))
                            elif event_type == "thought_signature":
                                sig = _signature_bytes_from_wire(data.get("value"))
                                if sig:
                                    message_extra = thought_signature_extra(sig, data.get("part_kind"))
                                    if ordinary_metadata:
                                        ordinary_metadata[-1]["thought_signature"] = _encode_wire_signature(sig)
                                    elif data.get("part_kind") == "signature_only":
                                        part_order.append({"type": "ordinary", "index": len(ordinary_metadata)})
                                        ordinary_metadata.append(ordinary_part_metadata("signature_only", "", sig))
                            elif event_type == "finish":
                                finish_raw = data
                            elif event_type == "safety":
                                safety_html = _safety_html_if_enabled(data)
                            elif event_type == "blocked":
                                blocked_msg = data
                            elif event_type == "usage":
                                usage_meta = data
                            elif event_type == "error":
                                err_msg = data.get("message", str(data)) if isinstance(data, dict) else str(data)
                                if _is_retryable_error(err_msg) and attempt < retry_max:
                                    api_error = err_msg
                                    break
                                full_text += f"\n[错误] {err_msg}"

                    if api_error and attempt < retry_max:
                        wait_sec = backoff_sec
                        stats.add_retry()
                        print(f"⚠️ [Studio] 429/限流 (尝试 {attempt+1}): {api_error[:100]}, {wait_sec}s 后重试...")
                        await asyncio.sleep(wait_sec)
                        continue

                    got_text = bool(full_text.strip())
                    got_output = got_text or bool(tool_calls)
                    elapsed = time.time() - start_time

                    if got_output:
                        # 预填充智能兼容：去重后把预填充文本拼回输出开头
                        if prefill_text and got_text:
                            full_text = prefill_text + strip_prefill_overlap(prefill_text, full_text)
                        finish_reason = "tool_calls" if tool_calls else _map_finish_reason(finish_raw)
                        stats.add_success()
                        print(f"✅ [Studio] {base_model_name} | {len(full_text)} 字符 | {elapsed:.1f}s")
                    else:
                        # 无正文（只有思考 / 完全空）：返回明确诊断而不是一个空格
                        detail = f"finishReason={finish_raw or '无'}"
                        if blocked_msg:
                            detail += f"；promptFeedback 拦截：{blocked_msg}"
                        desc = "只返回了思考、未返回正文" if reasoning_text else "未返回任何内容"
                        hint = (_THINKING_RUNAWAY_HINT if (reasoning_text and not strip_thoughts) else _NO_BODY_HINT)
                        stats.add_error()
                        print(f"❌ [Studio] {base_model_name} | 上游{desc}（{detail}）| {elapsed:.1f}s")
                        print(f"🔎 [Studio 诊断] 原始响应样本：\n{sampler.dump()}")
                        full_text = f"[Studio 提示] 上游{desc}（{detail}）。{hint}"
                        filtered = bool(blocked_msg) or ((finish_raw or "").upper() in _FINISH_FILTERED)
                        finish_reason = "content_filter" if filtered else "stop"

                    usage = _map_usage(usage_meta)

                    content_value = full_text + safety_html
                    message_obj = {"role": "assistant", "content": content_value if content_value else None}
                    if reasoning_text and not strip_thoughts:   # strip：batchGraphql 忽略 includeThoughts，输出侧剥离
                        message_obj["reasoning_content"] = reasoning_text
                    if tool_calls:
                        message_obj["tool_calls"] = tool_calls
                    if ordinary_metadata:
                        if message_extra is None:
                            message_extra = {"google": {}}
                        google = message_extra.setdefault("google", {})
                        google["ordinary_parts"] = ordinary_metadata
                        if tool_calls:
                            google["part_order"] = part_order
                    if message_extra:
                        message_obj["extra_content"] = message_extra

                    return JSONResponse(content={
                        "id": response_id,
                        "object": "chat.completion",
                        "created": int(time.time()),
                        "model": model_display,
                        "choices": [{
                            "index": 0,
                            "message": message_obj,
                            "finish_reason": finish_reason,
                        }],
                        "usage": usage,
                    })

                except Exception as e:
                    err_msg = str(e)
                    is_retryable = _is_retryable_error(err_msg) or "timeout" in err_msg.lower()

                    if is_retryable and attempt < retry_max:
                        wait_sec = backoff_sec
                        stats.add_retry()
                        print(f"⚠️ [Studio] 异常 (尝试 {attempt+1}): {err_msg[:100]}, {wait_sec}s 后重试...")
                        await asyncio.sleep(wait_sec)
                        continue

                    elapsed = time.time() - start_time
                    print(f"❌ [Studio] {base_model_name} | 异常 | {elapsed:.1f}s: {err_msg[:150]}")
                    traceback.print_exc()
                    return JSONResponse(status_code=500, content={
                        "error": {"message": f"batchGraphql proxy error: {err_msg}", "type": "proxy_error"}
                    })

            elapsed = time.time() - start_time
            print(f"❌ [Studio] {base_model_name} | 重试 {retry_max} 次后仍失败 | {elapsed:.1f}s")
            return JSONResponse(status_code=429, content={
                "error": {"message": "请求被限流，已重试多次仍失败。请稍后再试。", "type": "rate_limit_error"}
            })
