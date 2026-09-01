"""协议转换层（进阶报告 P1-⑦ 最小版拆分）。

从 api_helpers.py 机械搬移的**纯转换**函数与类：OpenAI 错误形状构造、
上游异常解析、请求参数（OpenAI→Gemini generationConfig）、Gemini 响应
有效性判定、流式 chunk（Gemini→OpenAI delta）、思考标签剥离器与工具调用
序号分配器。不做 IO、不碰统计、不含重试/故障转移语义——执行体
（execute_gemini_call / 流式 generator）仍留在 api_helpers.py。

兼容性：api_helpers 顶部 from protocol_conversion import *（显式名再导出），
既有 from api_helpers import X 的引用与测试零改动。
"""

import json
import re
import time
from typing import List, Dict, Any, Callable, Optional

from google.genai import types

from models import OpenAIRequest
import model_capabilities as mc
from runtime_state import app_state
import config as app_config
from config import VERTEX_REASONING_TAG

from message_processing import (
    parse_gemini_response_for_reasoning_and_content,
    build_tool_call_id,
    thought_signature_extra,
    ordinary_part_metadata,
    _signature_bytes,
    extract_reasoning_by_tags,
    PrefillDeduper,
    strip_prefill_overlap,
    _create_safety_ratings_html,
)
from signature_store import SignatureState
from anti_truncation import (
    has_synthetic_tool_call,
    strip_synthetic_from_openai_dict,
    strip_synthetic_from_stream_chunk,
)


def _safety_score_enabled() -> bool:
    """安全评分开关（随搬移的 convert_chunk_to_openai 一并迁来；纯读设置）。"""
    return bool(app_state.get_setting("safety_score", app_config.SAFETY_SCORE))


class StreamingReasoningProcessor:
    def __init__(self, tag_name: str = VERTEX_REASONING_TAG):
        self.tag_name = tag_name
        self.open_tag = f"<{tag_name}>"
        self.close_tag = f"</{tag_name}>"
        self.tag_buffer = ""
        self.inside_tag = False
        self._reasoning_chunks = []
        self.partial_tag_buffer = "" 

    def process_chunk(self, content: str) -> tuple[str, str]:
        if self.partial_tag_buffer:
            content = self.partial_tag_buffer + content
            self.partial_tag_buffer = ""
        self.tag_buffer += content
        
        processed_content_chunks = []
        current_reasoning_chunks = []
        
        while self.tag_buffer:
            if not self.inside_tag:
                open_pos = self.tag_buffer.find(self.open_tag)
                if open_pos == -1:
                    partial_match = False
                    for i in range(1, min(len(self.open_tag), len(self.tag_buffer) + 1)):
                        if self.tag_buffer[-i:] == self.open_tag[:i]:
                            partial_match = True
                            if len(self.tag_buffer) > i:
                                processed_content_chunks.append(self.tag_buffer[:-i])
                                self.partial_tag_buffer = self.tag_buffer[-i:]
                            else: 
                                self.partial_tag_buffer = self.tag_buffer
                            self.tag_buffer = ""
                            break
                    if not partial_match:
                        processed_content_chunks.append(self.tag_buffer)
                        self.tag_buffer = ""
                    break
                else:
                    processed_content_chunks.append(self.tag_buffer[:open_pos])
                    self.tag_buffer = self.tag_buffer[open_pos + len(self.open_tag):]
                    self.inside_tag = True
            else: 
                close_pos = self.tag_buffer.find(self.close_tag)
                if close_pos == -1:
                    partial_match = False
                    for i in range(1, min(len(self.close_tag), len(self.tag_buffer) + 1)):
                        if self.tag_buffer[-i:] == self.close_tag[:i]:
                            partial_match = True
                            if len(self.tag_buffer) > i:
                                new_reasoning = self.tag_buffer[:-i]
                                self._reasoning_chunks.append(new_reasoning)
                                if new_reasoning: current_reasoning_chunks.append(new_reasoning)
                                self.partial_tag_buffer = self.tag_buffer[-i:]
                            else: 
                                self.partial_tag_buffer = self.tag_buffer
                            self.tag_buffer = ""
                            break
                    if not partial_match:
                        if self.tag_buffer:
                            self._reasoning_chunks.append(self.tag_buffer)
                            current_reasoning_chunks.append(self.tag_buffer)
                            self.tag_buffer = ""
                    break
                else:
                    final_reasoning_chunk = self.tag_buffer[:close_pos]
                    self._reasoning_chunks.append(final_reasoning_chunk)
                    if final_reasoning_chunk: current_reasoning_chunks.append(final_reasoning_chunk)
                    
                    self.tag_buffer = self.tag_buffer[close_pos + len(self.close_tag):]
                    self.inside_tag = False
                    
        return "".join(processed_content_chunks), "".join(current_reasoning_chunks)
    
    def flush_remaining(self) -> tuple[str, str]:
        remaining_content_chunks = []
        if self.partial_tag_buffer:
            remaining_content_chunks.append(self.partial_tag_buffer)
            self.partial_tag_buffer = ""
            
        if not self.inside_tag:
            if self.tag_buffer: remaining_content_chunks.append(self.tag_buffer)
        else:
            if self.tag_buffer: self._reasoning_chunks.append(self.tag_buffer)
            self.inside_tag = False
            
        remaining_content = "".join(remaining_content_chunks)
        remaining_reasoning = "".join(self._reasoning_chunks)
        
        self.tag_buffer = ""
        self._reasoning_chunks.clear()
        
        return remaining_content, remaining_reasoning
def create_openai_error_response(status_code: int, message: str, error_type: str) -> Dict[str, Any]:
    safe_message = re.sub(r"([?&]key=)[^&\s'\"]+", r"\1***HIDDEN_API_KEY***", message)
    return {
        "error": {
            "message": safe_message,
            "type": error_type,
            "code": status_code,
            "param": None
        }
    }
def extract_upstream_error(e: Exception) -> tuple[int, str]:
    """从上游异常里尽力提取 (HTTP 状态码, 简明消息)。

    google-genai 的 ClientError/ServerError 带 .code 与结构化 message；
    其余异常回退 500 + 类名+摘要。用于把 404/403/400 等如实透传给客户端，
    避免笼统的 500 Internal Server Error。
    """
    code = getattr(e, "code", None) or getattr(e, "status_code", None)
    msg = str(e)
    # 从形如 "{'error': {'code':404,'message':...}}" 的 message 里提取更干净的说明
    try:
        m = re.search(r"'message':\s*'([^']+)'", msg) or re.search(r'"message":\s*"([^"]+)"', msg)
        if m:
            msg = m.group(1)
    except Exception:
        pass
    if not isinstance(code, int) or not (400 <= code <= 599):
        low = str(e).lower()
        if "not found" in low or "404" in low:
            code = 404
        elif "permission" in low or "403" in low or "denied" in low:
            code = 403
        elif "invalid" in low or "400" in low:
            code = 400
        else:
            code = 500
    return code, msg
def create_generation_config(request: OpenAIRequest) -> Dict[str, Any]:
    config: Dict[str, Any] = {}
    
    system_texts = []
    for msg in request.messages:
        if msg.role == "system" and msg.content:
            if isinstance(msg.content, str):
                system_texts.append(msg.content)
            elif isinstance(msg.content, list):
                for part in msg.content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        system_texts.append(part.get("text", ""))
                    elif hasattr(part, "text") and isinstance(part.text, str):
                        system_texts.append(part.text)
                        
    # P1-9：旧代码无条件剥离生图模型的 system_instruction，且没有注释理由。
    # 官方并未禁止生图模型使用系统指令，但为避免直接改变既有行为，
    # 改成控制台开关 image_system_instruction（默认关，保持旧行为）。
    if system_texts:
        _is_image_name = "image" in request.model.lower()
        _allow_sys = (not _is_image_name) or bool(
            app_state.get_setting("image_system_instruction", False))
        if _allow_sys:
            config["system_instruction"] = "\n".join(system_texts)
    
    if request.temperature is not None: config["temperature"] = request.temperature
    if request.max_tokens is not None: 
        config["max_output_tokens"] = request.max_tokens
    elif getattr(request, "max_completion_tokens", None) is not None:
        config["max_output_tokens"] = request.max_completion_tokens
        
    if request.top_p is not None: config["top_p"] = request.top_p
    if request.top_k is not None: config["top_k"] = request.top_k
    # P1-3：OpenAI 允许 stop 为字符串，统一规范成数组
    if request.stop is not None:
        config["stop_sequences"] = [request.stop] if isinstance(request.stop, str) else list(request.stop)
    if request.seed is not None: config["seed"] = request.seed
    if request.n is not None: config["candidate_count"] = request.n
    
    if getattr(request, "presence_penalty", None) is not None: config["presence_penalty"] = request.presence_penalty
    if getattr(request, "frequency_penalty", None) is not None: config["frequency_penalty"] = request.frequency_penalty
        
    # P1-3：OpenAI 语义 logprobs=bool + top_logprobs=int；Gemini 语义 response_logprobs=bool + logprobs=int。
    _logprobs = getattr(request, "logprobs", None)
    _top_logprobs = getattr(request, "top_logprobs", None)
    if getattr(request, "response_logprobs", None) is not None:
        config["response_logprobs"] = request.response_logprobs
    if isinstance(_logprobs, bool):
        if _logprobs:
            config["response_logprobs"] = True
            config["logprobs"] = _top_logprobs if isinstance(_top_logprobs, int) else 1
    elif isinstance(_logprobs, int):
        config["logprobs"] = _logprobs
    elif isinstance(_top_logprobs, int):
        config["logprobs"] = _top_logprobs

    if getattr(request, "response_format", None) is not None:
        fmt = request.response_format
        fmt_type = fmt.get("type", "") if isinstance(fmt, dict) else getattr(fmt, "type", "")
        if fmt_type == "json_object":
            config["response_mime_type"] = "application/json"
        elif fmt_type == "json_schema":
            # OpenAI 结构化输出：{"type":"json_schema","json_schema":{"name":...,"schema":{...}}}
            config["response_mime_type"] = "application/json"
            json_schema_obj = fmt.get("json_schema") if isinstance(fmt, dict) else getattr(fmt, "json_schema", None)
            schema = None
            if isinstance(json_schema_obj, dict):
                schema = json_schema_obj.get("schema")
            if isinstance(schema, dict):
                schema = {k: v for k, v in schema.items() if k != "$schema"}
                config["response_schema"] = schema
    
    # 官方 2026 最新基准配置
    safety_threshold = "BLOCK_NONE"
    
    config["safety_settings"] = [
        types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold=safety_threshold),
        types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold=safety_threshold),
        types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold=safety_threshold),
        types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold=safety_threshold),
        types.SafetySetting(category="HARM_CATEGORY_JAILBREAK", threshold=safety_threshold)
    ]
    
    tools_list = []
    if request.tools:
        function_declarations = []
        for tool in request.tools:
            if tool.get("type") == "function":
                func_data = tool.get("function")
                if func_data:
                    declaration = {
                        "name": func_data.get("name"),
                        "description": func_data.get("description"),
                    }
                    parameters = func_data.get("parameters")
                    if isinstance(parameters, dict) and "$schema" in parameters:
                        parameters = parameters.copy()
                        del parameters["$schema"]
                    if parameters is not None:
                        declaration["parameters"] = parameters
                    declaration = {k: v for k, v in declaration.items() if v is not None}
                    if declaration.get("name"): 
                        function_declarations.append(declaration)
        if function_declarations:
            tools_list.append({"function_declarations": function_declarations})

    # 读取控制台设置与模型能力档案（优先级：单次请求 > 模型专属 > 全局 > 内置默认）
    settings = app_state.get_effective_settings(request.model)
    # 控制台的「采样参数处理」可覆盖版本自动判定（新模型版本号更小却已废弃采样时用）
    profile = mc.apply_sampling_policy(mc.get_profile(request.model), settings)
    is_image_model = profile["is_image"]

    if is_image_model:
        config["response_modalities"] = ["TEXT", "IMAGE"]

        # 宽高比：两通道共用解析（请求额外字段 > OpenAI size 映射 > 提示词 > 控制台默认，按模型白名单校验）
        target_ar = mc.resolve_aspect_ratio(request.model, request, settings)

        # 分辨率：请求 > 控制台默认，按模型白名单校验并回退
        image_size = mc.resolve_image_size(request.model, request, settings)
        image_config_args = {"image_size": image_size}
        if target_ar:
            image_config_args["aspect_ratio"] = target_ar

        config["image_config"] = types.ImageConfig(**image_config_args)
        # 生图模型不支持自定义函数调用；只有客户端明确声明搜索工具时才保留
        # google_search。普通生图请求与 tool_choice=none 都不得偷偷启用搜索。
        _search_names = {"google_search", "googlesearch", "web_search", "websearch", "search"}
        _declared_search = any(
            str(((tool.get("function") or {}).get("name") or "")).lower() in _search_names
            for tool in (request.tools or []) if isinstance(tool, dict)
        )
        _tools_disabled = isinstance(request.tool_choice, str) and request.tool_choice.lower() == "none"
        tools_list = ([{"google_search": {}}]
                      if _declared_search and not _tools_disabled and profile.get("supports_search")
                      else [])

        # 生图不支持的键（采样类由 sanitize 统一剥离，这里清理其余）
        for key in ["response_mime_type", "response_schema", "response_logprobs", "logprobs"]:
            config.pop(key, None)
    else:
        # 文本/多模态：客户端未显式传采样值时，应用控制台默认（仅注入该模型支持的键）
        if config.get("temperature") is None and settings.get("default_temperature") is not None:
            config["temperature"] = settings["default_temperature"]
        if config.get("top_p") is None and settings.get("default_top_p") is not None:
            config["top_p"] = settings["default_top_p"]
        if config.get("max_output_tokens") is None and settings.get("default_max_tokens") is not None:
            config["max_output_tokens"] = settings["default_max_tokens"]

    # 按模型家族剥离不支持的采样参数（例如 Gemini 3.x 弃用 temperature/top_p/top_k、不支持 candidate_count）
    mc.sanitize_sampling(config, profile)

    if tools_list:
        config["tools"] = tools_list

    tool_config = None
    if request.tool_choice and not is_image_model:
        choice = request.tool_choice
        mode = None
        allowed_functions = None
        if isinstance(choice, str):
            if choice == "none": mode = "NONE"
            elif choice == "auto": mode = "AUTO"
            elif choice == "required": mode = "ANY"
        elif isinstance(choice, dict) and choice.get("type") == "function":
            func_name = choice.get("function", {}).get("name")
            if func_name:
                mode = "ANY"
                allowed_functions = [func_name]
        if mode:
            config_dict = {"mode": mode}
            if allowed_functions: config_dict["allowed_function_names"] = allowed_functions
            tool_config = {"function_calling_config": config_dict}
    
    if tool_config: config["tool_config"] = tool_config
        
    return config
def is_gemini_response_valid(response: Any) -> bool:
    if response is None: return False
    if hasattr(response, "text") and isinstance(response.text, str) and response.text.strip(): return True
    if hasattr(response, "candidates") and response.candidates:
        for cand in response.candidates:
            if hasattr(cand, "text") and isinstance(cand.text, str) and cand.text.strip(): return True
            if hasattr(cand, "content") and hasattr(cand.content, "parts") and cand.content.parts:
                for part in cand.content.parts:
                    if getattr(part, "function_call", None) is not None: return True
                    if getattr(part, "inline_data", None) is not None: return True
                    if hasattr(part, "text") and isinstance(getattr(part, "text", None), str) and getattr(part, "text", "").strip(): return True
    return False
class ToolCallIndexer:
    """一次流式响应内，为 tool_calls 分配稳定递增的 index（P0-3）。

    OpenAI 客户端按 delta.tool_calls[].index 累积函数调用。旧实现把 index 硬编码成 0，
    多个并行调用会被前端合并成一个；同时旧实现遇到第一个 function_call 就 break，
    直接丢掉其余并行调用——而官方要求并行调用必须完整按
    FC1,FC2,FR1,FR2 的顺序回传，缺一个就 400。

    序号必须由调用方持有：并行调用可能分布在不同 chunk 里。
    每次重试都要新建一个实例（重试会重发整轮，序号需要归零）。
    """

    def __init__(self):
        self._next: Dict[int, int] = {}
        self._seen_tool_calls: set[int] = set()
        self._ordinary_parts: Dict[int, list] = {}
        self._part_order: Dict[int, list] = {}

    def next_index(self, candidate_index: int = 0) -> int:
        i = self._next.get(candidate_index, 0)
        self._next[candidate_index] = i + 1
        self._seen_tool_calls.add(candidate_index)
        return i

    def record_tool_call(self, candidate_index: int, tool_index: int) -> None:
        self._part_order.setdefault(candidate_index, []).append(
            {"type": "tool_call", "index": tool_index})

    def record_ordinary_part(self, candidate_index: int, metadata: dict) -> int:
        items = self._ordinary_parts.setdefault(candidate_index, [])
        index = len(items)
        items.append(metadata)
        self._part_order.setdefault(candidate_index, []).append(
            {"type": "ordinary", "index": index})
        return index

    def topology(self, candidate_index: int = 0) -> tuple[list, list]:
        return (
            list(self._ordinary_parts.get(candidate_index, [])),
            list(self._part_order.get(candidate_index, [])),
        )

    def has_tool_calls(self, candidate_index: int = 0) -> bool:
        return candidate_index in self._seen_tool_calls
def convert_chunk_to_openai(chunk: Any, model_name: str, response_id: str, candidate_index: int = 0,
                            indexer: Optional[ToolCallIndexer] = None) -> str:
    from message_processing import (
        parse_gemini_response_for_reasoning_and_content,
        build_tool_call_id,
        thought_signature_extra,
        ordinary_part_metadata,
        _convert_image_to_markdown,
        _signature_bytes,
    )
    from signature_store import SignatureState
    delta_payload = {}
    openai_finish_reason = None

    if hasattr(chunk, "candidates") and chunk.candidates and len(chunk.candidates) > candidate_index:
        candidate = chunk.candidates[candidate_index]
        raw_gemini_finish_reason = getattr(candidate, "finish_reason", None)
        if raw_gemini_finish_reason:
            if hasattr(raw_gemini_finish_reason, "name"): raw_gemini_finish_reason_str = raw_gemini_finish_reason.name.upper()
            else: raw_gemini_finish_reason_str = str(raw_gemini_finish_reason).upper()

            if raw_gemini_finish_reason_str == "STOP": openai_finish_reason = "stop"
            elif raw_gemini_finish_reason_str == "MAX_TOKENS": openai_finish_reason = "length"
            elif raw_gemini_finish_reason_str == "SAFETY": openai_finish_reason = "content_filter"
            elif raw_gemini_finish_reason_str in ["TOOL_CODE", "FUNCTION_CALL"]: openai_finish_reason = "tool_calls"

        # Collect every parallel call and keep signatures on their exact tool-call
        # deltas. Message-level signatures cover ordinary/signature-only Parts.
        tool_call_deltas = []
        ordinary_metadata = []
        part_order = []
        ordinary_signature = None
        ordinary_signature_kind = None
        if hasattr(candidate, "content") and hasattr(candidate.content, "parts") and candidate.content.parts:
            for part in candidate.content.parts:
                fc = getattr(part, "function_call", None)
                sig = _signature_bytes(part, fc)
                if fc is None:
                    text_value = getattr(part, "text", None)
                    if text_value is None and getattr(part, "inline_data", None) is not None:
                        inline = part.inline_data
                        text_value = _convert_image_to_markdown(inline.data, inline.mime_type)
                    elif text_value is None and getattr(part, "file_data", None) is not None:
                        text_value = f"![Image]({part.file_data.file_uri})"
                    text_value = "" if text_value is None else str(text_value)
                    if getattr(part, "thought", None) is True:
                        kind = "thought"
                    elif text_value == "" and sig:
                        kind = "signature_only"
                    else:
                        kind = "text"
                    metadata = ordinary_part_metadata(kind, text_value, sig)
                    if indexer:
                        ordinary_index = indexer.record_ordinary_part(candidate_index, metadata)
                    else:
                        ordinary_index = len(ordinary_metadata)
                    part_order.append({"type": "ordinary", "index": ordinary_index})
                    ordinary_metadata.append(metadata)
                    if sig:
                        ordinary_signature = sig
                        ordinary_signature_kind = kind
                    continue
                tc_index = indexer.next_index(candidate_index) if indexer else len(tool_call_deltas)
                tc = {
                    "index": tc_index,
                    "id": build_tool_call_id(
                        fc, part,
                        missing_state=(SignatureState.UNSIGNED_FOLLOWER
                                       if tc_index > 0 else SignatureState.UNKNOWN)),
                    "type": "function",
                    "function": {
                        "name": fc.name,
                        "arguments": json.dumps(fc.args) if fc.args is not None else "",
                    },
                }
                extra = thought_signature_extra(sig)
                if extra:
                    tc["extra_content"] = extra
                part_order.append({"type": "tool_call", "index": tc_index})
                if indexer:
                    indexer.record_tool_call(candidate_index, tc_index)
                tool_call_deltas.append(tc)

        # Google often emits the functionCall in one chunk and a final STOP in a
        # later empty chunk. Once this candidate has produced any tool call, the
        # OpenAI finish reason for that turn must remain tool_calls.
        if (openai_finish_reason == "stop" and indexer
                and indexer.has_tool_calls(candidate_index)):
            openai_finish_reason = "tool_calls"

        if tool_call_deltas:
            delta_payload["tool_calls"] = tool_call_deltas

        reasoning_text, normal_text = parse_gemini_response_for_reasoning_and_content(candidate)

        # Only append safety ratings to the final chunk.
        if (openai_finish_reason and _safety_score_enabled()
                and hasattr(candidate, "safety_ratings") and candidate.safety_ratings):
            normal_text += _create_safety_ratings_html(candidate.safety_ratings)

        if reasoning_text:
            delta_payload["reasoning_content"] = reasoning_text
        if normal_text:
            delta_payload["content"] = normal_text
        elif tool_call_deltas:
            delta_payload["content"] = None
        elif not reasoning_text and openai_finish_reason is None:
            delta_payload["content"] = ""

        google_extra = {}
        if indexer:
            metadata_to_emit, order_to_emit = indexer.topology(candidate_index)
        else:
            metadata_to_emit, order_to_emit = ordinary_metadata, part_order

        # Emit cumulative topology on every later chunk (including the final STOP)
        # so standard OpenAI SSE aggregation—which replaces extension objects—keeps
        # the complete cross-chunk Part order and every ordinary signature.
        cumulative_signature = None
        cumulative_signature_kind = None
        for item in reversed(metadata_to_emit):
            if item.get("thought_signature"):
                cumulative_signature = item["thought_signature"]
                cumulative_signature_kind = item.get("kind")
                break
        if cumulative_signature:
            google_extra["thought_signature"] = cumulative_signature
            google_extra["thought_signature_part"] = cumulative_signature_kind
        elif ordinary_signature:
            google_extra.update((thought_signature_extra(
                ordinary_signature, ordinary_signature_kind) or {}).get("google", {}))
        if metadata_to_emit:
            google_extra["ordinary_parts"] = metadata_to_emit
        if order_to_emit:
            google_extra["part_order"] = order_to_emit
        if google_extra:
            delta_payload["extra_content"] = {"google": google_extra}
    
    if not delta_payload and openai_finish_reason is None:
        delta_payload["content"] = ""

    chunk_data = {
        "id": response_id, "object": "chat.completion.chunk", "created": int(time.time()), "model": model_name,
        "choices": [{"index": candidate_index, "delta": delta_payload, "finish_reason": openai_finish_reason}]
    }
    return f"data: {json.dumps(chunk_data)}\n\n"
def create_final_chunk(model: str, response_id: str, candidate_count: int = 1) -> str:
    choices = [{"index": i, "delta": {}, "finish_reason": "stop"} for i in range(candidate_count)]
    final_chunk_data = {"id": response_id, "object": "chat.completion.chunk", "created": int(time.time()), "model": model, "choices": choices}
    return f"data: {json.dumps(final_chunk_data)}\n\n"
