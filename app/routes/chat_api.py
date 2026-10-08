import json
import random
import time

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, StreamingResponse
from models import OpenAIRequest
from auth import get_api_key

# 引入运行状态管理器与多通道分发策略
from runtime_state import app_state
from upstreams.express_sdk import ExpressSDKUpstream
from upstreams.cookie_proxy import CookieProxyUpstream
from upstreams.service_account import ServiceAccountUpstream
from api_helpers import (
    extract_upstream_error, create_openai_error_response, is_retryable_exception,
    channel_display_name, make_usage_chunk, wants_usage,
)
from usage_mapping import map_usage
import outcome as outcome_mod
from failover import breaker, UpstreamUnstartedError
import config as app_config
import request_log

router = APIRouter()

# 实例化多通道策略
express_upstream = ExpressSDKUpstream()
cookie_upstream = CookieProxyUpstream()
sa_upstream = ServiceAccountUpstream()

CHANNELS = {
    "express": express_upstream,
    "cookie": cookie_upstream,
    "vertex": sa_upstream,
}

# 通道显示名（日志/错误聚合用）：真身是 api_helpers.CHANNEL_META（P0-1 唯一显示来源），
# 这里只保留兼容导入供既有代码/测试引用；新代码请用 channel_display_name()。
CHANNEL_NAMES = {
    "express": "Express API Key",
    "cookie": "Cookie 直连",
    "vertex": "服务账号",
}

# 可切换错误判定：统一失败语义（outcome.py）接管，本常量保留供既有代码/测试引用；
# 等价性回归见 tests/test_outcome.py（429/500/502/503/504 可切换，400/401/403 等如实报错）。
SWITCHABLE_STATUS_CODES = {429, 500, 502, 503, 504}

KNOWN_FAILURE_CATEGORIES = {
    outcome_mod.RATE_LIMITED, outcome_mod.TRANSIENT,
    outcome_mod.AUTH_REFRESHABLE, outcome_mod.CREDENTIAL_PERMANENT,
    outcome_mod.REQUEST_PERMANENT, outcome_mod.POLICY_BLOCKED,
    outcome_mod.EMPTY_OR_PROTOCOL, outcome_mod.CLIENT_CLOSED, outcome_mod.OTHER,
}


def _known_category(value):
    return value if isinstance(value, str) and value in KNOWN_FAILURE_CATEGORIES else None


def _json_body(resp: JSONResponse):
    try:
        body = resp.body
        if isinstance(body, bytes):
            return json.loads(body.decode("utf-8", errors="replace"))
        return json.loads(body)
    except Exception:
        return {}


def _json_response_category(resp: JSONResponse):
    data = _json_body(resp)
    error = data.get("error")
    category = _known_category(error.get("category")) if isinstance(error, dict) else None
    if category is not None:
        return category
    for choice in data.get("choices") or []:
        if isinstance(choice, dict) and choice.get("finish_reason") == "content_filter":
            return outcome_mod.POLICY_BLOCKED
    return None


def _explicit_exception_category(error):
    return _known_category(getattr(error, "category", None))


def _attach_exception_report(response, error):
    report = getattr(error, "conversion_report", None)
    if report is not None:
        payload = json.loads(response.body)
        payload["error"].setdefault("extra_content", {}).setdefault("vertex2openai", {})["conversion_report"] = report
        response.body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        response.headers["content-length"] = str(len(response.body))
    return response


def _explicit_exception_response(error, category):
    code = getattr(error, "code", None) or getattr(error, "status_code", None)
    if not isinstance(code, int) or not 400 <= code <= 599:
        code = outcome_mod.CATEGORY_HTTP_STATUS.get(category, 502)
    error_type = getattr(error, "error_type", None) or {
        outcome_mod.CREDENTIAL_PERMANENT: "upstream_account_disabled",
        outcome_mod.EMPTY_OR_PROTOCOL: "upstream_protocol_error",
    }.get(category, "upstream_error")
    payload = create_openai_error_response(code, str(error), error_type)
    error_payload = payload["error"]
    error_payload["error_type"] = error_type
    error_payload["category"] = category
    finish_reason = getattr(error, "upstream_finish_reason", None)
    if finish_reason is not None:
        error_payload["upstream_finish_reason"] = finish_reason
    return _attach_exception_report(JSONResponse(status_code=code, content=payload), error)


def _channel_order(strategy: str) -> list:
    """按策略返回通道尝试顺序：
    - express  -> [express]                只走 API Key（默认，向后兼容）
    - cookie   -> [cookie]                 只走 Cookie 直连反代
    - vertex   -> [vertex]                 只走服务账号（标准 Vertex）
    - hybrid   -> 控制台可配顺序（hybrid_channels 设置，默认 [express, cookie]，
                  可在「混合自动」标签页加入 vertex 并排序）；random 模式下
                  每次请求对参与的通道做一次等概率随机排列
    """
    if strategy == "cookie":
        return ["cookie"]
    if strategy == "vertex":
        return ["vertex"]
    if strategy == "hybrid":
        channels = list(app_state.get_hybrid_channels())
        if app_state.get_hybrid_dispatch_mode() == "random":
            # sample 而不是按请求轮询：所有排列等概率，且不改动持久化的优先级列表。
            return random.sample(channels, k=len(channels))
        return channels
    return ["express"]


def _available_channels(order: list) -> list:
    """可用性预检：凭证都没配的通道直接剔除，别等请求打到才报错。

    - express：至少配置了一个 VERTEX_EXPRESS_API_KEY
    - cookie ：控制台已保存 Cookie 与 Project ID（环境变量也算）
    - vertex ：控制台已保存服务账号 JSON（环境变量 VERTEX_SA_JSON/VERTEX_SA_FILE 也算）
    """
    out = []
    for channel in order:
        if channel == "express":
            # 实际生效的 key：控制台列表优先，其次环境变量（与 ExpressKeyManager 一致）
            if app_state.get_express_keys() or app_config.VERTEX_EXPRESS_API_KEY_VAL:
                out.append(channel)
            else:
                print("ℹ️ [通道预检] Express 通道跳过：未配置 VERTEX_EXPRESS_API_KEY。")
        elif channel == "cookie":
            if app_state.get_cookie_accounts():
                out.append(channel)
            else:
                print("ℹ️ [通道预检] Cookie 通道跳过：未配置 Google Cookie / Project ID。")
        else:  # vertex
            if app_state.get_sa_accounts():
                out.append(channel)
            else:
                print("ℹ️ [通道预检] 服务账号通道跳过：未配置 SA JSON 凭证。")
    return out


def _switchable_json(resp: JSONResponse) -> bool:
    """JSONResponse 是否属于"可切换通道"的失败（限流/上游 5xx）。

    经统一失败语义分类判定（outcome.status_switchable），等价于既有
    SWITCHABLE_STATUS_CODES 白名单。
    """
    explicit_category = _json_response_category(resp)
    if explicit_category is not None:
        return outcome_mod.category_switchable(explicit_category)
    return outcome_mod.status_switchable(resp.status_code)


def _current_credential_id(channel: str) -> str:
    """本次请求在该通道实际选中的凭证脱敏标识（进阶报告 P0-2 候选粒度）。

    与请求级账号快照共享状态（cookie/vertex 取选中账号；express 的 Key 在
    upstream 内部选择，路由层拿不到稳定标识——留空退回通道粒度，等 P2
    candidate planner 接管后再补）。取不到返回空串。
    """
    try:
        if channel == "cookie":
            return app_state.current_cookie_credential_id()
        if channel == "vertex":
            return app_state.current_sa_credential_id()
    except Exception:
        pass
    return ""


def _summarize_json_error(resp: JSONResponse) -> str:
    """从上游 JSONResponse 里提取一行错误摘要（不消费响应体，P0-6）。

    响应体形如 {"error": {"message": ...}}；提取失败回退为"无错误详情"。
    """
    try:
        import json as _json
        body = resp.body
        if isinstance(body, bytes):
            data = _json.loads(body.decode("utf-8", errors="replace"))
        else:
            data = _json.loads(body)
        msg = (((data or {}).get("error") or {}).get("message")) or data.get("message")
        if msg:
            return str(msg)[:200]
    except Exception:
        pass
    return "无错误详情"


def _extract_stream_message(chunk: str) -> str:
    """从一条 SSE chunk 里提取 OpenAI 错误 message（无错误返回空串）。"""
    try:
        if not isinstance(chunk, str) or not chunk.startswith("data:"):
            return ""
        payload = chunk[len("data:"):].strip()
        if payload == "[DONE]" or not payload:
            return ""
        data = json.loads(payload)
        err = (data or {}).get("error")
        if isinstance(err, dict) and err.get("message"):
            return str(err["message"])[:300]
    except Exception:
        pass
    return ""


def _sse_payload(chunk):
    if not isinstance(chunk, str) or not chunk.startswith("data:"):
        return None
    payload = chunk[len("data:"):].strip()
    if not payload or payload == "[DONE]":
        return None
    try:
        return json.loads(payload)
    except Exception:
        return None


def _sse_done(chunk):
    return isinstance(chunk, str) and chunk.startswith("data:") \
        and chunk[len("data:"):].strip() == "[DONE]"


def _sse_error_event(chunk):
    data = _sse_payload(chunk)
    return isinstance(data, dict) and isinstance(data.get("error"), dict)


def _sse_terminal_category(chunk):
    data = _sse_payload(chunk)
    error = data.get("error") if isinstance(data, dict) else None
    return _known_category(error.get("category")) if isinstance(error, dict) else None


def _sse_content_filter(chunk):
    data = _sse_payload(chunk)
    if not isinstance(data, dict):
        return False
    return any(isinstance(choice, dict) and choice.get("finish_reason") == "content_filter"
               for choice in data.get("choices") or [])


def _chunk_has_effective_output(chunk: str) -> bool:
    """一条 SSE chunk 是否携带"有效输出"（正文/图片/工具调用/明确错误事件，P0-7）。

    无效保活：SSE 注释心跳（`: keep-alive`）、空 content 的 role/usage 尾块、只有 [DONE]。
    """
    try:
        if not isinstance(chunk, str) or not chunk.startswith("data:"):
            return False   # SSE 注释行（心跳）等
        payload = chunk[len("data:"):].strip()
        if payload == "[DONE]" or not payload:
            return False
        data = json.loads(payload)
        if isinstance(data.get("error"), dict) and data["error"].get("message"):
            return True    # 明确的 OpenAI 错误事件
        for choice in (data.get("choices") or []):
            delta = (choice or {}).get("delta") or {}
            content = delta.get("content")
            if isinstance(content, str) and content.strip():
                return True
            if delta.get("tool_calls"):
                return True
            extra = delta.get("extra_content")
            if isinstance(extra, dict) and (extra.get("image") or extra.get("images")):
                return True
            if choice.get("finish_reason"):
                return True   # 正常收尾块（空流不会有）
    except Exception:
        pass
    return False


async def _dispatch(channels: list, request: OpenAIRequest,
                    fastapi_request: Request, failover_mode: bool):
    """按通道顺序尝试，第一个成功即返回；失败且可切换则依次兜底。

    P0-6：每个已尝试通道的名称/状态/错误摘要都记进 attempts；全部失败时聚合进
    最终错误响应，前序通道的失败原因不再丢失。
    """
    if not channels:
        return JSONResponse(
            status_code=503,
            content=create_openai_error_response(
                503, "当前策略下没有可用的上游通道：请配置 Express API Key、Google Cookie 或服务账号 JSON。",
                "upstream_error"),
        )

    last_status, last_msg = 500, "上游通道全部不可用"
    attempts: list = []   # [{channel, status, message, category, upstream?}]（全部失败聚合用）
    for idx, channel in enumerate(channels):
        # 熔断冷却中的通道直接跳过（日志由熔断器打出）
        if breaker.is_cooling(channel):
            attempts.append({"channel": channel, "status": 503,
                             "message": "通道处于熔断冷却中",
                             "category": outcome_mod.TRANSIENT})
            last_status, last_msg = 503, f"{channel_display_name(channel)} 通道处于熔断冷却中"
            continue

        upstream = CHANNELS[channel]
        request_log.note_channel(channel)
        # 候选粒度：本次请求在该通道选中的凭证（脱敏 id；express 留空=通道粒度）
        cred = _current_credential_id(channel)
        try:
            resp = await upstream.chat_completions(
                request, fastapi_request, failover_mode=failover_mode)

            if isinstance(resp, StreamingResponse):
                if failover_mode:
                    # 统一用外层包装 generator 包住流式响应：只有"未出流失败"
                    # （UpstreamUnstartedError）才允许切换，已出流后原样透传收尾。
                    return StreamingResponse(
                        _stream_with_failover(resp, channels[idx + 1:], request, fastapi_request,
                                              failover_mode, channel),
                        media_type="text/event-stream",
                    )
                # 非 hybrid：原样透传，行为与改造前完全一致（零包装开销）
                return resp

            # 非流式 JSONResponse
            if isinstance(resp, JSONResponse):
                _explicit_category = _json_response_category(resp)
                if _explicit_category is not None and resp.status_code >= 400 \
                        and not outcome_mod.category_switchable(_explicit_category):
                    # 明确的永久 generation 失败不能误按 502 切换，也不计作连接故障。
                    attempts.append({"channel": channel, "status": resp.status_code,
                                     "message": _summarize_json_error(resp),
                                     "category": _explicit_category, "upstream": True})
                    return resp
                if failover_mode and _switchable_json(resp):
                    _summary = _summarize_json_error(resp)
                    _cat = _explicit_category or outcome_mod.classify_failure(
                        resp.status_code, _summary)
                    # P0-2：429 类带 Retry-After 语义的按精确窗口冷却候选（解析不出走通用计数）
                    if _cat == outcome_mod.RATE_LIMITED:
                        breaker.report_rate_limited(channel, credential_id=cred or None,
                                                    message=_summary)
                    else:
                        breaker.report_failure(channel, credential_id=cred or None)
                    attempts.append({"channel": channel, "status": resp.status_code,
                                     "message": _summary, "category": _cat,
                                     "upstream": True})
                    if remaining_channels(channels, idx):
                        request_log.channel_switch(
                            channel, channels[idx + 1],
                            f"⚠️ [故障转移] {channel_display_name(channel)} 通道 HTTP {resp.status_code}"
                            f"（{attempts[-1]['message'][:120]}），切换至 {channel_display_name(channels[idx + 1])} 通道兜底。",
                            error_summary=attempts[-1]['message'])
                        last_status, last_msg = resp.status_code, attempts[-1]["message"]
                        continue
                    # 无兜底通道：聚合返回（含本通道与所有前序通道的错误）
                    return _all_failed_response(attempts, resp.status_code, "")

                # 不可切换错误（400/401/403 等）如实返回；也记入 attempts 供日志聚合
                if resp.status_code >= 400:
                    _summary = _summarize_json_error(resp)
                    attempts.append({"channel": channel, "status": resp.status_code,
                                     "message": _summary,
                                     "category": (_explicit_category or outcome_mod.classify_failure(
                                         resp.status_code, _summary)),
                                     "upstream": True})
                if _explicit_category is not None:
                    return resp   # 显式终止/过滤结果不清除连接健康计数

            breaker.report_success(channel)
            if cred:
                breaker.report_success((channel, cred))
            return resp

        except UpstreamUnstartedError as e:
            _explicit_category = _explicit_exception_category(e)
            if _explicit_category is not None and not outcome_mod.category_switchable(
                    _explicit_category):
                # 上游包装的显式永久结束不是连接未出流故障。
                return _explicit_exception_response(e, _explicit_category)
            if not failover_mode:
                raise  # 非 hybrid：upstream 不应抛此异常；透传给外层兜底
            _cat = outcome_mod.classify_exception(e)
            if _cat == outcome_mod.RATE_LIMITED:
                breaker.report_rate_limited(channel, credential_id=cred or None, message=str(e))
            else:
                breaker.report_failure(channel, credential_id=cred or None)
            attempts.append({"channel": channel, "status": 503, "message": str(e)[:200],
                             "category": _cat})
            if remaining_channels(channels, idx):
                request_log.channel_switch(
                    channel, channels[idx + 1],
                    f"⚠️ [故障转移] {channel_display_name(channel)} 通道未出流失败（{str(e)[:120]}），"
                    f"切换至 {channel_display_name(channels[idx + 1])} 通道兜底。",
                    error_summary=str(e))
                last_status, last_msg = 503, str(e)
                continue
            # 无兜底通道：聚合所有尝试结果（P0-6），如实转成 OpenAI 错误响应
            return _attach_exception_report(_all_failed_response(attempts, last_status, last_msg), e)
        except Exception as e:
            explicit_category = _explicit_exception_category(e)
            if explicit_category is not None and not outcome_mod.category_switchable(
                    explicit_category):
                # 显式永久 generation 失败既不尝试其它通道，也不记作连接故障。
                return _explicit_exception_response(e, explicit_category)
            # 非流式/其他异常：只对"可切换"类错误继续兜底（429/503/配额等）
            _cat = outcome_mod.classify_exception(e)
            if _cat == outcome_mod.RATE_LIMITED:
                breaker.report_rate_limited(channel, credential_id=cred or None, message=str(e))
            else:
                breaker.report_failure(channel, credential_id=cred or None)
            if remaining_channels(channels, idx) and _exception_switchable(e):
                attempts.append({"channel": channel, "status": 503, "message": str(e)[:200],
                                 "category": _cat})
                request_log.channel_switch(
                    channel, channels[idx + 1],
                    f"⚠️ [故障转移] {channel_display_name(channel)} 通道异常（{str(e)[:120]}），"
                    f"切换至 {channel_display_name(channels[idx + 1])} 通道兜底。",
                    error_summary=str(e))
                last_status, last_msg = 503, str(e)
                continue
            if not remaining_channels(channels, idx):
                attempts.append({"channel": channel, "status": 503, "message": str(e)[:200],
                                 "category": outcome_mod.classify_exception(e)})
                return _attach_exception_report(_all_failed_response(attempts, 503, str(e)), e)
            raise

    # 全部通道尝试完毕仍失败：聚合每个通道的具体错误（P0-6）
    return _all_failed_response(attempts, last_status, last_msg)


def _all_failed_response(attempts: list, last_status: int, last_msg: str) -> JSONResponse:
    """所有候选通道失败后的聚合错误（OpenAI error 形状，含每个通道的摘要）。

    最终 HTTP 状态由 outcome.final_http_status 决定（P0-3 修复）：优先取最后一次
    「真实上游 JSONResponse」的状态码，不再从聚合字符串反推——避免
    "Express 429 + Cookie 503" 被洗成 500，客户端重试策略/监控图因此失真。
    """
    parts = []
    for a in attempts:
        name = channel_display_name(a["channel"])
        msg = (a.get("message") or "").strip() or "无错误详情"
        code = a.get("status") or "—"
        if str(msg) != msg:
            msg = str(msg)
        parts.append(f"{name} HTTP {code}: {msg}")
    summary = "；".join(parts) if parts else (last_msg or "上游通道全部不可用")
    print(f"❌ [路由兜底] 所有候选通道均失败：{summary[:400]}")
    final_status = outcome_mod.final_http_status(attempts, fallback_status=last_status)
    return JSONResponse(status_code=final_status,
                         content=create_openai_error_response(
                             final_status,
                             f"所有候选通道均失败：{summary[:600]}",
                             "upstream_error"))


def remaining_channels(channels: list, idx: int) -> bool:
    return idx + 1 < len(channels)


def _exception_switchable(e: Exception) -> bool:
    """异常是否属于"可切换通道"类（限流/上游繁忙/网络故障）。

    经统一失败语义分类判定（outcome.exception_switchable），等价于既有
    is_retryable_exception。
    """
    return outcome_mod.exception_switchable(e)


def _error_usage_tail(request: OpenAIRequest) -> str:
    """包装器自己生成错误收尾时的全零 usage 尾块（客户端要求 usage 时）。

    下游网关见不到 usage 会按响应字节数估算输出 token 计费。
    """
    if not wants_usage(request):
        return ""
    return make_usage_chunk(f"chatcmpl-error-{time.time_ns()}", request.model, map_usage())


async def _stream_with_failover(primary_resp: StreamingResponse, remaining: list,
                                request: OpenAIRequest, fastapi_request: Request,
                                failover_mode: bool, primary_channel: str):
    """包装主通道的流式响应：

    - 正常：逐 chunk 透传（含 SSE 心跳），结束后给主通道记成功；
    - UpstreamUnstartedError：主通道"未出流"就挂了 → 有兜底则切剩余通道重发，
      无兜底则转成 SSE 错误流收尾。
    - P0-7 空流判定：generator "正常结束"但全程只有心跳/空 delta/[DONE]（无有效输出）
      时不算成功——有兜底则切换重发，无兜底则发送可见 SSE 错误再 [DONE]，
      不再静默空回。
      （已出流的错误由 upstream 内部发错误 chunk 收尾，这里不会看到异常。）
    """
    has_output = False
    has_terminal_error = False
    has_policy_terminal = False
    done_seen = False
    opened_iterators = [primary_resp.body_iterator]   # 退出时逐个关闭，清理沿调用链向内传递
    try:
        async for chunk in primary_resp.body_iterator:
            if _sse_done(chunk):
                done_seen = True
                continue
            if _sse_terminal_category(chunk) is not None:
                has_terminal_error = True
            elif _sse_content_filter(chunk):
                has_policy_terminal = True
            if not has_output and _chunk_has_effective_output(chunk):
                has_output = True
            yield chunk
        if has_terminal_error or has_policy_terminal:
            yield "data: [DONE]\n\n"
            return
        if has_output:
            if done_seen:
                yield "data: [DONE]\n\n"
            breaker.report_success(primary_channel)
            return
        # 空流（只有心跳/空 delta/[DONE]）：按未出流失败处理
        breaker.report_failure(primary_channel)
        if remaining:
            request_log.channel_switch(
                primary_channel, remaining[0],
                f"⚠️ [故障转移] {channel_display_name(primary_channel)} 通道流式结束但无有效输出"
                f"（上游返回空流），切换至 {channel_display_name(remaining[0])} 通道重新发起请求。",
                error_summary="上游返回空流（无有效输出）")
            switch_resp = await _dispatch(remaining, request, fastapi_request, failover_mode)
            if isinstance(switch_resp, StreamingResponse):
                opened_iterators.append(switch_resp.body_iterator)
                async for chunk in switch_resp.body_iterator:
                    yield chunk
            else:
                # 兜底通道非流式失败（JSONResponse）：转成 SSE 错误流收尾
                body = switch_resp.body
                yield f"data: {body.decode('utf-8', errors='replace')}\n\n"
                if (tail := _error_usage_tail(request)):
                    yield tail
                yield "data: [DONE]\n\n"
            return
        # 无兜底：发送可见的 SSE 错误（P0-7 不静默空回）
        print(f"❌ [故障转移] {channel_display_name(primary_channel)} 通道流式结束但无有效输出"
              f"（上游返回空流），且无兜底通道。")
        err_payload = create_openai_error_response(
            502, f"{channel_display_name(primary_channel)} 通道上游返回空流（无正文/工具调用/错误事件），本次请求未获得有效回复。", "upstream_error")
        yield f"data: {json.dumps(err_payload, ensure_ascii=False)}\n\n"
        if (tail := _error_usage_tail(request)):
            yield tail
        yield "data: [DONE]\n\n"
        return
    except UpstreamUnstartedError as e:
        explicit_category = _explicit_exception_category(e)
        if explicit_category is not None and not outcome_mod.category_switchable(explicit_category):
            payload = _explicit_exception_response(e, explicit_category)
            yield f"data: {payload.body.decode('utf-8', errors='replace')}\n\n"
            if (tail := _error_usage_tail(request)):
                yield tail
            yield "data: [DONE]\n\n"
            return
        breaker.report_failure(primary_channel)
        if not remaining:
            print(f"❌ [故障转移] {channel_display_name(primary_channel)} 通道未出流失败且无兜底通道（{str(e)[:120]}）。")
            payload = _attach_exception_report(
                JSONResponse(content=create_openai_error_response(502, str(e)[:500], 'upstream_error')), e)
            yield f"data: {payload.body.decode('utf-8')}\n\n"
            if (tail := _error_usage_tail(request)):
                yield tail
            yield "data: [DONE]\n\n"
            return
        request_log.channel_switch(
            primary_channel, remaining[0],
            f"⚠️ [故障转移] {channel_display_name(primary_channel)} 通道流式未出流失败（{str(e)[:120]}），"
            f"切换至 {channel_display_name(remaining[0])} 通道重新发起请求。",
            error_summary=str(e))
        switch_resp = await _dispatch(remaining, request, fastapi_request, failover_mode)
        if isinstance(switch_resp, StreamingResponse):
            opened_iterators.append(switch_resp.body_iterator)
            async for chunk in switch_resp.body_iterator:
                yield chunk
        else:
            # 兜底通道非流式失败（JSONResponse）：转成 SSE 错误流收尾
            body = switch_resp.body
            yield f"data: {body.decode('utf-8', errors='replace')}\n\n"
            if (tail := _error_usage_tail(request)):
                yield tail
            yield "data: [DONE]\n\n"
    finally:
        for opened in reversed(opened_iterators):
            await request_log.aclose_quietly(opened)


def _request_mode(request: OpenAIRequest) -> str:
    """请求日志里的模式：非流式 / 假流式（fake- 前缀）/ 流式（生图模型强制假流式由生成器事件更新）。"""
    if not request.stream:
        return "non_stream"
    name = (request.model or "").strip()
    if name.upper().startswith("[PAY]"):
        name = name[len("[PAY]"):].strip()
    return "fake_stream" if name.lower().startswith("fake-") else "stream"


async def _chat_completions_with_strategy(fastapi_request: Request, request: OpenAIRequest,
                                          strategy: str):
    """按指定策略执行一次聊天请求；显式渠道路径传入单渠道策略。"""
    # 一个客户端请求 = 一个 request_id（贯穿重试与故障转移）；早于通道预检，预检日志也能归组
    ctx = request_log.begin_request(request.model, _request_mode(request), strategy)
    try:
        order = _available_channels(_channel_order(strategy))
        if not order:
            return request_log.track_response(ctx, JSONResponse(
                status_code=503,
                content=create_openai_error_response(
                    503, f"当前策略（{strategy}）下没有可用通道：请配置 VERTEX_EXPRESS_API_KEY、"
                         "Google Cookie 与服务账号 JSON 中的至少一种，"
                         "或在大盘控制台「通道与凭证」页配置。",
                    "upstream_error"),
            ))
        try:
            return request_log.track_response(ctx, await _dispatch(
                order, request, fastapi_request, failover_mode=(strategy == "hybrid")))
        except Exception as e:
            explicit_category = _explicit_exception_category(e)
            if explicit_category is not None:
                return request_log.track_response(ctx, _explicit_exception_response(e, explicit_category))
            code, msg = extract_upstream_error(e)
            print(f"❌ [路由兜底] 模型 {request.model} 调用失败 | HTTP {code} | {msg[:200]}")
            return request_log.track_response(ctx, _attach_exception_report(JSONResponse(status_code=code, content=create_openai_error_response(code, msg, "upstream_error")), e))
    except BaseException as error:
        # CancelledError 等：连接断开或任务取消；其余未预期异常记为失败。仍要收一个终态事件（finish 幂等）
        if isinstance(error, Exception):
            request_log.finish(ctx, "failed", error_type=type(error).__name__,
                               error_summary=request_log.redact(error))
        else:
            request_log.finish(ctx, "cancelled")
        raise


@router.post("/v1/chat/completions")
async def chat_completions(fastapi_request: Request, request: OpenAIRequest, api_key: str = Depends(get_api_key)):
    """
    /v1/chat/completions 多通道动态分流路由器

    通道策略（控制台「通道与凭证」页可切）：
    - express  -> 只走 ExpressSDKUpstream（官方 API Key 标准通道）
    - cookie   -> 只走 CookieProxyUpstream（Cookie 直连反代通道，规避 429 限流）
    - vertex   -> 只走 ServiceAccountUpstream（服务账号 JSON，标准 Vertex 认证）
    - hybrid   -> 按控制台「混合自动」标签页的通道顺序尝试（默认 Express 优先，
                  限流/5xx/未出流失败自动切 Cookie 兜底，可在控制台加入/排序服务账号通道）；
                  `hybrid_dispatch_mode=random` 时每次请求随机排列已勾选通道；
                  任一通道连续失败自动熔断冷却（failover_threshold / failover_cooldown_seconds）

    统一异常兜底：把上游抛出的 404/403/400 等如实转成 OpenAI 错误格式，
    避免笼统的 500 Internal Server Error（非流式路径此前直接 raise）。
    统计由 main.py 的中间件按响应状态码统一计入，这里不重复计数。
    """
    return await _chat_completions_with_strategy(
        fastapi_request, request, app_state.get_channel_strategy())


@router.post("/{channel}/v1/chat/completions")
async def channel_chat_completions(channel: str, fastapi_request: Request,
                                   request: OpenAIRequest,
                                   api_key: str = Depends(get_api_key)):
    """显式渠道入口：/express/v1、/cookie/v1、/vertex/v1。"""
    if channel not in CHANNELS:
        return JSONResponse(
            status_code=404,
            content=create_openai_error_response(
                404, f"未知的独立渠道：{channel}。可用渠道为 express / cookie / vertex。",
                "invalid_request_error"),
        )
    # 显式路径永远只走指定渠道，不读取控制台当前策略，也不触发 hybrid 故障转移。
    return await _chat_completions_with_strategy(fastapi_request, request, channel)
