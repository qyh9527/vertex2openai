import json

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, StreamingResponse
from models import OpenAIRequest
from auth import get_api_key

# 引入运行状态管理器与多通道分发策略
from runtime_state import app_state
from upstreams.express_sdk import ExpressSDKUpstream
from upstreams.cookie_proxy import CookieProxyUpstream
from api_helpers import extract_upstream_error, create_openai_error_response, is_retryable_exception
from failover import breaker, UpstreamUnstartedError
import config as app_config

router = APIRouter()

# 实例化多通道策略
express_upstream = ExpressSDKUpstream()
cookie_upstream = CookieProxyUpstream()

CHANNELS = {
    "express": express_upstream,
    "cookie": cookie_upstream,
}

# 可切换错误白名单：只有这些状态码才触发跨通道故障转移。
# 400/401/403 等属于配置、鉴权、权限问题，切换通道不会变好，直接如实报错。
SWITCHABLE_STATUS_CODES = {429, 500, 502, 503, 504}


def _channel_order(strategy: str) -> list:
    """按策略返回通道尝试顺序：
    - express  -> [express]                只走 API Key（默认，向后兼容）
    - cookie   -> [cookie]                 只走 Cookie 直连反代
    - hybrid   -> [express, cookie]        Express 主，限流/故障自动切 Cookie 兜底
    """
    if strategy == "cookie":
        return ["cookie"]
    if strategy == "hybrid":
        return ["express", "cookie"]
    return ["express"]


def _available_channels(order: list) -> list:
    """可用性预检：凭证都没配的通道直接剔除，别等请求打到才报错。

    - express：至少配置了一个 VERTEX_EXPRESS_API_KEY
    - cookie ：控制台已保存 Cookie 与 Project ID（环境变量也算）
    """
    out = []
    for channel in order:
        if channel == "express":
            # 实际生效的 key：控制台列表优先，其次环境变量（与 ExpressKeyManager 一致）
            if app_state.get_express_keys() or app_config.VERTEX_EXPRESS_API_KEY_VAL:
                out.append(channel)
            else:
                print("ℹ️ [通道预检] Express 通道跳过：未配置 VERTEX_EXPRESS_API_KEY。")
        else:  # cookie
            if app_state.get_cookie_accounts():
                out.append(channel)
            else:
                print("ℹ️ [通道预检] Cookie 通道跳过：未配置 Google Cookie / Project ID。")
    return out


def _switchable_json(resp: JSONResponse) -> bool:
    """JSONResponse 是否属于"可切换通道"的失败（限流/上游 5xx）。"""
    return resp.status_code in SWITCHABLE_STATUS_CODES


async def _dispatch(channels: list, request: OpenAIRequest,
                    fastapi_request: Request, failover_mode: bool):
    """按通道顺序尝试，第一个成功即返回；失败且可切换则依次兜底。"""
    if not channels:
        return JSONResponse(
            status_code=503,
            content=create_openai_error_response(
                503, "当前策略下没有可用的上游通道：请配置 Express API Key 或 Google Cookie 与 Project ID。",
                "upstream_error"),
        )

    last_status, last_msg = 500, "上游通道全部不可用"
    for idx, channel in enumerate(channels):
        # 熔断冷却中的通道直接跳过（日志由熔断器打出）
        if breaker.is_cooling(channel):
            last_status, last_msg = 503, f"{channel} 通道处于熔断冷却中"
            continue

        upstream = CHANNELS[channel]
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
            if isinstance(resp, JSONResponse) and remaining_channels(channels, idx) \
                    and _switchable_json(resp):
                breaker.report_failure(channel)
                print(f"⚠️ [故障转移] {channel} 通道 HTTP {resp.status_code}（限流/上游故障），"
                      f"切换至 {channels[idx + 1]} 通道兜底。")
                last_status, last_msg = resp.status_code, "限流或上游故障"
                continue

            breaker.report_success(channel)
            return resp

        except UpstreamUnstartedError as e:
            if not failover_mode:
                raise  # 非 hybrid：upstream 不应抛此异常；透传给外层兜底
            if remaining_channels(channels, idx):
                breaker.report_failure(channel)
                print(f"⚠️ [故障转移] {channel} 通道未出流失败（{str(e)[:120]}），"
                      f"切换至 {channels[idx + 1]} 通道兜底。")
                last_status, last_msg = 503, str(e)
                continue
            # 无兜底通道：如实转成 OpenAI 错误响应（不再 raise，避免被外层误判成 500）
            breaker.report_failure(channel)
            code, msg = extract_upstream_error(ValueError(str(e)))
            return JSONResponse(status_code=code, content=create_openai_error_response(code, msg, "upstream_error"))
        except Exception as e:
            # 非流式/其他异常：只对"可切换"类错误继续兜底（429/503/配额等）
            if remaining_channels(channels, idx) and _exception_switchable(e):
                breaker.report_failure(channel)
                print(f"⚠️ [故障转移] {channel} 通道异常（{str(e)[:120]}），"
                      f"切换至 {channels[idx + 1]} 通道兜底。")
                last_status, last_msg = 503, str(e)
                continue
            raise

    # 全部通道尝试完毕仍失败：如实转成 OpenAI 错误格式
    code, msg = extract_upstream_error(
        ValueError(f"{last_msg}（最后尝试通道状态：{last_status}）"))
    return JSONResponse(status_code=code, content=create_openai_error_response(code, msg, "upstream_error"))


def remaining_channels(channels: list, idx: int) -> bool:
    return idx + 1 < len(channels)


def _exception_switchable(e: Exception) -> bool:
    """异常是否属于"可切换通道"类（限流/上游繁忙/网络故障）。"""
    return is_retryable_exception(e)


async def _stream_with_failover(primary_resp: StreamingResponse, remaining: list,
                                request: OpenAIRequest, fastapi_request: Request,
                                failover_mode: bool, primary_channel: str):
    """包装主通道的流式响应：

    - 正常：逐 chunk 透传（含 SSE 心跳），结束后给主通道记成功；
    - UpstreamUnstartedError：主通道"未出流"就挂了 → 有兜底则切剩余通道重发，
      无兜底则转成 SSE 错误流收尾。
      （已出流的错误由 upstream 内部发错误 chunk 收尾，这里不会看到异常。）
    """
    try:
        async for chunk in primary_resp.body_iterator:
            yield chunk
        breaker.report_success(primary_channel)
    except UpstreamUnstartedError as e:
        breaker.report_failure(primary_channel)
        if not remaining:
            print(f"❌ [故障转移] {primary_channel} 通道未出流失败且无兜底通道（{str(e)[:120]}）。")
            yield f"data: {json.dumps(create_openai_error_response(502, str(e)[:500], 'upstream_error'))}\n\n"
            yield "data: [DONE]\n\n"
            return
        print(f"⚠️ [故障转移] {primary_channel} 通道流式未出流失败（{str(e)[:120]}），"
              f"切换至 {remaining[0]} 通道重新发起请求。")
        switch_resp = await _dispatch(remaining, request, fastapi_request, failover_mode)
        if isinstance(switch_resp, StreamingResponse):
            async for chunk in switch_resp.body_iterator:
                yield chunk
        else:
            # 兜底通道非流式失败（JSONResponse）：转成 SSE 错误流收尾
            body = switch_resp.body
            yield f"data: {body.decode('utf-8', errors='replace')}\n\n"
            yield "data: [DONE]\n\n"


@router.post("/v1/chat/completions")
async def chat_completions(fastapi_request: Request, request: OpenAIRequest, api_key: str = Depends(get_api_key)):
    """
    /v1/chat/completions 多通道动态分流路由器

    通道策略（控制台「通道与凭证」页可切）：
    - express  -> 只走 ExpressSDKUpstream（官方 API Key 标准通道）
    - cookie   -> 只走 CookieProxyUpstream（Cookie 直连反代通道，规避 429 限流）
    - hybrid   -> Express 优先，限流/5xx/未出流失败自动切 Cookie 兜底；
                  任一通道连续失败自动熔断冷却（failover_threshold / failover_cooldown_seconds）

    统一异常兜底：把上游抛出的 404/403/400 等如实转成 OpenAI 错误格式，
    避免笼统的 500 Internal Server Error（非流式路径此前直接 raise）。
    统计由 main.py 的中间件按响应状态码统一计入，这里不重复计数。
    """
    strategy = app_state.get_channel_strategy()
    order = _available_channels(_channel_order(strategy))
    if not order:
        return JSONResponse(
            status_code=503,
            content=create_openai_error_response(
                503, f"当前策略（{strategy}）下没有可用通道：请配置 VERTEX_EXPRESS_API_KEY，"
                     "或在大盘控制台「通道与凭证」页粘贴 Google Cookie 与 Project ID。",
                "upstream_error"),
        )
    try:
        return await _dispatch(order, request, fastapi_request, failover_mode=(strategy == "hybrid"))
    except Exception as e:
        code, msg = extract_upstream_error(e)
        print(f"❌ [路由兜底] 模型 {request.model} 调用失败 | HTTP {code} | {msg[:200]}")
        return JSONResponse(status_code=code, content=create_openai_error_response(code, msg, "upstream_error"))
