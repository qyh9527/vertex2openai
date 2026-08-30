import re
from typing import Optional, Tuple

import httpx

from google.genai import types
import config as app_config


# ============================================================
# PayGo 流量等级（融合自 ST-Vertex-PayGo 的层级头方案）
#
# 官方"按量共享容量"层级通过请求头表达（仅对标准 Vertex 端点类通道有效，
# Express 与服务账号两通道共用；Cookie 通道走 batchGraphql 不适用）：
#   X-Vertex-AI-LLM-Request-Type: shared        标记按量共享（绕过预配吞吐）
#   X-Vertex-AI-LLM-Shared-Request-Type: flex    允许排队至 X-Server-Timeout（1800s）
#   X-Vertex-AI-LLM-Shared-Request-Type: priority  优先调度
# Flex/Priority 仅对 location=global 有效（非 global 自动降级并告警）。
# ============================================================
PRIORITY_PAYGO_HEADERS = {
    "X-Vertex-AI-LLM-Request-Type": "shared",
    "X-Vertex-AI-LLM-Shared-Request-Type": "priority",
}

PAYGO_TIERS = ("auto", "off", "standard", "flex", "priority")

# Flex 档允许上游排队 30 分钟，同步放大 httpx 超时
FLEX_TIMEOUT_SECONDS = 1800

# Flex 层级不支持 gemini-2.x 系（真机：gemini-2.5-flash 打 flex 头返回
# "400 Flex API is not supported for model"；3.x 及更新正常）。
# 自动化黑名单：只挡明确不支持的 2.x，其余（3.x / 未来模型）前向安全放行。
_FLEX_UNSUPPORTED_RE = re.compile(r"^gemini-?2\.\d", re.IGNORECASE)


def is_flex_supported(model_name: str) -> bool:
    """Flex 层级是否支持该模型（自动化黑名单判定，无需维护静态名单）。"""
    return not _FLEX_UNSUPPORTED_RE.match((model_name or "").strip())


def should_use_priority_paygo(model_path: str) -> bool:
    """Priority PayGo 只用于已明确钉定到 global 的模型资源路径。"""
    return "/locations/global/" in str(model_path or "")


def resolve_paygo_headers(tier: str, paygo_only: bool, is_global: bool) -> Tuple[dict, list]:
    """按流量等级设置解析要打的上游请求头；返回 (headers, 告警文案列表)。

    tier:
      auto     = 保持现状：global 请求打 priority 头（= should_use_priority_paygo 语义）
      off      = 不打任何层级头
      standard = 仅当 paygo_only=true 时打 "shared" 标记
      flex     = shared + flex + X-Server-Timeout: 1800（仅 global）
      priority = shared + priority（仅 global）
    非 global 打 flex/priority 会降级为空头并返回告警文案（调用方负责打印）。
    """
    tier = (tier or "auto").strip().lower()
    if tier == "off":
        return {}, []
    if tier == "auto":
        tier = "priority" if is_global else "off"
        if tier == "off":
            return {}, []
    warnings = []
    if tier in ("flex", "priority") and not is_global:
        return {}, [f"PayGo 层级 {tier} 仅对 global 请求有效，本次已降级为普通请求。"]
    headers: dict = {}
    if tier == "standard":
        if paygo_only:
            headers["X-Vertex-AI-LLM-Request-Type"] = "shared"
    elif tier == "flex":
        headers["X-Vertex-AI-LLM-Request-Type"] = "shared"
        headers["X-Vertex-AI-LLM-Shared-Request-Type"] = "flex"
        headers["X-Server-Timeout"] = str(FLEX_TIMEOUT_SECONDS)
    elif tier == "priority":
        headers["X-Vertex-AI-LLM-Request-Type"] = "shared"
        headers["X-Vertex-AI-LLM-Shared-Request-Type"] = "priority"
    return headers, warnings


def paygo_timeout(tier: str) -> Optional[int]:
    """流量等级要求的 httpx 超时（秒）；无特殊要求返回 None。"""
    return FLEX_TIMEOUT_SECONDS if (tier or "").strip().lower() == "flex" else None


def resolve_paygo_bundle(is_global: bool, settings: dict, model_name: str = ""):
    """按控制台设置 + 请求是否钉定 global + 目标模型，解析 PayGo 请求头三元组。

    返回 (headers, timeout, warnings)；headers 为空 = 不打层级头。
    供 Express 与服务账号两通道共用（cookie 通道不适用）。
    Flex 层级对不支持的模型（gemini-2.x 系）自动降级并告警，避免 400。
    """
    tier = (settings.get("paygo_tier", "auto") or "auto").strip().lower()
    paygo_only = bool(settings.get("paygo_only", False))
    headers, warnings = resolve_paygo_headers(tier, paygo_only, is_global)
    if tier == "flex" and headers and not is_flex_supported(model_name):
        return {}, None, warnings + [
            f"模型 {model_name} 不支持 Flex 层级（gemini-2.x 系打 flex 头会返回 400），本次已降级为普通请求。"]
    timeout = FLEX_TIMEOUT_SECONDS if headers.get("X-Vertex-AI-LLM-Shared-Request-Type") == "flex" else None
    return headers, timeout, warnings


def get_http_options(
    base_url: Optional[str] = None,
    *,
    priority_paygo: bool = False,
    headers: Optional[dict] = None,
    timeout: Optional[int] = None,
) -> Optional[types.HttpOptions]:
    """构造 google-genai HTTP 选项：代理、自定义证书、base_url、层级头与超时。

    priority_paygo（旧接口）为 True 时等价于打 Priority 头；新的调用方可直接传
    headers=resolve_paygo_headers(...) 的产物，timeout=paygo_timeout(...)。

    ⚠️ 代理/自定义证书走**预构建 httpx client**（httpx_client/httpx_async_client 字段），
    不传 client_args['proxy']：实测 genai 2.19 的 `_ensure_httpx_ssl_ctx` 会给 client_args
    注入一个 mTLS 默认 SSLContext，通过 HTTP 代理建 CONNECT 隧道时 start_tls 会 ConnectTimeout
    （Windows + 本地代理实测；VPS 直连无感）。预构建 client 用默认 SSL 栈，代理隧道正常。
    """
    client_args = {}
    if app_config.PROXY_URL:
        client_args["proxy"] = app_config.PROXY_URL
    if app_config.SSL_CERT_FILE:
        client_args["verify"] = app_config.SSL_CERT_FILE

    options = {}
    base_url = base_url or (app_config.VERTEX_BASE_URL or None)
    if base_url:
        options["base_url"] = base_url
    # 显式钉住 API 版本：thinking_level / thinking_budget 等参数只在 v1beta1 提供
    # （官方 REST 参考确认 v1beta1 同时有 generateContent 与 streamGenerateContent）。
    # 不钉住会随 google-genai 升级的默认值漂移，行为不可复现。
    options["api_version"] = "v1beta1"
    if client_args:
        hc_kwargs = dict(client_args)
        if timeout:
            hc_kwargs["timeout"] = timeout
        options["httpx_client"] = httpx.Client(**hc_kwargs)
        options["httpx_async_client"] = httpx.AsyncClient(**hc_kwargs)
    if headers:
        options["headers"] = dict(headers)
    elif priority_paygo:
        options["headers"] = dict(PRIORITY_PAYGO_HEADERS)
    if timeout and not client_args:
        options["timeout"] = int(timeout)

    return types.HttpOptions(**options) if options else None
