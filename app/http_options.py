from typing import Optional
from google.genai import types
import config as app_config


PRIORITY_PAYGO_HEADERS = {
    "X-Vertex-AI-LLM-Request-Type": "shared",
    "X-Vertex-AI-LLM-Shared-Request-Type": "priority",
}


def should_use_priority_paygo(model_path: str) -> bool:
    """Priority PayGo 只用于已明确钉定到 global 的模型资源路径。"""
    return "/locations/global/" in str(model_path or "")


def get_http_options(
    base_url: Optional[str] = None,
    *,
    priority_paygo: bool = False,
) -> Optional[types.HttpOptions]:
    """构造 google-genai HTTP 选项：代理、自定义证书、base_url 与请求头。"""
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
        options["client_args"] = client_args
        options["async_client_args"] = client_args
    if priority_paygo:
        options["headers"] = dict(PRIORITY_PAYGO_HEADERS)

    return types.HttpOptions(**options) if options else None
