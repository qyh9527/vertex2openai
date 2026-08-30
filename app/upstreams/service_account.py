"""
服务账号（Vertex SA）第三上游通道

用 Google Cloud 服务账号 JSON 凭证认证，走**标准 Vertex AI 端点**（与 Express 通道
同一套 generateContent / streamGenerateContent，但用 Bearer token 而非 API Key）。

认证流程（官方库等价实现，无需手写 OAuth2）：
  - service_account.Credentials.from_service_account_info(sa_json) 解析 SA JSON；
  - genai.Client(vertexai=True, project=, location=, credentials=) 构造客户端；
  - SDK 内部在 token 过期时自动走 OAuth2 JWT-bearer grant 换新并缓存
    （即 gproxy 手动 JWT 流程的官方等价物，token 约 1 小时惰性刷新）。

与 Express 通道的差异只有"客户端来源"与"模型路径"：
  - 客户端来自服务账号账号快照（contextvar，重试/流式/failover 重发不串号）；
  - 模型名传裸名，SDK 按 Client 的 project/location 自动拼完整资源路径
    （无需 Express 通道那套"钉定 location 靠模型路径透传"的 hack）。

请求管线（预填充/思考/工具/生图/重试/failover）全部继承 ExpressSDKUpstream。
"""

import json
import threading
from functools import partial
from typing import Any, Optional

from fastapi import Request
from fastapi.responses import JSONResponse
from google import genai
from google.oauth2 import service_account

from upstreams.express_sdk import ExpressSDKUpstream
from api_helpers import create_openai_error_response
from http_options import get_http_options, resolve_paygo_bundle
from runtime_state import app_state
import config as app_config

SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]


# ========== 服务账号 Client 复用缓存（与 Express 通道同款防护语义）==========
# 按 (sa_json, project_id, location, 层级头) 复用 genai.Client。SA 凭证对象自带
# token 缓存（惰性刷新），复用 Client 即省掉每次刷新与 TLS 握手。
# client_reuse 开关 / 连接级失败自动舍弃 / 硬错误立即舍弃 与 Express 通道共用同一套设置。
_SA_CLIENT_CACHE: dict = {}
_SA_CLIENT_CACHE_LOCK = threading.Lock()
_SA_CLIENT_FAILURES: dict = {}


def _on_sa_client_failure(cache_key: tuple, kind: str = "conn", reason: str = "") -> None:
    """复用 SA Client 的一次失败上报（同 Express 通道防护：kind=evict 立即舍弃 / conn 计数）。"""
    if kind == "evict":
        with _SA_CLIENT_CACHE_LOCK:
            _SA_CLIENT_CACHE.pop(cache_key, None)
            _SA_CLIENT_FAILURES.pop(cache_key, None)
        print(f"⚠️ [SA Client 复用] 已立即舍弃缓存 Client（{reason or '硬错误'}），下次请求重建连接池。")
        return
    try:
        threshold = int(app_state.get_setting(
            "client_reuse_evict_threshold",
            app_config.DEFAULT_SETTINGS["client_reuse_evict_threshold"]))
    except (TypeError, ValueError):
        threshold = app_config.DEFAULT_SETTINGS["client_reuse_evict_threshold"]
    if threshold <= 0:
        return
    with _SA_CLIENT_CACHE_LOCK:
        cnt = _SA_CLIENT_FAILURES.get(cache_key, 0) + 1
        if cnt >= threshold:
            _SA_CLIENT_CACHE.pop(cache_key, None)
            _SA_CLIENT_FAILURES.pop(cache_key, None)
            print(f"⚠️ [SA Client 复用] 缓存 Client 连续 {cnt} 次连接级失败，已舍弃，下次请求将重建连接池。")
        else:
            _SA_CLIENT_FAILURES[cache_key] = cnt


def build_sa_credentials(sa_json: str):
    """解析 SA JSON → google-auth Credentials（结构非法抛 ValueError）。"""
    info = json.loads(sa_json)
    return service_account.Credentials.from_service_account_info(info, scopes=SCOPES)


def _effective_project(sa_json: str, project_id: str) -> str:
    """project_id 为空时从 SA JSON 自身读取（凭证自带项目天然正确）。"""
    if project_id:
        return project_id
    try:
        return str(json.loads(sa_json).get("project_id") or "")
    except Exception:
        return ""


def validate_sa_credentials(sa_json: str) -> dict:
    """校验 SA JSON 结构；返回 {valid, message, project_id}（控制台新增账号时调用）。"""
    try:
        info = json.loads(sa_json)
    except json.JSONDecodeError as e:
        return {"valid": False, "message": f"不是合法 JSON：{e}", "project_id": ""}
    if not isinstance(info, dict):
        return {"valid": False, "message": "JSON 必须是对象。", "project_id": ""}
    for field in ("type", "client_email", "private_key", "project_id"):
        if not info.get(field):
            return {"valid": False, "message": f"缺少必需字段 {field}。", "project_id": ""}
    if info.get("type") != "service_account":
        return {"valid": False, "message": "type 字段必须是 service_account。", "project_id": ""}
    try:
        build_sa_credentials(sa_json)
    except Exception as e:
        return {"valid": False, "message": f"凭证构造失败：{e}", "project_id": ""}
    return {"valid": True, "message": "", "project_id": str(info.get("project_id") or "")}


def _new_sa_client(sa_json: str, project_id: str, location: str,
                   headers: Optional[dict], timeout: Optional[int],
                   cache_key: Any = None) -> Any:
    """构造服务账号认证的 genai.Client；复用模式下挂失败上报回调。"""
    creds = build_sa_credentials(sa_json)
    client = genai.Client(
        vertexai=True,
        project=_effective_project(sa_json, project_id) or None,
        location=location or "global",
        credentials=creds,
        http_options=get_http_options(headers=headers, timeout=timeout),
    )
    if cache_key is not None:
        client._vertex_cache_key = cache_key
        client._vertex_on_failure = partial(_on_sa_client_failure, cache_key)
    return client


def _get_cached_sa_client(sa_json: str, project_id: str, location: str,
                          headers: Optional[dict] = None, timeout: Optional[int] = None) -> Any:
    """取（必要时创建）复用的服务账号 genai.Client。

    缓存键含层级头：换 PayGo 层级必须换连接池语义（头不同不能复用旧连接）。
    client_reuse 关闭时每请求新建。
    """
    if not app_state.get_setting("client_reuse", True):
        return _new_sa_client(sa_json, project_id, location, headers, timeout, cache_key=None)
    headers_key = tuple(sorted((headers or {}).items()))
    cache_key = (sa_json, project_id, location, headers_key)
    with _SA_CLIENT_CACHE_LOCK:
        client = _SA_CLIENT_CACHE.get(cache_key)
        if client is None:
            client = _new_sa_client(sa_json, project_id, location, headers, timeout, cache_key=cache_key)
            _SA_CLIENT_CACHE[cache_key] = client
    return client


class ServiceAccountUpstream(ExpressSDKUpstream):
    """服务账号（Vertex SA）第三通道：继承 Express 通道完整请求管线，仅替换客户端来源。"""

    channel_name = "vertex"   # 每通道独立重试 / 熔断统计用

    def _resolve_client(self, fastapi_request: Request, base_model_name: str, settings: dict) -> dict:
        project_id, location, sa_json = app_state.get_current_sa_account()
        if not sa_json:
            error_msg = ("未配置服务账号凭证：请在控制台「通道与凭证」→ 服务账号 页粘贴 SA JSON，"
                         "或设置环境变量 VERTEX_SA_JSON / VERTEX_SA_FILE。")
            print(f"❌ [凭证配置] {error_msg}")
            return {"error": JSONResponse(
                status_code=401,
                content=create_openai_error_response(401, error_msg, "authentication_error"))}

        location = location or "global"
        is_global = location == "global"
        headers, timeout, warnings = resolve_paygo_bundle(is_global, settings, model_name=base_model_name)
        for w in warnings:
            print(f"⚠️ [流量等级] {w}")
        client_to_use = _get_cached_sa_client(sa_json, project_id, location, headers, timeout)
        return {
            "client": client_to_use,
            "model_to_call": base_model_name,   # 裸名，SDK 按 Client 的 project/location 拼路径
            "priority_paygo": bool(headers),
            "fallback_model": None,              # 无钉定失败回退需求
            "fallback_client_factory": None,
        }

    async def chat_completions(self, request_obj, fastapi_request: Request,
                               failover_mode: bool = False):
        # [PAY] 旧前缀兼容：剥掉后走本通道（旧配置模型名不改即用上服务账号通道）
        model = request_obj.model
        if isinstance(model, str) and model.startswith("[PAY]"):
            stripped = model[len("[PAY]"):]
            if stripped.startswith(" "):
                stripped = stripped[1:]
            request_obj = request_obj.model_copy(update={"model": stripped})
        return await super().chat_completions(request_obj, fastapi_request, failover_mode=failover_mode)
