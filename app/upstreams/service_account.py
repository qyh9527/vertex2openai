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

import asyncio
import json
import random
from functools import partial
from typing import Any, Optional

from fastapi import Request
from fastapi.responses import JSONResponse
from google import genai
from google.oauth2 import service_account

from upstreams.express_sdk import (ExpressSDKUpstream,
                                   pinned_resource_location, pinned_resource_project,
                                   resource_location_error)
from api_helpers import RetryStop, create_openai_error_response, upstream_exception_status
from http_options import get_http_options, resolve_paygo_bundle
from runtime_state import (_current_pricing_location, _current_pricing_tier,
                           _current_region_fallback, _current_sa_account,
                           _pending_region_failure, app_state, sa_breaker_id)
import request_log
import outcome
import sa_budget
from failover import breaker
import sa_affinity
import sa_region_pin
import config as app_config

SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]


# ========== 服务账号 Client 复用缓存（统一 ClientPool，进阶报告 P1-⑤）==========
# 原先与 Express 通道各自维护同构的"缓存字典 + 失败计数 + 锁"，已合并为
# upstreams.client_pool 的全进程唯一池。缓存键仍含层级头：换 PayGo 层级必须
# 换连接池语义（头不同不能复用旧连接）；client_reuse 关闭时每请求新建。
# 既有引用别名（_SA_CLIENT_CACHE 等）继续可用（测试按 dict clear 的地方
# 已改走 _clear_sa_client_cache）。
from upstreams.client_pool import client_pool as _POOL
_SA_CLIENT_CACHE_LOCK = _POOL._lock
_SA_CLIENT_CACHE = _POOL._cache


class _SaFailuresView:
    """兼容对象：_SA_CLIENT_FAILURES 旧名可用（clear 转发统一池）。"""

    def clear(self):
        _POOL.clear()

    def __len__(self):
        with _POOL._lock:
            return len(_POOL._failures)


_SA_CLIENT_FAILURES = _SaFailuresView()


def _clear_sa_client_cache() -> None:
    """清空统一池（测试隔离用；池是 Express/SA 共享的，整池清理）。"""
    _POOL.clear()


def _on_sa_client_failure(cache_key: tuple, kind: str = "conn", reason: str = "") -> None:
    """复用 SA Client 的一次失败上报（转发统一 ClientPool；同 Express 通道防护语义）：
    kind=evict 立即舍弃 / conn 计数达到阈值舍弃。"""
    _POOL.on_failure(cache_key, kind=kind, reason=reason,
                     log_prefix="[SA Client 复用]")


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


def _supports_region_fallback(model_name: str) -> bool:
    """该模型是否允许尝试 us/eu 多区域节点（Gemini 4 全家，含 Neon/cyber 变体）。

    某个区域是否真的开放由上游决定：尝试失败会回到原始 global 错误，
    不会把一个区域的 404 当成模型不存在返回给客户端。
    """
    name = str(model_name or "").strip().lower().rsplit("/", 1)[-1]
    if name.endswith("-search"):
        name = name[:-len("-search")]
    return name == "gemini-4" or name.startswith("gemini-4-")


def _region_attempt_count(settings: dict) -> int:
    """本次请求允许尝试的区域数：1（随机一个）或 2（us/eu 都试）。"""
    raw = (settings or {}).get("sa_region_fallback_attempts", 1)
    return 2 if str(raw) == "2" else 1


def _pinned_region_for(sa_json: str, location: str, base_model_name: str,
                       explicit_location: Optional[str] = None) -> Optional[str]:
    """该凭证是否有可用的 24 小时固定区域；显式路径/自定义主机/非 Gemini 4 不适用。"""
    if explicit_location is not None:
        return None
    if str(location or "").strip().lower() != "global":
        return None
    if app_config.VERTEX_BASE_URL or not _supports_region_fallback(base_model_name):
        return None
    try:
        return sa_region_pin.lookup(sa_budget.credential_id(sa_json))
    except (sa_budget.BudgetError, ValueError, OSError):
        return None


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
    """取（必要时创建）复用的服务账号 genai.Client（统一 ClientPool 承载，进阶报告 P1-⑤）。

    缓存键含层级头：换 PayGo 层级必须换连接池语义（头不同不能复用旧连接）。
    client_reuse 关闭时每请求新建。
    """
    headers_key = tuple(sorted((headers or {}).items()))
    cache_key = (sa_json, project_id, location, headers_key)

    def _factory(key):
        return _new_sa_client(sa_json, project_id, location, headers, timeout, cache_key=key)

    if not app_state.get_setting("client_reuse", True):
        _log_sa_client_reuse(reused=False)
        return _factory(None)
    client, reused = _POOL.get_or_create(cache_key, _factory)
    _log_sa_client_reuse(reused)
    return client


def _log_sa_client_reuse(reused: bool) -> None:
    """P1-3：SA Client 来源日志（new=新建连接池 / reused=命中缓存复用）。"""
    from api_helpers import channel_display_name
    print(f"🔌 [SA Client 复用] {channel_display_name('vertex')} 本次请求"
          f"{'复用缓存 Client' if reused else '新建 Client 连接池'}。")


class ServiceAccountUpstream(ExpressSDKUpstream):
    """服务账号（Vertex SA）第三通道：继承 Express 通道完整请求管线，仅替换客户端来源。"""

    channel_name = "vertex"   # 每通道独立重试 / 熔断统计用

    def _resolve_client(self, fastapi_request: Request, base_model_name: str, settings: dict) -> dict:
        account = app_state.get_current_sa_account(**sa_affinity.selection_preferences())
        project_id, location, sa_json = account
        if not sa_json:
            error_msg = ("未配置服务账号凭证：请在控制台「通道与凭证」→ 服务账号 页粘贴 SA JSON，"
                         "或设置环境变量 VERTEX_SA_JSON / VERTEX_SA_FILE。")
            print(f"❌ [凭证配置] {error_msg}")
            return {"error": JSONResponse(
                status_code=401,
                content=create_openai_error_response(401, error_msg, "authentication_error"))}

        # 路径写坏必须优先返回 400；否则空模型 ID 会先撞上预算/价表检查变成 429。
        location_error = resource_location_error(base_model_name)
        if location_error:
            return {"error": JSONResponse(status_code=400, content=create_openai_error_response(
                400, location_error, "invalid_request_error"))}
        pinned_location_raw = pinned_resource_location(base_model_name)
        pinned_location = pinned_location_raw.strip().lower() if pinned_location_raw else None
        location = (pinned_location or location or "global").strip().lower()
        try:
            key = sa_budget.credential_id(sa_json)
            budget = sa_budget.status(key)
            if budget.get("expired"):
                raise sa_budget.BudgetError("SA 赠金日期未确认或已到期，已触发日期熔断。")
            if not budget["initialized"] or budget["blocked"]:
                raise sa_budget.BudgetError("SA 历史花费未确认或独立额度已用尽，请检查控制台额度账本。")
            sa_budget.price(base_model_name, 1, 1)
        except (sa_budget.BudgetError, ValueError, OSError) as exc:
            return {"error": JSONResponse(status_code=429, content=create_openai_error_response(
                429, str(exc), "insufficient_quota"))}
        # 区域回退成功过的凭证，24 小时内直接使用成功区域；显式路径/自定义主机不参与。
        pinned_region = _pinned_region_for(
            sa_json, location, base_model_name, explicit_location=pinned_location)
        if pinned_region:
            location = pinned_region
            print(f"📌 [区域固定] 凭证 …{key[-4:]} 在 {location} 有 24 小时固定记录，"
                  f"本次直接使用该多区域节点（非 global 单价 +10%）。")
        is_global = location == "global"
        headers, timeout, warnings = resolve_paygo_bundle(is_global, settings, model_name=base_model_name)
        request_log.note_transport(headers, timeout)
        for w in warnings:
            print(f"⚠️ [流量等级] {w}")
        client_to_use = _get_cached_sa_client(sa_json, project_id, location, headers, timeout)
        client_to_use = sa_budget.BudgetClient(
            client_to_use, key, location,
            headers.get("X-Vertex-AI-LLM-Shared-Request-Type", "standard"))
        # 每次调用标明通道身份 + 脱敏 project/location（P0-2/12.3：SA 请求不再被误认为 Express）
        _proj = (project_id or "（取 SA JSON 自带 project_id）") if project_id else "（取 SA JSON 自带 project_id）"
        print(f"🌐 [上游请求] 服务账号（Vertex SA）通道：使用 SA 凭证（Bearer）调用模型 {base_model_name}，"
              f"project={_proj} location={location}（SDK 按 project/location 拼标准 Vertex 资源路径）。")
        return {
            "client": client_to_use,
            "model_to_call": base_model_name,   # 裸名，SDK 按 Client 的 project/location 拼路径
            "priority_paygo": bool(headers),
            "pricing_location": location,
            "pricing_tier": headers.get("X-Vertex-AI-LLM-Shared-Request-Type") or "standard",
            "fallback_model": None,              # 无钉定失败回退需求
            "fallback_client_factory": None,
            # 会话亲和：待写入的绑定决策（上游响应通过验证后提交）。
            # 亲和键必须用所选凭证自身的生效项目；资源路径里的目标项目只决定请求目标。
            "sa_affinity": sa_affinity.build_decision(
                _effective_project(sa_json, project_id), key, base_model_name),
            # 重试换号：可重试故障（429/503/配额）时换下一个未被本次请求试过的账号。
            "retry_client_factory": self.make_retry_client_factory(
                settings, base_model_name, sa_json, project_id, location,
                allow_region_fallback=(pinned_location is None
                                       and not app_config.VERTEX_BASE_URL)),
        }

    def make_retry_client_factory(self, settings: dict, base_model_name: str,
                                  current_sa_json: str, project_id: str = "",
                                  current_location: str = "global",
                                  allow_region_fallback: bool = True):
        """Prepare the next retry target off-loop; adopt it in the request task.

        第一次可重试失败优先用同一凭证换一个多区域节点（仅 global 429 → us/eu；
        控制台可设 1 或 2 次，2 次会无顺序尝试两个区域）。区域回退不修改账号配置、
        不计入凭证熔断；之后的重试仍按原逻辑轮换其它凭证。多区域端点由 SDK 拼为
        aiplatform.{us|eu}.rep.googleapis.com，单价按非 global +10% 计。
        """
        tried = {sa_budget.credential_id(current_sa_json)}
        current_id = sa_breaker_id(current_sa_json)
        region_candidates = list(("us", "eu"))
        random.shuffle(region_candidates)
        region_candidates = region_candidates[:_region_attempt_count(settings)]
        region_total = len(region_candidates)
        current_is_region = False
        region_original_reported = False
        pinned_project = pinned_resource_project(base_model_name)
        pinned_location_raw = pinned_resource_location(base_model_name)
        pinned_location = pinned_location_raw.strip().lower() if pinned_location_raw else None
        active_sa_json = current_sa_json
        active_project = project_id
        active_location = (pinned_location or current_location or "global").strip().lower()

        def _build(target_sa_json: str, target_project: str, location: str):
            key = sa_budget.credential_id(target_sa_json)
            headers, timeout, _warnings = resolve_paygo_bundle(
                location == "global", settings, model_name=base_model_name)
            client = _get_cached_sa_client(target_sa_json, target_project, location,
                                           headers, timeout)
            client = sa_budget.BudgetClient(
                client, key, location,
                headers.get("X-Vertex-AI-LLM-Shared-Request-Type", "standard"))
            return client, headers, timeout

        def prepare_region(location: str):
            project = _effective_project(active_sa_json, active_project)
            client, headers, timeout = _build(active_sa_json, project, location)
            return client, {"project_id": project, "location": location,
                            "sa_json": active_sa_json}, headers, timeout

        def prepare():
            # 显式完整资源路径已钉定项目；换号可能把请求发到别的项目/凭证权限下，直接不轮换。
            if pinned_project:
                return None
            while True:
                project_id_next, location, sa_json = app_state.next_sa_account_excluding(tried)
                if not sa_json:
                    return None
                key = sa_budget.credential_id(sa_json)
                tried.add(key)
                location = (location or "global").strip().lower()
                pinned_region = _pinned_region_for(
                    sa_json, location, base_model_name, explicit_location=pinned_location)
                if pinned_region:
                    location = pinned_region
                    print(f"📌 [区域固定] 备用凭证 …{key[-4:]} 使用固定区域 {location}。")
                try:
                    client, headers, timeout = _build(sa_json, project_id_next, location)
                except Exception as exc:
                    print(f"⚠️ [账号切换] 跳过无法构造的备用凭证（{type(exc).__name__}）。")
                    continue
                return client, {"project_id": project_id_next, "location": location,
                                "sa_json": sa_json}, headers, timeout

        def _adopt(account, headers, timeout, region_fallback=False):
            _current_sa_account.set(account)
            _current_pricing_location.set(account.get("location") or "global")
            _current_region_fallback.set(region_fallback)
            _current_pricing_tier.set(
                headers.get("X-Vertex-AI-LLM-Shared-Request-Type") or "standard")
            project = _effective_project(account["sa_json"], account["project_id"])
            key = sa_budget.credential_id(account["sa_json"])
            sa_affinity.begin_commit(sa_affinity.build_decision(project, key, base_model_name))
            request_log.note_transport(headers, timeout)

        async def switch(error: Exception):
            nonlocal current_id, current_is_region, region_original_reported
            nonlocal active_sa_json, active_project, active_location
            can_region = (allow_region_fallback
                          and pinned_location is None
                          and _supports_region_fallback(base_model_name)
                          and not isinstance(error, sa_budget.BudgetError))
            # 第一次换区：只在 global 明确 429 时触发，避免把普通限流以外的错误也搬去区域端点。
            if (can_region and region_candidates
                    and active_location == "global"
                    and upstream_exception_status(error) == 429):
                # 构造失败不消耗整条回退路径：继续试剩余区域；全失败再交回凭证轮换。
                while region_candidates:
                    location = region_candidates.pop(0)
                    try:
                        client, account, headers, timeout = await asyncio.to_thread(
                            prepare_region, location)
                    except Exception as exc:
                        print(f"⚠️ [区域回退] 多区域节点 {location} 客户端构造失败"
                              f"（{type(exc).__name__}），继续尝试下一个区域。")
                        continue
                    current_is_region = True
                    if not region_original_reported:
                        region_original_reported = True
                        # 原始 global 失败立即计入候选熔断一次；之后的区域尝试结果不再计入。
                        # 与普通失败路径同样豁免明确的永久/策略/客户端关闭类错误。
                        _category = outcome.classify_exception(error)
                        if _category not in (outcome.CLIENT_CLOSED, outcome.REQUEST_PERMANENT,
                                             outcome.POLICY_BLOCKED):
                            if _category == outcome.RATE_LIMITED:
                                breaker.report_candidate_rate_limited("vertex", current_id, str(error))
                            else:
                                breaker.report_failure(("vertex", current_id))
                        _pending_region_failure.set(error)
                    _adopt(account, headers, timeout, region_fallback=True)
                    active_location = account["location"]
                    print(f"🔁 [区域回退] global 上游 429，本次请求改用多区域节点 {location}"
                          f"（第 {region_total - len(region_candidates)}/{region_total} 次尝试；"
                          f"同一凭证，不修改账号配置，非 global 单价 +10%）。")
                    return client
            # 第一次换区也没有成功：按配置继续尝试另一个多区域节点（us/eu 无固定顺序）。
            if can_region and region_candidates and current_is_region:
                while region_candidates:
                    previous = active_location
                    location = region_candidates.pop(0)
                    try:
                        client, account, headers, timeout = await asyncio.to_thread(
                            prepare_region, location)
                    except Exception as exc:
                        print(f"⚠️ [区域回退] 多区域节点 {location} 客户端构造失败"
                              f"（{type(exc).__name__}），继续尝试下一个区域。")
                        continue
                    _adopt(account, headers, timeout, region_fallback=True)
                    active_location = account["location"]
                    print(f"🔁 [区域回退] {previous} 未成功，继续尝试多区域节点 {location}"
                          f"（第 {region_total - len(region_candidates)}/{region_total} 次尝试；"
                          f"同一凭证，不修改账号配置，非 global 单价 +10%）。")
                    return client
            prepared = await asyncio.to_thread(prepare)
            if prepared is None:
                if current_is_region:
                    original = (error if isinstance(error, sa_budget.BudgetError)
                                else (_pending_region_failure.get() or error))
                    raise RetryStop("区域回退已尝试，且没有其它可用凭证。",
                                    original_error=original)
                return None
            client, account, headers, timeout = prepared
            # 区域尝试本身不计熔断（原始 global 失败已在换区时上报）；普通轮换仍按当前错误上报。
            if not current_is_region and not isinstance(error, sa_budget.BudgetError):
                if outcome.classify_exception(error) == outcome.RATE_LIMITED:
                    breaker.report_candidate_rate_limited("vertex", current_id, str(error))
                else:
                    breaker.report_failure(("vertex", current_id))
            _pending_region_failure.set(None)
            current_is_region = False
            region_original_reported = False
            current_id = sa_breaker_id(account["sa_json"])
            active_sa_json = account["sa_json"]
            active_project = account["project_id"]
            active_location = account["location"]
            _adopt(account, headers, timeout)
            print(f"🔁 [账号切换] 本次请求改用另一份服务账号凭证（…{current_id[-4:]}）。")
            return client

        return switch

    async def chat_completions(self, request_obj, fastapi_request: Request,
                               failover_mode: bool = False, raw_mode: bool = False):
        # [PAY] 旧前缀兼容：剥掉后走本通道（旧配置模型名不改即用上服务账号通道）
        model = request_obj.model
        if isinstance(model, str) and model.startswith("[PAY]"):
            stripped = model[len("[PAY]"):]
            if stripped.startswith(" "):
                stripped = stripped[1:]
            request_obj = request_obj.model_copy(update={"model": stripped})
        parent_kwargs = {"failover_mode": failover_mode}
        if raw_mode:
            parent_kwargs["raw_mode"] = True
        return await super().chat_completions(request_obj, fastapi_request, **parent_kwargs)
