import re
import threading
from functools import partial

from typing import Any, Optional

import google.genai
from fastapi import Request
from fastapi.responses import JSONResponse
from google import genai

from models import OpenAIRequest
from upstreams.base import BaseUpstream
from api_helpers import (
    create_generation_config,
    execute_gemini_call,
    create_openai_error_response,
    FAKE_PREFIX,
    channel_display_name,
    channel_call_text,
)
from message_processing import (create_gemini_prompt, apply_prefill_compat,
                                apply_console_injection, DEFAULT_IMAGE_PREFILL_NUDGE)
from input_relay import apply_input_relay, get_input_relay_config, input_relay_active_for_stream
from top_input_injection import (
    apply_top_input_injection,
    get_top_input_injection_config,
    top_input_injection_active_for_channel,
)
from http_options import get_http_options, resolve_paygo_bundle
import model_capabilities as mc
from runtime_state import app_state
from failover import UpstreamUnstartedError
from anti_truncation import is_enabled_for_request, inject_request, get_enabled_field
import config as app_config
from schema_validation import SchemaValidationError, validate_request_schemas

LEGACY_EXPRESS_PREFIX = "[EXPRESS] "
LEGACY_PAY_PREFIX = "[PAY]"
OPENAI_DIRECT_SUFFIX = "-openai"
OPENAI_SEARCH_SUFFIX = "-openaisearch"

# Client 复用缓存：按 (api_key, base_url, priority_paygo) 复用 google-genai Client。
# google-genai 的 httpx 连接是惰性创建的，复用 Client 对象即复用 TLS/HTTP 连接池，
# 省掉每个请求重新握手（VPS → Google 链路的纯开销）。
# 额度按 API Key 与请求计费，与连接是否复用无关；429 限流也按请求判定，不受影响。
# dict 的 get/set 在 GIL 下原子，异步协程并发安全；极端并发下重复创建只是多费一个对象。
#
# 两个控制台可调行为（「Google GenAI Client 复用」卡片，Express 与服务账号通道共用）：
#   client_reuse=False              → 每请求新建 Client，彻底不用缓存
#   client_reuse_evict_threshold=N  → 缓存 Client 连续 N 次"连接级失败"（httpx.TransportError
#                                    类，如 RemoteProtocolError / ConnectError / 超时，不含 429
#                                    限流）自动舍弃，下次请求重建连接池（0=不自动舍弃）。
#                                    长驻进程里残留的失效 keep-alive 连接正是这类失败之源。
# ========== Client 复用：统一 ClientPool（进阶报告 P1-⑤）==========
# 原先 Express 与 SA 各自维护"缓存字典 + 失败计数 + 锁"的同构代码，已合并为
# upstreams.client_pool 的全进程唯一池。缓存键、开关语义、淘汰逻辑逐条保持：
#   client_reuse=False              → 每请求新建 Client，彻底不用缓存
#   client_reuse_evict_threshold=N  → 缓存 Client 连续 N 次"连接级失败"（httpx.TransportError
#                                    类，如 RemoteProtocolError / ConnectError / 超时，不含 429
#                                    限流）自动舍弃，下次请求重建连接池（0=不自动舍弃）。
#                                    长驻进程里残留的失效 keep-alive 连接正是这类失败之源。
# 既有引用别名：_CLIENT_CACHE_LOCK（池的锁）/ _CLIENT_CACHE（池的缓存 dict）/
# _CLIENT_FAILURES（兼容视图）仍可被测试/代码按原方式使用。
from upstreams.client_pool import client_pool as _POOL
_CLIENT_CACHE_LOCK = _POOL._lock
_CLIENT_CACHE = _POOL._cache


class _FailuresView:
    """兼容对象：既有测试按 dict 方式访问 _CLIENT_FAILURES（只用到 clear）。

    统一池把 cache 与 failures 拆开持有后，这里保持旧名字可用：
    clear() 同时清空统一池的失败计数与缓存（与旧行为一致——旧测试
    clear 两个字典总是一起清）。
    """

    def clear(self):
        with _CLIENT_CACHE_LOCK:
            _POOL._failures.clear()
            _POOL._cache.clear()

    def __len__(self):
        with _CLIENT_CACHE_LOCK:
            return len(_POOL._failures)


_CLIENT_FAILURES = _FailuresView()


def _clear_client_cache() -> None:
    """清空本通道在统一池中的缓存与失败计数（测试隔离用）。

    统一池是全进程共享的（Express/SA 同池），逐 key 清理无从知道哪些 key
    属于哪条通道，这里直接清整池——与改造前"各自 clear 自己的字典"在
    单测隔离语义上等价（SA 测试同样会先清池再建自己的 key）。
    """
    _POOL.clear()


def _new_express_client(express_api_key: str, priority_paygo: bool,
                        headers: dict | None = None, timeout: int | None = None,
                        cache_key: Any = None) -> Any:
    """创建 genai.Client；复用模式下挂上失败上报回调（由 api_helpers 调用）。

    headers/timeout 由 PayGo 流量等级解析得出（http_options.resolve_paygo_bundle）。
    """
    client = genai.Client(
        vertexai=True,
        api_key=express_api_key,
        http_options=get_http_options(headers=headers, timeout=timeout,
                                      priority_paygo=priority_paygo),
    )
    if cache_key is not None:
        client._vertex_cache_key = cache_key
        client._vertex_on_failure = partial(_on_client_failure, cache_key)
    return client


def _on_client_failure(cache_key: tuple, kind: str = "conn", reason: str = "") -> None:
    """复用 Client 的一次失败上报（hook，由 api_helpers 触发）——转发统一 ClientPool。

    kind="evict"：立即舍弃缓存 Client（真正的"连接/会话状态损坏"硬错误；安全拦截
                 已不再走此路径，P1-4），下次请求重建连接池。
    kind="conn"（默认）：连接级失败计数，累计达到 client_reuse_evict_threshold 才舍弃；
                 429 限流等 HTTP 状态错误不会走到这里（连接本身健康）。
    """
    _POOL.on_failure(cache_key, kind=kind, reason=reason,
                     log_prefix="[Client 复用]")


def _get_cached_client(express_api_key: str, priority_paygo: bool = False,
                       *, headers: dict | None = None, timeout: int | None = None) -> Any:
    """取（必要时创建）复用的 google-genai Client（统一 ClientPool 承载，进阶报告 P1-⑤）。

    缓存键含 priority_paygo 与 PayGo 层级头：Priority PayGo 请求头必须只用于钉定到
    global 的资源路径，绝不能复用到普通请求上（流量等级会标错）；换层级必须换连接池语义。
    控制台 client_reuse 关闭时不做缓存，每请求新建 Client。
    """
    base_url = app_config.VERTEX_BASE_URL or None
    headers_key = tuple(sorted((headers or {}).items()))
    cache_key = (express_api_key, base_url, priority_paygo, headers_key)

    def _factory(key):
        return _new_express_client(express_api_key, priority_paygo,
                                   headers=headers, timeout=timeout, cache_key=key)

    if not app_state.get_setting("client_reuse", True):
        _log_client_reuse("express", reused=False)
        return _factory(None)
    client, reused = _POOL.get_or_create(cache_key, _factory)
    _log_client_reuse("express", reused)
    return client


def _log_client_reuse(channel: str, reused: bool, evicted: bool = False) -> None:
    """P1-3：Client 来源日志（new=新建连接池 / reused=命中缓存复用 / evicted=缓存已被淘汰后重建）。

    排查连接复用问题（keep-alive 失效、代理切换后旧连接报错）时，
    这行日志能直接回答"这次请求用的到底是新建还是旧的连接池对象"。
    """
    if evicted:
        state = "缓存此前已被淘汰，本次新建连接池"
    elif reused:
        state = "复用缓存 Client"
    else:
        state = "新建 Client 连接池"
    print(f"🔌 [Client 复用] {channel_display_name(channel)} 本次请求{state}。")


def _normalize_model_name(model_name: str) -> tuple[str, bool, bool, str | None]:
    base_model_name = model_name
    is_fake = False

    # 假流式前缀与 legacy 前缀按任意顺序循环剥除：fake-[EXPRESS] x / [EXPRESS] fake-x 均支持
    while True:
        if base_model_name.startswith(FAKE_PREFIX):
            is_fake = True
            base_model_name = base_model_name[len(FAKE_PREFIX):]
        elif base_model_name.startswith(LEGACY_EXPRESS_PREFIX):
            base_model_name = base_model_name[len(LEGACY_EXPRESS_PREFIX):]
        else:
            break

    if base_model_name.startswith(LEGACY_PAY_PREFIX):
        return base_model_name, False, False, ("当前在 Express 通道：[PAY] 前缀仅由服务账号（vertex）/ 混合自动（hybrid）通道剥除，"
                                               "请把通道策略切到服务账号，或直接使用普通模型名。")

    if base_model_name.endswith(OPENAI_SEARCH_SUFFIX) or base_model_name.endswith(OPENAI_DIRECT_SUFFIX):
        return base_model_name, False, False, "当前版本已经移除 -openai/-openaisearch 直连上游路径，请直接使用普通模型名或 -search 模型名。"

    is_grounded_search = base_model_name.endswith("-search")
    if is_grounded_search:
        base_model_name = base_model_name[:-len("-search")]

    return base_model_name, is_grounded_search, is_fake, None


def _prefill_tpl(user_template: str, is_image_model: bool) -> str:
    """选用续写指令：生图模型在用户未自定义时改用要图片的那句。

    通用模板说的是"从断点处无缝往下写"，生图模型会老老实实继续写**文本**，
    结果吐出一段字符画而不是图片（实测复现）。用户填了自定义模板则以用户的为准。
    """
    tpl = (user_template or "").strip()
    if tpl:
        return tpl
    return DEFAULT_IMAGE_PREFILL_NUDGE if is_image_model else ""


def _build_thinking_config(base_model_name: str, request: OpenAIRequest, is_image_model: bool,
                           prefill_active: bool = False,
                           channel_name: Optional[str] = None) -> dict | None:
    """按模型能力档案 + 控制台设置 + 单次请求构建思考配置（SDK 线格式）。"""
    if is_image_model:
        return None

    settings = app_state.get_effective_settings(base_model_name)
    t = mc.resolve_thinking(base_model_name, request, settings, prefill_active=prefill_active)
    if t.get("mode") is None:
        return None

    thinking_config = {"include_thoughts": t.get("include_thoughts", True)}

    if t["mode"] == "level":
        genai_version_str = getattr(google.genai, "__version__", "1.0.0")
        try:
            parts = genai_version_str.split(".")
            sdk_supports_level = (int(parts[0]) >= 2) or (int(parts[0]) == 1 and int(parts[1]) >= 51)
        except Exception:
            sdk_supports_level = False

        if sdk_supports_level:
            thinking_config["thinking_level"] = t["level"]
        else:
            print(f"⚠️ [推理配置] 当前 google-genai 版本 {genai_version_str} 不支持 thinking_level，已自动跳过该参数。")
    else:  # budget（Gemini 2.5）
        thinking_config["thinking_budget"] = t["budget"]

    if app_state.get_setting("debug_outbound", False):
        print(f"🔎 [出站调试] {channel_display_name(channel_name)} 模型={base_model_name} thinkingConfig={thinking_config}")

    return thinking_config


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


# P0-5：端点诊断按 (channel, auth, base_url, project, location) 去重——
# Express 与服务账号两条通道的端点语义完全不同，各自至少打一次；
# 同一进程先 Express 后 SA（或反过来）时两种端点日志都必须出现。
# 账号 / project / location 变化后重新记录。
_ENDPOINT_LOGGED_KEYS: set = set()


def _log_resolved_endpoint(client: Any, channel_name: Optional[str] = None) -> None:
    """把 SDK 实际解析出的上游端点打进日志（按通道+端点身份去重）。

    为什么需要它：Express（api_key）模式下**无法**指定 location——SDK 里
    project/location 与 api_key 互斥，硬传会 `ValueError: Project/location and
    API key are mutually exclusive`。SDK 解析出的 base_url 就是**全局端点**
    https://aiplatform.googleapis.com/ ，project/location 均为 None，请求 URL 里
    根本没有 location 段。因此报错信息里出现的 `locations/xxx` 区域是 **Google 后端
    为该 Key 自行路由**的结果，不是本代理选的，客户端侧无从指定。

    服务账号（vertex）通道则由 SDK 按 Client 的 project/location 拼完整资源路径，
    日志明确输出脱敏 project 与 location，便于核实 403/404 到底落在哪个项目/区域。
    """
    try:
        api = client._api_client
        base_url = getattr(api._http_options, "base_url", None)
        api_version = getattr(api._http_options, "api_version", None)
        scope = "全局端点（URL 无 location 段）" if base_url == "https://aiplatform.googleapis.com/" \
            else "自定义/区域端点"
        ch = channel_name or "express"
        key = (ch, getattr(api, "project", None), getattr(api, "location", None),
               base_url, api_version)
        if key in _ENDPOINT_LOGGED_KEYS:
            return
        _ENDPOINT_LOGGED_KEYS.add(key)
        suffix = ""
        if ch == "vertex":
            proj = getattr(api, "project", None)
            suffix = (f"（服务账号：SDK 将按 project={proj} / location={getattr(api, 'location', None)} "
                      "拼标准 Vertex 资源路径）")
        else:
            suffix = ("Express Key 模式下具体区域由 Google 后端路由，无法在客户端指定"
                      "（如需强制指向某端点，可设环境变量 VERTEX_BASE_URL）。")
        print(f"🌐 [上游端点] {channel_display_name(ch)} 解析结果：base_url={base_url} "
              f"api_version={api_version} project={getattr(api, 'project', None)} "
              f"location={getattr(api, 'location', None)} → {scope}。{suffix}")
    except Exception as e:
        print(f"⚠️ [上游端点] 读取端点解析结果失败（不影响调用）：{e}")


def resolve_express_model_path(base_model_name: str, settings: dict) -> str:
    """把模型名解析成实际下发给 SDK 的 model 值。

    留空 express_location → 返回裸模型名，走 express 端点格式：
        https://aiplatform.googleapis.com/v1/publishers/google/models/{model}:generateContent
      此时 location 由 Google 后端自行路由，**可能落到该模型不提供服务的区域并 404**
      （实测：gemini-2.5-pro 被路由到 asia-southeast1 → 404 not found，
       同一 Key 换成下面的完整路径即 200）。

    填了 express_location → 返回完整资源路径，让区域由我们钉定：
        projects/{project}/locations/{location}/publishers/google/models/{model}
      google-genai 的 t_model() 对以 "projects/" 开头的 model 原样透传，
      因此不需要（也不能）给 Client 传 location —— api_key 与 project/location 互斥。

    项目 ID 直接取「通道与凭证」页填的那个（或环境变量 GOOGLE_PROJECT_ID）——
    一个人通常只有一个 Express 项目，不再单独配一份。
    拿不到项目 ID 就退回裸模型名（并提示），绝不拼出半截路径。
    """
    if base_model_name.startswith(("projects/", "publishers/", "models/")):
        return base_model_name          # 客户端已自带完整路径，尊重它

    location = str(settings.get("express_location", "") or "").strip()
    if not location:
        return base_model_name

    project = (app_config.GOOGLE_PROJECT_ID or app_state.get_project_id() or "").strip()
    if not project:
        print("⚠️ [上游端点] 已选择钉定 location，但没有可用的 Project ID"
              "（请在控制台「通道与凭证」页填写 Project ID，或设环境变量 GOOGLE_PROJECT_ID），"
              "本次退回默认路由。")
        return base_model_name

    return f"projects/{project}/locations/{location}/publishers/google/models/{base_model_name}"


class ExpressSDKUpstream(BaseUpstream):
    """
    官方 API Key Express Mode 渠道处理器
    封装了原有的多密钥切匙、代理挂载以及 SDK 运行时调用
    """
    # 通道名：用于每通道独立重试（channel_retry_overrides）与熔断统计；子类（服务账号）覆盖。
    channel_name = "express"

    def _resolve_client(self, fastapi_request: Request, base_model_name: str, settings: dict) -> dict:
        """解析本通道实际使用的 Client 与模型路径（子类覆盖点）。

        返回 dict：
          {"client", "model_to_call", "priority_paygo", "fallback_model", "fallback_client_factory"}
        无可用凭证时返回 {"error": JSONResponse}（路由层直接返回该错误）。
        """
        express_key_manager_instance = fastapi_request.app.state.express_key_manager
        if express_key_manager_instance.get_total_keys() == 0:
            error_msg = "未配置 VERTEX_EXPRESS_API_KEY，无法调用 Gemini Express Mode。"
            print(f"❌ [密钥配置] {error_msg}")
            return {"error": JSONResponse(
                status_code=401,
                content=create_openai_error_response(401, error_msg, "authentication_error"))}

        key_tuple = express_key_manager_instance.get_express_api_key()
        if not key_tuple:
            error_msg = "没有可用的 Express API Key。"
            print(f"❌ [密钥配置] {error_msg}")
            return {"error": JSONResponse(
                status_code=401,
                content=create_openai_error_response(401, error_msg, "authentication_error"))}

        _, express_api_key = key_tuple
        model_to_call = resolve_express_model_path(base_model_name, settings)
        is_global = "/locations/global/" in model_to_call
        headers, timeout, warnings = resolve_paygo_bundle(is_global, settings, model_name=base_model_name)
        for w in warnings:
            print(f"⚠️ [流量等级] {w}")
        priority_paygo = bool(headers)
        client_to_use = _get_cached_client(express_api_key, priority_paygo,
                                           headers=headers, timeout=timeout)
        fallback_model = base_model_name if model_to_call != base_model_name else None
        fallback_client_factory = (
            (lambda: _get_cached_client(express_api_key, False))
            if priority_paygo else None
        )
        return {
            "client": client_to_use,
            "model_to_call": model_to_call,
            "priority_paygo": priority_paygo,
            "fallback_model": fallback_model,
            "fallback_client_factory": fallback_client_factory,
        }

    async def chat_completions(self, request_obj: OpenAIRequest, fastapi_request: Request,
                               failover_mode: bool = False):
        try:
            validate_request_schemas(request_obj)
        except SchemaValidationError as exc:
            return JSONResponse(
                status_code=400,
                content=create_openai_error_response(400, str(exc), "invalid_request_error"),
            )

        base_model_name, is_grounded_search, is_fake, model_error = _normalize_model_name(request_obj.model)
        if model_error:
            print(f"❌ [模型名称] {model_error} 收到的模型名：{request_obj.model}")
            return JSONResponse(
                status_code=400,
                content=create_openai_error_response(400, model_error, "invalid_request_error"),
            )

        _inj_settings = app_state.get_effective_settings(base_model_name)
        resolved = self._resolve_client(fastapi_request, base_model_name, _inj_settings)
        if "error" in resolved:
            return resolved["error"]
        client_to_use = resolved["client"]
        model_to_call = resolved["model_to_call"]
        priority_paygo = resolved["priority_paygo"]
        fallback_model = resolved.get("fallback_model")
        fallback_client_factory = resolved.get("fallback_client_factory")

        _log_resolved_endpoint(client_to_use, channel_name=self.channel_name)
        print(f"🌐 [上游端点] 使用官方 Gemini SDK 通过 {channel_call_text(self.channel_name)} 调用模型 {base_model_name}。")
        if model_to_call != base_model_name:
            print(f"🌐 [上游端点] 已钉定 location：{model_to_call}")
        if priority_paygo:
            print("🚦 [流量等级] global 请求已附加 PayGo 层级请求头；实际是否命中以上游 traffic_type 为准。")
        else:
            print("ℹ️ [流量等级] 当前未钉定 global（或层级为 off），使用普通请求。")

        profile = mc.get_profile(base_model_name)
        is_image_model = profile["is_image"]

        _is_fake_stream = bool(request_obj.stream and (is_fake or is_image_model))
        _top_input_config, _top_input_config_note = get_top_input_injection_config(_inj_settings)
        _top_input_active = bool(
            _top_input_config and top_input_injection_active_for_channel(
                _top_input_config, self.channel_name))
        if _top_input_config_note:
            print(_top_input_config_note)
        if _top_input_active:
            _top_messages, _top_notes = apply_top_input_injection(
                request_obj.messages, _top_input_config, channel=self.channel_name)
            for _top_note in _top_notes:
                print(_top_note)
            if _top_messages is not request_obj.messages:
                request_obj = request_obj.model_copy(update={"messages": _top_messages})

        _relay_config, _relay_config_note = get_input_relay_config(_inj_settings)
        _relay_is_active = bool(
            _relay_config and input_relay_active_for_stream(
                _relay_config, _is_fake_stream))
        if _relay_config_note:
            print(_relay_config_note)
        if _relay_is_active:
            _relayed_messages, _relay_notes = apply_input_relay(request_obj.messages, _relay_config)
            for _relay_note in _relay_notes:
                print(_relay_note)
            if _relayed_messages is not request_obj.messages:
                request_obj = request_obj.model_copy(update={"messages": _relayed_messages})
        _relay_strip_tag = (_relay_config.tag if _relay_is_active and _relay_config.strip_generated else None)

        # 预填充智能兼容：按控制台模式 + 模型能力处理末尾 assistant 预填充（新模型自动生效）
        # - 2.5 及更早（允许 model 结尾）：原生透传，模型直接续写；
        # - 3.x（拒绝 model 结尾）：转成续写指令；
        # - 两者都会把预填充文本拼回输出开头（带去重）。
        prefill_text = ""
        prefill_active = False
        # 控制台注入（轻量前端用；两个字段都留空时是空操作）。
        # 必须在 apply_prefill_compat 之前，注入后的消息与前端自发预填充同形。
        _injected, _inj_notes = apply_console_injection(
            request_obj.messages,
            system_text=_inj_settings.get("inject_system_instruction", ""),
            prefill_text=_inj_settings.get("inject_prefill", ""),
            has_tools=bool(getattr(request_obj, "tools", None)),
            is_image_model=is_image_model,
            allow_image_prefill=bool(_inj_settings.get("inject_prefill_for_image", False)),
        )
        for _n in _inj_notes:
            print(_n)
        if _injected is not request_obj.messages:
            request_obj = request_obj.model_copy(update={"messages": _injected})

        _prefill_mode = app_state.get_setting("prefill_mode", app_config.DEFAULT_SETTINGS["prefill_mode"])
        if _prefill_mode != "off":
            new_msgs, prefill_text, prefill_active = apply_prefill_compat(
                request_obj.messages, _prefill_mode,
                allow_model_last=not profile["requires_user_last_turn"],
                instruction_template=_prefill_tpl(_inj_settings.get("prefill_instruction", ""), is_image_model),
                cot_guard=bool(_inj_settings.get("prefill_cot_guard", True)) and not is_image_model,
            )
            if new_msgs is not request_obj.messages:
                request_obj = request_obj.model_copy(update={"messages": new_msgs})
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

        # 防截断合成传输协议（可选单请求启用）：请求体带启用字段即注入合成工具 + 控制消息。
        # 注入必须在 create_generation_config 之前（合成工具声明要进 generationConfig）、
        # 在预填充处理之后（控制消息作为末尾 user，不与预填充兼容逻辑打架）。
        # 每次调用都打一行显眼日志（✅ 已启用 / ⛔ 未启用或被忽略），标注下游启用字段名
        # 与具体情况，滚动日志里一眼分辨哪些请求有防截断保护。
        synthetic_tool_name = None
        at_field = get_enabled_field()
        if is_enabled_for_request(request_obj, {"value": at_field}):
            if is_image_model:
                print(f"⛔ [防截断] 本次调用下游已启用（字段「{at_field}」=true），"
                      "但生图/非文本模型不支持工具参数输出，已忽略启用字段（走普通通道）。")
            else:
                request_obj, synthetic_tool_name = inject_request(request_obj)
                print(f"✅ [防截断] 本次调用下游已启用（字段「{at_field}」=true）→ "
                      f"已注入合成传输工具 {synthetic_tool_name}，回答改走工具参数输出绕开截断。")
        else:
            print(f"⛔ [防截断] 本次调用下游未启用（请求体无「{at_field}」字段或值非 true）→ "
                  "走普通文本生成，重提示词场景存在 max_output_tokens 截断风险。")

        gen_config_dict = create_generation_config(request_obj)
        thinking_config = _build_thinking_config(base_model_name, request_obj, is_image_model,
                                                 prefill_active=prefill_active,
                                                 channel_name=self.channel_name)
        if thinking_config:
            gen_config_dict["thinking_config"] = thinking_config

        _tools_disabled = isinstance(request_obj.tool_choice, str) and request_obj.tool_choice.lower() == "none"
        if is_grounded_search and not _tools_disabled:
            search_tool = {"google_search": {}}
            if "tools" in gen_config_dict and isinstance(gen_config_dict["tools"], list):
                if search_tool not in gen_config_dict["tools"]:
                    gen_config_dict["tools"].append(search_tool)
            else:
                gen_config_dict["tools"] = [search_tool]
            print(f"🔎 [搜索增强] 已为模型 {base_model_name} 启用 Google Search 工具。")

        # 传入真实模型名：create_gemini_prompt 需要它来判断是否对缺失的思考签名
        # 启用官方哨兵（仅 Gemini 3.x 强校验，见 message_processing.resolve_tool_call_signature）
        prompt_func = partial(create_gemini_prompt, model_name=base_model_name)

        if app_state.get_setting("debug_outbound", False):
            _dbg = {k: v for k, v in gen_config_dict.items()
                    if k in ("temperature", "top_p", "top_k", "candidate_count",
                             "max_output_tokens", "stop_sequences", "thinking_config",
                             "response_modalities", "image_config")}
            print(f"🔎 [出站调试] {channel_display_name(self.channel_name)} 生成参数={_dbg}")
            # P1-1：请求形状摘要——safety_settings / 工具声明 / system_instruction /
            # response_schema / 预填充 / 防截断。只记形状与名称，不打印参数里的敏感值。
            _shape = []
            _ss = gen_config_dict.get("safety_settings") or []
            if _ss:
                _shape.append("safety_settings=[" + ", ".join(
                    f"{getattr(s, 'category', s)}/{getattr(s, 'threshold', '')}"
                    for s in _ss) + "]")
            _tools_dbg = gen_config_dict.get("tools") or []
            _tool_names = [f.get("name") for t in _tools_dbg if isinstance(t, dict)
                           for f in (t.get("function_declarations") or [])
                           if isinstance(f, dict) and f.get("name")]
            if _tool_names:
                _shape.append(f"工具声明={_tool_names}（共 {len(_tool_names)} 个）")
            if gen_config_dict.get("tool_config"):
                _shape.append(f"tool_config={gen_config_dict['tool_config']}")
            _shape.append(f"system_instruction={'有' if gen_config_dict.get('system_instruction') else '无'}")
            _shape.append(f"response_schema={'有' if gen_config_dict.get('response_schema') else '无'}")
            _shape.append(f"response_mime_type={gen_config_dict.get('response_mime_type') or '无'}")
            if synthetic_tool_name:
                _shape.append(f"防截断合成工具={synthetic_tool_name}")
            _shape.append(f"预填充={'有（' + str(len(prefill_text)) + ' 字）' if prefill_text else '无'}")
            print(f"🔎 [出站调试] {channel_display_name(self.channel_name)} 请求形状：{'；'.join(_shape)}")

        return await execute_gemini_call(
            client_to_use, model_to_call, prompt_func, gen_config_dict, request_obj,
            fastapi_request=fastapi_request, prefill_text=prefill_text, failover_mode=failover_mode,
            # fake- 前缀请求：强制假流式输出（该请求级别，不影响其它模型）
            force_fake_streaming=is_fake,
            # 钉定路径万一不对（Project ID 与 Key 不同项目、该区域没有此模型），
            # 自动退回裸模型名重试一次，并改用不带 PayGo 层级头的普通客户端。
            fallback_model=fallback_model,
            fallback_client_factory=fallback_client_factory,
            channel_name=self.channel_name,
            synthetic_tool_name=synthetic_tool_name,
            input_relay_strip_tag=_relay_strip_tag,
        )
