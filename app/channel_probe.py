"""全渠道最小探活（控制台「通道与凭证」→ 渠道探活卡片）

回答一个正式请求路径回答不了的问题：**这条凭证现在还能不能用？**

两级判定：
  ① 本地结构校验（零消耗、毫秒级）：Express Key 非空、Cookie 含 SAPISID 族、
     SA JSON 字段齐备且私钥可构造。挡掉"格式就不对"的凭证，避免它们拿到
     一句看不懂的上游报错。
  ② 真实最小请求（每条凭证 1 次、max_output_tokens=1）：确认凭证在线上确实可用。
     只有这一级能发现"本地合法但线上已失效"——Cookie 过期、SA 未开计费、
     Key 被吊销、Project ID 填错、区域没有该模型。

探活只读不写：不写 web_state.json、不动 ClientPool、不碰熔断计数、不计入请求统计。
每条凭证新起一个临时 Client（用后关闭），句柄不进复用池，探活不会污染正式请求的连接状态。

并发：每条凭证一个任务，信号量限制同时进行的请求数；单条硬超时兜底，
保证接口一定能在有限时间内返回，不会让浏览器一直转圈。
"""

import asyncio
import time

import httpx
from google import genai
from google.genai import types

import config as app_config
import model_capabilities as mc
import outcome as outcome_mod
from api_helpers import CHANNEL_META, channel_display_name
from cookie_auth import BATCH_GRAPHQL_URL, build_headers, validate_cookie
from model_loader import get_express_models
from protocol_conversion import extract_upstream_error
from runtime_state import app_state
from upstreams.cookie_proxy import (
    PROJECT_ERROR_HINT,
    COOKIE_REFRESH_HINT,
    _build_batch_graphql_body,
    _extract_from_results,
    _is_cookie_expired_error,
    _is_project_error,
    _iter_json_objects,
)
from upstreams.express_sdk import resolve_express_model_path
from upstreams.service_account import (
    _effective_project,
    build_sa_credentials,
    validate_sa_credentials,
)
from models import OpenAIRequest

# 探活默认参与通道（前端不传时全探）。
DEFAULT_CHANNELS = ("express", "cookie", "vertex")

# 单条凭证的上游请求超时（毫秒，genai HttpOptions 的单位）与 httpx 侧秒数。
PROBE_TIMEOUT_MS = 20000
PROBE_TIMEOUT_SECONDS = 20.0

# 单条凭证的硬超时：比上游超时略长，兜住 SDK 内部重试/连接建立等额外耗时。
PROBE_HARD_TIMEOUT_SECONDS = 26.0

# 并行度：探活是人工触发的低频操作，但要避免一次点出十几条凭证时把上游打满。
PROBE_CONCURRENCY = 4

# 探活请求体：一个词 + 1 个输出 token，只验证"能不能调通"，不产生有效内容。
PROBE_PROMPT = "hi"
PROBE_MAX_OUTPUT_TOKENS = 1

# 优先用于探活的模型（实测首字最快的一档）；都不在模型列表里时退回列表首个非生图模型。
_PREFERRED_PROBE_MODELS = ("gemini-3.1-flash-lite", "gemini-3.5-flash-lite", "gemini-3.6-flash")

# 失败分类 → 给控制台看的中文归因（分类真身是 outcome.classify_failure，此处只补文案）。
_PROBE_HINTS = {
    outcome_mod.RATE_LIMITED: "限流/配额用尽：凭证本身有效，稍后重试或换一条。",
    outcome_mod.AUTH_REFRESHABLE: "鉴权被拒：Key/凭证可能已失效或被吊销。",
    outcome_mod.CREDENTIAL_PERMANENT: "权限/计费问题：项目未开启计费、账号缺 aiplatform.user 权限，或 Cookie 已失效。",
    outcome_mod.REQUEST_PERMANENT: "请求被拒（400）：多半是 Project ID / 区域与所选模型不匹配。",
    outcome_mod.CLIENT_CLOSED: "探活被中断。",
}


# ========== 凭证枚举（与正式请求路径同源：控制台优先，其次环境变量兜底）==========

def _express_keys() -> list:
    """控制台保存的 Key 列表优先；未保存过时回落环境变量（同 ExpressKeyManager 语义）。"""
    controlled = app_state.get_express_keys()
    if controlled:
        return list(controlled)
    return list(app_config.VERTEX_EXPRESS_API_KEY_VAL)


def _cookie_accounts() -> list:
    """Cookie 账号列表；控制台为空时回落环境变量单账号。"""
    accounts = app_state.get_cookie_accounts()
    if accounts:
        return accounts
    if app_config.GOOGLE_COOKIE:
        return [{"cookie": app_config.GOOGLE_COOKIE,
                 "project_id": app_config.GOOGLE_PROJECT_ID or ""}]
    return []


def _sa_accounts() -> list:
    """服务账号列表（get_sa_accounts 已含环境变量 VERTEX_SA_JSON / VERTEX_SA_FILE 兜底）。"""
    return app_state.get_sa_accounts()


# ========== 凭证标签（探活结果里指认"是哪一条"，绝不回显完整凭证）==========
# 与 /api/settings/runtime 的掩码同格式，但这里是"探活结果行的标签"，按各通道
# 最能指认身份的那几个字段拼，不直接复用运行时掩码函数（两者的用途不同）。

def _label_key(key: str, index: int) -> str:
    masked = f"{key[:4]}…{key[-4:]}（{len(key)} 字符）" if len(key) > 8 else f"****（{len(key)} 字符）"
    return f"Key #{index + 1} · {masked}"


def _label_cookie(account: dict, index: int) -> str:
    project = account.get("project_id") or "（未填 Project ID）"
    return f"Cookie #{index + 1} · project={project}"


def _label_sa(account: dict, index: int) -> str:
    try:
        import json
        email = str(json.loads(account.get("sa_json") or "{}").get("client_email") or "") or "（未解析 email）"
    except Exception:
        email = "（未解析 email）"
    return f"SA #{index + 1} · {email} · location={account.get('location') or 'global'}"


# ========== 探活模型选择 ==========

def pick_probe_model(models: list, requested: str = "") -> tuple[str, str]:
    """挑一个探活用模型；返回 (模型名, 说明)。

    生图模型一律不用于探活：它会真的生成一张图（慢且费额度），而探活只需要
    "这条凭证能不能调通"这一个答案。
    """
    text_models = [m for m in (models or []) if not mc.get_profile(m)["is_image"]]
    requested = (requested or "").strip()
    if requested:
        if mc.get_profile(requested)["is_image"]:
            fallback, note = _pick_default_model(text_models)
            return fallback, f"所选模型 {requested} 是生图模型（会真的生成图片），已改用它探活。{note}"
        return requested, ""
    return _pick_default_model(text_models)


def _pick_default_model(text_models: list) -> tuple[str, str]:
    for preferred in _PREFERRED_PROBE_MODELS:
        if preferred in text_models:
            return preferred, ""
    if text_models:
        return text_models[0], ""
    return "gemini-2.5-flash", "（模型列表为空，已回退到内置默认模型）"


# ========== 探活请求构造 ==========

def _probe_http_options() -> types.HttpOptions:
    """探活专用 HttpOptions：只带代理/证书与超时，**刻意不打 PayGo 层级头**。

    flex 档允许上游排队至 30 分钟，会把探活拖成"假死"；层级行为属于正式请求路径
    的职责，探活只回答"凭证现在能不能用"。代理/证书走预构建 httpx client，
    原因同 http_options.get_http_options（genai 注入的 mTLS ctx 会破坏代理隧道）。
    """
    client_args = {}
    if app_config.PROXY_URL:
        client_args["proxy"] = app_config.PROXY_URL
    if app_config.SSL_CERT_FILE:
        client_args["verify"] = app_config.SSL_CERT_FILE
    options = {"api_version": "v1beta1"}
    if client_args:
        hc_kwargs = dict(client_args, timeout=PROBE_TIMEOUT_SECONDS)
        options["httpx_client"] = httpx.Client(**hc_kwargs)
        options["httpx_async_client"] = httpx.AsyncClient(**hc_kwargs)
    else:
        options["timeout"] = PROBE_TIMEOUT_MS
    return types.HttpOptions(**options)


def _probe_config() -> types.GenerateContentConfig:
    """最小生成配置；显式关掉自动函数调用，避免 SDK 打印 AFC 提示污染运行日志。"""
    return types.GenerateContentConfig(
        max_output_tokens=PROBE_MAX_OUTPUT_TOKENS,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )


async def _aclose_client(client) -> None:
    """尽力关闭临时 Client；关闭失败不影响探活结论。"""
    try:
        await client.aio.aclose()
    except Exception:
        pass


def _ok(message: str, latency_ms: int) -> dict:
    return {"ok": True, "stage": "upstream", "message": message, "latency_ms": latency_ms}


def _local_fail(message: str) -> dict:
    return {"ok": False, "stage": "local", "message": message, "latency_ms": 0}


def _upstream_fail(e: Exception, latency_ms: int) -> dict:
    code, msg = extract_upstream_error(e)
    category = outcome_mod.classify_failure(code, msg)
    hint = _PROBE_HINTS.get(category)
    text = f"HTTP {code}：{msg}"
    if hint:
        text += f"\n💡 {hint}"
    return {"ok": False, "stage": "upstream", "message": text, "latency_ms": latency_ms}


# ========== 各通道单条凭证探活 ==========

async def _probe_express(key: str, model: str, settings: dict) -> dict:
    """Express 通道：真实请求走 resolve_express_model_path（含 express_location 钉定）。"""
    model_to_call = resolve_express_model_path(model, settings)
    client = genai.Client(vertexai=True, api_key=key, http_options=_probe_http_options())
    started = time.monotonic()
    try:
        await client.aio.models.generate_content(
            model=model_to_call, contents=PROBE_PROMPT, config=_probe_config())
        return _ok(f"✅ 上游返回 200（model={model_to_call}）",
                   int((time.monotonic() - started) * 1000))
    except Exception as e:
        return _upstream_fail(e, int((time.monotonic() - started) * 1000))
    finally:
        await _aclose_client(client)


async def _probe_sa(account: dict, model: str) -> dict:
    """服务账号通道：凭证直接构造 Client（project/location 取自该账号自身）。"""
    sa_json = account.get("sa_json") or ""
    project = _effective_project(sa_json, account.get("project_id") or "")
    location = account.get("location") or "global"
    client = genai.Client(
        vertexai=True,
        project=project or None,
        location=location,
        credentials=build_sa_credentials(sa_json),
        http_options=_probe_http_options(),
    )
    started = time.monotonic()
    try:
        await client.aio.models.generate_content(
            model=model, contents=PROBE_PROMPT, config=_probe_config())
        return _ok(f"✅ 上游返回 200（project={project or '取 SA 自带'} location={location}）",
                   int((time.monotonic() - started) * 1000))
    except Exception as e:
        return _upstream_fail(e, int((time.monotonic() - started) * 1000))
    finally:
        await _aclose_client(client)


class _TextAsResponse:
    """把已读到的响应文本伪装成异步文本流，复用 cookie_proxy 的 JSON 分块解析器。"""

    def __init__(self, text: str):
        self._text = text

    async def aiter_text(self):
        yield self._text


async def _probe_cookie(account: dict, model: str) -> dict:
    """Cookie 直连通道：内容与正式路径逐字同构（同 body 构造器 + 同响应解析器）。

    只有 HTTP 200 且解析出真实候选（正文/finish/函数调用）才算通过——200 里
    照样可能是 cookie_error / project_error 的 GraphQL 错误信封。
    """
    cookie = account.get("cookie") or ""
    project_id = account.get("project_id") or app_config.GOOGLE_PROJECT_ID or ""
    if not project_id:
        return _local_fail("未填 Project ID：batchGraphql 的 requestContext 必须带项目，请先在下方补上。")

    headers = build_headers(cookie)
    if not headers:
        return _local_fail("Cookie 中未找到 SAPISID 族字段，无法计算认证头。")

    probe_request = OpenAIRequest(model=model, messages=[{"role": "user", "content": PROBE_PROMPT}])
    body = await asyncio.to_thread(_build_batch_graphql_body, project_id, model, probe_request)

    client_kwargs = {
        "timeout": httpx.Timeout(connect=10.0, read=PROBE_TIMEOUT_SECONDS, write=10.0, pool=10.0),
        "follow_redirects": True,
    }
    if app_config.PROXY_URL:
        client_kwargs["proxy"] = app_config.PROXY_URL

    started = time.monotonic()
    try:
        async with httpx.AsyncClient(**client_kwargs) as client:
            response = await client.post(BATCH_GRAPHQL_URL, headers=headers, json=body)
            latency = int((time.monotonic() - started) * 1000)
            if response.status_code != 200:
                return _cookie_http_fail(response.status_code, response.text[:600], latency)

            saw_content = False
            async for obj in _iter_json_objects(_TextAsResponse(response.text)):
                for event_type, data in _extract_from_results(obj):
                    if event_type in ("text", "thought", "function_call"):
                        saw_content = True
                    elif event_type == "finish":
                        saw_content = True
                    elif event_type == "blocked":
                        return {"ok": False, "stage": "upstream", "latency_ms": latency,
                                "message": f"⚠️ 请求被上游安全策略拦截：{data}"
                                           "\n（这仍说明 Cookie 是通的，只是这条探活提示词被拦了。）"}
                    elif event_type == "error":
                        em = data.get("message", str(data)) if isinstance(data, dict) else str(data)
                        if _is_cookie_expired_error(em) or _is_project_error(em):
                            hint = PROJECT_ERROR_HINT if _is_project_error(em) else COOKIE_REFRESH_HINT
                            return {"ok": False, "stage": "upstream", "latency_ms": latency,
                                    "message": f"❌ {em}{hint}"}
                        return {"ok": False, "stage": "upstream", "latency_ms": latency,
                                "message": f"❌ 上游返回错误：{em}"}
            if not saw_content:
                return {"ok": False, "stage": "upstream", "latency_ms": latency,
                        "message": "❌ HTTP 200 但没有解析出任何候选内容：Cookie 可能已失效，"
                                   "或该项目在此模型上没有返回体。"}
            return _ok(f"✅ 上游返回 200 且有候选内容（project={project_id}）", latency)
    except Exception as e:
        return _upstream_fail(e, int((time.monotonic() - started) * 1000))


def _cookie_http_fail(status_code: int, body_text: str, latency_ms: int) -> dict:
    """非 200 的 Cookie 探活失败：沿用正式路径的错误归类与提示文案。"""
    if _is_project_error(body_text):
        return {"ok": False, "stage": "upstream", "latency_ms": latency_ms,
                "message": f"HTTP {status_code}：{body_text}{PROJECT_ERROR_HINT}"}
    if status_code in (401, 403) or _is_cookie_expired_error(body_text):
        return {"ok": False, "stage": "upstream", "latency_ms": latency_ms,
                "message": f"HTTP {status_code}：{body_text}{COOKIE_REFRESH_HINT}"}
    return {"ok": False, "stage": "upstream", "latency_ms": latency_ms,
            "message": f"HTTP {status_code}：{body_text}"}


# ========== 单条凭证的完整探活（本地校验 + 上游请求 + 硬超时）==========

async def _probe_one(entry: dict, sem: asyncio.Semaphore) -> dict:
    """entry 需含 channel / index / label / 以及该通道的凭证明细。"""
    async with sem:
        local_error = entry["local_check"]()
        if local_error:
            return {**entry["identity"], **_local_fail(local_error)}
        try:
            result = await asyncio.wait_for(entry["probe"](), timeout=PROBE_HARD_TIMEOUT_SECONDS)
        except asyncio.TimeoutError:
            result = {"ok": False, "stage": "upstream", "latency_ms": int(PROBE_HARD_TIMEOUT_SECONDS * 1000),
                      "message": f"⏱️ 超过 {PROBE_HARD_TIMEOUT_SECONDS:.0f}s 未返回：网络不通、"
                                 "上游无响应或该凭证被静默挂起。"}
        except Exception as e:
            result = _upstream_fail(e, 0)
        return {**entry["identity"], **result}


def _entries_for_channel(channel: str, model: str, settings: dict) -> list:
    """构造该通道下每一条凭证的探活任务描述。"""
    entries = []
    if channel == "express":
        for i, key in enumerate(_express_keys()):
            entries.append({
                "identity": {"channel": channel, "index": i, "label": _label_key(key, i)},
                "local_check": (lambda k=key: None if k.strip() else "Key 为空。"),
                "probe": (lambda k=key: _probe_express(k, model, settings)),
            })
    elif channel == "cookie":
        for i, account in enumerate(_cookie_accounts()):
            entries.append({
                "identity": {"channel": channel, "index": i, "label": _label_cookie(account, i)},
                "local_check": (lambda a=account: _local_check_cookie(a)),
                "probe": (lambda a=account: _probe_cookie(a, model)),
            })
    elif channel == "vertex":
        for i, account in enumerate(_sa_accounts()):
            entries.append({
                "identity": {"channel": channel, "index": i, "label": _label_sa(account, i)},
                "local_check": (lambda a=account: _local_check_sa(a)),
                "probe": (lambda a=account: _probe_sa(a, model)),
            })
    return entries


def _local_check_cookie(account: dict) -> str | None:
    """Cookie 本地结构校验：复用正式保存路径的 validate_cookie（同一条判定标准）。"""
    validation = validate_cookie(account.get("cookie") or "")
    return None if validation["valid"] else validation["message"]


def _local_check_sa(account: dict) -> str | None:
    """SA 本地结构校验：复用正式保存路径的 validate_sa_credentials。"""
    validation = validate_sa_credentials(account.get("sa_json") or "")
    return None if validation["valid"] else validation["message"]


# ========== 对外入口 ==========

async def probe_all(channels=None, model: str = "") -> dict:
    """按通道逐条凭证探活，返回可直接 JSON 化的结果。

    channels 为空 = 探全部三通道；model 为空 = 自动挑一个最快的非生图模型。
    """
    requested = [c for c in (channels or DEFAULT_CHANNELS) if c in CHANNEL_META]
    if not requested:
        requested = list(DEFAULT_CHANNELS)

    models = await get_express_models()
    probe_model, model_note = pick_probe_model(models, model)
    settings = app_state.get_effective_settings(probe_model)

    started = time.monotonic()
    sem = asyncio.Semaphore(PROBE_CONCURRENCY)
    tasks = []
    for channel in requested:
        for entry in _entries_for_channel(channel, probe_model, settings):
            tasks.append(_probe_one(entry, sem))
    results = await asyncio.gather(*tasks)

    by_channel: dict = {c: [] for c in requested}
    for r in results:
        by_channel[r["channel"]].append(r)

    channel_reports = []
    for channel in requested:
        creds = sorted(by_channel[channel], key=lambda r: r["index"])
        ok_count = sum(1 for c in creds if c["ok"])
        if not creds:
            message, ok = "未配置任何凭证。", False
        elif ok_count == len(creds):
            message, ok = f"全部 {len(creds)} 条凭证可用。", True
        else:
            message, ok = f"{len(creds) - ok_count}/{len(creds)} 条凭证不可用。", False
        channel_reports.append({
            "channel": channel,
            "display": channel_display_name(channel),
            "configured": bool(creds),
            "ok": ok,
            "message": message,
            "credentials": creds,
        })

    total = len(results)
    ok_total = sum(1 for r in results if r["ok"])
    print(f"🩺 [渠道探活] 模型 {probe_model}：{ok_total}/{total} 条凭证通过，"
          f"耗时 {time.monotonic() - started:.1f}s。")
    return {
        "model": probe_model,
        "model_note": model_note,
        "elapsed_sec": round(time.monotonic() - started, 2),
        "summary": {"total": total, "ok": ok_total, "failed": total - ok_total},
        "channels": channel_reports,
    }
