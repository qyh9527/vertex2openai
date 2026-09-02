import asyncio
import json
import os
import secrets
import threading
import time
from fastapi import FastAPI, Depends, Request, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic import BaseModel

from auth import get_api_key
from express_key_manager import ExpressKeyManager
from routes import models_api, chat_api

from logger import rt_logger, stats, read_recent_log_lines
import config
from runtime_state import app_state
import model_capabilities as mc
from model_loader import get_express_models

from cookie_auth import validate_cookie
from upstreams.service_account import validate_sa_credentials

express_key_manager = ExpressKeyManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 [服务启动] agentplatform2api 适配器已启动（Express API Key / Cookie 直连 / 服务账号 三通道）。")

    # S-1：默认口令 + 公开托管 + 明文 Cookie 是很危险的组合，必须让人看见。
    if config.API_KEY == DEFAULT_API_KEY:
        public_host = any(os.environ.get(k) for k in ("SPACE_ID", "SPACE_HOST", "HF_SPACE_ID"))
        if public_host and os.environ.get("ALLOW_DEFAULT_KEY", "").lower() not in ("1", "true", "yes"):
            raise RuntimeError(
                "检测到公开托管环境（HuggingFace Space 等）且 API_KEY 仍为默认值 123456。\n"
                "该口令同时是控制台登录密码，而控制台可以读写完整的 Google 会话 Cookie。\n"
                "请设置一个强 API_KEY 后重启；确需临时放行可设 ALLOW_DEFAULT_KEY=true。"
            )
        print("🔴 [安全警告] API_KEY 仍是默认值 123456！它既是本代理的 Key，也是控制台登录口令，"
              "请立刻改成强口令。")
    if express_key_manager.get_total_keys() > 0:
        print(f"✅ [密钥配置] 已加载 {express_key_manager.get_total_keys()} 个 Express API Key。")
    else:
        print("⚠️ [密钥配置] 未检测到 VERTEX_EXPRESS_API_KEY。若不启用 Cookie 直连模式，聊天请求将会报错。")
    yield

app = FastAPI(title="agentplatform2api", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    # 本服务使用 Bearer / Basic 鉴权，不依赖浏览器 Cookie；
    # 关闭 allow_credentials 以符合 CORS 规范（通配符 + 凭证不合法）。
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.state.express_key_manager = express_key_manager


@app.middleware("http")
async def stats_tracker_middleware(request: Request, call_next):
    if "chat/completions" in request.url.path:
        stats.increment_total()
        try:
            response = await call_next(request)
            if response.status_code >= 400:
                stats.add_error(status=response.status_code)
            return response
        except Exception as e:
            stats.add_error(message=str(e))
            raise e
    return await call_next(request)


# ====== 仅密码登录（Cookie 会话，免输账号）======
AUTH_COOKIE = "ap_session"
DEFAULT_API_KEY = "123456"
SESSION_TTL_SECONDS = 60 * 60 * 24 * 30

# P2-2：会话 token 改为随机值存内存，不再用 sha256(常量 + API_KEY) 这种确定值。
# 确定值意味着同一个 API_KEY 永远对应同一个 cookie，无法单独失效某个会话。
_sessions: dict = {}          # token -> 过期时间戳
_sessions_lock = threading.Lock()

# 登录失败计数：{ip: [失败次数, 最近失败时间]}，指数退避
_login_failures: dict = {}
_login_lock = threading.Lock()
LOGIN_LOCK_BASE_SECONDS = 2
LOGIN_LOCK_MAX_SECONDS = 300


def _issue_session() -> str:
    token = secrets.token_urlsafe(32)
    now = time.time()
    with _sessions_lock:
        for t, exp in list(_sessions.items()):     # 顺手清理过期会话
            if exp < now:
                _sessions.pop(t, None)
        _sessions[token] = now + SESSION_TTL_SECONDS
    return token


def _revoke_session(token: str) -> None:
    with _sessions_lock:
        _sessions.pop(token, None)


def _is_authed(request: Request) -> bool:
    token = request.cookies.get(AUTH_COOKIE, "")
    if not token:
        return False
    with _sessions_lock:
        exp = _sessions.get(token)
        if exp is None:
            return False
        if exp < time.time():
            _sessions.pop(token, None)
            return False
    return True


def _login_retry_after(ip: str) -> int:
    """该 IP 还需等待多少秒才能再次尝试登录（0 = 可以尝试）。"""
    with _login_lock:
        rec = _login_failures.get(ip)
        if not rec:
            return 0
        count, last = rec
        if count < 3:
            return 0
        wait = min(LOGIN_LOCK_BASE_SECONDS * (2 ** (count - 3)), LOGIN_LOCK_MAX_SECONDS)
        remain = int(last + wait - time.time())
        return max(0, remain)


def _record_login_failure(ip: str) -> None:
    with _login_lock:
        count, _ = _login_failures.get(ip, (0, 0.0))
        _login_failures[ip] = (count + 1, time.time())


def _clear_login_failure(ip: str) -> None:
    with _login_lock:
        _login_failures.pop(ip, None)


def mask_cookie(cookie_str: str) -> str:
    """S-1：控制台只回显掩码，不再把完整 Google 会话 Cookie 明文吐回前端。"""
    if not cookie_str:
        return ""
    names = []
    for seg in cookie_str.split(";"):
        name = seg.strip().split("=", 1)[0].strip()
        if name:
            names.append(name)
    return f"已配置（共 {len(names)} 个 cookie 字段，{len(cookie_str)} 字符）"


def mask_key(key: str) -> str:
    """Express API Key 掩码：只露前后 4 位 + 长度。"""
    if not key:
        return ""
    if len(key) <= 8:
        return f"****（{len(key)} 字符）"
    return f"{key[:4]}…{key[-4:]}（{len(key)} 字符）"


def mask_sa_account(a: dict, index: int) -> dict:
    """服务账号掩码回显：只暴露 client_email（公开标识）+ project + location，绝不回填 SA JSON。"""
    email = ""
    try:
        email = str(json.loads(a.get("sa_json", "")).get("client_email") or "")
    except Exception:
        pass
    return {
        "index": index,
        "client_email": email,
        "project_id": a.get("project_id", ""),
        "location": a.get("location", "global"),
    }


async def require_auth(request: Request):
    if not _is_authed(request):
        raise HTTPException(status_code=401, detail="未登录")
    return True


# 进阶报告 P1-6：Console 前端资源拆分为独立静态文件（app/console/），
# main.py 只保留加载逻辑；HTML 内容未做任何改动（字节级一致，脚本抽取）。
from console_assets import LOGIN_HTML, DASHBOARD_HTML


@app.get("/", response_class=HTMLResponse)
async def dashboard_ui(request: Request):
    if _is_authed(request):
        return HTMLResponse(DASHBOARD_HTML)
    return HTMLResponse(LOGIN_HTML)


class LoginBody(BaseModel):
    password: str


@app.post("/api/login")
async def login(body: LoginBody, request: Request):
    ip = request.client.host if request.client else "unknown"

    # P2-2：失败三次后指数退避，避免对口令（同时也是 API Key）无限爆破
    remain = _login_retry_after(ip)
    if remain > 0:
        return JSONResponse(status_code=429,
                            content={"error": f"尝试过于频繁，请 {remain} 秒后再试"})

    if config.API_KEY and secrets.compare_digest(body.password, config.API_KEY):
        _clear_login_failure(ip)
        resp = JSONResponse(content={"ok": True})
        resp.set_cookie(
            AUTH_COOKIE, _issue_session(),
            httponly=True, samesite="lax", max_age=SESSION_TTL_SECONDS, path="/",
            # 反代后 request.url.scheme 可能是 http，这里同时看 x-forwarded-proto
            secure=(request.url.scheme == "https"
                    or request.headers.get("x-forwarded-proto", "") == "https"),
        )
        return resp

    _record_login_failure(ip)
    print(f"🔐 [登录失败] 来自 {ip} 的密码尝试失败。")
    return JSONResponse(status_code=401, content={"error": "密码错误"})


@app.post("/api/logout")
async def logout(request: Request):
    _revoke_session(request.cookies.get(AUTH_COOKIE, ""))
    resp = JSONResponse(content={"ok": True})
    resp.delete_cookie(AUTH_COOKIE, path="/")
    return resp


@app.get("/api/stats")
async def get_stats_api(_auth: bool = Depends(require_auth)):
    return JSONResponse(content=stats.get_json_stats())


# ==========================================
# 设置与通道控制
# ==========================================
class ModeSetting(BaseModel):
    mode: str


@app.get("/api/settings/runtime")
async def get_runtime_settings(_auth: bool = Depends(require_auth)):
    cookie = app_state.get_google_cookie()
    accounts = app_state.get_cookie_accounts()
    keys = express_key_manager.express_keys
    sa_accounts = app_state.get_sa_accounts()
    return JSONResponse(content={
        "channel_strategy": app_state.get_channel_strategy(),
        # 旧前端兼容：布尔开关仍回显
        "use_web_proxy": app_state.is_web_proxy_enabled(),
        # S-1：只回显掩码。完整 Cookie 等价于该 Google 账号的完整访问权，
        # 没有任何理由让它出现在前端 JS / 浏览器缓存 / 截图里。
        "google_cookie": mask_cookie(cookie),
        "google_cookie_configured": bool(cookie),
        "google_project_id": app_state.get_project_id(),
        # 多账号凭证（掩码回显）
        "express_keys": [mask_key(k) for k in keys],
        "express_keys_controlled": app_state.get_express_keys() is not None,
        "express_keys_env_count": len(config.VERTEX_EXPRESS_API_KEY_VAL),
        "cookie_accounts": [
            {"index": i, "cookie": mask_cookie(a["cookie"]),
             "project_id": a.get("project_id", "")}
            for i, a in enumerate(accounts)
        ],
        # 服务账号（第三通道）掩码回显：只暴露 client_email + project + location
        "sa_accounts": [mask_sa_account(a, i) for i, a in enumerate(sa_accounts)],
        "sa_accounts_controlled": app_state.get_sa_accounts_console() != [],
        "sa_env_configured": bool(config.VERTEX_SA_JSON or config.VERTEX_SA_FILE),
        # 混合自动可配置项（hybrid 通道顺序 / 调度方式 / 每通道重试覆盖）
        "hybrid_channels": app_state.get_hybrid_channels(),
        "hybrid_dispatch_mode": app_state.get_hybrid_dispatch_mode(),
        "channel_retry_overrides": app_state.get_setting(
            "channel_retry_overrides", config.DEFAULT_SETTINGS["channel_retry_overrides"]),
        # PayGo 流量等级
        "paygo_tier": app_state.get_setting("paygo_tier", config.DEFAULT_SETTINGS["paygo_tier"]),
        "paygo_only": app_state.get_setting("paygo_only", config.DEFAULT_SETTINGS["paygo_only"]),
    })


@app.post("/api/express-keys")
async def save_express_keys(request: Request, _auth: bool = Depends(require_auth)):
    """整表覆盖保存 Express API Key 列表（控制台管理，覆盖环境变量）。"""
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": "请求体必须是 JSON。"})
    if not isinstance(body, dict) or not isinstance(body.get("keys"), list):
        return JSONResponse(status_code=400, content={"error": "请求体需要 keys 数组。"})
    clean = app_state.set_express_keys(body["keys"])
    express_key_manager.refresh_keys()
    print(f"🔑 [密钥管理] 控制台已保存 {len(clean)} 个 Express API Key。")
    return JSONResponse(content={"status": "success", "count": len(clean)})


class CookieAccountItem(BaseModel):
    index: int = -1          # >=0 更新已有账号；-1 新增
    cookie: str = ""         # 留空 = 保持该账号原 Cookie 不变
    project_id: str = ""
    delete: bool = False


@app.post("/api/cookie-account")
async def save_cookie_account(item: CookieAccountItem, _auth: bool = Depends(require_auth)):
    """多 Cookie 账号单行增/改/删（旧 /api/cookie 单账号端点保留兼容）。"""
    accounts = app_state.get_cookie_accounts()
    pid = (item.project_id or "").strip()

    if item.delete:
        if 0 <= item.index < len(accounts):
            removed = accounts.pop(item.index)
            app_state.set_cookie_accounts(accounts)
            print(f"🔑 [账号管理] 已删除 Cookie 账号 #{item.index + 1}。")
            return JSONResponse(content={"status": "success", "count": len(accounts)})
        return JSONResponse(status_code=400, content={"error": "账号索引无效。"})

    new_cookie = (item.cookie or "").strip()
    if 0 <= item.index < len(accounts):
        if new_cookie:
            accounts[item.index] = {"cookie": new_cookie, "project_id": pid}
        elif pid:
            accounts[item.index] = {**accounts[item.index], "project_id": pid}
        else:
            return JSONResponse(status_code=400, content={"error": "没有要更新的内容（Cookie 与 Project ID 都为空）。"})
        app_state.set_cookie_accounts(accounts)
        print(f"🔑 [账号管理] 已更新 Cookie 账号 #{item.index + 1}。")
        return JSONResponse(content={"status": "success", "count": len(accounts)})

    if not new_cookie:
        return JSONResponse(status_code=400, content={"error": "新增账号必须填写 Cookie。"})
    validation = validate_cookie(new_cookie)
    if not validation["valid"]:
        return JSONResponse(status_code=400, content={"error": validation["message"]})
    accounts.append({"cookie": new_cookie, "project_id": pid})
    app_state.set_cookie_accounts(accounts)
    print(f"🔑 [账号管理] 已新增 Cookie 账号（当前共 {len(accounts)} 个）。")
    return JSONResponse(content={"status": "success", "count": len(accounts)})


class ServiceAccountItem(BaseModel):
    index: int = -1          # >=0 更新已有账号；-1 新增
    sa_json: str = ""        # 留空 = 保持该账号原 SA JSON 不变
    project_id: str = ""     # 留空 = 取 SA JSON 自身 project_id
    location: str = "global"
    delete: bool = False


@app.post("/api/sa-account")
async def save_sa_account(item: ServiceAccountItem, _auth: bool = Depends(require_auth)):
    """服务账号（第三通道）单行增/改/删（凭证永不明文回显）。"""
    accounts = app_state.get_sa_accounts_console()
    loc = (item.location or "global").strip() or "global"
    new_sa = (item.sa_json or "").strip()
    pid = (item.project_id or "").strip()

    if item.delete:
        if 0 <= item.index < len(accounts):
            removed = accounts.pop(item.index)
            app_state.set_sa_accounts(accounts)
            print(f"🔑 [账号管理] 已删除服务账号 #{item.index + 1}。")
            return JSONResponse(content={"status": "success", "count": len(accounts)})
        return JSONResponse(status_code=400, content={"error": "账号索引无效。"})

    if 0 <= item.index < len(accounts):
        if new_sa:
            v = validate_sa_credentials(new_sa)
            if not v["valid"]:
                return JSONResponse(status_code=400, content={"error": v["message"]})
            accounts[item.index] = {"project_id": pid or v["project_id"], "location": loc, "sa_json": new_sa}
        else:
            accounts[item.index] = {
                **accounts[item.index],
                "project_id": pid or accounts[item.index].get("project_id", ""),
                "location": loc,
            }
        app_state.set_sa_accounts(accounts)
        print(f"🔑 [账号管理] 已更新服务账号 #{item.index + 1}（location={loc}）。")
        return JSONResponse(content={"status": "success", "count": len(accounts)})

    if not new_sa:
        return JSONResponse(status_code=400, content={"error": "新增账号必须填写服务账号 JSON。"})
    v = validate_sa_credentials(new_sa)
    if not v["valid"]:
        return JSONResponse(status_code=400, content={"error": v["message"]})
    accounts.append({"project_id": pid or v["project_id"], "location": loc, "sa_json": new_sa})
    app_state.set_sa_accounts(accounts)
    print(f"🔑 [账号管理] 已新增服务账号（当前共 {len(accounts)} 个，location={loc}）。")
    return JSONResponse(content={"status": "success", "count": len(accounts)})


@app.post("/api/settings/mode")
async def set_settings_mode(setting: ModeSetting, _auth: bool = Depends(require_auth)):
    # 前端取值：api_key|web_proxy（旧）/ express|cookie|vertex|hybrid（新），统一映射到四档策略
    raw = (setting.mode or "").strip().lower()
    mapping = {
        "api_key": "express",
        "web_proxy": "cookie",
        "express": "express",
        "cookie": "cookie",
        "vertex": "vertex",
        "hybrid": "hybrid",
    }
    strategy = mapping.get(raw)
    if strategy is None:
        return JSONResponse(status_code=400, content={"error": "无效的通道模式，应为 express / cookie / vertex / hybrid（vertex = 服务账号 SA 通道）。"})
    if not app_state.set_channel_strategy(strategy):
        return JSONResponse(status_code=400, content={"error": "设置通道策略失败。"})
    return JSONResponse(content={"status": "success", "channel_strategy": strategy})


@app.get("/api/settings")
async def get_settings_api(_auth: bool = Depends(require_auth)):
    return JSONResponse(content=app_state.get_settings())


@app.post("/api/settings")
async def update_settings_api(request: Request, _auth: bool = Depends(require_auth)):
    try:
        patch = await request.json()
    except Exception:
        patch = {}
    if not isinstance(patch, dict):
        return JSONResponse(status_code=400, content={"error": "请求体必须是 JSON 对象。"})
    updated = app_state.update_settings(patch)
    return JSONResponse(content=updated)


@app.get("/api/capabilities")
async def get_capabilities_api(_auth: bool = Depends(require_auth)):
    try:
        models = await get_express_models()
    except Exception:
        models = []
    # 能力摘要要反映该模型生效的采样策略，否则控制台提示与实际下发不一致
    caps = {m: mc.capabilities_summary(m, app_state.get_effective_settings(m)) for m in models}
    # 附带各模型是否已有专属参数覆盖，供前端标示
    overrides = app_state.get_model_overrides()
    return JSONResponse(content={"models": models, "capabilities": caps, "overrides": overrides})


@app.post("/api/models/refresh")
async def refresh_models_manual_api(_auth: bool = Depends(require_auth)):
    """控制台「获取远程模型」：手动拉取远程模型配置并持久化到磁盘（不再自动获取）。"""
    from model_loader import refresh_models_config_cache, get_express_models
    ok = await refresh_models_config_cache()
    models = await get_express_models()
    return JSONResponse(content={"ok": ok, "models": models})


@app.get("/api/models/manage")
async def list_models_manage_api(_auth: bool = Depends(require_auth)):
    """模型管理数据：全部模型（含自定义）+ 自定义列表（供「编辑模型」弹窗）。"""
    models = await get_express_models()
    return JSONResponse(content={"models": models, "custom": app_state.get_custom_models()})


class CustomModelsBody(BaseModel):
    models: list = []


@app.post("/api/models/custom")
async def set_custom_models_api(body: CustomModelsBody, _auth: bool = Depends(require_auth)):
    app_state.set_custom_models(body.models)
    return JSONResponse(content={"custom": app_state.get_custom_models()})


# ==========================================
# 按模型参数覆盖（per-model overrides）
# ==========================================
@app.get("/api/model-overrides")
async def list_model_overrides(_auth: bool = Depends(require_auth)):
    return JSONResponse(content=app_state.get_model_overrides())


@app.post("/api/model-overrides/{model_name}")
async def save_model_override(model_name: str, request: Request, _auth: bool = Depends(require_auth)):
    try:
        patch = await request.json()
    except Exception:
        patch = {}
    if not isinstance(patch, dict):
        return JSONResponse(status_code=400, content={"error": "请求体必须是 JSON 对象。"})
    saved = app_state.set_model_override(model_name, patch)
    return JSONResponse(content={"status": "success", "model": model_name, "override": saved})


@app.delete("/api/model-overrides/{model_name}")
async def delete_model_override(model_name: str, _auth: bool = Depends(require_auth)):
    ok = app_state.clear_model_override(model_name)
    return JSONResponse(content={"status": "success" if ok else "not_found", "model": model_name})


class CookieSetting(BaseModel):
    cookie: str = ""          # 留空 = 保持现有 Cookie
    project_id: str = ""


@app.post("/api/cookie")
async def set_google_cookie(setting: CookieSetting, _auth: bool = Depends(require_auth)):
    """保存 Cookie 与 Project ID。

    S-1：cookie 传空字符串表示「保持现有 Cookie 不变，只更新 Project ID」，
    这样前端就不需要为了改 Project ID 而把完整 Cookie 再取回来一次。
    """
    new_cookie = (setting.cookie or "").strip()
    project_id = (setting.project_id or "").strip()

    if new_cookie:
        validation = validate_cookie(new_cookie)
        if not validation["valid"]:
            return JSONResponse(status_code=400, content={"error": validation["message"]})
        app_state.set_google_cookie(new_cookie)
        message = validation["message"]
    else:
        if not app_state.get_google_cookie():
            return JSONResponse(status_code=400, content={
                "error": "尚未配置 Cookie，请粘贴完整的 Google Cookie。"})
        message = "✅ 已保留原有 Cookie，仅更新 Project ID。"

    if project_id:
        app_state.set_project_id(project_id)
    return JSONResponse(content={"status": "success", "message": message})


@app.get("/stream-logs")
async def stream_logs_endpoint(request: Request, _auth: bool = Depends(require_auth)):
    async def log_generator():
        # 先补发持久化日志的历史尾部，重建容器后前端也能看到此前的日志；再走实时流。
        for msg in read_recent_log_lines(200):
            yield f"data: {msg}\n\n"
        q = rt_logger.subscribe()
        try:
            for msg in rt_logger.snapshot_history():
                yield f"data: {msg}\n\n"
            while True:
                if await request.is_disconnected():
                    break
                try:
                    msg = await asyncio.wait_for(q.get(), timeout=1.0)
                    yield f"data: {msg}\n\n"
                except asyncio.TimeoutError:
                    yield ": keep-alive heartbeat\n\n"
        finally:
            rt_logger.unsubscribe(q)
    return StreamingResponse(log_generator(), media_type="text/event-stream")


app.include_router(models_api.router)
app.include_router(chat_api.router)
