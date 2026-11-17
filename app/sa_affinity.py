"""SA 会话亲和：把同一会话钉在同一个 GCP project（保住上游隐式缓存）。

隐式缓存按 project 隔离、永不跨项目共享；多个 SA 账号按轮询/随机选号会让同一
会话每轮 cold start。本模块用**内容指纹**识别会话：

  输入 = 全部 system 消息 + 直到首条 user 为止的消息（含首条 user）；
  规范编码（长度前缀、对象键排序、类型标记）后 SHA-256，取前 16 字节 = 32 hex。

绑定存独立 SQLite（STATE_DIR/sa_affinity.sqlite3，与 sa_budget 分开）：

  affinity(session_id PK, project_id, created_at, last_seen)

  - TTL 1 小时，命中即续期（滑动）；
  - 容量 5000 条，超出按 last_seen 淘汰最旧；
  - 上游成功后才写绑定；瞬时故障（熔断冷却/单次失败/临时限流）保留原绑定；
  - 结构性不可用（SA 被删、该 project 全部账号额度熔断/赠金到期）重绑，且不受
    缓存门槛限制；
  - 新会话只有本次请求真实 prompt token 达到模型门槛才建绑定。
"""

import hashlib
import os
import sqlite3
import threading
import time
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Optional

TTL_SECONDS = 3600
MAX_BINDINGS = 5000

# 官方上下文缓存最小前缀：以下 4 个模型为 6144，Gemini 2 家族 2048，其余 Gemini 3/4 为 4096。
LARGE_PREFIX_MODELS = {
    "gemini-3.0-flash-preview",
    "gemini-3.1-pro-preview",
    "gemini-3.7-flash",
    "gemini-3.8-flash",
}
LARGE_PREFIX_MIN_TOKENS = 6144
DEFAULT_MIN_TOKENS = 4096
GEMINI2_MIN_TOKENS = 2048

_lock = threading.Lock()

# 请求级 ContextVar：会话指纹、绑定查询结果、待写入的绑定决策。
_session_id: ContextVar[Optional[str]] = ContextVar("sa_affinity_session", default=None)
_selection: ContextVar[Optional[dict]] = ContextVar("sa_affinity_selection", default=None)
_pending: ContextVar[Optional[dict]] = ContextVar("sa_affinity_pending", default=None)


# ---------------------------------------------------------------- 开关 / 存储

def enabled() -> bool:
    """控制台全局开关，默认关闭。"""
    try:
        from runtime_state import app_state
        import config as app_config
        return bool(app_state.get_setting(
            "sa_session_affinity", app_config.DEFAULT_SETTINGS.get("sa_session_affinity", False)))
    except Exception:
        return False


def _db_path() -> Path:
    return Path(os.getenv("STATE_DIR", ".")) / "sa_affinity.sqlite3"


def _connect() -> sqlite3.Connection:
    path = _db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path, timeout=10)
    db.execute(
        "CREATE TABLE IF NOT EXISTS affinity ("
        "session_id TEXT PRIMARY KEY, project_id TEXT NOT NULL, "
        "created_at REAL NOT NULL, last_seen REAL NOT NULL)")
    db.execute("CREATE INDEX IF NOT EXISTS affinity_last_seen ON affinity(last_seen)")
    db.commit()
    return db


def _mask(project: str) -> str:
    project = str(project or "")
    if len(project) <= 14:
        return project
    return f"{project[:8]}…{project[-4:]}"


# ---------------------------------------------------------------- 指纹

def _encode(value: Any) -> bytes:
    """确定性规范编码：类型标记 + 长度前缀 + 对象键排序。"""
    if value is None:
        return b"n"
    if isinstance(value, bool):
        return b"b1" if value else b"b0"
    if isinstance(value, (int, float)):
        return b"i" + repr(value).encode("utf-8") + b";"
    if isinstance(value, str):
        raw = value.encode("utf-8")
        return b"s" + str(len(raw)).encode("ascii") + b":" + raw
    if isinstance(value, (list, tuple)):
        parts = [_encode(item) for item in value]
        return b"a" + str(len(parts)).encode("ascii") + b"[" + b"".join(parts)
    if isinstance(value, dict):
        items = []
        for key in sorted(value, key=lambda k: str(k)):
            items.append(_encode(str(key)) + _encode(value[key]))
        return b"o" + str(len(items)).encode("ascii") + b"{" + b"".join(items)
    return _encode(str(value))


def _role(message: Any) -> str:
    if isinstance(message, dict):
        return str(message.get("role") or "")
    return str(getattr(message, "role", "") or "")


def _message_payload(message: Any) -> dict:
    content = (message.get("content") if isinstance(message, dict)
               else getattr(message, "content", None))
    if isinstance(content, list):
        content = [part.model_dump() if hasattr(part, "model_dump") else part
                   for part in content]
    return {"role": _role(message), "content": content}


def _stable_prefix(messages: list) -> Optional[list]:
    """全部 system 消息 + 直到首条 user 为止的非 system 消息（含首条 user）。

    上游把所有位置的 system 汇总成 system_instruction，因此后置 system 也属于
    缓存前缀；先按原序取全部 system，再拼接首条 user 前的非 system 消息。
    """
    messages = list(messages or [])
    systems = [m for m in messages if _role(m) == "system"]
    has_user = any(_role(m) == "user" for m in messages)
    if not systems and not has_user:
        return None
    leading = []
    for message in messages:
        if _role(message) == "system":
            continue
        leading.append(message)
        if _role(message) == "user":
            break
    return systems + leading


def session_id_for_messages(messages: list) -> str:
    """会话内容指纹（32 hex）；无法得到稳定前缀时返回空串。"""
    prefix = _stable_prefix(messages)
    if prefix is None:
        return ""
    payload = [_message_payload(message) for message in prefix]
    return hashlib.sha256(_encode(payload)).hexdigest()[:32]


def effective_project(project_id: Any, sa_json: Any) -> str:
    """SA 账号的生效 project：账号配置优先，空则取 SA JSON 自带 project_id。"""
    project = str(project_id or "").strip()
    if project:
        return project
    try:
        import json
        return str(json.loads(sa_json or "{}").get("project_id") or "").strip()
    except Exception:
        return ""


def min_tokens_for_model(model: Any) -> int:
    """该模型的隐式缓存最小前缀门槛（token）。"""
    name = str(model or "").rsplit("/", 1)[-1].strip().lower()
    if name in LARGE_PREFIX_MODELS:
        return LARGE_PREFIX_MIN_TOKENS
    if name.startswith("gemini-2"):
        return GEMINI2_MIN_TOKENS
    return DEFAULT_MIN_TOKENS


# ---------------------------------------------------------------- 绑定读写

def _lookup(session_id: str) -> tuple[Optional[str], float]:
    """返回 (project_id, 距上次访问秒数)；过期/不存在返回 (None, 0)。命中即续期。"""
    now = time.time()
    with _lock:
        db = _connect()
        try:
            row = db.execute(
                "SELECT project_id, last_seen FROM affinity WHERE session_id=?",
                (session_id,)).fetchone()
            if row is None:
                return None, 0.0
            project, last_seen = str(row[0]), float(row[1])
            age = max(0.0, now - last_seen)
            if age > TTL_SECONDS:
                db.execute("DELETE FROM affinity WHERE session_id=?", (session_id,))
                db.commit()
                return None, 0.0
            db.execute("UPDATE affinity SET last_seen=? WHERE session_id=?", (now, session_id))
            db.commit()
            return project, age
        finally:
            db.close()


def _evict_overflow(db) -> None:
    count = db.execute("SELECT COUNT(*) FROM affinity").fetchone()[0]
    if count > MAX_BINDINGS:
        db.execute(
            "DELETE FROM affinity WHERE session_id IN ("
            "SELECT session_id FROM affinity ORDER BY last_seen ASC LIMIT ?)",
            (count - MAX_BINDINGS,))


def _write(session_id: str, project_id: str) -> None:
    """无条件写入（测试/显式覆盖用）。"""
    now = time.time()
    with _lock:
        db = _connect()
        try:
            db.execute(
                "INSERT INTO affinity(session_id, project_id, created_at, last_seen) "
                "VALUES (?, ?, ?, ?) ON CONFLICT(session_id) DO UPDATE SET "
                "project_id=excluded.project_id, last_seen=excluded.last_seen",
                (session_id, project_id, now, now))
            _evict_overflow(db)
            db.commit()
        finally:
            db.close()


def _write_decision(decision: dict) -> None:
    """按决策写绑定：首次建绑只补空位；结构重绑只替换自己看到的旧绑定。

    并发首请求可能都查到“无绑定”，先成功者写下绑定后，后到者不得覆盖；
    结构重绑以决策里的 bound_project 做条件更新，避免覆盖其他请求刚修复的绑定。
    """
    session_id = decision["session_id"]
    project_id = decision["project_id"]
    state = decision.get("state")
    bound_project = decision.get("bound_project")
    now = time.time()
    with _lock:
        db = _connect()
        try:
            if state == "structural" and bound_project:
                cursor = db.execute(
                    "UPDATE affinity SET project_id=?, last_seen=? "
                    "WHERE session_id=? AND project_id=?",
                    (project_id, now, session_id, bound_project))
                if cursor.rowcount == 0:
                    db.execute(
                        "INSERT INTO affinity(session_id, project_id, created_at, last_seen) "
                        "VALUES (?, ?, ?, ?) ON CONFLICT(session_id) DO NOTHING",
                        (session_id, project_id, now, now))
            else:
                db.execute(
                    "INSERT INTO affinity(session_id, project_id, created_at, last_seen) "
                    "VALUES (?, ?, ?, ?) ON CONFLICT(session_id) DO NOTHING",
                    (session_id, project_id, now, now))
            _evict_overflow(db)
            db.commit()
        finally:
            db.close()


def stats() -> dict:
    """控制台展示：绑定数、最近命中/新建时间（epoch 秒）。"""
    with _lock:
        db = _connect()
        try:
            count, last_hit, last_created = db.execute(
                "SELECT COUNT(*), MAX(last_seen), MAX(created_at) FROM affinity").fetchone()
        finally:
            db.close()
    return {
        "count": int(count or 0),
        "capacity": MAX_BINDINGS,
        "ttl_seconds": TTL_SECONDS,
        "last_hit_at": last_hit,
        "last_created_at": last_created,
    }


def clear_all() -> int:
    with _lock:
        db = _connect()
        try:
            removed = db.execute("SELECT COUNT(*) FROM affinity").fetchone()[0]
            db.execute("DELETE FROM affinity")
            db.commit()
        finally:
            db.close()
    return int(removed or 0)


# ---------------------------------------------------------------- 请求级流程

def reset_request() -> None:
    """请求任务内调用：重置本请求的亲和状态（不碰数据库）。"""
    _pending.set(None)
    _selection.set(None)
    _session_id.set(None)


def set_session_id(session_id: Optional[str]) -> None:
    """请求任务内写回指纹（纯计算可在线程里完成）。"""
    _session_id.set(session_id or None)


def prepare_request(request_obj: Any) -> Optional[str]:
    """同步便捷入口（测试/直接调用）：reset + 计算指纹 + 写回 ContextVar。

    生产异步热路径请用 reset_request() + to_thread(session_id_for_messages) +
    set_session_id()，避免大提示词/媒体指纹阻塞事件循环。
    """
    reset_request()
    if not enabled():
        return None
    try:
        session_id = session_id_for_messages(getattr(request_obj, "messages", []) or [])
    except Exception:
        return None
    set_session_id(session_id)
    return session_id or None


def lookup_binding(session_id: str) -> tuple[Optional[str], float]:
    """线程内查询绑定（命中即滑动续期）；存储异常只放弃本次亲和，不影响聊天。"""
    if not session_id:
        return None, 0.0
    try:
        return _lookup(session_id)
    except Exception as exc:
        print(f"⚠️ [会话亲和] 读取绑定失败，已忽略本次亲和：{exc}")
        return None, 0.0


def apply_lookup(session_id: str, bound: Optional[str], age: float = 0.0) -> None:
    """请求任务内应用绑定查询结果（ContextVar 写回主任务）。"""
    _selection.set({
        "session_id": session_id,
        "bound_project": bound,
        "selected_project": None,
        "state": "new" if bound is None else "hit",
    })
    if bound:
        print(f"↔️ [会话亲和] 命中绑定 project={_mask(bound)}"
              f"（距上次访问 {age:.0f}s，TTL {TTL_SECONDS}s）。")


def note_request(request_obj: Any) -> Optional[str]:
    """同步便捷入口（测试/直接调用）：prepare + lookup + apply。"""
    session_id = prepare_request(request_obj)
    if not session_id:
        return None
    bound, age = lookup_binding(session_id)
    apply_lookup(session_id, bound, age)
    return bound


def current_session_id() -> Optional[str]:
    return _session_id.get()


def preferred_project() -> Optional[str]:
    selection = _selection.get() or {}
    return selection.get("bound_project")


def current_selection() -> Optional[dict]:
    return _selection.get()


def note_selection(selected_project: str, state: str) -> None:
    """记录账号选择结果（state: new / hit / transient / structural）。"""
    selection = dict(_selection.get() or {})
    if not selection.get("session_id"):
        return
    bound = selection.get("bound_project")
    selection.update({"selected_project": selected_project, "state": state})
    _selection.set(selection)
    if state == "transient":
        print(f"⚠️ [会话亲和] 绑定 project={_mask(bound)} 暂不可用，本次改用 "
              f"project={_mask(selected_project)}，绑定保留。")
    elif state == "structural":
        print(f"🔁 [会话亲和] 原绑定 project={_mask(bound)} 已不可用，"
              f"成功后将重绑至 project={_mask(selected_project)}。")


def build_decision(project_id: str, model: str) -> Optional[dict]:
    """在 `_resolve_client` 里组装「待写入的绑定决策」，随 resolved dict 返回。"""
    selection = _selection.get() or {}
    if not selection.get("session_id") or not project_id:
        return None
    return {
        "session_id": selection["session_id"],
        "project_id": str(project_id),
        "state": selection.get("state") or "new",
        "bound_project": selection.get("bound_project"),
        "model": str(model or ""),
    }


def begin_commit(decision: Optional[dict]) -> None:
    """在请求任务（主线程上下文）里登记待写入决策；上游成功后再由 commit 落库。"""
    _pending.set(dict(decision) if decision else None)


def note_prompt_tokens(tokens: Any) -> None:
    """记录本次请求真实 prompt token 数（取自 BudgetModels 的 countTokens 结果）。"""
    pending = _pending.get()
    if pending is None:
        return
    if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens < 0:
        return
    pending["prompt_tokens"] = tokens
    _pending.set(pending)


def take_pending() -> Optional[dict]:
    """请求任务内取出并校验待写入决策；返回可直接交给写库线程的决策。

    命中/瞬时故障/未达门槛都返回 None，并清空本请求的待写入状态。
    """
    pending = _pending.get()
    _pending.set(None)
    if pending is None:
        return None
    state = str(pending.get("state") or "")
    if state in ("hit", "transient"):
        return None
    tokens = pending.get("prompt_tokens")
    threshold = min_tokens_for_model(pending.get("model", ""))
    if state != "structural":
        if tokens is None or tokens < threshold:
            if tokens is not None:
                print(f"ℹ️ [会话亲和] 未建立绑定：prompt {tokens} 低于缓存门槛 {threshold}。")
            return None
    if not pending.get("session_id") or not pending.get("project_id"):
        return None
    pending["threshold"] = threshold
    return pending


def write_decision(decision: dict) -> None:
    """写库线程内执行；存储异常只放弃亲和，不改变聊天结果。"""
    try:
        _write_decision(decision)
    except Exception as exc:
        print(f"⚠️ [会话亲和] 写入绑定失败，已忽略本次亲和：{exc}")
        return
    project = decision.get("project_id")
    if decision.get("state") == "structural":
        print(f"🔁 [会话亲和] 已重绑至 project={_mask(project)}（不受缓存门槛限制）。")
    else:
        print(f"🆕 [会话亲和] 新建绑定 project={_mask(project)}"
              f"（prompt {decision.get('prompt_tokens')} ≥ 门槛 {decision.get('threshold')}）。")


def commit_success() -> None:
    """同步便捷入口：取决策并写库（生产路径应把写库放到线程里）。"""
    decision = take_pending()
    if decision is not None:
        write_decision(decision)
