"""SA 会话亲和：把同一会话绑定到一个 SA 凭证和 GCP project（保住上游隐式缓存）。

隐式缓存按 project 隔离、永不跨项目共享；多个 SA 账号按轮询/随机选号会让同一
会话每轮 cold start。本模块用**内容指纹**识别会话：

  输入 = 全部 system 消息 + 直到首条 user 为止的消息（含首条 user）；
  规范编码（长度前缀、对象键排序、类型标记）后 SHA-256，取前 16 字节 = 32 hex。

绑定存独立 SQLite（STATE_DIR/sa_affinity.sqlite3，与 sa_budget 分开）：

  affinity(session_id PK, project_id, created_at, last_seen, credential_id)

  - TTL 1 小时，命中即续期（滑动）；
  - 容量 5000 条，超出按 last_seen 淘汰最旧；
  - 改用不同凭证或 project 时，只在上游成功后更新，不受缓存门槛限制；
  - 旧 project 绑定保留，首次成功后补齐凭证标识；
  - 新会话只有本次请求真实 prompt token 达到模型门槛才建绑定。
"""

import asyncio
import hashlib
import math
import os
import sqlite3
import threading
import time
from contextvars import ContextVar
from dataclasses import dataclass
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

@dataclass(frozen=True)
class Binding:
    project_id: str
    credential_id: str
    # 每次重绑换版本；滑动续期只更新 last_seen。
    created_at: float


# 请求级 ContextVar：会话指纹、绑定查询结果、待写入的绑定决策。
_session_id: ContextVar[Optional[str]] = ContextVar("sa_affinity_session", default=None)
_binding: ContextVar[Optional[Binding]] = ContextVar("sa_affinity_binding", default=None)
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
    try:
        db.execute(
            "CREATE TABLE IF NOT EXISTS affinity ("
            "session_id TEXT PRIMARY KEY, project_id TEXT NOT NULL, "
            "created_at REAL NOT NULL, last_seen REAL NOT NULL, "
            "credential_id TEXT NOT NULL DEFAULT '')")
        if "credential_id" not in {row[1] for row in db.execute("PRAGMA table_info(affinity)")}:
            # 多个进程可能同时打开旧表；取得写锁后再检查一次。
            db.execute("BEGIN IMMEDIATE")
            if "credential_id" not in {row[1] for row in db.execute("PRAGMA table_info(affinity)")}:
                db.execute("ALTER TABLE affinity ADD COLUMN credential_id TEXT NOT NULL DEFAULT ''")
        db.execute("CREATE INDEX IF NOT EXISTS affinity_last_seen ON affinity(last_seen)")
        db.commit()
        return db
    except BaseException:
        db.close()
        raise


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


def min_tokens_for_model(model: Any) -> int:
    """该模型的隐式缓存最小前缀门槛（token）。"""
    name = str(model or "").rsplit("/", 1)[-1].strip().lower()
    if name in LARGE_PREFIX_MODELS:
        return LARGE_PREFIX_MIN_TOKENS
    if name.startswith("gemini-2"):
        return GEMINI2_MIN_TOKENS
    return DEFAULT_MIN_TOKENS


# ---------------------------------------------------------------- 绑定读写

def _lookup(session_id: str) -> tuple[Optional[Binding], float]:
    """读取并续期绑定；过期/不存在返回 (None, 0)。"""
    with _lock:
        db = _connect()
        try:
            db.execute("BEGIN IMMEDIATE")
            now = time.time()
            row = db.execute(
                "SELECT project_id, credential_id, created_at, last_seen FROM affinity WHERE session_id=?",
                (session_id,)).fetchone()
            if row is None:
                return None, 0.0
            project, credential, created_at, last_seen = row
            age = max(0.0, now - last_seen)
            if age > TTL_SECONDS:
                db.execute("DELETE FROM affinity WHERE session_id=?", (session_id,))
                db.commit()
                return None, 0.0
            db.execute("UPDATE affinity SET last_seen=MAX(last_seen, ?) WHERE session_id=?", (now, session_id))
            db.commit()
            return Binding(project, credential, created_at), age
        finally:
            db.close()


def _evict_overflow(db) -> None:
    count = db.execute("SELECT COUNT(*) FROM affinity").fetchone()[0]
    if count > MAX_BINDINGS:
        db.execute(
            "DELETE FROM affinity WHERE session_id IN ("
            "SELECT session_id FROM affinity ORDER BY last_seen ASC LIMIT ?)",
            (count - MAX_BINDINGS,))


def _write_decision(decision: dict) -> bool:
    """只在绑定未被其他请求替换时更新；首次建绑只补空位。"""
    session_id = decision["session_id"]
    project_id = decision["project_id"]
    credential_id = decision["credential_id"]
    previous = decision.get("previous")
    with _lock:
        db = _connect()
        try:
            now = time.time()
            if previous is None:
                cursor = db.execute(
                    "INSERT INTO affinity(session_id, project_id, credential_id, created_at, last_seen) "
                    "VALUES (?, ?, ?, ?, ?) ON CONFLICT(session_id) DO NOTHING",
                    (session_id, project_id, credential_id, now, now))
                if cursor.rowcount:
                    _evict_overflow(db)
            else:
                version = max(now, math.nextafter(previous.created_at, math.inf))
                cursor = db.execute(
                    "UPDATE affinity SET project_id=?, credential_id=?, created_at=?, last_seen=MAX(last_seen, ?) "
                    "WHERE session_id=? AND project_id=? AND credential_id=? AND created_at=? AND last_seen>=?",
                    (project_id, credential_id, version, now, session_id, previous.project_id,
                     previous.credential_id, previous.created_at, now - TTL_SECONDS))
            db.commit()
            return cursor.rowcount > 0
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
    _binding.set(None)
    _session_id.set(None)


def lookup_binding(session_id: str) -> tuple[Optional[Binding], float]:
    """在线程中读取；存储异常不影响聊天。"""
    try:
        return _lookup(session_id)
    except Exception as exc:
        print(f"⚠️ [会话亲和] 读取绑定失败，已忽略本次亲和：{exc}")
        return None, 0.0


def apply_lookup(session_id: str, bound: Optional[Binding], age: float) -> None:
    """在请求任务中应用绑定查询结果。"""
    _session_id.set(session_id)
    _binding.set(bound)
    if bound:
        print(f"↔️ [会话亲和] 找到绑定 project={_mask(bound.project_id)}"
              f"（距上次访问 {age:.0f}s，TTL {TTL_SECONDS}s）。")


def selection_preferences() -> dict:
    """绑定凭证优先；旧记录只有 project 信息。"""
    bound = _binding.get()
    if not bound:
        return {}
    prefs = {"preferred_project": bound.project_id}
    if bound.credential_id:
        prefs["preferred_credential"] = bound.credential_id
    return prefs


def build_decision(project_id: str, credential_id: str, model: str) -> Optional[dict]:
    session_id = _session_id.get()
    previous = _binding.get()
    if not session_id or not project_id or not credential_id:
        return None
    if previous and (project_id, credential_id) == (previous.project_id, previous.credential_id):
        return None
    return dict(session_id=session_id, project_id=project_id, credential_id=credential_id,
                previous=previous, model=model)


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
    """取出待提交的决策；只有初次建绑需要达到缓存门槛。"""
    pending = _pending.get()
    _pending.set(None)
    if pending is not None and pending.get("previous") is None:
        tokens = pending.get("prompt_tokens")
        threshold = min_tokens_for_model(pending.get("model", ""))
        if tokens is None or tokens < threshold:
            if tokens is not None:
                print(f"ℹ️ [会话亲和] 未建立绑定：prompt {tokens} 低于缓存门槛 {threshold}。")
            return None
        pending["threshold"] = threshold
    return pending


def write_decision(decision: dict) -> None:
    """写库线程内执行；存储异常只放弃亲和，不改变聊天结果。"""
    try:
        if not _write_decision(decision):
            return
    except Exception as exc:
        print(f"⚠️ [会话亲和] 写入绑定失败，已忽略本次亲和：{exc}")
        return
    project = decision.get("project_id")
    if decision.get("previous") is not None:
        print(f"🔁 [会话亲和] 已刷新凭证绑定 project={_mask(project)}。")
    else:
        print(f"🆕 [会话亲和] 新建绑定 project={_mask(project)}"
              f"（prompt {decision.get('prompt_tokens')} ≥ 门槛 {decision.get('threshold')}）。")


async def commit_success() -> None:
    """上游响应通过验证后，在请求任务取决策并交给写库线程。"""
    decision = take_pending()
    if decision is not None:
        await asyncio.to_thread(write_decision, decision)
