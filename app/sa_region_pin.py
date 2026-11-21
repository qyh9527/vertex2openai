"""SA 区域固定：区域回退成功后，把该凭证固定到成功的 us/eu 区域 24 小时。

只记录「区域回退实际成功」的凭证与区域；账号本身配置为 us/eu 时不需要固定。
存储使用 STATE_DIR/sa_region_pins.sqlite3，进程重启后仍然有效；TTL 从固定时刻
起算，不因后续成功请求续期。读取失败只影响本次区域选择，不改变聊天结果。
"""

import asyncio
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Optional

PIN_TTL_SECONDS = 24 * 3600
VALID_REGIONS = ("us", "eu")

_lock = threading.Lock()


def _now() -> float:
    return time.time()


def _db_path() -> Path:
    return Path(os.getenv("STATE_DIR", ".")) / "sa_region_pins.sqlite3"


def _connect() -> sqlite3.Connection:
    path = _db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path, timeout=10)
    try:
        db.execute(
            "CREATE TABLE IF NOT EXISTS region_pins ("
            "credential_id TEXT PRIMARY KEY, location TEXT NOT NULL, "
            "pinned_at REAL NOT NULL)")
        db.commit()
        return db
    except BaseException:
        db.close()
        raise


def lookup(credential_id: str) -> Optional[str]:
    """返回该凭证仍在 24 小时内的固定区域；不存在/过期返回 None。"""
    if not credential_id:
        return None
    try:
        with _lock:
            db = _connect()
            try:
                row = db.execute(
                    "SELECT location, pinned_at FROM region_pins WHERE credential_id=?",
                    (credential_id,)).fetchone()
                if row is None:
                    return None
                location, pinned_at = row
                if _now() - float(pinned_at) > PIN_TTL_SECONDS:
                    db.execute("DELETE FROM region_pins WHERE credential_id=?",
                               (credential_id,))
                    db.commit()
                    return None
                return location if location in VALID_REGIONS else None
            finally:
                db.close()
    except Exception as exc:
        print(f"⚠️ [区域固定] 读取固定区域失败，已忽略：{exc}")
        return None


def pin(credential_id: str, location: str) -> bool:
    """记录凭证的固定区域；已有未过期固定时保留原记录，返回是否新写入。"""
    location = str(location or "").strip().lower()
    if not credential_id or location not in VALID_REGIONS:
        return False
    try:
        with _lock:
            db = _connect()
            try:
                now = _now()
                row = db.execute(
                    "SELECT location, pinned_at FROM region_pins WHERE credential_id=?",
                    (credential_id,)).fetchone()
                if row is not None and now - float(row[1]) <= PIN_TTL_SECONDS:
                    return False
                db.execute(
                    "INSERT INTO region_pins(credential_id, location, pinned_at) "
                    "VALUES (?, ?, ?) ON CONFLICT(credential_id) DO UPDATE SET "
                    "location=excluded.location, pinned_at=excluded.pinned_at",
                    (credential_id, location, now))
                db.commit()
                return True
            finally:
                db.close()
    except Exception as exc:
        print(f"⚠️ [区域固定] 写入固定区域失败，已忽略：{exc}")
        return False


async def commit_success() -> None:
    """上游响应通过验证后调用：仅当本次成功来自区域回退时固定该区域。"""
    from runtime_state import (_current_pricing_location, _current_region_fallback,
                               _current_sa_account)
    if not _current_region_fallback.get():
        return
    account = _current_sa_account.get()
    if not account:
        return
    location = str(_current_pricing_location.get() or account.get("location") or "").strip().lower()
    if location not in VALID_REGIONS:
        return
    import sa_budget
    try:
        key = sa_budget.credential_id(account.get("sa_json") or "")
    except Exception:
        return
    if await asyncio.to_thread(pin, key, location):
        print(f"📌 [区域固定] 凭证 …{key[-4:]} 在 {location} 调用成功，"
              f"未来 {PIN_TTL_SECONDS // 3600} 小时固定使用该区域（非 global 单价 +10%）。")


def clear_all() -> int:
    """测试/维护入口：清空全部固定记录。"""
    with _lock:
        db = _connect()
        try:
            removed = db.execute("SELECT COUNT(*) FROM region_pins").fetchone()[0]
            db.execute("DELETE FROM region_pins")
            db.commit()
        finally:
            db.close()
    return int(removed or 0)
