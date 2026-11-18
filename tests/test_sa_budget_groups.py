"""sa_budget 账号归并（budget_members 分组）测试。

只使用 tmp_path 下的 sqlite：不发起真实上游请求、不读取真实凭证或日志。
覆盖：旧库无 budget_members、默认历史金额相加、自定义历史金额边界与确认指纹、
spent/holds 归并与合并后旧 token 仍可结算、并发预留/结算共享一本账、跨成员保护、
陈旧确认无副作用、失败事务回滚、同组冲突、未初始化成员沿用已记账日期、
链式归并成员与组基线去重，以及 POST /api/sa-budget/merge 的 DTO 预览/应用接线。
"""
import datetime
import json
import os
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

import sa_budget

NANO = 1_000_000_000          # 1 美元 = 10^9 纳美元


@pytest.fixture(autouse=True)
def _isolated_state(tmp_path, monkeypatch):
    """每个测试独立 STATE_DIR，账本只落在本次 tmp_path。"""
    monkeypatch.setenv("STATE_DIR", str(tmp_path))
    return tmp_path


def _db_path():
    return Path(os.environ["STATE_DIR"]) / "sa_budget.sqlite3"


def _query(sql, params=()):
    db = sqlite3.connect(_db_path())
    try:
        return db.execute(sql, params).fetchall()
    finally:
        db.close()


def _tables():
    return {row[0] for row in _query("SELECT name FROM sqlite_master WHERE type='table'")}


def _init(key, amount):
    sa_budget.initialize(key, amount)
    return key


def _preview(source, target, **kwargs):
    return sa_budget.merge_accounts(source, target, **kwargs)


def _apply(source, target, **kwargs):
    preview = _preview(source, target, **kwargs)
    return sa_budget.merge_accounts(source, target, confirmation=preview["confirmation"], **kwargs)


def _total(key):
    row = _query("SELECT baseline + spent FROM accounts WHERE id=?", (key,))
    holds = _query("SELECT COALESCE(SUM(amount), 0) FROM holds WHERE account=?", (key,))[0][0]
    return (row[0][0] if row else 0) + holds


# ---------- 旧 SQLite 无 budget_members ----------

def test_legacy_database_without_budget_members_table_still_merges():
    source, target = _init("legacy-a", 100), _init("legacy-b", 20)
    with sa_budget._connect() as db:
        db.execute("DROP TABLE budget_members")
    assert "budget_members" not in _tables()

    assert _apply(source, target) == {"account_id": target}

    assert "budget_members" in _tables()
    assert sa_budget.status(source)["account_id"] == target
    assert sa_budget.status(target)["baseline"] == 120
    assert _query("SELECT COUNT(*) FROM accounts") == [(1,)]


# ---------- 默认历史金额相加 / spent 与 holds 归并 ----------

def test_merge_defaults_to_sum_of_baselines_and_preserves_spent_and_holds():
    source, target = _init("src", 100), _init("dst", 50)
    spend = sa_budget.reserve(source, 4 * NANO)
    sa_budget.settle(spend, 3 * NANO)                 # 源侧 spent 3
    src_hold = sa_budget.reserve(source, 7 * NANO)
    dst_hold = sa_budget.reserve(target, 2 * NANO)
    sa_budget.settle(dst_hold, 1 * NANO)              # 目标侧 spent 1

    preview = _preview(source, target)
    assert preview["source"]["baseline"] == 100
    assert preview["target"]["baseline"] == 50
    # 预览不落库
    assert _query("SELECT COUNT(*) FROM budget_members") == [(0,)]
    assert dict(_query("SELECT id, baseline FROM accounts")) == {source: 100 * NANO, target: 50 * NANO}

    assert _apply(source, target) == {"account_id": target}

    data = sa_budget.status(target)
    assert data["baseline"] == 150                     # 默认 = 两基线相加
    assert data["spent"] == 4                          # 3 + 1
    assert data["reserved"] == 7                       # 源侧未结算 hold 一并归并
    assert _query("SELECT account FROM holds WHERE id=?", (src_hold,)) == [(target,)]

    sa_budget.settle(src_hold, 7 * NANO)               # 合并后旧 token 仍可结算
    after = sa_budget.status(target)
    assert after["spent"] == 11
    assert after["reserved"] == 0
    assert _query("SELECT COUNT(*) FROM accounts WHERE id=?", (source,)) == [(0,)]


# ---------- 自定义历史金额：边界与确认指纹 ----------

def test_custom_historical_amount_bounds_and_confirmation_fingerprint():
    source, target = _init("src", 100), _init("dst", 50)

    default = _preview(source, target)
    custom = _preview(source, target, historical_usd="120")
    assert default["confirmation"] != custom["confirmation"]   # 自定义金额进入确认指纹

    # 用默认指纹去确认自定义金额必须失败，且不落库。
    with pytest.raises(sa_budget.BudgetConflict):
        sa_budget.merge_accounts(source, target, confirmation=default["confirmation"],
                                 historical_usd="120")
    assert dict(_query("SELECT id, baseline FROM accounts")) == {source: 100 * NANO, target: 50 * NANO}
    assert _query("SELECT COUNT(*) FROM budget_members") == [(0,)]

    # 自定义必须落在 max(两基线)..sum(两基线) 内（100..150）。
    for bad in ("99.99", "150.000000001", "160"):
        with pytest.raises(ValueError) as err:
            _preview(source, target, historical_usd=bad)
        assert not isinstance(err.value, sa_budget.BudgetConflict)

    assert _apply(source, target, historical_usd="120") == {"account_id": target}
    assert sa_budget.status(target)["baseline"] == 120
    assert sa_budget.status(target)["spent"] == 0


def test_custom_historical_amount_accepts_both_bounds():
    source, target = _init("src", 100), _init("dst", 50)
    assert _apply(source, target, historical_usd="100") == {"account_id": target}
    assert sa_budget.status(target)["baseline"] == 100

    other_source, other_target = _init("src2", 100), _init("dst2", 50)
    assert _apply(other_source, other_target, historical_usd="150") == {"account_id": other_target}
    assert sa_budget.status(other_target)["baseline"] == 150


# ---------- 陈旧确认无副作用 ----------

def test_stale_confirmation_is_rejected_without_side_effects():
    source, target = _init("src", 100), _init("dst", 50)
    sa_budget.reserve(source, 2 * NANO)
    preview = _preview(source, target)
    sa_budget.reserve(target, NANO)                   # 预览后账本发生变化

    accounts_before = _query("SELECT id, baseline, spent FROM accounts ORDER BY id")
    holds_before = _query("SELECT id, account, amount FROM holds ORDER BY id")
    dates_before = dict(_query("SELECT id, start_date FROM trial_dates"))

    with pytest.raises(sa_budget.BudgetConflict):
        sa_budget.merge_accounts(source, target, confirmation=preview["confirmation"])

    assert _query("SELECT id, baseline, spent FROM accounts ORDER BY id") == accounts_before
    assert _query("SELECT id, account, amount FROM holds ORDER BY id") == holds_before
    assert dict(_query("SELECT id, start_date FROM trial_dates")) == dates_before
    assert _query("SELECT COUNT(*) FROM budget_members") == [(0,)]
    assert sa_budget.status(source)["account_id"] == source


# ---------- 同组冲突 ----------

def test_merging_members_of_same_group_is_a_conflict():
    source, target = _init("src", 100), _init("dst", 20)
    _apply(source, target)

    with pytest.raises(sa_budget.BudgetConflict):
        _preview(source, target)
    with pytest.raises(sa_budget.BudgetConflict):
        _preview(target, source)


# ---------- 失败事务回滚 ----------

def test_failed_merge_rolls_back_all_changes():
    source, target = _init("src", 100), _init("dst", 50)
    hold = sa_budget.reserve(source, 2 * NANO)
    preview = _preview(source, target)
    accounts_before = dict(_query("SELECT id, baseline FROM accounts"))
    dates_before = dict(_query("SELECT id, start_date FROM trial_dates"))

    # 用触发器让归并最后一步 DELETE accounts 失败。
    with sa_budget._connect() as db:
        db.execute("CREATE TRIGGER block_account_delete BEFORE DELETE ON accounts "
                   "BEGIN SELECT RAISE(ABORT, 'blocked by test'); END")

    with pytest.raises(sqlite3.DatabaseError):
        sa_budget.merge_accounts(source, target, confirmation=preview["confirmation"])

    assert dict(_query("SELECT id, baseline FROM accounts")) == accounts_before
    assert dict(_query("SELECT id, start_date FROM trial_dates")) == dates_before
    assert _query("SELECT account FROM holds WHERE id=?", (hold,)) == [(source,)]
    assert _query("SELECT COUNT(*) FROM budget_members") == [(0,)]

    sa_budget.settle(hold, 2 * NANO)                  # 回滚后源账本仍可正常结算
    assert sa_budget.status(source)["spent"] == 2


# ---------- 合并后并发预留/结算共享一本账 ----------

def test_merged_members_reserve_and_settle_one_shared_ledger_concurrently():
    source, target = _init("src", 100), _init("dst", 100)
    assert _apply(source, target) == {"account_id": target}
    assert sa_budget.status(target)["baseline"] == 200

    charge = 5 * NANO
    attempts = 20                                     # 可用额度 (295-200)/5 = 19

    def _reserve(index):
        key = source if index % 2 else target         # 两个旧成员 ID 交替使用
        try:
            return sa_budget.reserve(key, charge)
        except sa_budget.BudgetError:
            return None

    with ThreadPoolExecutor(max_workers=attempts) as pool:
        tokens = list(pool.map(_reserve, range(attempts)))

    accepted = [token for token in tokens if token]
    assert len(accepted) == 19
    assert _total(target) == sa_budget.LIMIT
    assert sa_budget.status(target)["blocked"] is True

    with ThreadPoolExecutor(max_workers=len(accepted)) as pool:
        list(pool.map(lambda token: sa_budget.settle(token, charge), accepted))

    data = sa_budget.status(target)
    assert data["spent"] == 95
    assert data["reserved"] == 0
    assert _total(target) == sa_budget.LIMIT
    assert _query("SELECT COUNT(*) FROM holds") == [(0,)]


# ---------- 跨成员保护 ----------

def test_merged_members_cannot_open_a_second_ledger():
    source, target = _init("src", 100), _init("dst", 50)
    _apply(source, target)

    assert sa_budget.status(source)["account_id"] == target
    with pytest.raises(ValueError):
        sa_budget.initialize(source, 10)              # 组账本已存在，不能再初始化
    with pytest.raises(sa_budget.BudgetConflict):
        sa_budget.initialize(source, 10, expected_owner=source)
    with pytest.raises(sa_budget.BudgetConflict):
        sa_budget.set_trial_date(source, "2026-01-01", expected_owner=source)

    assert _query("SELECT COUNT(*) FROM accounts") == [(1,)]


# ---------- 日期：保守取较早已存在日期，未初始化默认日期不覆盖 ----------

def test_uninitialized_source_adopts_initialized_target_date(monkeypatch):
    monkeypatch.setattr(sa_budget, "_today", lambda: datetime.date(2025, 6, 1))
    source = "imported"
    sa_budget.ensure_trial_date(source)               # 未初始化来源的导入默认日期 2025-06-01
    monkeypatch.setattr(sa_budget, "_today", lambda: datetime.date(2026, 10, 14))
    target = _init("recorded", 100)
    sa_budget.set_trial_date(target, "2026-10-01")

    _apply(source, target)

    data = sa_budget.status(target)
    assert data["start_date"] == "2026-10-01"          # 不采用未初始化来源的默认日期
    assert data["expires_on"] == "2026-12-30"
    assert not data["expired"]


def test_uninitialized_target_default_does_not_override_recorded_source_date(monkeypatch):
    monkeypatch.setattr(sa_budget, "_today", lambda: datetime.date(2026, 10, 14))
    source = _init("recorded", 100)
    sa_budget.set_trial_date(source, "2026-09-01")
    target = "imported"
    sa_budget.ensure_trial_date(target, "2026-10-14")  # 当天默认日期，晚于已记账日期

    _apply(source, target)

    assert sa_budget.status(target)["start_date"] == "2026-09-01"


def test_merge_keeps_earliest_recorded_start_date(monkeypatch):
    monkeypatch.setattr(sa_budget, "_today", lambda: datetime.date(2026, 10, 14))
    source = _init("src", 100)
    sa_budget.set_trial_date(source, "2026-09-01")
    target = _init("dst", 50)
    sa_budget.set_trial_date(target, "2026-08-01")

    _apply(source, target)

    data = sa_budget.status(target)
    assert data["start_date"] == "2026-08-01"
    assert data["expires_on"] == "2026-10-30"
    assert not data["expired"]


# ---------- 链式归并：成员全部保留，组基线去重 ----------

def test_chained_merge_keeps_all_members_and_deduplicates_baseline():
    first = _init("a", 100)
    token_a = sa_budget.reserve(first, 10 * NANO)
    sa_budget.settle(token_a, 10 * NANO)              # a spent 10
    second = _init("b", 100)
    token_b = sa_budget.reserve(second, 3 * NANO)
    third = _init("c", 50)
    token_c = sa_budget.reserve(third, 2 * NANO)

    _apply(first, second)
    _apply(second, third)

    for member in (first, second, third):
        assert sa_budget.status(member)["account_id"] == third
    data = sa_budget.status(third)
    assert data["baseline"] == 250                    # 组基线只计一次，不是 350
    assert data["spent"] == 10
    assert data["reserved"] == 5
    assert set(_query("SELECT id FROM budget_members WHERE owner=?", (third,))) == {
        (first,), (second,)}
    assert _query("SELECT COUNT(*) FROM accounts") == [(1,)]

    sa_budget.settle(token_b, 3 * NANO)
    sa_budget.settle(token_c, 2 * NANO)
    settled = sa_budget.status(third)
    assert settled["spent"] == 15
    assert settled["reserved"] == 0


# ---------- API 接线：凭证 ID DTO 预览并确认 ----------

def test_merge_api_previews_and_applies_with_credential_ids(monkeypatch):
    import main

    source_sa = json.dumps({"client_email": "merge-source@example.com"})
    target_sa = json.dumps({"client_email": "merge-target@example.com"})
    source = sa_budget.credential_id(source_sa)
    target = sa_budget.credential_id(target_sa)
    _init(source, 100)
    _init(target, 50)
    monkeypatch.setattr(main.app_state, "get_sa_accounts",
                        lambda: [{"sa_json": source_sa}, {"sa_json": target_sa}])

    preview = main.merge_sa_budgets(
        main.SABudgetMerge(source_credential_id=source, target_credential_id=target), True)
    assert "confirmation" in preview

    applied = main.merge_sa_budgets(
        main.SABudgetMerge(source_credential_id=source, target_credential_id=target,
                           confirmation=preview["confirmation"]), True)
    assert applied == {"account_id": target}
    assert sa_budget.status(target)["baseline"] == 150
