"""sa_budget（SA 独立额度账本）单元测试。

只使用 tmp_path 下的 sqlite 与假 SDK Client：不发起真实请求、不读取真实凭证。
覆盖：金额校验、初始化唯一性、稳定身份、295 边界、并发预留、settle 幂等、
重载后 holds 保留、未知模型拒绝，以及 BudgetModels 非流/流/中断的预留与结算语义。
"""
import importlib
import json
import os
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest

import sa_budget

MODEL = "gemini-3.6-flash"
NANO = 1_000_000_000          # 1 美元 = 10^9 纳美元
SA_JSON = json.dumps({"type": "service_account", "client_id": "sa-abc-123",
                      "client_email": "svc@example.com", "private_key": "key-v1",
                      "project_id": "demo"})
USAGE = SimpleNamespace(prompt_token_count=100, candidates_token_count=20,
                        thoughts_token_count=30, cached_content_token_count=0)


@pytest.fixture(autouse=True)
def _isolated_state(tmp_path, monkeypatch):
    """每个测试独立 STATE_DIR，账本只落在本次 tmp_path。"""
    monkeypatch.setenv("STATE_DIR", str(tmp_path))
    return tmp_path


def _db_path() -> Path:
    return Path(os.environ["STATE_DIR"]) / "sa_budget.sqlite3"


def _query(sql, params=()):
    db = sqlite3.connect(_db_path())
    try:
        return db.execute(sql, params).fetchall()
    finally:
        db.close()


def _init(key="acct", amount=100):
    sa_budget.initialize(key, amount)
    return key


# ---------- 金额校验 ----------

@pytest.mark.parametrize("value", [-1, "-0.01", "NaN", "Infinity", "-Infinity",
                                   "1000000000.000000001", 1e10, Decimal("-0.5")])
def test_money_rejects_negative_non_finite_or_oversized(value):
    with pytest.raises(ValueError):
        sa_budget.money(value)


@pytest.mark.parametrize("value,expected", [
    (0, 0),
    ("1.5", 1_500_000_000),
    (Decimal("295"), 295 * NANO),
    ("0.0000000001", 1),          # 0.1 纳美元向上取整到 1
])
def test_money_accepts_non_negative_and_rounds_up(value, expected):
    assert sa_budget.money(value) == expected


# ---------- 稳定身份 ----------

def test_credential_identity_is_stable_across_key_rotation():
    rotated = json.dumps({"client_id": "changed-or-omitted", "client_email": "svc@example.com",
                          "private_key": "key-v2"})
    assert sa_budget.credential_id(SA_JSON) == sa_budget.credential_id(rotated)
    # client_id 增删不改变认证身份或预算。
    a = json.dumps({"client_email": "Svc@Example.com"})
    b = json.dumps({"client_email": " svc@example.com "})
    assert sa_budget.credential_id(a) == sa_budget.credential_id(b)
    assert sa_budget.credential_id(a) == sa_budget.credential_id(SA_JSON)


def test_credential_identity_requires_stable_identity():
    with pytest.raises(sa_budget.BudgetError):
        sa_budget.credential_id(json.dumps({"private_key": "k"}))


# ---------- 初始化仅一次 ----------

def test_initialize_runs_only_once():
    key = _init("acct", 120)
    with pytest.raises(ValueError):
        sa_budget.initialize(key, 500)
    assert sa_budget.status(key)["baseline"] == 120.0
    assert _query("SELECT baseline FROM accounts WHERE id=?", (key,)) == [(120 * NANO,)]


# ---------- 295 边界 ----------

def test_reserve_295_boundary_allows_exact_limit_and_rejects_overage():
    key = _init("acct", 294)
    assert sa_budget.status(key)["remaining"] == pytest.approx(1.0)
    token = sa_budget.reserve(key, 1 * NANO)      # 恰好到 295
    assert token

    budget = sa_budget.status(key)
    assert budget["reserved"] == pytest.approx(1.0)
    assert budget["remaining"] == 0.0
    assert budget["blocked"] is True

    with pytest.raises(sa_budget.BudgetError):
        sa_budget.reserve(key, 1)                 # 再多 1 纳美元即超限


def test_reserve_at_baseline_limit_blocks_any_charge():
    key = _init("acct", 295)
    assert sa_budget.status(key)["blocked"] is True
    with pytest.raises(sa_budget.BudgetError):
        sa_budget.reserve(key, 1)


# ---------- 并发预留不超限 ----------

def test_concurrent_reserves_never_exceed_limit():
    key = _init("acct", 294)
    attempts = 8

    def _try(_):
        try:
            sa_budget.reserve(key, 1 * NANO)
            return "reserved"
        except sa_budget.BudgetError:
            return "rejected"

    with ThreadPoolExecutor(max_workers=attempts) as pool:
        results = list(pool.map(_try, range(attempts)))

    assert results.count("reserved") == 1
    assert results.count("rejected") == attempts - 1
    total = _query(
        "SELECT baseline + spent + (SELECT COALESCE(SUM(amount), 0) FROM holds WHERE account = ?)"
        " FROM accounts WHERE id = ?", (key, key))[0][0]
    assert total == sa_budget.LIMIT
    assert sa_budget.status(key)["blocked"] is True


# ---------- settle 幂等 ----------

def test_settle_is_idempotent():
    key = _init("acct", 100)
    token = sa_budget.reserve(key, 5 * NANO)

    sa_budget.settle(token, 3 * NANO)
    first = sa_budget.status(key)
    assert first["spent"] == pytest.approx(3.0)
    assert first["reserved"] == 0.0

    sa_budget.settle(token, 3 * NANO)             # 重复结算不再累加
    again = sa_budget.status(key)
    assert again["spent"] == pytest.approx(3.0)
    assert again["reserved"] == 0.0

    sa_budget.settle("no-such-token", 7 * NANO)   # 未知 token 静默忽略
    assert sa_budget.status(key)["spent"] == pytest.approx(3.0)


# ---------- 重载保留 holds ----------

def test_holds_survive_module_reload():
    key = _init("acct", 294)
    token = sa_budget.reserve(key, 1 * NANO)
    assert _query("SELECT amount FROM holds WHERE id=?", (token,)) == [(1 * NANO,)]

    importlib.reload(sa_budget)                   # 模块无内存状态，账本以 sqlite 为准
    assert sa_budget.status(key)["reserved"] == pytest.approx(1.0)
    with pytest.raises(sa_budget.BudgetError):
        sa_budget.reserve(key, 1)                 # 重载后 hold 仍占用额度


# ---------- 未知模型拒绝 ----------

def test_price_rejects_unknown_or_image_model():
    with pytest.raises(sa_budget.BudgetError):
        sa_budget.price("gemini-does-not-exist", 1, 1)
    assert sa_budget.price("gemini-3.1-flash-image", 1_000_000, 1_000_000,
                           image_tokens=500_000) == sa_budget.money("32")
    assert sa_budget.price(MODEL, 1000, 1000) > 0


# ---------- BudgetModels：非流 / 流 / 中断 ----------

class FakeModels:
    """假 SDK models：count_tokens 与生成调用可编程，记录调用次数。"""

    def __init__(self, total_tokens=100, response=None, chunks=()):
        self.total_tokens = total_tokens
        self.response = response
        self.chunks = list(chunks)
        self.count_calls = []
        self.generate_calls = 0

    async def count_tokens(self, **kwargs):
        self.count_calls.append(kwargs)
        return SimpleNamespace(total_tokens=self.total_tokens)

    async def generate_content(self, **kwargs):
        self.generate_calls += 1
        return self.response

    async def generate_content_stream(self, **kwargs):
        self.generate_calls += 1

        async def stream():
            for chunk in self.chunks:
                yield chunk

        return stream()


def _budget_models(sdk):
    return sa_budget.BudgetModels(sdk, "acct", "global", "standard")


def _call_kwargs(max_output_tokens=10):
    return {"model": MODEL, "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
            "config": {"max_output_tokens": max_output_tokens}}


def _expected_settlement():
    # 结算按真实 usage：prompt=100，completion=candidates+thoughts=50
    return sa_budget.price(MODEL, 100, 50, 0, tier="standard", location="global") / NANO


def _expected_hold(max_output_tokens=10):
    # 预留按 countTokens 输入 + max_output_tokens（candidate_count 默认 1）
    return sa_budget.price(MODEL, 100, max_output_tokens, 0,
                           tier="standard", location="global") / NANO


async def test_budget_models_rejects_unknown_model_before_counting():
    sdk = FakeModels()
    bm = _budget_models(sdk)
    with pytest.raises(sa_budget.BudgetError):
        await bm.generate_content(model="gemini-unknown", contents=[], config={})
    assert sdk.count_calls == []
    assert sdk.generate_calls == 0


async def test_budget_models_nonstream_settles_each_attempt():
    key = _init("acct", 100)
    sdk = FakeModels(response=SimpleNamespace(usage_metadata=USAGE))
    bm = _budget_models(sdk)

    await bm.generate_content(**_call_kwargs())
    after_first = sa_budget.status(key)
    assert sdk.generate_calls == 1
    assert len(sdk.count_calls) == 1
    assert after_first["reserved"] == 0.0
    assert after_first["spent"] == pytest.approx(_expected_settlement())
    assert _query("SELECT COUNT(*) FROM holds") == [(0,)]

    await bm.generate_content(**_call_kwargs())    # 第二次尝试独立预留并结算
    after_second = sa_budget.status(key)
    assert sdk.generate_calls == 2
    assert after_second["reserved"] == 0.0
    assert after_second["spent"] == pytest.approx(2 * _expected_settlement())


async def test_budget_models_stream_settles_once_at_end():
    key = _init("acct", 100)
    chunks = [SimpleNamespace(usage_metadata=None),
              SimpleNamespace(usage_metadata=USAGE)]
    sdk = FakeModels(chunks=chunks)
    bm = _budget_models(sdk)

    stream = await bm.generate_content_stream(**_call_kwargs())
    seen = [chunk async for chunk in stream]

    assert len(seen) == 2
    after = sa_budget.status(key)
    assert after["reserved"] == 0.0
    assert after["spent"] == pytest.approx(_expected_settlement())
    assert _query("SELECT COUNT(*) FROM holds") == [(0,)]


async def test_budget_models_stream_interrupt_keeps_hold():
    key = _init("acct", 100)
    sdk = FakeModels(chunks=[SimpleNamespace(usage_metadata=USAGE)])
    bm = _budget_models(sdk)

    stream = await bm.generate_content_stream(**_call_kwargs())
    await stream.__anext__()
    await stream.aclose()                          # 消费端提前断开，未产生 usage 结算

    after = sa_budget.status(key)
    assert after["spent"] == 0.0
    assert after["reserved"] == pytest.approx(_expected_hold())
    assert _query("SELECT COUNT(*) FROM holds") == [(1,)]


@pytest.mark.parametrize("code,released", [(429, True), (503, False)])
async def test_stream_first_iteration_failure_reservation(code, released):
    _init("acct", 0)
    class Failure(RuntimeError):
        pass
    error = Failure("upstream")
    error.code = code
    class Models(FakeModels):
        async def generate_content_stream(self, **kwargs):
            async def stream():
                raise error
                yield
            return stream()
    stream = await _budget_models(Models()).generate_content_stream(**_call_kwargs())
    with pytest.raises(Failure):
        await stream.__anext__()
    assert (sa_budget.status("acct")["reserved"] == 0) is released


async def test_budget_client_preserves_async_close():
    from unittest.mock import AsyncMock
    close = AsyncMock()
    raw = SimpleNamespace(aio=SimpleNamespace(models=FakeModels(), aclose=close))
    client = sa_budget.BudgetClient(raw, "acct", "global", "standard")
    await client.aio.aclose()
    close.assert_awaited_once()


async def test_search_queries_settle_additional_charge():
    _init("acct", 0)
    response = SimpleNamespace(usage_metadata=USAGE, candidates=[SimpleNamespace(grounding_metadata=SimpleNamespace(web_search_queries=["q1", "q2"]))])
    bm = _budget_models(FakeModels(response=response))
    kwargs = _call_kwargs()
    kwargs["config"]["tools"] = [{"google_search": {}}]
    await bm.generate_content(**kwargs)
    assert sa_budget.status("acct")["spent"] == pytest.approx(_expected_settlement() + 0.028)


def test_image_output_is_not_double_charged():
    assert sa_budget.price("gemini-nano-banana-2.1", 0, 1120, image_tokens=1120) == sa_budget.money("0.0336")


def test_audio_input_uses_audio_price():
    assert sa_budget.price("gemini-2.5-flash", 1_000_000, 0, audio_tokens=1_000_000) == sa_budget.money("1")
    assert sa_budget.price("gemini-3.1-flash-lite", 1_000_000, 0, audio_tokens=1_000_000) == sa_budget.money("0.5")


async def test_cancelled_reservation_releases_unsent_hold(monkeypatch):
    import asyncio
    import threading
    _init("acct", 0)
    entered = threading.Event()
    release = threading.Event()
    original = sa_budget.reserve
    def delayed(key, amount):
        entered.set()
        release.wait(5)
        return original(key, amount)
    monkeypatch.setattr(sa_budget, "reserve", delayed)
    sdk = FakeModels(response=SimpleNamespace(usage_metadata=USAGE))
    task = asyncio.create_task(_budget_models(sdk).generate_content(**_call_kwargs()))
    await asyncio.to_thread(entered.wait, 5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    release.set()
    await asyncio.gather(*list(sa_budget._pending_cleanup))
    assert sa_budget.status("acct")["reserved"] == 0
    assert sdk.generate_calls == 0


def test_initialization_rejects_stale_account_identity(monkeypatch):
    import main
    monkeypatch.setattr(main.app_state, "get_sa_accounts", lambda: [{"sa_json": SA_JSON}])
    result = main.initialize_sa_budget(main.SABudgetInitialization(index=0, credential_id="stale", historical_usd="0"), True)
    assert result.status_code == 409
    key = sa_budget.credential_id(SA_JSON)
    assert not sa_budget.status(key)["initialized"]
    result = main.initialize_sa_budget(main.SABudgetInitialization(index=0, credential_id=key, historical_usd="294"), True)
    assert result.status_code == 200
    assert sa_budget.status(key)["baseline"] == 294


async def test_flex_reserves_standard_fallback_price():
    _init("acct", 0)
    sdk = FakeModels(response=SimpleNamespace(usage_metadata=None))
    bm = sa_budget.BudgetModels(sdk, "acct", "global", "flex")
    await bm.generate_content(**_call_kwargs())
    expected = sa_budget.price(MODEL, 100, 10, tier="standard") / NANO
    assert sa_budget.status("acct")["reserved"] == pytest.approx(expected)


@pytest.mark.parametrize("model", ["gemini-2.5-flash", "projects/p/locations/global/publishers/google/models/gemini-2.5-flash"])
async def test_media_without_modality_details_keeps_hold(model):
    _init("acct", 0)
    sdk = FakeModels(response=SimpleNamespace(usage_metadata=USAGE))
    kwargs = _call_kwargs()
    kwargs["model"] = model
    kwargs["contents"] = [{"parts": [{"inline_data": {"mime_type": "audio/wav", "data": "AA=="}}]}]
    await _budget_models(sdk).generate_content(**kwargs)
    assert sa_budget.status("acct")["spent"] == 0
    assert sa_budget.status("acct")["reserved"] > 0


def test_trial_date_persists_without_resetting_spend(monkeypatch):
    import datetime
    monkeypatch.setattr(sa_budget, "_today", lambda: datetime.date(2026, 1, 1))
    key = _init("dated", 100)
    token = sa_budget.reserve(key, NANO)
    sa_budget.settle(token, NANO)
    sa_budget.set_trial_date(key, "2025-12-15")
    sa_budget.ensure_trial_date(key, "2026-01-01")
    data = sa_budget.status(key)
    assert data["start_date"] == "2025-12-15"
    assert data["spent"] == 1
    assert data["baseline"] == 100


def test_trial_expiry_exact_boundary_and_settlement(monkeypatch):
    import datetime
    monkeypatch.setattr(sa_budget, "_today", lambda: datetime.date(2026, 1, 1))
    key = _init("dated", 0)
    token = sa_budget.reserve(key, NANO)
    monkeypatch.setattr(sa_budget, "_today", lambda: datetime.date(2026, 3, 31))
    assert not sa_budget.status(key)["expired"]
    monkeypatch.setattr(sa_budget, "_today", lambda: datetime.date(2026, 4, 1))
    assert sa_budget.status(key)["expired"]
    assert sa_budget.status(key)["blocked"]
    with pytest.raises(sa_budget.BudgetError, match="日期"):
        sa_budget.reserve(key, 1)
    sa_budget.settle(token, NANO)
    assert sa_budget.status(key)["spent"] == 1


@pytest.mark.parametrize("day", ["2026-02-30", "2026-1-01", "2027-01-01", ""])
def test_invalid_trial_date_rejected(day, monkeypatch):
    import datetime
    monkeypatch.setattr(sa_budget, "_today", lambda: datetime.date(2026, 1, 1))
    with pytest.raises(ValueError):
        sa_budget.set_trial_date("dated", day)


def test_legacy_account_without_date_fails_closed():
    key = _init("legacy", 0)
    with sa_budget._connect() as db:
        db.execute("DELETE FROM trial_dates WHERE id=?", (key,))
    assert sa_budget.status(key)["expired"]
    with pytest.raises(sa_budget.BudgetError):
        sa_budget.reserve(key, 1)


def test_trial_date_api_rejects_stale_identity(monkeypatch):
    import main
    key = _init(sa_budget.credential_id(SA_JSON), 123)
    monkeypatch.setattr(main.app_state, "get_sa_accounts", lambda: [{"sa_json": SA_JSON}])
    result = main.save_sa_trial_date(main.SATrialDate(index=0, credential_id="wrong", start_date="2026-01-01"), True)
    assert result.status_code == 409
    result = main.save_sa_trial_date(main.SATrialDate(index=0, credential_id=key, start_date="2026-01-01"), True)
    assert result["budget"]["expired"]
    assert result["budget"]["baseline"] == 123


def test_trial_date_default_does_not_reset_on_reimport(monkeypatch):
    import datetime
    monkeypatch.setattr(sa_budget, "_today", lambda: datetime.date(2026, 1, 1))
    sa_budget.ensure_trial_date("imported")
    monkeypatch.setattr(sa_budget, "_today", lambda: datetime.date(2026, 2, 1))
    sa_budget.ensure_trial_date("imported")
    assert sa_budget.status("imported")["start_date"] == "2026-01-01"


def test_trial_status_reads_clock_once(monkeypatch):
    import datetime
    monkeypatch.setattr(sa_budget, "_today", lambda: datetime.date(2026, 1, 1))
    _init("midnight", 0)
    calls = []
    def clock():
        calls.append(1)
        return datetime.date(2026, 3, 31) if len(calls) == 1 else datetime.date(2026, 4, 1)
    monkeypatch.setattr(sa_budget, "_today", clock)
    data = sa_budget.status("midnight")
    assert len(calls) == 1
    assert data["days_remaining"] == 1
    assert not data["expired"]


@pytest.mark.parametrize("start,end", [("2026-01-31", "2026-04-30"), ("2026-05-01", "2026-07-30"), ("2025-11-30", "2026-02-28")])
def test_trial_uses_earlier_of_90_days_and_three_months(start,end,monkeypatch):
    import datetime
    monkeypatch.setattr(sa_budget, "_today", lambda: datetime.date(2026, 6, 1))
    sa_budget.ensure_trial_date("month-end", start)
    assert sa_budget.status("month-end")["expires_on"] == end
