"""全渠道最小探活（channel_probe）回归测试。

覆盖：
- 探活模型挑选（生图排除、显式指定、空列表兜底）；
- 凭证枚举与控制台/环境变量优先级；
- 本地结构校验失败 → 不发上游请求（stage=local）；
- 三类通道各自的探活结果判定（含 Cookie 的 HTTP 200 内错误信封）；
- 失败归因文案与并发编排（probe_all 的汇总结构）。

真实上游请求不在单测内（打网络不稳定），由真机验证补：见 PROJECT_SUMMARY/11。
"""
import asyncio
import json

import pytest

import channel_probe as cp
from channel_probe import (
    DEFAULT_CHANNELS, PROBE_HARD_TIMEOUT_SECONDS, _label_cookie, _label_key, _label_sa,
    pick_probe_model, probe_all,
)
import config as app_config
from runtime_state import app_state


@pytest.fixture(autouse=True)
def _clean_credentials():
    """每个用例前后清空控制台凭证与环境变量兜底（避免相互污染）。"""
    saved = {
        "express": app_state.get_express_keys(),
        "cookie_accounts": app_state.get_cookie_accounts(),
        "sa_accounts": app_state.get_sa_accounts_console(),
        "env_key": app_config.VERTEX_EXPRESS_API_KEY_VAL,
        "env_cookie": app_config.GOOGLE_COOKIE,
        "env_project": app_config.GOOGLE_PROJECT_ID,
        "env_sa_json": app_config.VERTEX_SA_JSON,
        "env_sa_file": app_config.VERTEX_SA_FILE,
    }
    app_state.set_express_keys([])
    app_state.set_cookie_accounts([])
    app_state.set_sa_accounts([])
    app_config.GOOGLE_COOKIE = None
    app_config.GOOGLE_PROJECT_ID = None
    app_config.VERTEX_SA_JSON = None
    app_config.VERTEX_SA_FILE = None
    app_config.VERTEX_EXPRESS_API_KEY_VAL = []
    yield
    app_state.set_express_keys(saved["express"] or [])
    app_state.set_cookie_accounts(saved["cookie_accounts"])
    app_state.set_sa_accounts(saved["sa_accounts"])
    app_config.VERTEX_EXPRESS_API_KEY_VAL = saved["env_key"]
    app_config.GOOGLE_COOKIE = saved["env_cookie"]
    app_config.GOOGLE_PROJECT_ID = saved["env_project"]
    app_config.VERTEX_SA_JSON = saved["env_sa_json"]
    app_config.VERTEX_SA_FILE = saved["env_sa_file"]


# ---------- 探活模型挑选 ----------

def test_probe_model_prefers_fast_flash_lite():
    model, note = pick_probe_model(["gemini-2.5-pro", "gemini-3.1-flash-lite", "gemini-3.6-flash"])
    assert model == "gemini-3.1-flash-lite"
    assert note == ""


def test_probe_model_excludes_image_models():
    """生图模型一律不用于探活（会真的生成图片），列表里只剩生图时退内置默认。"""
    model, note = pick_probe_model(["gemini-3-pro-image"])
    assert model == "gemini-2.5-flash"
    assert "为空" in note


def test_probe_model_explicit_request_wins():
    model, note = pick_probe_model(["gemini-3.1-flash-lite", "gemini-2.5-pro"], "gemini-2.5-pro")
    assert model == "gemini-2.5-pro"
    assert note == ""


def test_probe_model_explicit_image_request_falls_back_with_note():
    model, note = pick_probe_model(["gemini-3.1-flash-lite", "gemini-3-pro-image"], "gemini-3-pro-image")
    assert model == "gemini-3.1-flash-lite"
    assert "gemini-3-pro-image" in note and "生图" in note


def test_probe_model_falls_back_to_first_text_model():
    model, note = pick_probe_model(["gemini-9-unknown-flash"])
    assert model == "gemini-9-unknown-flash"
    assert note == ""


# ---------- 凭证枚举：控制台优先、环境变量兜底 ----------

def test_express_keys_console_wins_then_env():
    app_config.VERTEX_EXPRESS_API_KEY_VAL = ["env-key-1", "env-key-2"]
    assert cp._express_keys() == ["env-key-1", "env-key-2"]
    app_state.set_express_keys(["console-key"])
    assert cp._express_keys() == ["console-key"]


def test_cookie_accounts_console_wins_then_env():
    app_config.GOOGLE_COOKIE = "SAPISID=env; SID=env"
    app_config.GOOGLE_PROJECT_ID = "env-proj"
    accounts = cp._cookie_accounts()
    assert len(accounts) == 1 and accounts[0]["project_id"] == "env-proj"
    app_state.set_cookie_accounts([{"cookie": "SAPISID=c; SID=c", "project_id": "c-proj"}])
    accounts = cp._cookie_accounts()
    assert len(accounts) == 1 and accounts[0]["project_id"] == "c-proj"


def test_no_credentials_anywhere_yields_empty():
    assert cp._express_keys() == []
    assert cp._cookie_accounts() == []
    assert cp._sa_accounts() == []


# ---------- 标签（只露掩码，绝不回显完整凭证）----------

def test_labels_never_expose_full_credentials():
    long_key = "FAKE-KEY-FOR-TEST-000000000000000000000000000000000000"
    key_label = _label_key(long_key, 0)
    assert long_key not in key_label
    assert key_label.startswith("Key #1")

    cookie_label = _label_cookie({"cookie": "SAPISID=secret-value; SID=another", "project_id": "p1"}, 1)
    assert "secret-value" not in cookie_label
    assert "project=p1" in cookie_label

    sa_label = _label_sa({"sa_json": json.dumps({"client_email": "svc@proj.iam.gserviceaccount.com"}),
                          "location": "global"}, 2)
    assert "private_key" not in sa_label
    assert "svc@proj.iam.gserviceaccount.com" in sa_label


def test_label_cookie_marks_missing_project():
    assert "未填 Project ID" in _label_cookie({"cookie": "x", "project_id": ""}, 0)


# ---------- 本地结构校验：不合格就不发上游请求 ----------

class _ProbeSpy:
    """记录上游探活是否被调用的替身。"""

    def __init__(self, result=None):
        self.calls = 0
        self.result = result or {"ok": True, "stage": "upstream", "message": "✅", "latency_ms": 1}

    async def __call__(self):
        self.calls += 1
        return self.result


def _entry(channel="express", index=0, label="x", local=None, spy=None):
    return {
        "identity": {"channel": channel, "index": index, "label": label},
        "local_check": (lambda: local),
        "probe": spy or _ProbeSpy(),
    }


def test_local_check_failure_skips_upstream_request():
    spy = _ProbeSpy()
    entry = _entry(local="不是合法 JSON：Expecting value", spy=spy)
    result = asyncio.run(cp._probe_one(entry, asyncio.Semaphore(1)))
    assert result["ok"] is False
    assert result["stage"] == "local"
    assert result["latency_ms"] == 0
    assert spy.calls == 0, "本地校验没过就不该发上游请求（省配额、报错更可读）"


def test_local_check_pass_runs_upstream_probe():
    spy = _ProbeSpy()
    result = asyncio.run(cp._probe_one(_entry(local=None, spy=spy), asyncio.Semaphore(1)))
    assert spy.calls == 1
    assert result["ok"] is True and result["channel"] == "express"


def test_probe_hard_timeout_is_reported_not_raised():
    """单条探活必须硬兜底：上游静默挂起时接口仍要在有限时间内返回。"""
    class _Hang:
        async def __call__(self):
            await asyncio.sleep(PROBE_HARD_TIMEOUT_SECONDS + 5)

    import channel_probe
    monkeypatch_timeout = 0.05
    original = channel_probe.PROBE_HARD_TIMEOUT_SECONDS
    channel_probe.PROBE_HARD_TIMEOUT_SECONDS = monkeypatch_timeout
    try:
        result = asyncio.run(cp._probe_one(_entry(spy=_Hang()), asyncio.Semaphore(1)))
    finally:
        channel_probe.PROBE_HARD_TIMEOUT_SECONDS = original
    assert result["ok"] is False and result["stage"] == "upstream"
    assert "未返回" in result["message"]


# ---------- 分通道的本地校验接线 ----------

def test_cookie_local_check_uses_validate_cookie():
    assert cp._local_check_cookie({"cookie": "SAPISID=a; SID=b"}) is None
    message = cp._local_check_cookie({"cookie": "foo=bar"})
    assert message and "无效" in message


def test_sa_local_check_uses_validate_sa_credentials():
    valid = json.dumps({
        "type": "service_account", "client_email": "a@b.iam.gserviceaccount.com",
        "private_key": _real_pem(), "project_id": "p",
        "token_uri": "https://oauth2.googleapis.com/token",
    })
    assert cp._local_check_sa({"sa_json": valid}) is None
    message = cp._local_check_sa({"sa_json": '{"type":"user"}'})
    assert message and "缺少必需字段" in message


# ---------- 上游失败归类（复用 outcome 分类 + 中文归因）----------

def test_upstream_failure_is_classified_with_hint():
    import outcome as outcome_mod

    class _Err(Exception):
        code = 403

        def __str__(self):
            return "PERMISSION_DENIED: The caller does not have permission"

    result = cp._upstream_fail(_Err(), 120)
    assert result["ok"] is False and result["stage"] == "upstream" and result["latency_ms"] == 120
    assert "403" in result["message"] and "💡" in result["message"]
    assert outcome_mod.CREDENTIAL_PERMANENT  # 分类常量真身来自 outcome，不在本模块另立一套


def test_rate_limited_failure_hint_keeps_credential_valid():
    class _Err(Exception):
        code = 429

        def __str__(self):
            return "429 RESOURCE_EXHAUSTED"

    result = cp._upstream_fail(_Err(), 90)
    assert "限流" in result["message"] and "凭证本身有效" in result["message"]


# ---------- Cookie 通道：HTTP 200 内的错误信封同样判失败 ----------

def test_cookie_probe_rejects_missing_project_before_request():
    result = asyncio.run(cp._probe_cookie({"cookie": "SAPISID=a; SID=b", "project_id": ""}, "gemini-3.1-flash-lite"))
    assert result["ok"] is False and result["stage"] == "local"
    assert "Project ID" in result["message"]


def test_cookie_probe_rejects_cookie_without_sapisid():
    result = asyncio.run(cp._probe_cookie({"cookie": "foo=bar", "project_id": "p"}, "gemini-3.1-flash-lite"))
    assert result["ok"] is False and result["stage"] == "local"
    assert "SAPISID" in result["message"]


def test_cookie_http_fail_keeps_official_hint_text():
    project_fail = cp._cookie_http_fail(403, "This API method requires billing to be enabled", 30)
    assert "项目层面" in project_fail["message"]
    cookie_fail = cp._cookie_http_fail(403, "PERMISSION_DENIED: caller does not have permission", 30)
    assert cookie_fail["ok"] is False and "Cookie" in cookie_fail["message"]


# ---------- 端到端编排：probe_all 的汇总结构 ----------

def test_probe_all_reports_per_channel_and_totals(monkeypatch):
    app_state.set_express_keys(["k1", "k2"])
    app_state.set_cookie_accounts([{"cookie": "foo=bar", "project_id": "p"}])   # 本地校验会挂
    _install_fake_probe(monkeypatch)

    report = asyncio.run(probe_all())
    assert [c["channel"] for c in report["channels"]] == list(DEFAULT_CHANNELS)
    assert report["summary"]["total"] == 3
    assert report["summary"]["ok"] == 2 and report["summary"]["failed"] == 1
    assert report["channels"][0]["ok"] is True and "全部 2 条" in report["channels"][0]["message"]
    assert report["channels"][1]["ok"] is False and "1/1 条凭证不可用" in report["channels"][1]["message"]
    assert report["channels"][2]["configured"] is False


def test_probe_all_only_probes_requested_channels(monkeypatch):
    app_state.set_express_keys(["k1"])
    app_state.set_sa_accounts([{"sa_json": '{"type":"service_account","client_email":"a@b.c",'
                                          '"private_key":"x","project_id":"p"}', "location": "global"}])
    _install_fake_probe(monkeypatch)
    report = asyncio.run(probe_all(["vertex"]))
    assert [c["channel"] for c in report["channels"]] == ["vertex"]
    assert report["summary"]["total"] == 1


def test_probe_all_ignores_unknown_channel_keys(monkeypatch):
    _install_fake_probe(monkeypatch)
    report = asyncio.run(probe_all(["express", "不存在"]))
    assert [c["channel"] for c in report["channels"]] == ["express"]


def _install_fake_probe(monkeypatch):
    """把上游探活换成不联网的替身，只验证编排与汇总逻辑。"""
    async def _fake_express(key, model, settings):
        return cp._ok("✅ 上游返回 200", 12)

    async def _fake_sa(account, model):
        return cp._ok("✅ 上游返回 200", 15)

    async def _fake_cookie(account, model):
        return cp._ok("✅ 上游返回 200", 20)

    monkeypatch.setattr(cp, "_probe_express", _fake_express)
    monkeypatch.setattr(cp, "_probe_sa", _fake_sa)
    monkeypatch.setattr(cp, "_probe_cookie", _fake_cookie)


# ---------- HTTP 层 ----------

def test_channel_probe_endpoint_requires_login():
    from fastapi.testclient import TestClient
    import main
    with TestClient(main.app) as client:
        assert client.post("/api/channel-probe", json={}).status_code == 401


# ---------- 探活 HTTP 选项：刻意不打 PayGo 层级头 ----------

def test_probe_http_options_omit_paygo_headers():
    """flex 档允许上游排队 30 分钟，会把探活拖成假死；层级头属于正式请求路径的职责。"""
    options = cp._probe_http_options()
    assert not (options.headers or {}).get("X-Vertex-AI-LLM-Request-Type")
    assert options.api_version == "v1beta1"


def test_probe_config_disables_afc_and_caps_tokens():
    config = cp._probe_config()
    assert config.max_output_tokens == cp.PROBE_MAX_OUTPUT_TOKENS
    assert config.automatic_function_calling.disable is True


_FAKE_PEM = (
    "-----BEGIN PRIVATE KEY-----\n"
    "MIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQC7VJTUt9Us8cKj\n"
    "-----END PRIVATE KEY-----\n"
)


def _real_pem() -> str:
    """现生成一把真 RSA 私钥：SA 校验会真的去反序列化私钥，
    手写的假 PEM 过不了（"ASN.1 parsing error"），无法用于「本地校验通过」的用例。"""
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    return key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    ).decode()
