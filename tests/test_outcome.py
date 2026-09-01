"""统一失败语义（outcome）的等价性与分类测试（进阶报告 P0-1）。

兼容铁律：outcome 的 switchable 判定必须与改造前的
SWITCHABLE_STATUS_CODES {429,500,502,503,504} / is_retryable_exception 逐条等价，
否则升级分类会悄悄改变故障转移行为。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))

import httpx
import pytest

import outcome
from api_helpers import is_retryable_exception


def _exc(text):
    return ValueError(text)


class TestStatusSwitchableEquivalence:
    """改造前白名单 {429,500,502,503,504} 逐码等价。"""

    @pytest.mark.parametrize("code", [429, 500, 502, 503, 504])
    def test_switchable_codes(self, code):
        assert outcome.status_switchable(code) is True

    @pytest.mark.parametrize("code", [400, 401, 403, 404, 405, 409, 422, 499, 501, 505])
    def test_non_switchable_codes(self, code):
        assert outcome.status_switchable(code) is False


class TestExceptionSwitchableEquivalence:
    """exception_switchable 与 is_retryable_exception 逐例等价（不扩大识别面）。"""

    CASES = [
        "429 too many requests",
        "Quota exceeded for project",
        "503 Service Unavailable",
        "permission denied: roles/aiplatform.user missing",
        "not found: model gemini-x",
        "invalid argument: bad schema",
        "Response blocked by Gemini safety filter: PROHIBITED_CONTENT",
    ]

    @pytest.mark.parametrize("text", CASES)
    def test_equivalent(self, text):
        assert outcome.exception_switchable(_exc(text)) == is_retryable_exception(_exc(text))

    def test_httpx_status_error_429(self):
        e = httpx.HTTPStatusError(
            "Client error '429'", request=httpx.Request("POST", "https://x"),
            response=httpx.Response(429, request=httpx.Request("POST", "https://x")))
        assert outcome.exception_switchable(e) is True
        assert is_retryable_exception(e) is True

    def test_httpx_status_error_403(self):
        e = httpx.HTTPStatusError(
            "Client error '403'", request=httpx.Request("POST", "https://x"),
            response=httpx.Response(403, request=httpx.Request("POST", "https://x")))
        assert outcome.exception_switchable(e) is False
        assert is_retryable_exception(e) is False

    def test_code_attribute_503(self):
        class _Err(Exception):
            code = 503
        assert outcome.exception_switchable(_Err("upstream busy")) is True


class TestClassifyFailure:
    def test_policy_blocked_keywords(self):
        for kw in ("prohibited_content", "safety filter", "recitation",
                   "blocklist", "spii", "image_safety", "安全策略拦截"):
            assert outcome.classify_failure(message=kw) == outcome.POLICY_BLOCKED

    def test_policy_blocked_wins_over_400(self):
        """同一个 400 可能是安全拦截：语义优先于状态码。"""
        assert outcome.classify_failure(status=400, message="PROHIBITED_CONTENT") \
            == outcome.POLICY_BLOCKED

    def test_rate_limited(self):
        assert outcome.classify_failure(status=429, message="") == outcome.RATE_LIMITED
        assert outcome.classify_failure(message="too many requests") == outcome.RATE_LIMITED
        assert outcome.classify_failure(message="resource_exhausted") == outcome.RATE_LIMITED

    def test_auth_refreshable(self):
        assert outcome.classify_failure(status=401, message="") == outcome.AUTH_REFRESHABLE

    def test_credential_permanent(self):
        assert outcome.classify_failure(status=403, message="") == outcome.CREDENTIAL_PERMANENT
        assert outcome.classify_failure(message="requires billing") == outcome.CREDENTIAL_PERMANENT
        assert outcome.classify_failure(message="permission denied") == outcome.CREDENTIAL_PERMANENT

    def test_transient_only_whitelist_5xx(self):
        """501/505 不属于瞬态（不扩大切换范围）。"""
        assert outcome.classify_failure(status=503) == outcome.TRANSIENT
        assert outcome.classify_failure(status=501) == outcome.OTHER

    def test_request_permanent(self):
        assert outcome.classify_failure(status=400, message="bad field") == outcome.REQUEST_PERMANENT

    def test_empty_or_protocol(self):
        assert outcome.classify_failure(message="上游返回空流") == outcome.EMPTY_OR_PROTOCOL

    def test_client_closed(self):
        assert outcome.classify_failure(status=499, message="") == outcome.CLIENT_CLOSED


class TestFinalHttpStatus:
    """P0-3：全失败状态码不再从聚合字符串反推，429 不再被洗成 500。"""

    def test_prefers_last_upstream_status(self):
        attempts = [
            {"channel": "express", "status": 429, "upstream": True},
            {"channel": "cookie", "status": 429, "upstream": True},
        ]
        assert outcome.final_http_status(attempts) == 429

    def test_last_upstream_wins_over_earlier(self):
        attempts = [
            {"channel": "express", "status": 429, "upstream": True},
            {"channel": "cookie", "status": 503, "upstream": True},
        ]
        assert outcome.final_http_status(attempts) == 503

    def test_breaker_skip_only_maps_category(self):
        attempts = [{"channel": "express", "status": 503, "category": outcome.RATE_LIMITED}]
        assert outcome.final_http_status(attempts) == 429

    def test_fallback_when_no_signal(self):
        assert outcome.final_http_status([], fallback_status=503) == 503
        assert outcome.final_http_status([{}], fallback_status=500) == 500

    def test_invalid_upstream_status_ignored(self):
        attempts = [{"channel": "express", "status": 200, "upstream": True},
                    {"channel": "cookie", "status": 0, "upstream": True, "category": outcome.TRANSIENT}]
        assert outcome.final_http_status(attempts) == 503


class TestParseRetryAfter:
    """P0-2：从 429/RESOURCE_EXHAUSTED 响应解析精确冷却时间。"""

    def test_retry_delay_seconds(self):
        assert outcome.parse_retry_after('... "retryDelay": "30s" ...') == 30.0

    def test_retry_after_seconds(self):
        assert outcome.parse_retry_after("Retry-After: 45") == 45.0

    def test_reset_epoch_seconds(self):
        import time
        ts = time.time() + 120
        assert outcome.parse_retry_after(f'"retry_time": {int(ts)}') == pytest.approx(120, abs=5)

    def test_none_when_unparseable(self):
        assert outcome.parse_retry_after("plain quota error") is None

    def test_capped_at_max(self):
        assert outcome.parse_retry_after("Retry-After: 999999") == outcome.MAX_COOLDOWN_SECONDS

    def test_past_reset_returns_none(self):
        assert outcome.parse_retry_after('"retry_time": 1000000000') is None
