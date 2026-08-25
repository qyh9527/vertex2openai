"""Cookie 通道错误分类测试。

README「Studio(Cookie) 通道」一节的核心行为：
项目级错误（Project ID / 计费 / 权限）与 Cookie 失效必须区分开，
否则会让人反复重取 Cookie 却永远好不了。
"""
import pytest

from upstreams.cookie_proxy import _is_retryable_error, _is_cookie_expired_error, _is_project_error


class TestRetryableError:
    @pytest.mark.parametrize("msg", [
        "RESOURCE EXHAUSTED: quota exceeded",
        "429 too many requests, try again later",
        "upstream overloaded, temporarily unavailable",
        "rate limit exceeded, please slow down",
        "internal error occurred",
    ])
    def test_retryable_keywords(self, msg):
        assert _is_retryable_error(msg)

    @pytest.mark.parametrize("msg", [
        "invalid argument: bad request",
        "model not found",
        "",
    ])
    def test_not_retryable(self, msg):
        assert not _is_retryable_error(msg)


class TestCookieExpired:
    @pytest.mark.parametrize("msg", [
        "Permission denied on resource",
        "user not authorized, login required",
        "session expired, please sign in",
        "unauthenticated request",
        "invalid credentials",
    ])
    def test_expired_keywords(self, msg):
        assert _is_cookie_expired_error(msg)

    def test_normal_message_not_expired(self):
        assert not _is_cookie_expired_error("everything is fine")


class TestProjectErrorPriority:
    """项目级错误必须被判为项目问题（README 里专门纠正过的误导行为）。"""

    def test_billing_error_is_project(self):
        assert _is_project_error(
            "PERMISSION DENIED: requires billing to be enabled on project #123")

    def test_project_resource_denied_is_project(self):
        assert _is_project_error(
            "Permission 'aiplatform.endpoints.predict' denied on resource "
            "//aiplatform.googleapis.com/projects/myproj/locations/global/...")

    def test_pure_session_error_not_project(self):
        assert not _is_project_error("Your session has expired. Please sign in again.")

    def test_unknown_error_not_project(self):
        assert not _is_project_error("something went wrong, internal error")

    def test_project_error_also_hits_cookie_keyword(self):
        """同时命中两类关键词时，两路判定都为真——指引以项目为准（调用方分流）。"""
        msg = "Permission 'aiplatform.endpoints.predict' denied on resource " \
              "//aiplatform.googleapis.com/projects/myproj/locations/global"
        assert _is_cookie_expired_error(msg)
        assert _is_project_error(msg)
