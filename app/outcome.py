"""统一失败语义（Outcome/Disposition，进阶研究报告 P0-1）。

全项目的失败分类「唯一真相表」：重试、跨通道切换、熔断、全失败状态码选择、
凭证冷却都消费这里的分类结果，不再各自 grep 状态码/关键词。

分类表（报告 §7，适配 Vertex 场景）：

| 分类                | 典型来源                                   | 默认动作                        |
|---------------------|--------------------------------------------|---------------------------------|
| success             | 2xx + 有效输出                             | 返回；清相关健康计数            |
| rate_limited        | 429 / RESOURCE_EXHAUSTED / quota          | 退避重试或换候选；冷却凭证      |
| transient           | 502/503/504 / 连接超时 / RemoteProtocol    | 短重试或换候选                  |
| auth_refreshable    | 401 / token expired                        | 刷新凭证（SDK 自动）            |
| credential_permanent| 403 权限/计费 / Cookie 失效 / Key revoked  | 长冷却该凭证；换同类凭证        |
| request_permanent   | 400 schema/参数错误                       | 直接回客户端；不污染健康        |
| policy_blocked      | PROHIBITED_CONTENT / SAFETY / BLOCKLIST 等 | 直接回可读错误；不重试不 evict  |
| empty_or_protocol   | 空流 / envelope 解析异常                   | 未出流可换候选；已出流收尾      |
| client_closed       | 499 客户端断开                             | 停止一切重试                    |
| other               | 无法归类的异常                             | 保持既有兜底行为                |

兼容性铁律：status_switchable / exception_switchable 的判定结果与改造前的
SWITCHABLE_STATUS_CODES {429,500,502,503,504}、is_retryable_exception 逐条等价
（见 tests/test_outcome.py 的等价性回归），升级分类不允许悄悄改变切换/重试行为。
"""

import re
from typing import Any, Optional

import httpx

# ---------- 分类常量（字符串而非 Enum：日志/JSON 序列化零转换）----------

RATE_LIMITED = "rate_limited"
TRANSIENT = "transient"
AUTH_REFRESHABLE = "auth_refreshable"
CREDENTIAL_PERMANENT = "credential_permanent"
REQUEST_PERMANENT = "request_permanent"
POLICY_BLOCKED = "policy_blocked"
EMPTY_OR_PROTOCOL = "empty_or_protocol"
CLIENT_CLOSED = "client_closed"
OTHER = "other"

# 可触发跨通道切换 / 内部重试的分类（= 改造前白名单语义）
_SWITCHABLE = {RATE_LIMITED, TRANSIENT}

# 全部候选失败、无真实上游状态码时，按分类映射的稳定网关状态码
CATEGORY_HTTP_STATUS = {
    RATE_LIMITED: 429,
    TRANSIENT: 503,
    AUTH_REFRESHABLE: 401,
    CREDENTIAL_PERMANENT: 502,
    REQUEST_PERMANENT: 400,
    POLICY_BLOCKED: 400,
    EMPTY_OR_PROTOCOL: 502,
    CLIENT_CLOSED: 499,
}

# 安全/政策拦截关键词（Google 官方 blockReason 集合 + 本项目安全拦截文案）
_POLICY_KEYWORDS = (
    "安全策略拦截", "安全拦截", "prohibited_content", "content_filter",
    "safety", "recitation", "blocklist", "spii", "image_safety",
)
# 凭证/权限永久失效关键词（403 类：IAM/计费/Cookie 失效/Key 吊销）
_CREDENTIAL_KEYWORDS = (
    "permission denied", "permission_denied", "forbidden",
    "unauthenticated", "login required", "session expired",
    "invalid credentials", "cookie 已失效", "凭证已失效", "revoked",
    "requires billing", "billing to be enabled",
)
# 请求参数永久错误关键词（400 schema/参数类）
_REQUEST_PERMANENT_KEYWORDS = (
    "invalid argument", "invalid_request", "invalid request",
    "unsupported", "not supported",
)
# 空流/协议异常关键词（本项目空流文案 + 上游 envelope 异常）
_EMPTY_OR_PROTOCOL_KEYWORDS = (
    "空流", "未返回任何响应", "无有效内容", "未返回任何内容",
    "remote protocol error", "peer closed connection",
)


def classify_failure(status: Optional[int] = None, message: str = "") -> str:
    """把一次失败归入唯一分类（语义消息优先于状态码，同 logger.classify_error 的排序思路）。

    只做分类，不做动作：调用方按分类决定重试/切换/冷却/透传。
    """
    s = status if isinstance(status, int) else None
    low = str(message or "").lower()
    # 1) 客户端断开（499/文案）：停止一切重试
    if s == 499 or "客户端已断开" in low or "client_closed" in low:
        return CLIENT_CLOSED
    # 2) 安全/政策拦截：优先于状态码（同一个 400/200 都可能是安全拦截）
    if any(k in low for k in _POLICY_KEYWORDS):
        return POLICY_BLOCKED
    # 3) 限流/配额（429 或关键词）
    if s == 429 or "429" in low or "too many requests" in low \
            or "quota" in low or "resource_exhausted" in low or "resource exhausted" in low:
        return RATE_LIMITED
    # 4) 鉴权可刷新（401/token 过期；google-auth/SDK 自动刷新，不算凭证死亡）
    if s == 401 or "token expired" in low or "unauthenticated" in low:
        return AUTH_REFRESHABLE
    # 5) 凭证/权限/计费永久失效（403 类；Cookie 通道的文案关键词也在这里）
    if s == 403 or any(k in low for k in _CREDENTIAL_KEYWORDS):
        return CREDENTIAL_PERMANENT
    # 6) 空流/协议异常
    if any(k in low for k in _EMPTY_OR_PROTOCOL_KEYWORDS):
        return EMPTY_OR_PROTOCOL
    # 7) 请求参数永久错误
    if s == 400 or any(k in low for k in _REQUEST_PERMANENT_KEYWORDS):
        return REQUEST_PERMANENT
    # 8) 瞬态上游错误：只认改造前白名单里的三个 5xx（501/505 等不扩大切换范围）
    if s in (500, 502, 503, 504) or "503" in low:
        return TRANSIENT
    return OTHER


def classify_exception(e: Optional[BaseException]) -> str:
    """把上游异常归入唯一分类。

    与 api_helpers.is_retryable_exception 的识别面严格等价（见模块 docstring 兼容铁律）：
    httpx.HTTPStatusError / e.code 属性 / 错误文本关键词，不额外扩大。
    """
    if e is None:
        return OTHER
    # httpx 状态错误：按真实状态码分类
    if isinstance(e, httpx.HTTPStatusError):
        return classify_failure(status=e.response.status_code, message=str(e))
    code = getattr(e, "code", None)
    if isinstance(code, int):
        return classify_failure(status=code, message=str(e))
    return classify_failure(message=str(e))


def category_switchable(category: str) -> bool:
    """该分类是否允许跨通道切换（= 改造前 SWITCHABLE_STATUS_CODES 语义）。"""
    return category in _SWITCHABLE


def status_switchable(status: int) -> bool:
    """HTTP 状态码是否可切换（等价改造前 {429,500,502,503,504}）。"""
    return category_switchable(classify_failure(status=status))


def exception_switchable(e: Optional[BaseException]) -> bool:
    """异常是否可切换（等价改造前 is_retryable_exception）。"""
    return category_switchable(classify_exception(e))


def final_http_status(attempts: list, fallback_status: int = 500) -> int:
    """全部候选失败后的最终 HTTP 状态码（修复「聚合字符串被洗成 500」的缺口，报告 §8）。

    规则：
      1) 优先取最后一个「真实上游 JSONResponse」记录的 HTTP 状态（attempts 项带
         upstream=True 标记），不再从拼接后的字符串反推；
      2) 没有真实上游状态（全是熔断跳过/未出流异常）时，按最后一个分类经
         CATEGORY_HTTP_STATUS 映射到稳定网关状态；
      3) 仍无分类 → 沿用调用方传入的 fallback（保持旧行为的 500/503 兜底）。
    """
    for a in reversed(attempts or []):
        if isinstance(a, dict) and a.get("upstream"):
            s = a.get("status")
            if isinstance(s, int) and 400 <= s <= 599:
                return s
    for a in reversed(attempts or []):
        cat = (a or {}).get("category") if isinstance(a, dict) else None
        if cat in CATEGORY_HTTP_STATUS:
            return CATEGORY_HTTP_STATUS[cat]
    if isinstance(fallback_status, int) and 400 <= fallback_status <= 599:
        return fallback_status
    return 500


# ---------- 429 冷却窗口解析（报告 §15.2：优先精确时间，再保守退避）----------

# retryDelay: "30s"（Google RetryInfo 元数据）；Retry-After: 30 / HTTP 头同形
_RETRY_DELAY_RE = re.compile(r'retrydelay["\']?\s*[:=]\s*"?(\d{1,6})', re.I)
_RETRY_AFTER_RE = re.compile(r'retry-after["\']?\s*[:=]\s*"?(\d{1,6})', re.I)
# resetTime / retry_time 的 epoch（秒或毫秒）
_RESET_EPOCH_RE = re.compile(r'(?:resettime|retry_time|resets? in)["\']?\s*[:=]\s*"?(\d{10,14})', re.I)

# 冷却上限：防止解析出一个畸形大数值把凭证冻结到天荒地老（gcli2api 的 4 小时数值不照抄）
MAX_COOLDOWN_SECONDS = 4 * 3600


def parse_retry_after(message: str, now: Optional[float] = None) -> Optional[float]:
    """从错误文本解析冷却秒数（解析不出返回 None，调用方走保守默认）。

    识别三种形态（按优先级）：
      - retryDelay: "30s"（RESOURCE_EXHAUSTED 的 RetryInfo 元数据）
      - Retry-After: 30（HTTP 头文本化进 message 时）
      - resetTime/retry_time 的 epoch 时间戳（秒或毫秒，超上限按上限截断）
    """
    import time
    low = str(message or "")
    for pattern in (_RETRY_DELAY_RE, _RETRY_AFTER_RE):
        m = pattern.search(low)
        if m:
            try:
                return min(float(m.group(1)), MAX_COOLDOWN_SECONDS)
            except (TypeError, ValueError):
                continue
    m = _RESET_EPOCH_RE.search(low)
    if m:
        try:
            ts = float(m.group(1))
            if ts > 1e12:       # 毫秒
                ts /= 1000.0
            sec = ts - (now if now is not None else time.time())
            if sec > 0:
                return min(sec, MAX_COOLDOWN_SECONDS)
        except (TypeError, ValueError):
            pass
    return None
