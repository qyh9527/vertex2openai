"""Gemini 官方按量定价表（Agent Platform，USD / 1M tokens，global ≤200K 输入档）。

来源：https://cloud.google.com/gemini-enterprise-agent-platform/generative-ai/pricing
核对于 2026-08-31（gemini-4-argon / gemini-3.8-flash 补核于 2026-10-14）。
官方按「Standard / Priority / Flex-Batch」三张独立表计价（global 档）。
规律：Priority = Standard × 1.8，Flex = Standard × 0.5（半价）；缓存命中输入默认 90% 折扣，
个别模型另有折扣（见 CACHED_INPUT_RATIO_OVERRIDES）。
未知模型返回 None（不计费，避免误报）；生图模型图片输出按张计费（usage 无张数），
美刀估算仅覆盖文本输入/输出部分。
"""

import datetime

import model_capabilities as mc

# 模型名（前缀匹配，越长越优先）→ Standard 档 (input $/1M, output $/1M)，即当前生效价；
# 官方已公布的调价见 PRICE_SCHEDULE，到日期后自动切换，无需改这里。
MODEL_PRICING: dict = {
    "gemini-4-argon": (2.00, 10.00),           # introductory，见 PRICE_SCHEDULE
    "gemini-3.8-flash": (0.75, 3.75),          # introductory，见 PRICE_SCHEDULE
    "gemini-3.7-flash": (0.75, 3.75),          # introductory，见 PRICE_SCHEDULE
    "gemini-3.6-flash": (0.75, 3.75),          # introductory，见 PRICE_SCHEDULE
    "gemini-3.5-flash-lite": (0.30, 2.50),
    "gemini-3.5-flash": (1.50, 9.00),
    "gemini-3.1-pro-preview": (2.00, 12.00),
    "gemini-3.1-pro": (2.00, 12.00),
    "gemini-3.1-flash-lite": (0.25, 1.50),
    "gemini-2.5-pro": (1.25, 10.00),
    "gemini-2.5-flash": (0.30, 2.50),
    # 生图模型：图片输出按张计费（如 3.1-flash-image 图片输出 $30/张），美刀估算只覆盖文本
    "gemini-3.1-flash-image": (0.25, 1.50),
}

# 官方已公布的定时调价：MODEL_PRICING 的键 → [(生效日期, Standard 档价), ...]（按日期升序）。
# 按服务器本地日期（日志时区，默认 Asia/Shanghai）切换；官方生效时刻与本地零点可能差几个小时。
PRICE_SCHEDULE: dict = {
    "gemini-4-argon": [(datetime.date(2026, 12, 1), (4.00, 20.00))],
    "gemini-3.8-flash": [(datetime.date(2027, 1, 1), (1.50, 7.50))],
    "gemini-3.7-flash": [(datetime.date(2027, 1, 1), (1.50, 7.50))],
    "gemini-3.6-flash": [(datetime.date(2027, 1, 1), (1.50, 7.50))],
}

# 官方三档系数：Priority = Standard × 1.8，Flex = Standard × 0.5
# off / standard → 1.0；auto（默认）语义 = 钉定 global 打 Priority 头 → 1.8
PRICING_FACTORS: dict = {
    "standard": 1.0,
    "off": 1.0,
    "priority": 1.8,
    "auto": 1.8,
    "flex": 0.5,
}

# 缓存命中输入的折扣后单价系数：官方隐式/显式缓存命中 90% 折扣 → 按 10% 计费
CACHED_INPUT_DISCOUNT = 0.1

# 个别模型的缓存命中系数（同样按最长前缀匹配）：gemini-4-argon 缓存输入 $0.1 / 输入 $2 → 95% 折扣
CACHED_INPUT_RATIO_OVERRIDES: dict = {
    "gemini-4-argon": 0.05,
}


def get_model_price(model_name: str, on_date: datetime.date = None):
    """按最长前缀匹配模型价格；返回 Standard 档 (input $/1M, output $/1M) 或 None。

    自动剥掉代理层前缀/后缀（fake- / -search / [EXPRESS] / [PAY]）。
    on_date 缺省为服务器当天，按 PRICE_SCHEDULE 取当天生效价。
    """
    match = _longest_prefix_match(model_name, MODEL_PRICING)
    if match is None:
        return None
    key, price = match
    day = on_date or _today()
    for start, scheduled in PRICE_SCHEDULE.get(key, ()):
        if day >= start:
            price = scheduled
    return price


def get_cached_input_ratio(model_name: str) -> float:
    """缓存命中输入按输入单价的多少计费（未单列的模型用 CACHED_INPUT_DISCOUNT）。"""
    match = _longest_prefix_match(model_name, CACHED_INPUT_RATIO_OVERRIDES)
    return CACHED_INPUT_DISCOUNT if match is None else match[1]


def _today() -> datetime.date:
    return datetime.date.today()


def _longest_prefix_match(model_name: str, table: dict):
    """返回 (命中的键, 值)；未命中返回 None。"""
    name = (model_name or "").strip().lower()
    for prefix in ("fake-", "[express] ", "[pay] "):
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    for suffix in ("-search", "-openai", "-openaisearch"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break

    best = None  # (key, value)
    for key, value in table.items():
        if name == key or name.startswith(key):
            if best is None or len(key) > len(best[0]):
                best = (key, value)
    return best


def estimate_cost(model_name: str, prompt_tokens: int, completion_tokens: int,
                  cached_tokens: int = 0, tier: str = "standard",
                  on_date: datetime.date = None):
    """估算一次请求的美刀成本；未知模型或全 0 返回 None（不计费）。

    计费拆三块（官方按量价，tier 决定 Standard/Priority/Flex 档）：
      - 未命中缓存的输入：  档 input 价 × (prompt - cached) / 1M
      - 命中缓存的输入：    档 input 价 × 缓存系数 × cached / 1M（默认 10%，即 90% 折扣）
      - 输出：              档 output 价 × completion / 1M
    Flex 档对 2.x 模型不可用（请求侧会自动降级为普通请求），这里同样降级为 Standard 计费。
    单价按 on_date（缺省服务器当天）取 PRICE_SCHEDULE 生效价。
    """
    price = get_model_price(model_name, on_date)
    if price is None:
        return None
    # 2.x 模型不支持 flex：打 flex 头的请求实际被降级为标准，计费保持一致
    if tier == "flex" and (model_name or "").startswith("gemini-2"):
        tier = "standard"
    # auto 档对生图模型不打 Priority 头（http_options.resolve_paygo_bundle），按 Standard 计费
    if tier == "auto" and mc.get_profile(model_name)["is_image"]:
        tier = "standard"
    factor = PRICING_FACTORS.get(tier or "standard", 1.0)

    prompt_tokens = max(0, prompt_tokens or 0)
    completion_tokens = max(0, completion_tokens or 0)
    cached = max(0, min(cached_tokens or 0, prompt_tokens))
    if prompt_tokens == 0 and completion_tokens == 0:
        return None
    input_price, output_price = price
    input_price *= factor
    output_price *= factor
    input_cost = (prompt_tokens - cached) / 1_000_000 * input_price
    cached_cost = cached / 1_000_000 * input_price * get_cached_input_ratio(model_name)
    output_cost = completion_tokens / 1_000_000 * output_price
    return input_cost + cached_cost + output_cost
