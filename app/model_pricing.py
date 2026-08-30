"""Gemini 官方按量定价表（Agent Platform，USD / 1M tokens，global ≤200K 输入档）。

来源：https://cloud.google.com/gemini-enterprise-agent-platform/generative-ai/pricing
核对于 2026-08-31。官方按「Standard / Priority / Flex-Batch」三张独立表计价（global 档）。
规律：Priority = Standard × 1.8，Flex = Standard × 0.5（半价）；缓存命中输入统一 90% 折扣。
未知模型返回 None（不计费，避免误报）；生图模型图片输出按张计费（usage 无张数），
美刀估算仅覆盖文本输入/输出部分。
"""

# 模型名（前缀匹配，越长越优先）→ Standard 档 (input $/1M, output $/1M)
MODEL_PRICING: dict = {
    "gemini-3.7-flash": (0.75, 3.75),          # introductory：2027-01-01 起 1.5 / 7.5
    "gemini-3.6-flash": (0.75, 3.75),          # introductory：2027-01-01 起 1.5 / 7.5
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


def get_model_price(model_name: str):
    """按最长前缀匹配模型价格；返回 Standard 档 (input $/1M, output $/1M) 或 None。

    自动剥掉代理层前缀/后缀（fake- / -search / [EXPRESS] / [PAY]）。
    """
    name = (model_name or "").strip().lower()
    for prefix in ("fake-", "[express] ", "[pay] "):
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    for suffix in ("-search", "-openai", "-openaisearch"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break

    best = None  # (key_len, price)
    for key, price in MODEL_PRICING.items():
        if name == key or name.startswith(key):
            if best is None or len(key) > best[0]:
                best = (len(key), price)
    return best[1] if best else None


def estimate_cost(model_name: str, prompt_tokens: int, completion_tokens: int,
                  cached_tokens: int = 0, tier: str = "standard"):
    """估算一次请求的美刀成本；未知模型或全 0 返回 None（不计费）。

    计费拆三块（官方按量价，tier 决定 Standard/Priority/Flex 档）：
      - 未命中缓存的输入：  档 input 价 × (prompt - cached) / 1M
      - 命中缓存的输入：    档 input 价 × 10% × cached / 1M（缓存命中 90% 折扣）
      - 输出：              档 output 价 × completion / 1M
    Flex 档对 2.x 模型不可用（请求侧会自动降级为普通请求），这里同样降级为 Standard 计费。
    """
    price = get_model_price(model_name)
    if price is None:
        return None
    # 2.x 模型不支持 flex：打 flex 头的请求实际被降级为标准，计费保持一致
    if tier == "flex" and (model_name or "").startswith("gemini-2"):
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
    cached_cost = cached / 1_000_000 * input_price * CACHED_INPUT_DISCOUNT
    output_cost = completion_tokens / 1_000_000 * output_price
    return input_cost + cached_cost + output_cost
