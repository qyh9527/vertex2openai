"""raw 纯净入口：只做 OpenAI ↔ Gemini 协议转换时的「已开启但被忽略」提示。

raw 入口固定走服务账号（Vertex SA）通道，并跳过全部提示词改写功能；生成参数
一律使用代码内置默认（`config.DEFAULT_SETTINGS`），不读取控制台全局值、模型专属
覆盖与单请求扩展字段。本模块只负责检测哪些提示词功能已被配置、本次却被跳过，
供请求日志 `request_start.raw_skipped` 与逐项 `⏭️ [raw 入口]` 提示使用。
"""

from typing import Any, Optional

import config as app_config
from runtime_state import app_state
import model_capabilities as mc
from top_input_injection import (
    get_top_input_injection_config,
    top_input_injection_active_for_channel,
)
from input_relay import get_input_relay_config, input_relay_active_for_stream
from history_tool import history_tool_active
from anti_truncation import get_enabled_field, is_enabled_for_request

RAW_STRATEGY = "raw"
RAW_CHANNEL = "vertex"

# 跳过项顺序固定（日志与结构化字段共用同一顺序，便于测试断言）。
RAW_SKIPPED_ORDER = (
    "top_injection",
    "input_relay",
    "history_tool",
    "console_injection",
    "prefill_compat",
    "anti_truncation",
    "model_overrides",
    "generation_defaults",
)

RAW_SKIPPED_LABELS = {
    "top_injection": "顶部注入",
    "input_relay": "输入搬运",
    "history_tool": "历史工具承载",
    "console_injection": "控制台 system/prefill 注入",
    "prefill_compat": "预填充兼容",
    "anti_truncation": "防截断合成工具",
    "model_overrides": "模型专属参数",
    "generation_defaults": "控制台生成参数",
}

# raw 入口一律走内置默认的生成参数键：控制台值即使被改动也不生效。
RAW_GENERATION_SETTING_KEYS = (
    "default_temperature",
    "default_top_p",
    "default_max_tokens",
    "native_thinking_mode",
    "thinking_g3_level",
    "image_size",
    "image_aspect_ratio",
    "image_system_instruction",
    "sampling_policy",
)


def base_model_name(model: Any) -> str:
    """请求模型名归一（只用于读取设置/能力档案，不改变下发模型名）。

    raw 入口固定走 SA，`[PAY]` 前缀由 ServiceAccountUpstream 自行剥除；
    `fake-` 与 `-search` 属于保留的传输层能力，同样从基础名里去掉。
    """
    name = str(model or "").strip()
    if name.upper().startswith("[PAY]"):
        name = name[len("[PAY]"):].strip()
    while name.startswith("fake-"):
        name = name[len("fake-"):]
    if name.endswith("-search"):
        name = name[:-len("-search")]
    return name


def _is_fake_model(model: Any) -> bool:
    """与 SA 上游一致：先剥 [PAY] 前缀，再看 fake-。"""
    name = str(model or "").strip()
    if name.upper().startswith("[PAY]"):
        name = name[len("[PAY]"):].strip()
    return name.startswith("fake-")


def _last_role(request_obj: Any) -> Optional[str]:
    messages = getattr(request_obj, "messages", None) or []
    if not messages:
        return None
    return str(getattr(messages[-1], "role", "") or "")


def detect_skipped_features(request_obj: Any,
                            settings: Optional[dict] = None) -> list:
    """返回本次 raw 请求「已开启但被忽略」的功能键（按 RAW_SKIPPED_ORDER 排序）。"""
    base = base_model_name(getattr(request_obj, "model", ""))
    settings = settings if settings is not None else app_state.get_effective_settings(base)
    is_fake = _is_fake_model(getattr(request_obj, "model", ""))
    is_image = bool(mc.get_profile(base)["is_image"])
    is_fake_stream = bool(getattr(request_obj, "stream", False) and (is_fake or is_image))

    found = set()

    top_config, _ = get_top_input_injection_config(settings)
    if top_config and top_input_injection_active_for_channel(
            top_config, RAW_CHANNEL, is_fake=is_fake):
        found.add("top_injection")

    relay_config, _ = get_input_relay_config(settings)
    if relay_config and input_relay_active_for_stream(relay_config, is_fake_stream):
        found.add("input_relay")

    if history_tool_active(settings, is_fake):
        found.add("history_tool")

    if (str(settings.get("inject_system_instruction") or "").strip()
            or str(settings.get("inject_prefill") or "").strip()):
        found.add("console_injection")

    if (str(settings.get("prefill_mode") or "off") != "off"
            and _last_role(request_obj) == "assistant"):
        found.add("prefill_compat")

    if is_enabled_for_request(request_obj, {"value": get_enabled_field(settings)}):
        found.add("anti_truncation")

    overrides = app_state.get_model_overrides().get(base)
    if isinstance(overrides, dict) and overrides:
        found.add("model_overrides")

    for key in RAW_GENERATION_SETTING_KEYS:
        if settings.get(key) != app_config.DEFAULT_SETTINGS.get(key):
            found.add("generation_defaults")
            break

    return [key for key in RAW_SKIPPED_ORDER if key in found]


def log_skipped_features(keys: list) -> None:
    """逐项打印被 raw 跳过的提示词功能（未命中时不打任何日志）。"""
    for key in keys:
        label = RAW_SKIPPED_LABELS.get(key, key)
        print(f"⏭️ [raw 入口] {label} 未应用于本请求（raw 只做协议转换）。")
