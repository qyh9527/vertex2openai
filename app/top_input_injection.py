"""可配置的顶部输入注入（Top Input Injection）。

把最新 user 纯文本复制进控制台维护的模板方案，并作为 system / assistant / user
消息插在整个 OpenAI 消息序列开头。它不替换原 user 消息，故与普通问答、工具往返
和输入搬运可以独立组合；未完整配置时严格空操作。
"""

from __future__ import annotations

from dataclasses import dataclass
import secrets
from typing import Any, Mapping, Optional

from models import OpenAIMessage
from input_relay import (
    MODE_ALWAYS,
    MODE_FAKE_STREAM_ONLY,
    MODE_OFF,
    input_relay_active_for_stream,
)

SETTING_MODE = "top_input_injection_mode"
SETTING_PLANS = "top_input_injection_plans"
SETTING_SELECTED_PLAN_ID = "top_input_injection_selected_plan_id"
SETTING_RANDOM = "top_input_injection_random"
INPUT_PLACEHOLDER = "{{input}}"
MAX_PLANS = 32
MAX_PLAN_CONTENT_CHARS = 100_000


@dataclass(frozen=True)
class TopInputPlan:
    """一套持久化的顶部输入注入模板。"""

    plan_id: str
    name: str
    role: str
    content: str


@dataclass(frozen=True)
class TopInputInjectionConfig:
    """已校验的顶部输入注入配置。"""

    mode: str
    plans: tuple[TopInputPlan, ...]
    selected_plan_id: str
    random_enabled: bool


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


def _normalize_role(value: Any) -> Optional[str]:
    role = str(value or "").strip().lower()
    if role == "system":
        return "system"
    if role == "user":
        return "user"
    if role in ("assistant", "ai", "model"):
        return "assistant"
    return None


def _read_plans(raw: Any) -> tuple[TopInputPlan, ...]:
    if not isinstance(raw, list):
        return ()

    plans: list[TopInputPlan] = []
    seen_ids: set[str] = set()
    for item in raw[:MAX_PLANS]:
        if not isinstance(item, Mapping):
            continue
        plan_id = str(item.get("id") or "").strip()[:96]
        name = str(item.get("name") or "").strip()[:120]
        role = _normalize_role(item.get("role"))
        content = str(item.get("content") or "")[:MAX_PLAN_CONTENT_CHARS]
        if not plan_id or plan_id in seen_ids or not role or not content.strip():
            continue
        seen_ids.add(plan_id)
        plans.append(TopInputPlan(
            plan_id=plan_id,
            name=name or f"方案 {len(plans) + 1}",
            role=role,
            content=content,
        ))
    return tuple(plans)


def get_top_input_injection_config(
    settings: Mapping[str, Any],
) -> tuple[Optional[TopInputInjectionConfig], Optional[str]]:
    """读取顶部输入注入设置；模式关闭或无有效方案时返回空配置。"""
    mode = str(settings.get(SETTING_MODE) or MODE_OFF).strip().lower()
    if mode == MODE_OFF:
        return None, None
    if mode not in (MODE_FAKE_STREAM_ONLY, MODE_ALWAYS):
        return None, "⛔ [顶部输入注入] 模式无效，已保持空操作。"

    plans = _read_plans(settings.get(SETTING_PLANS))
    if not plans:
        return None, "⛔ [顶部输入注入] 当前模式已启用但没有有效方案，已保持空操作。"
    return TopInputInjectionConfig(
        mode=mode,
        plans=plans,
        selected_plan_id=str(settings.get(SETTING_SELECTED_PLAN_ID) or "").strip(),
        random_enabled=_as_bool(settings.get(SETTING_RANDOM, False)),
    ), None


def top_input_injection_active_for_stream(
    config: TopInputInjectionConfig,
    is_fake_stream: bool,
) -> bool:
    """复用输入搬运的三态语义，保证两种输入处理模式口径一致。"""
    return input_relay_active_for_stream(config, is_fake_stream)


def _choose_plan(config: TopInputInjectionConfig) -> TopInputPlan:
    if config.random_enabled:
        return secrets.choice(config.plans)
    return next(
        (plan for plan in config.plans if plan.plan_id == config.selected_plan_id),
        config.plans[0],
    )


def _render_plan(plan: TopInputPlan, latest_user_text: str) -> str:
    """渲染模板；未使用内部占位符时将原始输入追加在模板尾部。"""
    if INPUT_PLACEHOLDER in plan.content:
        return plan.content.replace(INPUT_PLACEHOLDER, latest_user_text)
    return f"{plan.content.rstrip()}\n\n{latest_user_text}"


def apply_top_input_injection(
    messages: list[OpenAIMessage],
    config: TopInputInjectionConfig,
) -> tuple[list[OpenAIMessage], list[str]]:
    """把所选方案渲染为一条消息并插入消息列表首位。

    原始 user 消息绝不替换；其余消息对象也不原地修改，便于混合通道故障转移复用。
    """
    source = next((
        message for message in reversed(messages)
        if str(getattr(message, "role", "")).lower() == "user"
    ), None)
    if source is None:
        return messages, ["ℹ️ [顶部输入注入] 未找到 user 消息，未改写。"]
    if not isinstance(source.content, str) or not source.content.strip():
        return messages, ["ℹ️ [顶部输入注入] 最新 user 消息不是非空纯文本，未改写（避免丢失图片/多段内容）。"]

    plan = _choose_plan(config)
    injected_content = _render_plan(plan, source.content)
    if not injected_content.strip():
        return messages, ["ℹ️ [顶部输入注入] 所选方案渲染为空，未改写。"]

    injected = OpenAIMessage(role=plan.role, content=injected_content)
    return [injected, *messages], [
        f"✅ [顶部输入注入] 已以 {plan.role} 身份在提示词首位注入方案「{plan.name}」"
        f"（{len(injected_content)} 字，{'随机' if config.random_enabled else '指定'}选择）。"
    ]
