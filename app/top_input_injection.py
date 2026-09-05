"""可配置的顶部注入（Top Injection）。

把控制台维护的内容方案作为 system / assistant / user 消息插在整个 OpenAI
消息序列开头。它不读取或改写 user 输入，也不依赖输入搬运；未完整配置时严格空操作。
"""

from __future__ import annotations

from dataclasses import dataclass
import secrets
from typing import Any, Mapping, Optional

from models import OpenAIMessage

MODE_OFF = "off"
MODE_ALWAYS = "always"

MODE_NON_VERTEX_ONLY = "non_vertex_only"
_LEGACY_FAKE_STREAM_ONLY = "fake_stream_only"

SETTING_MODE = "top_input_injection_mode"
SETTING_PLANS = "top_input_injection_plans"
SETTING_SELECTED_PLAN_ID = "top_input_injection_selected_plan_id"
SETTING_RANDOM = "top_input_injection_random"
MAX_PLANS = 32
MAX_PLAN_CONTENT_CHARS = 100_000


@dataclass(frozen=True)
class TopInputPlan:
    """一套持久化的顶部注入模板。"""

    plan_id: str
    name: str
    role: str
    content: str


@dataclass(frozen=True)
class TopInputInjectionConfig:
    """已校验的顶部注入配置。"""

    mode: str
    plans: tuple[TopInputPlan, ...]
    selected_plan_id: str
    random_mode: str


def normalize_random_mode(value: Any) -> str:
    """兼容旧布尔值；非法值退回固定方案。"""
    if value is True:
        return MODE_ALWAYS
    if isinstance(value, str):
        value = value.strip().lower()
        if value in (MODE_OFF, MODE_ALWAYS, MODE_NON_VERTEX_ONLY):
            return value
        if value in ("1", "true", "yes", "on"):
            return MODE_ALWAYS
    return MODE_OFF


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
    """读取顶部注入设置；模式关闭或无有效方案时返回空配置。"""
    mode = str(settings.get(SETTING_MODE) or MODE_OFF).strip().lower()
    random_mode = normalize_random_mode(settings.get(SETTING_RANDOM))
    # 旧路由限定档迁移为开启注入、仅 Express/Cookie 随机；SA 固定注入。
    if mode in (_LEGACY_FAKE_STREAM_ONLY, MODE_NON_VERTEX_ONLY):
        mode = MODE_ALWAYS
        random_mode = MODE_NON_VERTEX_ONLY
    if mode == MODE_OFF:
        return None, None
    if mode != MODE_ALWAYS:
        return None, "⛔ [顶部注入] 模式无效，已保持空操作。"

    plans = _read_plans(settings.get(SETTING_PLANS))
    if not plans:
        return None, "⛔ [顶部注入] 当前模式已启用但没有有效方案，已保持空操作。"
    return TopInputInjectionConfig(
        mode=mode,
        plans=plans,
        selected_plan_id=str(settings.get(SETTING_SELECTED_PLAN_ID) or "").strip(),
        random_mode=random_mode,
    ), None


def top_input_injection_active_for_channel(
    config: TopInputInjectionConfig,
    channel: str,
) -> bool:
    """注入只受总开关控制，任何实际通道都可注入。"""
    return config.mode == MODE_ALWAYS


def random_active_for_channel(config: TopInputInjectionConfig, channel: str) -> bool:
    """随机按实际候选通道判断，不读取四选一上游策略。"""
    mode = normalize_random_mode(config.random_mode)
    return mode == MODE_ALWAYS or (
        mode == MODE_NON_VERTEX_ONLY
        and str(channel or "").strip().lower() in ("express", "cookie")
    )


def _choose_plan(config: TopInputInjectionConfig, random_enabled: bool) -> TopInputPlan:
    if random_enabled:
        return secrets.choice(config.plans)
    return next(
        (plan for plan in config.plans if plan.plan_id == config.selected_plan_id),
        config.plans[0],
    )


def _render_plan(plan: TopInputPlan) -> str:
    """返回方案正文原文，不解析宏或占位符。"""
    return plan.content


def apply_top_input_injection(
    messages: list[OpenAIMessage],
    config: TopInputInjectionConfig,
    channel: str = "",
) -> tuple[list[OpenAIMessage], list[str]]:
    """把所选方案原样置于 messages 第 1 条，必要时与同角色首条消息融合。

    不读取、不替换、也不依赖原始 user 消息；其余消息对象不原地修改，便于混合
    通道故障转移复用。
    """
    if config.mode != MODE_ALWAYS:
        return messages, []
    random_enabled = random_active_for_channel(config, channel)
    plan = _choose_plan(config, random_enabled)
    injected_content = _render_plan(plan)
    if not injected_content.strip():
        return messages, ["ℹ️ [顶部注入] 所选方案为空，未改写。"]

    injected = OpenAIMessage(role=plan.role, content=injected_content)
    first = messages[0] if messages else None
    if (first is not None
            and _normalize_role(getattr(first, "role", "")) == plan.role
            and isinstance(first.content, str)):
        # 同角色第一条按请求顺序融合：顶部方案在前，原始提示在后，且只隔一个换行。
        merged = first.model_copy(update={"content": f"{injected_content}\n{first.content}"})
        return [merged, *messages[1:]], [
            f"✅ [顶部注入] 已以 {plan.role} 身份将方案「{plan.name}」"
            f"并入 messages 第 1 条（{len(injected_content)} 字，"
            f"{'随机' if random_enabled else '指定'}选择）。"
        ]

    return [injected, *messages], [
        f"✅ [顶部注入] 已以 {plan.role} 身份插入 messages 第 1 条方案「{plan.name}」"
        f"（{len(injected_content)} 字，{'随机' if random_enabled else '指定'}选择）。"
    ]
