"""可配置的输入搬运（Input Relay）协议。

该模块在 OpenAI 请求层工作：当最新 user 文本严格等于一个由控制台配置的 XML
包裹块时，提取其中的载荷，追加到其前最近的一条普通 assistant 消息末尾，并把
user 消息替换为同样由控制台配置的占位语。标签名和占位语均没有内置默认值；未完整
配置时始终是空操作。
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Mapping, Optional

from models import OpenAIMessage

SETTING_MODE = "input_relay_mode"
SETTING_TAG = "input_relay_tag"
SETTING_PLACEHOLDER = "input_relay_placeholder"
SETTING_STRIP_GENERATED = "input_relay_strip_generated"

MODE_OFF = "off"
MODE_FAKE_STREAM_ONLY = "fake_stream_only"
MODE_ALWAYS = "always"
_VALID_MODES = {MODE_OFF, MODE_FAKE_STREAM_ONLY, MODE_ALWAYS}

# XML 元素名的保守子集：足够覆盖普通英文/数字/命名空间式标签，同时拒绝会破坏动态
# 正则的控制字符、尖括号与超长输入。配置值是“标签名”，不是完整 XML 文本。
_TAG_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_.:-]{0,63}\Z")


@dataclass(frozen=True)
class InputRelayConfig:
    """已校验的输入搬运配置。"""

    tag: str
    placeholder: str
    mode: str
    strip_generated: bool = False


def get_input_relay_config(settings: Mapping[str, Any]) -> tuple[Optional[InputRelayConfig], Optional[str]]:
    """从运行时设置读取配置；模式关闭或未填完整配置时返回 ``(None, 原因)``。

    返回原因仅供调用方记录日志，绝不把半配置状态降级成任意默认标签或占位语。
    """
    mode = str(settings.get(SETTING_MODE) or MODE_OFF).strip().lower()
    if mode == MODE_OFF:
        return None, None
    if mode not in _VALID_MODES:
        return None, "⛔ [输入搬运] 模式无效，已保持空操作。"

    tag = str(settings.get(SETTING_TAG) or "").strip()
    placeholder = str(settings.get(SETTING_PLACEHOLDER) or "").strip()
    if not tag or not placeholder:
        return None, "⛔ [输入搬运] 当前模式已启用但标签名或占位语为空，已保持空操作。"
    if not _TAG_NAME_RE.fullmatch(tag):
        return None, "⛔ [输入搬运] 标签名无效，已保持空操作（仅允许字母/下划线开头及字母数字 _ . : -，最长 64 字）。"
    return InputRelayConfig(
        tag=tag,
        placeholder=placeholder,
        mode=mode,
        strip_generated=bool(settings.get(SETTING_STRIP_GENERATED, False)),
    ), None


def input_relay_active_for_stream(
    config: InputRelayConfig,
    is_fake_stream: bool,
    treat_fake_only_as_always: bool = False,
) -> bool:
    """按控制台三态模式判断当前请求是否应用搬运。

    Cookie 文本通道没有 `fake-` 假流实现。调用方可设
    ``treat_fake_only_as_always=True``，把用户选择的「只在假流开」在该通道
    降级为全请求开启，而不是静默变成永远不开。
    """
    return config.mode == MODE_ALWAYS or (
        config.mode == MODE_FAKE_STREAM_ONLY
        and (is_fake_stream or treat_fake_only_as_always)
    )


def _tag_block_pattern(tag: str) -> re.Pattern[str]:
    """生成严格匹配配置标签的 XML 块正则。

    支持开闭标签内的空白、跨行载荷与大小写差异；标签名由 ``re.escape`` 转义，且
    开标签必须紧接空白或 ``>``，不会把 ``<tag_extra>`` 误识别为 ``<tag>``。
    本协议不接受属性，避免把任意富文本/XML 解析语义带进请求改写路径。
    """
    escaped = re.escape(tag)
    return re.compile(
        rf"<\s*{escaped}\s*>(?P<body>.*?)<\s*/\s*{escaped}\s*>",
        re.IGNORECASE | re.DOTALL,
    )


def format_input_relay_block(tag: str, payload: str) -> str:
    """用用户配置的标签把提取出的载荷规范化成注入块。"""
    return f"<{tag}>\n{payload.strip()}\n</{tag}>"


def apply_input_relay(messages: list[OpenAIMessage], config: InputRelayConfig) -> tuple[list[OpenAIMessage], list[str]]:
    """将最新匹配的用户输入搬运进前一条 assistant 消息尾部。

    为避免静默吞掉对话内容，只有同时满足下列条件才会改写：
    1. 最新 user 消息与配置的 XML 块完全一致（允许块外空白）；
    2. 该块恰好出现一次且载荷非空；
    3. 前面存在 content 为字符串、且没有 tool_calls 的 assistant 消息。

    返回新列表而不原地修改 Pydantic 消息，故同一请求因故障转移重试时仍是幂等的。
    """
    if not messages:
        return messages, []

    source_idx = next((
        i for i in range(len(messages) - 1, -1, -1)
        if str(getattr(messages[i], "role", "")).lower() == "user"
    ), -1)
    if source_idx < 0:
        return messages, ["ℹ️ [输入搬运] 未找到 user 消息，未改写。"]

    source = messages[source_idx]
    if not isinstance(source.content, str):
        return messages, ["ℹ️ [输入搬运] 最新 user 消息不是纯文本，未改写（避免丢失图片/多段内容）。"]

    pattern = _tag_block_pattern(config.tag)
    matches = list(pattern.finditer(source.content))
    if len(matches) != 1:
        return messages, []
    match = matches[0]
    if source.content[:match.start()].strip() or source.content[match.end():].strip():
        return messages, ["ℹ️ [输入搬运] 配置标签必须包住整条最新 user 文本，未改写。"]

    payload = match.group("body")
    if not payload.strip():
        return messages, ["ℹ️ [输入搬运] 配置标签内为空，未改写。"]

    target_idx = next((
        i for i in range(source_idx - 1, -1, -1)
        if str(getattr(messages[i], "role", "")).lower() in ("assistant", "model")
        and not getattr(messages[i], "tool_calls", None)
        and isinstance(messages[i].content, str)
    ), -1)
    if target_idx < 0:
        return messages, ["ℹ️ [输入搬运] 未找到可追加的前一条 assistant 消息，未改写。"]

    block = format_input_relay_block(config.tag, payload)
    target = messages[target_idx]
    target_text = target.content
    # 防御性幂等：极少数外部中间件会把同一请求对象重复送入本函数；此时不能再追加。
    if target_text.rstrip().endswith(block):
        appended_text = target_text
    else:
        separator = "\n\n" if target_text and not target_text.endswith("\n") else "\n"
        appended_text = target_text + separator + block

    updated = list(messages)
    updated[target_idx] = target.model_copy(update={"content": appended_text})
    updated[source_idx] = source.model_copy(update={"content": config.placeholder})
    return updated, [
        f"✅ [输入搬运] 已从最新 user 消息提取 {len(payload.strip())} 字，"
        f"追加到前一条 assistant 消息尾部；user 已替换为配置占位语。"
    ]


def strip_generated_relay_blocks(text: str, tag: str) -> str:
    """从模型文本中移除已闭合的配置标签块（非流式路径）。"""
    if not isinstance(text, str) or not text:
        return text
    return _tag_block_pattern(tag).sub("", text)


class RelayBlockStreamStripper:
    """流式移除已闭合配置标签块，兼容 XML 标签跨 SSE chunk 切分。

    不完整标签/块在 ``flush`` 时原样放行，防止网络中断或模型半途输出时静默丢字；
    单个待闭合块最多暂存 64 KiB，超过上限也 fail-open 原样输出。
    """

    _MAX_PENDING_BLOCK = 64 * 1024

    def __init__(self, tag: str):
        escaped = re.escape(tag)
        self._open_re = re.compile(rf"<\s*{escaped}\s*>", re.IGNORECASE)
        self._close_re = re.compile(rf"<\s*/\s*{escaped}\s*>", re.IGNORECASE)
        self._tag_lower = tag.lower()
        self._buffer = ""
        self._inside_block = False
        self._opening = ""

    def _possible_opening_suffix(self) -> int:
        """返回可能是跨 chunk 开标签前缀的起点；没有则返回缓冲长度。"""
        pos = self._buffer.rfind("<")
        if pos < 0:
            return len(self._buffer)
        tail = self._buffer[pos:]
        partial = re.fullmatch(r"<\s*([A-Za-z0-9_.:-]*)", tail)
        if partial is None:
            return len(self._buffer)
        candidate = partial.group(1).lower()
        if self._tag_lower.startswith(candidate):
            return pos
        return len(self._buffer)

    def feed(self, text: str) -> str:
        """输入一个文本 delta，返回可安全下发给客户端的文本。"""
        if not text:
            return ""
        self._buffer += text
        output: list[str] = []

        while True:
            if self._inside_block:
                close = self._close_re.search(self._buffer)
                if close:
                    self._buffer = self._buffer[close.end():]
                    self._inside_block = False
                    self._opening = ""
                    continue
                if len(self._buffer) > self._MAX_PENDING_BLOCK:
                    # fail-open：模型长期不闭合时绝不无限吃内存，也不吞掉可见回复。
                    output.append(self._opening + self._buffer)
                    self._buffer = ""
                    self._inside_block = False
                    self._opening = ""
                break

            opening = self._open_re.search(self._buffer)
            if opening:
                output.append(self._buffer[:opening.start()])
                self._opening = self._buffer[opening.start():opening.end()]
                self._buffer = self._buffer[opening.end():]
                self._inside_block = True
                continue

            safe_end = self._possible_opening_suffix()
            output.append(self._buffer[:safe_end])
            self._buffer = self._buffer[safe_end:]
            break

        return "".join(output)

    def flush(self) -> str:
        """流结束时放行无法判定/未闭合部分，确保不丢文本。"""
        if self._inside_block:
            out = self._opening + self._buffer
        else:
            out = self._buffer
        self._buffer = ""
        self._inside_block = False
        self._opening = ""
        return out
