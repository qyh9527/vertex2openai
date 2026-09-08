"""统一 Gemini 用量口径及 OpenAI 流式用量契约，不依赖应用状态。"""
from collections.abc import Mapping
from typing import Any


def map_usage(metadata: Any = None) -> dict:
    """兼容 SDK snake_case 对象和 Cookie camelCase 字典；思考计入输出。"""
    def value(snake: str, camel: str):
        if isinstance(metadata, Mapping):
            return metadata.get(camel, metadata.get(snake))
        return getattr(metadata, snake, None)

    prompt = int(value("prompt_token_count", "promptTokenCount") or 0)
    candidates = int(value("candidates_token_count", "candidatesTokenCount") or 0)
    thoughts = int(value("thoughts_token_count", "thoughtsTokenCount") or 0)
    cached = int(value("cached_content_token_count", "cachedContentTokenCount") or 0)
    total = value("total_token_count", "totalTokenCount")
    return {
        "prompt_tokens": prompt,
        "completion_tokens": candidates + thoughts,
        "total_tokens": int(total) if total is not None else prompt + candidates + thoughts,
        "prompt_tokens_details": {"cached_tokens": cached},
        "completion_tokens_details": {"reasoning_tokens": thoughts},
    }


async def with_usage_null(source, include_usage: bool):
    """为本项目序列化的普通 SSE 块补 usage:null，保留尾块/错误/注释。

    序列化器固定将顶层 choices 放在 delta 之前；只检查此前的短头部，
    不解析或重序列化图片和工具参数等大正文。正文内的字符串键已 JSON 转义。
    """
    try:
        async for line in source:
            if include_usage and line.startswith("data: {"):
                choices_at = line.find('"choices":')
                if (choices_at >= 0
                        and '"object": "chat.completion.chunk"' in line[:choices_at]
                        and '"usage":' not in line[:choices_at]):
                    choices_start = choices_at + len('"choices":')
                    if not line[choices_start:choices_start + 4].lstrip().startswith("[]"):
                        line = 'data: {"usage": null,' + line[len("data: {"):]
            yield line
    finally:
        await source.aclose()
