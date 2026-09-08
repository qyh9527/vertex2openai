"""三通道共享用量映射，不依赖真实上游或凭证。"""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from google.genai import types

import api_helpers
from message_processing import convert_to_openai_format
from upstreams.cookie_proxy import _map_usage
from usage_mapping import map_usage


FIELDS = {
    "promptTokenCount": "prompt_token_count",
    "candidatesTokenCount": "candidates_token_count",
    "thoughtsTokenCount": "thoughts_token_count",
    "cachedContentTokenCount": "cached_content_token_count",
    "totalTokenCount": "total_token_count",
    "toolUsePromptTokenCount": "tool_use_prompt_token_count",
}
META = {"promptTokenCount": 100, "candidatesTokenCount": 20,
        "thoughtsTokenCount": 30, "cachedContentTokenCount": 40,
        "totalTokenCount": 150}
EXPECTED = {
    "prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150,
    "prompt_tokens_details": {"cached_tokens": 40},
    "completion_tokens_details": {"reasoning_tokens": 30},
}


def sdk_meta(meta):
    return types.GenerateContentResponseUsageMetadata(**{FIELDS[k]: v for k, v in meta.items()})


@pytest.mark.parametrize("meta,prompt,completion,total,cached,thoughts", [
    (META, 100, 50, 150, 40, 30),
    ({"promptTokenCount": 10, "candidatesTokenCount": 5}, 10, 5, 15, 0, 0),
    ({"thoughtsTokenCount": 7}, 0, 7, 7, 0, 7),
    ({"promptTokenCount": 10, "thoughtsTokenCount": 7, "totalTokenCount": 99}, 10, 7, 99, 0, 7),
    ({**META, "totalTokenCount": 0}, 100, 50, 0, 40, 30),
    ({**META, "totalTokenCount": None}, 100, 50, 150, 40, 30),
    ({**META, "totalTokenCount": 157, "toolUsePromptTokenCount": 7}, 100, 50, 157, 40, 30),
    ({key: None for key in FIELDS}, 0, 0, 0, 0, 0),
    ({}, 0, 0, 0, 0, 0),
])
def test_mapping_matches_across_channels(meta, prompt, completion, total, cached, thoughts):
    expected = {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": total,
                "prompt_tokens_details": {"cached_tokens": cached},
                "completion_tokens_details": {"reasoning_tokens": thoughts}}
    sdk = sdk_meta(meta)
    assert map_usage(sdk) == _map_usage(meta) == expected
    response = types.GenerateContentResponse(usage_metadata=sdk)
    assert convert_to_openai_format(response, "gemini-3.6-flash")["usage"] == expected
    assert api_helpers._extract_usage(response) == (prompt, completion, total)
    assert api_helpers._extract_cached_tokens(response) == cached


@pytest.mark.parametrize("response", [SimpleNamespace(), SimpleNamespace(usage_metadata=None)])
def test_missing_metadata(response):
    assert api_helpers._extract_usage(response) == (0, 0, 0)
    assert api_helpers._extract_cached_tokens(response) == 0
    assert map_usage() == _map_usage(None)


def test_stats_count_thoughts_once(monkeypatch):
    stats = Mock()
    monkeypatch.setattr(api_helpers, "stats", stats)
    monkeypatch.setattr(api_helpers, "_effective_pricing_tier", lambda: "standard")
    response = types.GenerateContentResponse(usage_metadata=sdk_meta(META))
    assert api_helpers._record_usage(response, "gemini-3.6-flash") == EXPECTED
    stats.add_tokens.assert_called_once_with(100, 50, cached=40, model="gemini-3.6-flash", tier="standard")
