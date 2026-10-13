"""2026-10 模型更新：nano-banana-2.1 / 生图思考档位 / 4-argon 输出上限 / 生图不打 Priority 头。"""
import pytest

import model_capabilities as mc
from http_options import resolve_paygo_bundle
from models import OpenAIRequest
from protocol_conversion import create_generation_config
from runtime_state import app_state
from upstreams.cookie_proxy import _build_batch_graphql_body
from upstreams.express_sdk import _build_thinking_config

_IMAGE_SAFETY = {"HARM_CATEGORY_IMAGE_HATE", "HARM_CATEGORY_IMAGE_DANGEROUS_CONTENT",
                 "HARM_CATEGORY_IMAGE_HARASSMENT", "HARM_CATEGORY_IMAGE_SEXUALLY_EXPLICIT"}


def _req(model, **kw):
    return OpenAIRequest(model=model, messages=[{"role": "user", "content": "画一只鲸鱼"}], **kw)


@pytest.fixture(autouse=True)
def _no_settings(monkeypatch):
    monkeypatch.setattr(app_state, "get_effective_settings", lambda _: {})


@pytest.mark.parametrize("model", ["gemini-nano-banana-2.1", "fake-gemini-nano-banana-2.1"])
def test_nano_banana_is_image_profile(model):
    prof = mc.get_profile(model)
    assert prof["is_image"] is True
    assert prof["thinking_levels"] == {"minimal", "medium", "high"}
    assert prof["image_sizes"] == {"1K", "2K", "4K"}
    assert len(prof["image_aspect_ratios"]) == 15
    assert prof["allowed_sampling"] == set()


@pytest.mark.parametrize("model,levels", [
    ("gemini-3.1-flash-image", {"minimal", "high"}),
    ("gemini-nano-banana-2.1", {"minimal", "medium", "high"}),
])
def test_image_models_send_thinking_level_default_high(model, levels):
    assert mc.get_profile(model)["thinking_levels"] == levels
    cfg = _build_thinking_config(model, _req(model), True)
    assert cfg == {"include_thoughts": True, "thinking_level": "high"}


def test_pro_image_sends_no_thinking_config():
    # 真机：3-pro-image 带 thinking_level 返回 400 "thinking_level is not supported by this model"
    assert mc.get_profile("gemini-3-pro-image")["thinking_kind"] is None
    assert _build_thinking_config("gemini-3-pro-image", _req("gemini-3-pro-image"), True) is None


def test_image_thinking_suppressed_clamps_to_lowest_legal_level(monkeypatch):
    monkeypatch.setattr(app_state, "get_effective_settings", lambda _: {"native_thinking_mode": "off"})
    assert _build_thinking_config("gemini-3.1-flash-image", _req("gemini-3.1-flash-image"), True)["thinking_level"] == "minimal"
    assert _build_thinking_config("gemini-3-pro-image", _req("gemini-3-pro-image"), True) is None


def test_image_aspect_ratios_expanded_except_lite():
    assert mc.validate_aspect_ratio("gemini-3-pro-image", "9:21") == "9:21"
    assert mc.validate_aspect_ratio("gemini-3-pro-image", "1:8") == "1:8"
    assert mc.validate_aspect_ratio("gemini-3.1-flash-lite-image", "1:8") is None


def test_nano_banana_generation_config_sdk():
    config = create_generation_config(_req("gemini-nano-banana-2.1", seed=7, temperature=0.3))
    assert config["response_modalities"] == ["TEXT", "IMAGE"]
    assert config["media_resolution"] == "MEDIA_RESOLUTION_HIGH"
    cats = {str(getattr(s.category, "value", s.category)) for s in config["safety_settings"]}
    assert _IMAGE_SAFETY <= cats
    assert {s.threshold for s in config["safety_settings"] if str(getattr(s.category, "value", s.category)) in _IMAGE_SAFETY} == {"OFF"}
    assert "seed" not in config and "temperature" not in config


def test_old_image_model_keeps_text_safety_only():
    config = create_generation_config(_req("gemini-3.1-flash-image"))
    cats = {str(getattr(s.category, "value", s.category)) for s in config["safety_settings"]}
    assert not (_IMAGE_SAFETY & cats)
    assert "media_resolution" not in config


def test_nano_banana_cookie_body():
    body = _build_batch_graphql_body("p", "gemini-nano-banana-2.1", _req("gemini-nano-banana-2.1"))
    variables = body["variables"] if "variables" in body else body
    gen = variables["generationConfig"]
    assert gen["responseModalities"] == ["TEXT", "IMAGE"]
    assert gen["mediaResolution"] == "HIGH"
    assert gen["thinkingConfig"]["thinkingLevel"] == "HIGH"
    assert _IMAGE_SAFETY <= {s["category"] for s in variables["safetySettings"]}


def test_old_image_cookie_body_gets_thinking_only():
    body = _build_batch_graphql_body("p", "gemini-3.1-flash-image", _req("gemini-3.1-flash-image"))
    variables = body["variables"] if "variables" in body else body
    assert variables["generationConfig"]["thinkingConfig"]["thinkingLevel"] == "HIGH"
    assert "thinkingConfig" not in _build_batch_graphql_body("p", "gemini-3-pro-image", _req("gemini-3-pro-image"))["variables"]["generationConfig"]
    assert "mediaResolution" not in variables["generationConfig"]
    assert not (_IMAGE_SAFETY & {s["category"] for s in variables["safetySettings"]})


@pytest.mark.parametrize("max_tokens,expected", [(200000, 65536), (65536, 65536), (8192, 8192)])
def test_argon_max_output_tokens_capped(max_tokens, expected):
    config = create_generation_config(_req("gemini-4-argon", max_tokens=max_tokens))
    assert config["max_output_tokens"] == expected
    body = _build_batch_graphql_body("p", "gemini-4-argon", _req("gemini-4-argon", max_tokens=max_tokens))
    variables = body["variables"] if "variables" in body else body
    assert variables["generationConfig"]["maxOutputTokens"] == expected


def test_argon_profile_and_non_argon_uncapped():
    prof = mc.get_profile("gemini-4-argon")
    assert prof["thinking_levels"] == {"low", "medium", "high"}
    assert prof["default_level"] == "medium"
    assert not ({"temperature", "top_p", "top_k", "presence_penalty", "frequency_penalty", "candidate_count"}
                & prof["allowed_sampling"])
    assert create_generation_config(_req("gemini-3.8-flash", max_tokens=200000))["max_output_tokens"] == 200000


@pytest.mark.parametrize("model", ["gemini-nano-banana-2.1", "gemini-3.1-flash-image", "fake-gemini-3-pro-image"])
def test_auto_tier_skips_priority_for_image(model):
    assert resolve_paygo_bundle(True, {"paygo_tier": "auto"}, model_name=model) == ({}, None, [])


def test_auto_tier_keeps_priority_for_text_and_explicit_tier_for_image():
    headers, _, _ = resolve_paygo_bundle(True, {"paygo_tier": "auto"}, model_name="gemini-4-argon")
    assert headers.get("X-Vertex-AI-LLM-Shared-Request-Type") == "priority"
    headers, _, _ = resolve_paygo_bundle(True, {"paygo_tier": "priority"}, model_name="gemini-nano-banana-2.1")
    assert headers.get("X-Vertex-AI-LLM-Shared-Request-Type") == "priority"


def test_auto_tier_image_cost_billed_as_standard():
    from model_pricing import estimate_cost
    assert estimate_cost("gemini-3.1-flash-image", 1_000_000, 0, tier="auto") == \
        estimate_cost("gemini-3.1-flash-image", 1_000_000, 0, tier="standard")
    assert estimate_cost("gemini-3.1-flash-lite", 1_000_000, 0, tier="auto") == \
        estimate_cost("gemini-3.1-flash-lite", 1_000_000, 0, tier="priority")


def test_fallback_catalog_matches_vertex_models():
    import json
    from pathlib import Path
    import model_loader as ml
    catalog = json.loads((Path(__file__).resolve().parent.parent / "vertexModels.json").read_text(encoding="utf-8"))
    assert catalog["models"] == ml._DEFAULT_FALLBACK_MODELS
    assert "gemini-4-argon" in catalog["models"] and "gemini-nano-banana-2.1" in catalog["models"]
    assert not any(m.startswith("gemini-2.5") for m in catalog["models"])
