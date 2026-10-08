import json
from pathlib import Path

import pytest

import model_capabilities as mc
import model_loader as ml
from models import OpenAIRequest
from protocol_conversion import create_generation_config
from runtime_state import app_state
from upstreams.express_sdk import _build_thinking_config


@pytest.mark.parametrize("model", ["gemini-3.8-flash", "fake-gemini-3.8-flash-search", "models/gemini-3.8-flash", "projects/p/locations/global/publishers/google/models/gemini-3.8-flash"])
def test_hard_sampling_limits(model, monkeypatch):
    monkeypatch.setattr(app_state, "get_effective_settings", lambda _: {"sampling_policy": "allowed"})
    request = OpenAIRequest(model=model, messages=[], temperature=0.5, top_p=0.8, top_k=10, n=2)
    config = create_generation_config(request)
    assert not ({"temperature", "top_p", "top_k", "candidate_count"} & config.keys())
    assert not ({"temperature", "top_p", "top_k", "candidate_count"} & set(mc.capabilities_summary(model, {"sampling_policy": "allowed"})["sampling"]))


@pytest.mark.parametrize("fields,settings,expected", [
    ({"thinking_budget": 0}, {}, "low"),
    ({"thinking_budget": 0, "reasoning_effort": "invalid"}, {}, "low"),
    ({"thinking_budget": 0, "reasoning_effort": "high"}, {}, "high"),
    ({"thinking_budget": 0}, {"native_thinking_mode": "console", "thinking_g3_level": "high"}, "high"),
    ({"reasoning_effort": "off"}, {}, "low"),
    ({"reasoning_effort": "minimal"}, {}, "low"),
])
def test_sdk_thinking(fields, settings, expected, monkeypatch):
    monkeypatch.setattr(app_state, "get_effective_settings", lambda _: settings)
    request = OpenAIRequest(model="gemini-3.8-flash", messages=[], **fields)
    config = _build_thinking_config(request.model, request, False)
    assert config["thinking_level"] == expected
    assert "thinking_budget" not in config


@pytest.mark.parametrize("model,stripped", [
    ("gemini-3.7-flash", True), ("gemini-3.6-flash", True), ("fake-gemini-3.8-flash", True),
    ("gemini-3.5-flash", False), ("gemini-2.5-flash", False),
])
@pytest.mark.parametrize("policy", ["auto", "allowed"])
def test_penalty_stripped_from_36(model, stripped, policy, monkeypatch):
    monkeypatch.setattr(app_state, "get_effective_settings", lambda _: {"sampling_policy": policy})
    request = OpenAIRequest(model=model, messages=[], presence_penalty=0.5, frequency_penalty=0.3)
    keys = {"presence_penalty", "frequency_penalty"} & create_generation_config(request).keys()
    assert keys == (set() if stripped else {"presence_penalty", "frequency_penalty"})


def test_25_controls_unchanged(monkeypatch):
    monkeypatch.setattr(app_state, "get_effective_settings", lambda _: {})
    request = OpenAIRequest(model="gemini-2.5-flash", messages=[], temperature=0.5, top_p=0.8, top_k=10, n=2, thinking_budget=0)
    config = create_generation_config(request)
    assert config["candidate_count"] == 2
    assert config["temperature"] == 0.5
    assert _build_thinking_config(request.model, request, False) == {"thinking_budget": 0, "include_thoughts": False}


def test_38_catalog():
    catalog = json.loads((Path(__file__).resolve().parent.parent / "vertexModels.json").read_text(encoding="utf-8"))
    assert "gemini-3.8-flash" in catalog["models"]
    assert "gemini-3.8-flash" in ml._DEFAULT_FALLBACK_MODELS


async def test_old_disk_models_preserved_with_custom_38(tmp_path, monkeypatch):
    from routes.models_api import list_models

    disk = tmp_path / "models.json"
    disk.write_text('{"models":["gemini-old-custom"]}', encoding="utf-8")
    monkeypatch.setattr(ml, "_MODELS_DISK_FILE", str(disk))
    monkeypatch.setattr(ml, "_model_cache", None)
    monkeypatch.setattr(app_state, "get_custom_models", lambda: ["gemini-3.8-flash"])
    result = await list_models(fastapi_request=object(), api_key="test")
    ids = {model["id"] for model in result["data"]}
    assert {"gemini-old-custom", "gemini-3.8-flash"} <= ids
    assert json.loads(disk.read_text()) == {"models": ["gemini-old-custom"]}


@pytest.mark.parametrize("channel", ["express", "vertex"])
async def test_sdk_actual_outbound_controls(channel, monkeypatch):
    from types import SimpleNamespace
    from google.genai import types
    import upstreams.express_sdk as sdk
    from upstreams.service_account import ServiceAccountUpstream

    captured = {}

    async def generate_content(**kwargs):
        captured.update(kwargs)
        return types.GenerateContentResponse(candidates=[types.Candidate(content=types.Content(role="model", parts=[types.Part(text="ok")]), finish_reason="STOP")])

    client = SimpleNamespace(aio=SimpleNamespace(models=SimpleNamespace(generate_content=generate_content)))
    upstream = sdk.ExpressSDKUpstream() if channel == "express" else ServiceAccountUpstream()
    monkeypatch.setattr(upstream, "_resolve_client", lambda *args: {"client": client, "model_to_call": "gemini-3.8-flash", "priority_paygo": False})
    monkeypatch.setattr(app_state, "get_effective_settings", lambda _: {"sampling_policy": "allowed", "native_thinking_mode": "request"})
    request = OpenAIRequest(model="gemini-3.8-flash", messages=[{"role": "user", "content": "hi"}], temperature=0.5, top_p=0.8, top_k=10, n=2, thinking_budget=0, compatibility_report=True)
    response = await upstream.chat_completions(request, None)
    assert response.status_code == 200
    config = captured["config"]
    assert not ({"temperature", "top_p", "top_k", "candidate_count"} & config.keys())
    assert config["thinking_config"]["thinking_level"] == "low"
    entries = json.loads(response.body)["choices"][0]["message"]["extra_content"]["vertex2openai"]["conversion_report"]
    assert {"generation_config.temperature", "generation_config.top_p", "generation_config.top_k", "generation_config.candidate_count", "thinking_budget"} <= {e["path"] for e in entries}
    assert {e["channel"] for e in entries} == {channel}
