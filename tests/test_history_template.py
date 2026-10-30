import copy
import json

import pytest

from models import OpenAIRequest
from history_template import default_template, parse_template, inject_template_history, validate_template_request
DEFAULT_TEMPLATE = default_template()
from anti_truncation import restore_synthetic_args, strip_synthetic_from_openai_dict
from outcome import GenerationEndError


def make_request(**kwargs):
    return OpenAIRequest(model="fake-gemini-3.8-flash", messages=[
        {"role": "system", "content": "RULE"},
        {"role": "assistant", "content": "HISTORY"},
        {"role": "user", "content": "LATEST"},
        {"role": "assistant", "content": "PREFILL"},
    ], **kwargs)


def test_template_roundtrip():
    assert parse_template(json.dumps(DEFAULT_TEMPLATE)) == DEFAULT_TEMPLATE
    fields = copy.deepcopy(DEFAULT_TEMPLATE)
    fields[0]["required"] = False
    assert parse_template(fields)[0]["required"] is False


@pytest.mark.parametrize("value", [[], None, "bad", [{}],
    [{"name": "content", "required": False}],
    [{"name": "content", "required": True, "tag": "<content>"}],
    [{"name": "content", "required": True}] * 2])
def test_invalid_template(value):
    with pytest.raises(ValueError):
        parse_template(value)


def test_eight_messages_and_original_input():
    request = make_request()
    before = request.model_dump()
    result, name = inject_template_history(request, DEFAULT_TEMPLATE, "CONSOLE_PREFILL")
    assert request.model_dump() == before
    assert len(result.messages) == 8
    assert [m.role for m in result.messages] == ["user", "assistant", "assistant", "tool"] * 2
    description = result.tools[1]["function"]["description"]
    assert all(s in description for s in ("RULE", "HISTORY", "LATEST"))
    assert "PREFILL" not in description
    assert all(s in result.messages[-1].content for s in ("LATEST", "PREFILL", "CONSOLE_PREFILL"))
    assert result.tool_choice["function"]["name"] == name == "sanbox_rp_content"
    schema = result.tools[0]["function"]["parameters"]
    assert schema["required"] == ["planning", "content"]
    assert schema["propertyOrdering"] == ["planning", "content", "extra"]
    from message_processing import create_gemini_prompt
    prompt = create_gemini_prompt(result.messages, "gemini-3.8-flash")
    calls = [p for m in prompt for p in m.parts if p.function_call]
    assert len(calls) == 2
    assert all(p.thought_signature == b"skip_thought_signature_validator" for p in calls)


def test_system_after_latest_user_is_context_not_prefill():
    from models import OpenAIMessage
    request = make_request()
    request.messages.insert(-1, OpenAIMessage(role="system", content="TAIL_RULE"))
    result, _ = inject_template_history(request, DEFAULT_TEMPLATE)
    assert "TAIL_RULE" in result.tools[1]["function"]["description"]
    assert "PREFILL" in result.messages[-1].content
    assert "TAIL_RULE" not in result.messages[-1].content


@pytest.mark.parametrize("kwargs", [{"tools": [{"type": "function", "function": {"name": "weather"}}]}])
def test_reject_client_tools(kwargs):
    with pytest.raises(ValueError, match="工具声明"):
        validate_template_request(make_request(**kwargs))


def test_reject_tool_history():
    request = make_request()
    request.messages[1] = request.messages[1].model_copy(update={"tool_calls": [{"id": "c", "function": {"name": "weather"}}]})
    with pytest.raises(ValueError, match="工具调用历史"):
        validate_template_request(request)


def test_restore_order_and_no_duplicate_tags():
    result = restore_synthetic_args({"planning": "PLAN", "content": "<content>BODY<tucao>X</tucao></content>", "extra": "<edit>Y</edit>"}, DEFAULT_TEMPLATE)
    assert result.structurally_complete
    assert result.text == "<konatan_planning~>\nPLAN\n</konatan_planning~>\n<content>BODY<tucao>X</tucao></content>\n<edit>Y</edit>"


@pytest.mark.parametrize("args", [{"content": "body"}, {"planning": "p", "content": " "},
    {"planning": "p", "content": "body", "extra": 1},
    '{"planning":"p","content":"body",',
    '{"planning":"p","content":"one","content":"two"}',
    {"planning": "p", "content": "body", "unknown": "x"}])
def test_strict_integrity(args):
    assert not restore_synthetic_args(args, DEFAULT_TEMPLATE).structurally_complete


@pytest.mark.parametrize("fake", [False, True])
@pytest.mark.parametrize("valid", [False, True])
async def test_response_pipeline(fake, valid):
    from api_helpers import execute_gemini_call
    from types import SimpleNamespace
    from google.genai import types
    args = {"content": "BODY"}
    if valid:
        args["planning"] = "PLAN"
    async def generate(**kwargs):
        return types.GenerateContentResponse(candidates=[types.Candidate(index=0, finish_reason="STOP",
            content=types.Content(role="model", parts=[types.Part(function_call=types.FunctionCall(
                name="sanbox_rp_content", args=args))]))])
    client = SimpleNamespace(aio=SimpleNamespace(models=SimpleNamespace(generate_content=generate)))
    request = make_request(stream=fake)
    response = await execute_gemini_call(client, "gemini-3.8-flash", lambda _: [], {}, request,
        synthetic_tool_name="sanbox_rp_content", force_fake_streaming=True, template_fields=DEFAULT_TEMPLATE)
    if fake:
        wire = "".join([item async for item in response.body_iterator])
        assert ('"error"' in wire) is not valid
        assert ('"finish_reason": "stop"' in wire) is valid
        if valid:
            text = "".join(choice.get("delta", {}).get("content", "")
                for line in wire.splitlines() if line.startswith("data: {")
                for choice in json.loads(line[6:]).get("choices", []))
            assert "BODY" in text and "konatan_planning~" in text
    else:
        assert response.status_code == (200 if valid else 502)
        if valid:
            assert "<content>" in json.loads(response.body)["choices"][0]["message"]["content"]


@pytest.mark.parametrize("fake", [False, True])
async def test_upstream_scope_and_guidance(fake, monkeypatch):
    from fastapi.responses import JSONResponse
    from runtime_state import app_state
    from upstreams import express_sdk
    monkeypatch.setattr(app_state, "get_effective_settings", lambda _: {
        "history_tool_mode": "all", "input_relay_mode": "always", "input_relay_tag": "advice",
        "input_relay_placeholder": "PLACEHOLDER", "inject_prefill": "CONSOLE_GUIDE"})
    upstream = express_sdk.ExpressSDKUpstream()
    monkeypatch.setattr(upstream, "_resolve_client", lambda *a: {"client": object(), "model_to_call": "gemini-3.8-flash", "priority_paygo": False})
    monkeypatch.setattr(express_sdk, "_log_resolved_endpoint", lambda *a, **kw: None)
    captured = {}
    async def execute(*args, **kwargs):
        captured["req"] = args[4]
        captured.update(kwargs)
        return JSONResponse({"ok": True})
    monkeypatch.setattr(express_sdk, "execute_gemini_call", execute)
    req = make_request()
    req.model = "fake-gemini-3.8-flash" if fake else "gemini-3.8-flash"
    req.messages[2].content = "<advice>LATEST</advice>"
    before = req.model_dump()
    assert (await upstream.chat_completions(req, None)).status_code == 200
    assert req.model_dump() == before
    assert (captured["template_fields"] is not None) is fake
    if fake:
        assert captured["prefill_text"] == ""
        tail = captured["req"].messages[-1].content
        assert "LATEST" in tail and "CONSOLE_GUIDE" in tail and "PLACEHOLDER" not in tail
    else:
        assert any(m.role == "system" and m.content == "RULE" for m in captured["req"].messages)


async def test_invalid_template_settings_rejected_atomically():
    from main import app, require_auth
    from runtime_state import app_state
    from httpx import AsyncClient, ASGITransport
    old = app_state.get_settings()
    app.dependency_overrides[require_auth] = lambda: True
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post("/api/settings", json={"history_tool_template": "broken", "history_tool_mode": "all"})
            assert response.status_code == 400
            assert app_state.get_settings() == old
            response = await client.post("/api/settings", json={"history_tool_template": json.dumps(DEFAULT_TEMPLATE)})
            assert response.status_code == 200
            assert response.json()["history_tool_template"] == DEFAULT_TEMPLATE
    finally:
        app.dependency_overrides.pop(require_auth, None)
        app_state.update_settings(old)


def test_strip_and_reject_missing_call():
    payload = {"choices": [{"index": 0, "finish_reason": "tool_calls", "message": {"role": "assistant", "tool_calls": [
        {"function": {"name": "sanbox_rp_content", "arguments": json.dumps({"planning": "p", "content": "body"})}}
    ]}}]}
    result = strip_synthetic_from_openai_dict(payload, "sanbox_rp_content", template_fields=DEFAULT_TEMPLATE)
    assert result["choices"][0]["finish_reason"] == "stop"
    assert "tool_calls" not in result["choices"][0]["message"]
    with pytest.raises(GenerationEndError):
        strip_synthetic_from_openai_dict({"choices": [{"message": {"content": "plain"}}]}, "sanbox_rp_content", template_fields=DEFAULT_TEMPLATE)
