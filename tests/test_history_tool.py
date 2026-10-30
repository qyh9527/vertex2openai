import pytest

from models import OpenAIRequest
from history_tool import carry_history, history_tool_active, validate_history_tool


def request(messages, tools=None):
    return OpenAIRequest(model="fake-gemini-3.8-flash", messages=messages, tools=tools)


@pytest.fixture(autouse=True)
def restore_settings():
    from runtime_state import app_state
    before = app_state.get_settings()
    yield
    app_state.update_settings(before)


def test_modes():
    for mode, fake, expected in [("off", True, False), ("fake_stream_only", True, True),
                                 ("fake_stream_only", False, False), ("all", False, True)]:
        assert history_tool_active({"history_tool_mode": mode}, fake) == expected


def test_first_system_latest_user_and_prefill_kept():
    req = request([{"role":"system","content":"first"}, {"role":"user","content":"old"},
                   {"role":"system","content":"middle rule"}, {"role":"assistant","content":"reply"},
                   {"role":"user","content":"continue"}, {"role":"assistant","content":"<planning>"}])
    original = req.model_dump()
    result = carry_history(req)
    assert [m.role for m in result.messages] == ["system", "user", "assistant"]
    assert result.messages[0] is req.messages[0]
    assert result.messages[-1] is req.messages[-1]
    assert result.messages[1].content.endswith("continue")
    description = result.tools[-1]["function"]["description"]
    assert "[system]\nmiddle rule" in description
    assert "[user]\nold" in description
    assert "first" not in description
    assert req.model_dump() == original


def test_real_tool_suffix_preserved():
    req = request([{"role":"system","content":"first"}, {"role":"user","content":"old"},
        {"role":"assistant","content":None,"tool_calls":[{"id":"c1","type":"function","function":{"name":"weather","arguments":"{}"}}]},
        {"role":"tool","tool_call_id":"c1","content":"sunny"},
        {"role":"assistant","content":"later"}, {"role":"user","content":"next"}])
    result = carry_history(req)
    assert result.messages[1:4] == req.messages[2:5]
    assert result.messages[-1].content.endswith("next")
    assert "sunny" not in result.tools[-1]["function"]["description"]


def test_collision_does_not_replace_client_tool():
    tools = [{"type":"function","function":{"name":"context","parameters":{"type":"object","properties":{}}}}]
    req = request([{"role":"user","content":"old"}, {"role":"assistant","content":"reply"}, {"role":"user","content":"next"}], tools)
    result = carry_history(req)
    assert result.tools[0] == tools[0]
    assert result.tools[-1]["function"]["name"] == "context_1"
    assert "context_1" in result.messages[-1].content


@pytest.mark.parametrize("part", [{"type":"image_url","image_url":{"url":"https://invalid.test/a.png"}},
                                    {"type":"file","file":{"file_id":"example"}}])
def test_reject_media_even_in_retained_messages(part):
    req = request([{"role":"system","content":"rule"}, {"role":"user","content":[part]}])
    with pytest.raises(ValueError, match="不支持"):
        validate_history_tool(req)


def test_no_history_is_noop():
    req = request([{"role":"system","content":"rule"}, {"role":"user","content":"next"}])
    assert carry_history(req) is req


def test_relay_then_history():
    from input_relay import InputRelayConfig, apply_input_relay
    req = request([{"role":"system","content":"rule"}, {"role":"assistant","content":"previous"},
                   {"role":"user","content":"<advice>new action</advice>"}])
    messages, _ = apply_input_relay(req.messages, InputRelayConfig("advice", "continue", "fake_stream_only"))
    result = carry_history(req.model_copy(update={"messages":messages}))
    assert "new action" in result.tools[-1]["function"]["description"]
    assert result.messages[-1].content.endswith("continue")


async def test_pipeline_top_injection_after_history_and_auto_transport(monkeypatch):
    from fastapi.responses import JSONResponse
    from runtime_state import app_state
    from upstreams import express_sdk
    app_state.update_settings({"history_tool_mode":"all", "input_relay_mode":"off",
        "top_input_injection_mode":"always", "top_input_injection_random":"off",
        "top_input_injection_fake_only":False,
        "top_input_injection_selected_plan_id":"p",
        "top_input_injection_plans":[{"id":"p","name":"test","role":"system","content":"TOP_ONLY"}]})
    captured = {}
    upstream = express_sdk.ExpressSDKUpstream()
    monkeypatch.setattr(upstream, "_resolve_client", lambda *a: {"client":object(), "model_to_call":"gemini-3.8-flash", "priority_paygo":False})
    monkeypatch.setattr(express_sdk, "_log_resolved_endpoint", lambda *a, **k: None)
    async def execute(client, model, prompt, config, req, **kwargs):
        captured["request"] = req
        captured["name"] = kwargs["synthetic_tool_name"]
        return JSONResponse({"ok":True})
    monkeypatch.setattr(express_sdk, "execute_gemini_call", execute)
    req = request([{"role":"system","content":"ORIGINAL"}, {"role":"assistant","content":"HISTORY"}, {"role":"user","content":"next"}])
    response = await upstream.chat_completions(req, None)
    assert response.status_code == 200
    output = captured["request"]
    assert captured["name"] == "sanbox_rp_content"
    description = output.tools[1]["function"]["description"]
    assert "HISTORY" in description
    assert "ORIGINAL" in description
    assert "TOP_ONLY" not in description
    assert output.messages[0].content == "TOP_ONLY"
    assert output.messages[-1].role == "tool"


async def test_media_rejected_before_client_resolution(monkeypatch):
    from runtime_state import app_state
    from upstreams.express_sdk import ExpressSDKUpstream
    app_state.update_settings({"history_tool_mode":"all"})
    upstream = ExpressSDKUpstream()
    def forbidden(*a):
        pytest.fail("must reject before client resolution")
    monkeypatch.setattr(upstream, "_resolve_client", forbidden)
    req = request([{"role":"user","content":[{"type":"image_url","image_url":{"url":"https://invalid.test/p.png"}}]}])
    response = await upstream.chat_completions(req, None)
    assert response.status_code == 400
