"""防截断合成传输协议测试：注入 / 启用判定 / 非流式解构 / 流式剥离 / 工具名生成。

不触碰真实网络：构造 pydantic 请求对象与模拟 SDK chunk（google.genai types）。
"""
import json

import pytest
from google.genai import types

from anti_truncation import (
    generate_synthetic_tool_name, build_synthetic_tool, build_control_message,
    inject_request, is_enabled_for_request, extract_content_from_args,
    strip_synthetic_from_openai_dict, strip_synthetic_from_stream_chunk,
    TOOL_PREFIX,
)
from models import OpenAIRequest, OpenAIMessage


def _req(**kw):
    defaults = {"model": "gemini-3.6-flash", "messages": [{"role": "user", "content": "hi"}]}
    defaults.update(kw)
    return OpenAIRequest(**defaults)


class TestToolName:
    def test_unique_and_prefixed(self):
        names = {generate_synthetic_tool_name() for _ in range(50)}
        assert len(names) == 50
        for n in names:
            assert n.startswith(TOOL_PREFIX)

    def test_avoids_existing_names(self):
        n = generate_synthetic_tool_name(["v2o_emit_abc", "weather_api"])
        assert n not in ("v2o_emit_abc", "weather_api")


class TestBuilders:
    def test_synthetic_tool_structure(self):
        t = build_synthetic_tool("v2o_emit_x")
        assert t["type"] == "function"
        fn = t["function"]
        assert fn["name"] == "v2o_emit_x"
        assert fn["parameters"]["required"] == ["content"]

    def test_control_message_is_user(self):
        m = build_control_message("v2o_emit_x")
        assert m["role"] == "user"
        assert "v2o_emit_x" in m["content"]


class TestInjectRequest:
    def test_appends_tool_and_control_message(self):
        req = _req()
        new_req, tool_name = inject_request(req)
        assert new_req is not req
        assert new_req.tools[-1]["function"]["name"] == tool_name
        assert new_req.messages[-1].role == "user"
        assert tool_name in new_req.messages[-1].content
        # 原请求不被污染
        assert req.tools is None
        assert len(req.messages) == 1

    def test_tool_choice_none_overridden_to_auto(self):
        req = _req(tool_choice="none")
        new_req, _ = inject_request(req)
        assert new_req.tool_choice == "auto"

    def test_tool_choice_specific_function_kept(self):
        req = _req(tool_choice={"type": "function", "function": {"name": "weather_api"}})
        new_req, _ = inject_request(req)
        assert new_req.tool_choice == req.tool_choice

    def test_existing_tools_preserved(self):
        req = _req(tools=[{"type": "function", "function": {"name": "weather_api"}}])
        new_req, tool_name = inject_request(req)
        assert len(new_req.tools) == 2
        assert new_req.tools[0]["function"]["name"] == "weather_api"
        assert tool_name != "weather_api"


class TestExtractContent:
    def test_dict(self):
        assert extract_content_from_args({"content": "回答"}) == "回答"

    def test_json_string(self):
        assert extract_content_from_args('{"content": "回答"}') == "回答"

    def test_missing_or_empty(self):
        assert extract_content_from_args({}) is None
        assert extract_content_from_args({"content": ""}) is None
        assert extract_content_from_args(None) is None
        assert extract_content_from_args("not json") is None


class TestStripOpenaiDict:
    def _msg_with_calls(self, calls, finish_reason="tool_calls"):
        return {
            "message": {"role": "assistant", "content": None, "tool_calls": calls},
            "finish_reason": finish_reason,
        }

    def test_mixed_real_and_synthetic(self):
        syn_name = "v2o_emit_x"
        calls = [
            {"index": 0, "type": "function", "function": {"name": "weather_api", "arguments": '{"city":"北京"}'}},
            {"index": 1, "type": "function", "function": {"name": syn_name, "arguments": '{"content":"完整回答正文"}'}},
        ]
        d = {"choices": [self._msg_with_calls(calls)]}
        out = strip_synthetic_from_openai_dict(d, syn_name)
        msg = out["choices"][0]["message"]
        assert msg["content"] == "完整回答正文"
        assert len(msg["tool_calls"]) == 1
        assert msg["tool_calls"][0]["function"]["name"] == "weather_api"
        assert msg["tool_calls"][0]["index"] == 0
        assert out["choices"][0]["finish_reason"] == "tool_calls"

    def test_only_synthetic(self):
        syn_name = "v2o_emit_x"
        calls = [
            {"index": 0, "type": "function", "function": {"name": syn_name, "arguments": {"content": "纯合成回答"}}},
        ]
        d = {"choices": [self._msg_with_calls(calls)]}
        out = strip_synthetic_from_openai_dict(d, syn_name)
        msg = out["choices"][0]["message"]
        assert msg["content"] == "纯合成回答"
        assert "tool_calls" not in msg
        assert out["choices"][0]["finish_reason"] == "stop"

    def test_no_synthetic_untouched(self):
        syn_name = "v2o_emit_x"
        calls = [
            {"index": 0, "type": "function", "function": {"name": "weather_api", "arguments": "{}"}},
        ]
        d = {"choices": [self._msg_with_calls(calls)]}
        out = strip_synthetic_from_openai_dict(d, syn_name)
        assert out["choices"][0]["message"]["tool_calls"] == calls
        assert out["choices"][0]["finish_reason"] == "tool_calls"

    def test_no_tool_name_untouched(self):
        calls = [{"index": 0, "type": "function", "function": {"name": "x", "arguments": "{}"}}]
        d = {"choices": [self._msg_with_calls(calls)]}
        out = strip_synthetic_from_openai_dict(d, None)
        assert out is d

    def test_synthetic_call_with_empty_content_not_leaked(self):
        """模型调用合成工具但 content 为空：合成调用必须被剥离，不得泄漏给客户端。"""
        syn_name = "v2o_emit_x"
        calls = [
            {"index": 0, "type": "function", "function": {"name": syn_name, "arguments": {"content": ""}}},
        ]
        d = {"choices": [self._msg_with_calls(calls)]}
        out = strip_synthetic_from_openai_dict(d, syn_name)
        msg = out["choices"][0]["message"]
        assert "tool_calls" not in msg
        assert out["choices"][0]["finish_reason"] == "stop"


class TestStripStreamChunk:
    def _chunk(self, parts):
        return types.GenerateContentResponse(candidates=[types.Candidate(content=types.Content(parts=parts))])

    def test_synthetic_part_removed_content_extracted(self):
        syn_name = "v2o_emit_x"
        chunk = self._chunk([
            types.Part(function_call=types.FunctionCall(name=syn_name, args={"content": "流式正文"})),
            types.Part(function_call=types.FunctionCall(name="weather_api", args={"city": "上海"})),
        ])
        stripped, contents = strip_synthetic_from_stream_chunk(chunk, 0, syn_name)
        assert contents == ["流式正文"]
        assert stripped is not None
        fc_names = [p.function_call.name for p in stripped.candidates[0].content.parts]
        assert fc_names == ["weather_api"]

    def test_only_synthetic_returns_none(self):
        syn_name = "v2o_emit_x"
        chunk = self._chunk([
            types.Part(function_call=types.FunctionCall(name=syn_name, args={"content": "全部内容"})),
        ])
        stripped, contents = strip_synthetic_from_stream_chunk(chunk, 0, syn_name)
        assert stripped is None
        assert contents == ["全部内容"]

    def test_no_synthetic_untouched(self):
        chunk = self._chunk([types.Part(function_call=types.FunctionCall(name="weather_api", args={}))])
        stripped, contents = strip_synthetic_from_stream_chunk(chunk, 0, "v2o_emit_x")
        assert stripped is chunk
        assert contents == []

    def test_no_tool_name_untouched(self):
        chunk = self._chunk([types.Part(function_call=types.FunctionCall(name="v2o_emit_x", args={}))])
        stripped, contents = strip_synthetic_from_stream_chunk(chunk, 0, None)
        assert stripped is chunk
        assert contents == []


class TestEnabledField:
    def test_true_variants(self):
        for v in (True, "true", "True"):
            req = _req(**{"anti_truncation": v})
            assert is_enabled_for_request(req) is True

    def test_false_or_missing(self):
        assert is_enabled_for_request(_req()) is False
        assert is_enabled_for_request(_req(**{"anti_truncation": False})) is False
        assert is_enabled_for_request(_req(**{"anti_truncation": "false"})) is False

    def test_custom_field_name(self):
        req = _req(**{"my_field": True})
        assert is_enabled_for_request(req, {"anti_truncation_field": "my_field"}) is True
        assert is_enabled_for_request(req) is False


class TestPersistence:
    def test_field_saved_and_reloaded(self, tmp_path, monkeypatch):
        """控制台新增设置必须落盘：update_settings 接受 → 重启后读回（模拟持久化）。"""
        import runtime_state
        monkeypatch.setattr(runtime_state, "STATE_FILE", str(tmp_path / "web_state.json"))
        st = runtime_state.AppState()
        assert st.get_setting("anti_truncation_field") == "anti_truncation"
        st.update_settings({"anti_truncation_field": "my_custom_field"})
        assert st.get_setting("anti_truncation_field") == "my_custom_field"
        # 新实例（模拟容器重启）从磁盘读回
        st2 = runtime_state.AppState()
        assert st2.get_setting("anti_truncation_field") == "my_custom_field"
        # 未知键不会被接受
        st2.update_settings({"bogus_field": "x"})
        assert st2.get_setting("bogus_field", "fallback") == "fallback"


class TestFakeStreamCompatibility:
    """防截断 + fake- 假流式组合：假流式路径的解构必须生效（fake 管传输方式，
    防截断管生成方式，两者可叠加且互不破坏）。"""

    async def test_fake_stream_with_anti_truncation(self):
        import json
        from api_helpers import gemini_fake_stream_generator
        from models import OpenAIRequest

        tool_name = "v2o_emit_x"
        resp = types.GenerateContentResponse(candidates=[types.Candidate(
            content=types.Content(parts=[types.Part(
                function_call=types.FunctionCall(
                    name=tool_name, args={"content": "假流式下的完整回答"}))],
            )
        )])

        class FakeModels:
            async def generate_content(self, model, contents, config):
                return resp

        class FakeAio:
            models = FakeModels()

        class FakeClient:
            aio = FakeAio()

        req = OpenAIRequest(model="fake-gemini-3.6-flash",
                            messages=[{"role": "user", "content": "hi"}], stream=True)
        chunks = []
        async for sse in gemini_fake_stream_generator(
            FakeClient(), "gemini-3.6-flash", [], {}, req, False,
            synthetic_tool_name=tool_name,
        ):
            chunks.append(sse)

        contents = []
        tool_names = []
        for sse in chunks:
            if not sse.startswith("data: "):
                continue
            try:
                payload = json.loads(sse[len("data: "):].strip())
            except Exception:
                continue
            for choice in payload.get("choices", []):
                delta = choice.get("delta") or {}
                if delta.get("content"):
                    contents.append(delta["content"])
                for tc in (delta.get("tool_calls") or []):
                    fn = tc.get("function") or {}
                    if fn.get("name"):
                        tool_names.append(fn["name"])
        assert "".join(contents) == "假流式下的完整回答"
        assert tool_names == []          # 合成工具调用被剥离，不泄漏
