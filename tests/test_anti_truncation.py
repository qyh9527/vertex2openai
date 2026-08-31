"""防截断合成传输协议测试：注入 / 启用判定 / 非流式解构 / 流式剥离 / 工具名生成。

不触碰真实网络：构造 pydantic 请求对象与模拟 SDK chunk（google.genai types）。
"""
import json

import pytest
from google.genai import types

from anti_truncation import (
    generate_synthetic_tool_name, build_synthetic_tool, build_control_message,
    inject_request, is_enabled_for_request, extract_content_from_args,
    has_synthetic_tool_call, strip_synthetic_from_openai_dict, strip_synthetic_from_stream_chunk,
    TOOL_PREFIX,
)
from fastapi.responses import StreamingResponse
from google.genai import types
from message_processing import create_gemini_prompt
from models import OpenAIRequest, OpenAIMessage


class FakeModels:
    def __init__(self, owner):
        self.owner = owner

    async def generate_content_stream(self, model, contents, config):
        async def _gen():
            for chunk in self.owner.stream_chunks:
                yield chunk
        return _gen()


class FakeAio:
    def __init__(self, owner):
        self.models = FakeModels(owner)


class FakeClient:
    def __init__(self, stream_chunks=()):
        self.aio = FakeAio(self)
        self.stream_chunks = list(stream_chunks)


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
        # strict 约束：禁止模型在参数里加额外字段
        assert fn["parameters"]["additionalProperties"] is False

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

    def test_markdown_code_fences_stripped(self):
        """模型把参数包进 ```json 围栏：剥围栏后仍能提取。"""
        assert extract_content_from_args(
            '```json\n{"content": "围栏里的回答"}\n```') == "围栏里的回答"

    def test_nested_content_fallback(self):
        """content 被塞进嵌套对象（shape 写错）：递归兜底提取。"""
        assert extract_content_from_args(
            {"result": {"content": "嵌套里的回答"}}) == "嵌套里的回答"
        assert extract_content_from_args(
            {"a": {"b": {"c": {"d": {"content": "深层回答"}}}}}) == "深层回答"

    def test_nested_depth_bounded(self):
        """递归兜底有界（depth ≤ 4）：过深嵌套不无限下钻。"""
        deep = {"content": "太深了"}
        for _ in range(8):
            deep = {"x": deep}
        assert extract_content_from_args(deep) is None

    def test_non_string_content_rejected(self):
        assert extract_content_from_args({"content": 123}) is None
        assert extract_content_from_args({"content": {"text": "x"}}) is None


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


class TestHasSyntheticToolCall:
    """解构前检测合成工具是否被调用（用于"防截断未生效"提示）。"""

    def _msg_with_calls(self, calls, finish_reason="tool_calls"):
        return {"message": {"role": "assistant", "content": None, "tool_calls": calls},
                "finish_reason": finish_reason}

    def test_present(self):
        syn = "v2o_emit_x"
        d = {"choices": [self._msg_with_calls([
            {"index": 0, "type": "function", "function": {"name": syn, "arguments": '{"content":"x"}'}},
        ])]}
        assert has_synthetic_tool_call(d, syn) is True

    def test_mixed_present(self):
        syn = "v2o_emit_x"
        d = {"choices": [self._msg_with_calls([
            {"index": 0, "type": "function", "function": {"name": "weather_api", "arguments": "{}"}},
            {"index": 1, "type": "function", "function": {"name": syn, "arguments": '{"content":"y"}'}},
        ])]}
        assert has_synthetic_tool_call(d, syn) is True

    def test_absent(self):
        syn = "v2o_emit_x"
        d = {"choices": [self._msg_with_calls([
            {"index": 0, "type": "function", "function": {"name": "weather_api", "arguments": "{}"}},
        ])]}
        assert has_synthetic_tool_call(d, syn) is False

    def test_no_tool_calls(self):
        d = {"choices": [{"message": {"role": "assistant", "content": "普通输出"},
                          "finish_reason": "stop"}]}
        assert has_synthetic_tool_call(d, "v2o_emit_x") is False

    def test_empty_or_none_tool_name(self):
        d = {"choices": [self._msg_with_calls([
            {"index": 0, "type": "function", "function": {"name": "v2o_emit_x", "arguments": "{}"}},
        ])]}
        assert has_synthetic_tool_call(d, None) is False
        assert has_synthetic_tool_call(d, "") is False

    def test_not_dict(self):
        assert has_synthetic_tool_call(None, "v2o_emit_x") is False


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


class TestStreamSideBuffer:
    """真流式 side-buffer（参考 Antigravity-gateway stream.go）：
    - 合成调用出现前的普通文本先入缓冲，命中合成调用即丢弃（单来源原则）；
    - 全程未命中合成调用则流末 flush 兜底（防截断未生效时正文不丢）；
    - 未启用防截断（无 synthetic_tool_name）时零行为变化，正文直接透传。
    """

    def _make_stream(self, stream_chunks):
        from api_helpers import execute_gemini_call
        req = OpenAIRequest(model="gemini-3.6-flash",
                            messages=[{"role": "user", "content": "hi"}], stream=True)
        client = FakeClient(stream_chunks=stream_chunks)
        prompt = create_gemini_prompt(req.messages)
        return client, req, prompt

    async def _run(self, client, req, prompt, synthetic_tool_name):
        from api_helpers import execute_gemini_call
        resp = await execute_gemini_call(
            client, "gemini-3.6-flash", lambda m: prompt, {}, req,
            synthetic_tool_name=synthetic_tool_name)
        assert isinstance(resp, StreamingResponse)
        return "".join([c async for c in resp.body_iterator])

    def _collect_content(self, sse_text):
        out = []
        for line in sse_text.splitlines():
            if not line.startswith("data: "):
                continue
            payload = line[len("data: "):].strip()
            if not payload or payload == "[DONE]":
                continue
            for choice in json.loads(payload).get("choices", []):
                c = (choice.get("delta") or {}).get("content")
                if c:
                    out.append(c)
        return "".join(out)

    async def test_preamble_text_dropped_on_synthetic_hit(self):
        """模型先吐几个字再调合成工具：普通文本必须被丢弃，只输出合成正文。"""
        syn = "v2o_emit_x"
        chunks = [
            types.GenerateContentResponse(candidates=[types.Candidate(
                content=types.Content(parts=[types.Part(text="开头几个字")], role="model"))]),
            types.GenerateContentResponse(candidates=[types.Candidate(
                content=types.Content(parts=[types.Part(function_call=types.FunctionCall(
                    name=syn, args={"content": "合成正文"}))], role="model"))]),
        ]
        client, req, prompt = self._make_stream(chunks)
        body = await self._run(client, req, prompt, syn)
        content = self._collect_content(body)
        assert content == "合成正文"
        assert "开头几个字" not in content

    async def test_buffer_flushed_when_no_synthetic_hit(self):
        """全程未命中合成调用：缓冲的普通文本流末完整 flush（防截断未生效正文不丢）。"""
        chunks = [
            types.GenerateContentResponse(candidates=[types.Candidate(
                content=types.Content(parts=[types.Part(text="正常回答")], role="model"))]),
            types.GenerateContentResponse(candidates=[types.Candidate(
                content=types.Content(parts=[], role="model"),
                finish_reason=types.FinishReason.STOP)]),
        ]
        client, req, prompt = self._make_stream(chunks)
        body = await self._run(client, req, prompt, "v2o_emit_x")
        content = self._collect_content(body)
        assert content == "正常回答"

    async def test_no_synthetic_tool_name_passthrough(self):
        """未启用防截断：正文 chunk 直接透传，不缓冲（零行为变化）。"""
        chunks = [
            types.GenerateContentResponse(candidates=[types.Candidate(
                content=types.Content(parts=[types.Part(text="普通正文")], role="model"),
                finish_reason=types.FinishReason.STOP)]),
        ]
        client, req, prompt = self._make_stream(chunks)
        body = await self._run(client, req, prompt, None)
        assert self._collect_content(body) == "普通正文"



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
