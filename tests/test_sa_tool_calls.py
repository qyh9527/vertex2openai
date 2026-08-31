"""P1-9：SA（服务账号）通道工具调用专项测试。

验证 ServiceAccountUpstream 完整管线（继承自 ExpressSDKUpstream）下的工具调用：
- 非流式：模型发起 function_call → OpenAI tool_calls 形状正确
- 真流式：工具调用 chunk 转 SSE、并行 tool call 各自 index、role=tool 回传还原成 FunctionResponse
- 假流式（fake-）：合成工具解构路径不破坏工具调用
- 通道断言：请求确实经过 SA Client（channel_name=vertex）

不触碰真实网络：用假 SDK Client + 真实 types.* 对象走完 execute_gemini_call 的转换管线。
"""
import json

import pytest
from fastapi.responses import JSONResponse, StreamingResponse
from google.genai import types

from api_helpers import execute_gemini_call, gemini_fake_stream_generator
from message_processing import create_gemini_prompt
from models import OpenAIRequest
from upstreams.service_account import ServiceAccountUpstream
import signature_store
from signature_store import SignatureRecord, SignatureState


# ---------- 测试用假 SDK Client（记录调用，返回可编程响应） ----------

class FakeModels:
    def __init__(self, owner):
        self.owner = owner

    async def generate_content(self, model, contents, config):
        self.owner.calls.append(("nonstream", model, contents, config))
        return self.owner.nonstream_resp

    async def generate_content_stream(self, model, contents, config):
        self.owner.calls.append(("stream", model, contents, config))

        async def _gen():
            for chunk in self.owner.stream_chunks:
                yield chunk
        return _gen()


class FakeAio:
    def __init__(self, owner):
        self.models = FakeModels(owner)


class FakeClient:
    """SA 通道用的假 genai.Client：记录所有调用并断言用什么模型名/contents。"""

    def __init__(self, nonstream_resp=None, stream_chunks=()):
        self.calls = []
        self.aio = FakeAio(self)
        # 挂上失败回调钩子（复用 Client 才有；这里模拟 SA 复用 Client 的形状）
        self._vertex_cache_key = ("fake-sa-key",)
        self.nonstream_resp = nonstream_resp
        self.stream_chunks = list(stream_chunks)


# ---------- 公共工具 ----------

def _weather_tool():
    return {"type": "function", "function": {
        "name": "weather_api",
        "description": "查询天气",
        "parameters": {"type": "object", "properties": {
            "city": {"type": "string"}}, "required": ["city"]}}}


def _fc_part(name="weather_api", args=None, call_id="call_sa_1", sig=None):
    kwargs = {"function_call": types.FunctionCall(
        name=name, args=args or {"city": "上海"}, id=call_id)}
    if sig:
        kwargs["thought_signature"] = sig
    return types.Part(**kwargs)


def _tool_call_response(parts, finish=types.FinishReason.STOP):
    return types.GenerateContentResponse(candidates=[types.Candidate(
        content=types.Content(parts=parts, role="model"), finish_reason=finish)])


def _parse_sse(chunks):
    """把 SSE chunk 列表解析成 payload dict 列表（跳过心跳与 [DONE]）。"""
    out = []
    for c in chunks:
        if not c.startswith("data:"):
            continue
        payload = c[len("data:"):].strip()
        if not payload or payload == "[DONE]":
            continue
        out.append(json.loads(payload))
    return out


class TestSaChannelIdentity:
    """前置断言：这些测试跑的确实是 SA 通道语义。"""

    def test_channel_name_vertex(self):
        assert ServiceAccountUpstream().channel_name == "vertex"

    async def test_execute_gemini_call_receives_vertex_channel(self, capsys):
        """execute_gemini_call 用 channel_name=vertex 调用时，日志必须显示 SA 身份。"""
        req = OpenAIRequest(model="gemini-3.6-flash",
                            messages=[{"role": "user", "content": "hi"}], stream=True)
        client = FakeClient(stream_chunks=[
            types.GenerateContentResponse(candidates=[types.Candidate(
                content=types.Content(parts=[
                    types.Part(text="你好")], role="model"),
                finish_reason=types.FinishReason.STOP)]),
        ])
        prompt = create_gemini_prompt(req.messages)
        resp = await execute_gemini_call(
            client, "gemini-3.6-flash", lambda msgs: prompt, {}, req,
            channel_name="vertex")
        assert isinstance(resp, StreamingResponse)
        body = "".join([c async for c in resp.body_iterator])
        # SSE 正常出流（正文在 ensure_ascii 转义后也在场，且以 [DONE] 收尾）
        assert "[DONE]" in body
        assert '"finish_reason": "stop"' in body
        log = capsys.readouterr().out
        assert "服务账号" in log
        assert "Express" not in log.replace("ExpressSDKUpstream", "")


class TestSaNonstreamToolCall:
    async def test_function_call_to_openai_tool_calls(self):
        """非流式：SA Client 返回 function_call → OpenAI tool_calls 形状完整。"""
        sig = b"\x01\x02sa-signature"
        req = OpenAIRequest(model="gemini-3.6-flash",
                            messages=[{"role": "user", "content": "上海天气"}],
                            tools=[_weather_tool()])
        client = FakeClient(nonstream_resp=_tool_call_response([_fc_part(sig=sig)]))
        prompt = create_gemini_prompt(req.messages)
        resp = await execute_gemini_call(
            client, "gemini-3.6-flash", lambda msgs: prompt, {}, req,
            channel_name="vertex")
        assert isinstance(resp, JSONResponse)
        body = json.loads(resp.body.decode("utf-8"))
        msg = body["choices"][0]["message"]
        assert msg["tool_calls"][0]["function"]["name"] == "weather_api"
        assert json.loads(msg["tool_calls"][0]["function"]["arguments"]) == {"city": "上海"}
        # 思考签名放进 extra_content.google（文档化的 OpenAI 扩展载体）
        extra = msg["tool_calls"][0].get("extra_content") or {}
        assert extra.get("google", {}).get("thought_signature")
        # 签名已进旁路缓存，下一轮回传能还原
        rec = signature_store.signature_store.get_record(msg["tool_calls"][0]["id"])
        assert rec is not None and rec.state is SignatureState.SIGNED
        assert body["choices"][0]["finish_reason"] == "tool_calls"


class TestSaStreamToolCall:
    async def test_stream_tool_call_chunks(self):
        """真流式：工具调用 chunk → SSE tool_calls delta + finish_reason=tool_calls。"""
        req = OpenAIRequest(model="gemini-3.6-flash",
                            messages=[{"role": "user", "content": "上海天气"}],
                            tools=[_weather_tool()], stream=True)
        client = FakeClient(stream_chunks=[
            _tool_call_response([_fc_part()], finish=types.FinishReason.MALFORMED_FUNCTION_CALL),
            # 官方常见形态：functionCall 一个 chunk，STOP 尾块另一个 chunk
            _tool_call_response([], finish=types.FinishReason.STOP),
        ])
        prompt = create_gemini_prompt(req.messages)
        resp = await execute_gemini_call(
            client, "gemini-3.6-flash", lambda msgs: prompt, {}, req,
            channel_name="vertex")
        chunks = _parse_sse([c async for c in resp.body_iterator])
        tool_chunks = [c for c in chunks
                       if (c["choices"] and c["choices"][0]["delta"].get("tool_calls"))]
        assert tool_chunks, "必须输出 tool_calls delta"
        tc = tool_chunks[0]["choices"][0]["delta"]["tool_calls"][0]
        assert tc["function"]["name"] == "weather_api"
        assert json.loads(tc["function"]["arguments"]) == {"city": "上海"}
        # 收尾 finish_reason 必须是 tool_calls（不是 stop）
        finishes = [c["choices"][0]["finish_reason"] for c in chunks if c["choices"]]
        assert "tool_calls" in finishes

    async def test_parallel_tool_calls_distinct_index(self):
        """并行 tool call：两个 function_call 必须各占一个 index（不合并成一个）。"""
        req = OpenAIRequest(model="gemini-3.6-flash",
                            messages=[{"role": "user", "content": "上海和北京天气"}],
                            tools=[_weather_tool()], stream=True)
        client = FakeClient(stream_chunks=[
            _tool_call_response([
                _fc_part(args={"city": "上海"}, call_id="call_sa_a"),
                _fc_part(args={"city": "北京"}, call_id="call_sa_b"),
            ], finish=types.FinishReason.MALFORMED_FUNCTION_CALL),
        ])
        prompt = create_gemini_prompt(req.messages)
        resp = await execute_gemini_call(
            client, "gemini-3.6-flash", lambda msgs: prompt, {}, req,
            channel_name="vertex")
        chunks = _parse_sse([c async for c in resp.body_iterator])
        indexes = []
        names = []
        for c in chunks:
            for choice in c["choices"]:
                for tc in (choice["delta"].get("tool_calls") or []):
                    indexes.append(tc["index"])
                    if tc["function"].get("name"):
                        names.append(tc["function"]["name"])
        assert len(indexes) == 2
        assert indexes[0] != indexes[1]      # 两个调用必须可区分
        assert names.count("weather_api") == 2


class TestSaToolResultRoundTrip:
    """role=tool 回传：SA 管线下 FunctionResponse 还原正确（含签名恢复）。"""

    async def test_tool_message_becomes_function_response(self):
        sig = b"\x01\x02sa-signature"
        req = OpenAIRequest(model="gemini-3.6-flash",
                            messages=[
                                {"role": "user", "content": "上海天气"},
                                {"role": "assistant", "content": None,
                                 "tool_calls": [{
                                     "id": "call_sa_1", "type": "function",
                                     "function": {"name": "weather_api",
                                                  "arguments": json.dumps({"city": "上海"})}}]},
                                {"role": "tool", "tool_call_id": "call_sa_1",
                                 "content": json.dumps({"temp": 25})},
                            ],
                            tools=[_weather_tool()])
        # 先让签名进旁路缓存（模拟上一轮 SA 通道产出的 tool_call id）
        signature_store.signature_store.put_record(
            "call_sa_1", SignatureRecord(SignatureState.SIGNED, sig))
        client = FakeClient(nonstream_resp=_tool_call_response([
            types.Part(text="上海 25 度")]))
        prompt = create_gemini_prompt(req.messages, model_name="gemini-3.6-flash")
        resp = await execute_gemini_call(
            client, "gemini-3.6-flash", lambda msgs: prompt, {}, req,
            channel_name="vertex")
        assert isinstance(resp, JSONResponse)
        # 请求侧：发给 SA 的 contents 必须含 FunctionResponse（而不是 mock 文本）
        kind, _, contents, _ = client.calls[0]
        fr_parts = []
        for c in contents:
            for p in (c.parts or []):
                if getattr(p, "function_response", None) is not None:
                    fr_parts.append(p.function_response)
        assert len(fr_parts) == 1
        assert fr_parts[0].name == "weather_api"
        assert fr_parts[0].response == {"temp": 25}
        assert fr_parts[0].id == "call_sa_1"
        # assistant 工具调用 part 上的思考签名已恢复（SA 通道不丢签名）
        has_sig = False
        for c in contents:
            for p in (c.parts or []):
                if getattr(p, "function_call", None) is not None and \
                   getattr(p, "thought_signature", None):
                    has_sig = True
        assert has_sig

    async def test_tool_result_without_name_degrades_gracefully(self):
        """无法关联函数名的 tool 消息降级为 System Observation 文本，不炸。"""
        req = OpenAIRequest(model="gemini-3.6-flash",
                            messages=[
                                {"role": "user", "content": "hi"},
                                {"role": "tool", "tool_call_id": "call_unknown",
                                 "content": "结果"},
                            ])
        client = FakeClient(nonstream_resp=_tool_call_response([
            types.Part(text="ok")]))
        prompt = create_gemini_prompt(req.messages)
        resp = await execute_gemini_call(
            client, "gemini-3.6-flash", lambda msgs: prompt, {}, req,
            channel_name="vertex")
        assert isinstance(resp, JSONResponse)
        body = json.loads(resp.body.decode("utf-8"))
        assert body["choices"][0]["message"]["content"] == "ok"


class TestSaFakeStreamToolCall:
    async def test_fake_stream_strips_synthetic_keeps_real_tool(self):
        """假流式（fake-）× 防截断：合成工具被剥离，真实工具调用原样输出。"""
        from api_helpers import gemini_fake_stream_generator
        from upstreams.service_account import ServiceAccountUpstream

        syn_name = "v2o_emit_abc123"
        real_sig = b"\x01sa-sig"
        # 一次返回同时含合成调用与真实调用（合成在前）
        resp = types.GenerateContentResponse(candidates=[types.Candidate(
            content=types.Content(parts=[
                types.Part(function_call=types.FunctionCall(
                    name=syn_name, args={"content": "防截断正文"})),
                _fc_part(args={"city": "上海"}, call_id="call_sa_r", sig=real_sig),
            ], role="model"), finish_reason=types.FinishReason.MALFORMED_FUNCTION_CALL)])

        class _FM:
            async def generate_content(self, model, contents, config):
                return resp

        class _FA:
            models = _FM()

        class _FC:
            aio = _FA()

        req = OpenAIRequest(model="fake-gemini-3.6-flash",
                            messages=[{"role": "user", "content": "hi"}],
                            tools=[_weather_tool()], stream=True)
        chunks = []
        async for sse in gemini_fake_stream_generator(
            _FC(), "gemini-3.6-flash", [], {}, req, False,
            channel_name="vertex", synthetic_tool_name=syn_name,
        ):
            chunks.append(sse)

        payloads = _parse_sse(chunks)
        contents_out = []
        # 假流式按 OpenAI 惯例分两个 delta（先 name 后 arguments），按 index 合并
        merged_calls = {}
        finish = None
        for p in payloads:
            for choice in p["choices"]:
                d = choice["delta"]
                if d.get("content"):
                    contents_out.append(d["content"])
                for tc in (d.get("tool_calls") or []):
                    m = merged_calls.setdefault(tc["index"], {"name": None, "arguments": ""})
                    fn = tc.get("function") or {}
                    if fn.get("name"):
                        m["name"] = fn["name"]
                    if fn.get("arguments"):
                        m["arguments"] += fn["arguments"]
                if choice.get("finish_reason"):
                    finish = choice["finish_reason"]
        assert "".join(contents_out) == "防截断正文"       # 合成工具内容 → 正文
        assert len(merged_calls) == 1                       # 真实工具调用保留（合成被剥离）
        real = next(iter(merged_calls.values()))
        assert real["name"] == "weather_api"
        assert json.loads(real["arguments"]) == {"city": "上海"}
        assert finish == "tool_calls"
        # 合成工具名绝不泄漏到下游
        all_text = "".join(chunks)
        assert syn_name not in all_text


class TestSaThoughtSignatureContinuity:
    async def test_signature_survives_round_trip(self):
        """签名跨轮连续性：第一轮产出签名 → 第二轮回传还原（SA 通道不变签）。"""
        # 第一轮：SA 通道产出带签名的工具调用
        sig = b"\x03sa-roundtrip"
        req1 = OpenAIRequest(model="gemini-3.6-flash",
                             messages=[{"role": "user", "content": "上海天气"}],
                             tools=[_weather_tool()])
        client1 = FakeClient(nonstream_resp=_tool_call_response([_fc_part(sig=sig)]))
        prompt1 = create_gemini_prompt(req1.messages)
        resp1 = await execute_gemini_call(
            client1, "gemini-3.6-flash", lambda m: prompt1, {}, req1,
            channel_name="vertex")
        body1 = json.loads(resp1.body.decode("utf-8"))
        tc1 = body1["choices"][0]["message"]["tool_calls"][0]
        call_id = tc1["id"]

        # 第二轮：客户端把 tool_calls（不含 extra_content）+ role=tool 回传
        req2 = OpenAIRequest(model="gemini-3.6-flash",
                             messages=[
                                 {"role": "user", "content": "上海天气"},
                                 {"role": "assistant", "content": None,
                                  "tool_calls": [tc1]},
                                 {"role": "tool", "tool_call_id": call_id,
                                  "content": json.dumps({"temp": 25})},
                             ],
                             tools=[_weather_tool()])
        client2 = FakeClient(nonstream_resp=_tool_call_response([
            types.Part(text="上海 25 度")]))
        prompt2 = create_gemini_prompt(req2.messages, model_name="gemini-3.6-flash")
        await execute_gemini_call(
            client2, "gemini-3.6-flash", lambda m: prompt2, {}, req2,
            channel_name="vertex")
        _, _, contents2, _ = client2.calls[0]
        restored_sig = None
        for c in contents2:
            for p in (c.parts or []):
                fc = getattr(p, "function_call", None)
                if fc is not None:
                    restored_sig = getattr(p, "thought_signature", None)
        assert restored_sig == sig   # 旁路缓存恢复，与第一轮逐字节一致
