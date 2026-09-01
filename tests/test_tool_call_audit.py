"""工具调用差集审计回归（进阶报告 P0-4 / §15.5）。

对照 gcli2api 2026-08-28~30 修复的几类流式/多轮工具调用问题，逐项验证本项目
对应行为不缺位：
  1. 相邻同 role contents 归一化（tool result 后紧跟 user 消息等场景）；
  2. 流式响应收尾 finish_reason 必须是 tool_calls（Google 把 functionCall 与
     最终 STOP 分在两个 chunk 时尤其关键）；
  3. 并行 functionCall 分散在多个流式 chunk 时 index 持续递增（不归零）；
  4. 非法/畸形 tool arguments 不让 FunctionCall/FunctionResponse 失配。
既有覆盖（test_sa_tool_calls 等）已包含大部分场景，这里补齐"差集"缺口。
"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))

from google import genai as genai_types
from google.genai import types

from api_helpers import ToolCallIndexer, convert_chunk_to_openai
from message_processing import create_gemini_prompt
from models import OpenAIMessage, OpenAIRequest


def _weather_tool():
    return {"type": "function", "function": {
        "name": "weather_api",
        "description": "query weather",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}


class TestAdjacentSameRoleMerge:
    """差集 1：相邻同 role 的 contents 必须合并（gcli2api f0f6d3c 修的 0-token 空回）。"""

    def _contents_roles(self, messages):
        prompt = create_gemini_prompt(messages, model_name="gemini-3.6-flash")
        return [c.role for c in prompt]

    def test_tool_result_then_user_not_merged(self):
        """tool 结果（user role）后紧跟普通 user 消息：保持两个独立 Content。

        这是刻意行为（上游 merge 逻辑的 previous_is_results != current_is_results
        分支）：并行 tool result 必须连续，而其后的新 user 轮次要开新 Content，
        否则会破坏 Gemini 签名校验器的轮次边界。gcli2api 修的"相邻同 role
        合并"缺口在本项目已由「两条普通文本 user 才合并」的更精确规则覆盖
        （见 test_two_plain_user_messages_merge）。"""
        messages = [
            OpenAIMessage(role="user", content="上海天气"),
            OpenAIMessage(role="assistant", content=None, tool_calls=[{
                "id": "call_1", "type": "function",
                "function": {"name": "weather_api", "arguments": json.dumps({"city": "上海"})}}]),
            OpenAIMessage(role="tool", tool_call_id="call_1", name="weather_api",
                          content=json.dumps({"temp": 25})),
            OpenAIMessage(role="user", content="顺便北京呢？"),
        ]
        roles = self._contents_roles(messages)
        assert roles == ["user", "model", "user", "user"]

    def test_parallel_tool_results_still_merge(self):
        """连续两条 tool 结果（并行工具）必须合并进同一个 user Content。"""
        messages = [
            OpenAIMessage(role="user", content="两地天气"),
            OpenAIMessage(role="assistant", content=None, tool_calls=[
                {"id": "call_1", "type": "function",
                 "function": {"name": "weather_api", "arguments": json.dumps({"city": "上海"})}},
                {"id": "call_2", "type": "function",
                 "function": {"name": "weather_api", "arguments": json.dumps({"city": "北京"})}}]),
            OpenAIMessage(role="tool", tool_call_id="call_1", name="weather_api",
                          content=json.dumps({"temp": 25})),
            OpenAIMessage(role="tool", tool_call_id="call_2", name="weather_api",
                          content=json.dumps({"temp": 23})),
        ]
        roles = self._contents_roles(messages)
        assert roles == ["user", "model", "user"]   # 两条 tool 结果合并为一个 user

    def test_two_plain_user_messages_merge(self):
        messages = [
            OpenAIMessage(role="user", content="第一句"),
            OpenAIMessage(role="user", content="第二句"),
        ]
        roles = self._contents_roles(messages)
        assert roles == ["user"]

    def test_user_then_assistant_not_merged(self):
        messages = [
            OpenAIMessage(role="user", content="问"),
            OpenAIMessage(role="assistant", content="答"),
        ]
        roles = self._contents_roles(messages)
        assert roles == ["user", "model"]


class TestStreamFinishReasonToolCalls:
    """差集 2：含工具调用的流式响应，收尾 finish_reason 必须是 tool_calls。"""

    def _chunk(self, parts, finish=None, candidates_idx=0):
        return types.GenerateContentResponse(candidates=[
            types.Candidate(
                content=types.Content(parts=parts, role="model"),
                finish_reason=finish)])

    def test_function_call_then_stop_chunk_keeps_tool_calls(self):
        """Google 常把 functionCall 与最终 STOP 分在两个 chunk：第二个空 STOP chunk
        的 finish_reason 必须被改写为 tool_calls（客户端靠它判断要执行工具）。"""
        indexer = ToolCallIndexer()
        # chunk 1：functionCall，无 finish_reason
        c1 = self._chunk([types.Part(function_call=types.FunctionCall(
            name="weather_api", args={"city": "上海"}))])
        sse1 = convert_chunk_to_openai(c1, "gemini-3.6-flash", "resp-1",
                                       candidate_index=0, indexer=indexer)
        d1 = json.loads(sse1.removeprefix("data: ").strip())
        assert d1["choices"][0]["delta"]["tool_calls"]
        # chunk 2：只有 STOP，没有新 part
        c2 = self._chunk([], finish=types.FinishReason.STOP)
        sse2 = convert_chunk_to_openai(c2, "gemini-3.6-flash", "resp-1",
                                       candidate_index=0, indexer=indexer)
        d2 = json.loads(sse2.removeprefix("data: ").strip())
        assert d2["choices"][0]["finish_reason"] == "tool_calls"

    def test_no_tool_call_stop_stays_stop(self):
        indexer = ToolCallIndexer()
        c = self._chunk([types.Part(text="你好")], finish=types.FinishReason.STOP)
        sse = convert_chunk_to_openai(c, "gemini-3.6-flash", "resp-1",
                                      candidate_index=0, indexer=indexer)
        d = json.loads(sse.removeprefix("data: ").strip())
        assert d["choices"][0]["finish_reason"] == "stop"


class TestParallelIndexAcrossChunks:
    """差集 3：并行 functionCall 分散在多个 chunk 时 index 持续递增。"""

    def test_two_chunks_two_indices(self):
        indexer = ToolCallIndexer()
        c1 = types.GenerateContentResponse(candidates=[types.Candidate(
            content=types.Content(parts=[types.Part(function_call=types.FunctionCall(
                name="f1", args={}))], role="model"))])
        c2 = types.GenerateContentResponse(candidates=[types.Candidate(
            content=types.Content(parts=[types.Part(function_call=types.FunctionCall(
                name="f2", args={}))], role="model"))])
        d1 = json.loads(convert_chunk_to_openai(
            c1, "m", "r", indexer=indexer).removeprefix("data: ").strip())
        d2 = json.loads(convert_chunk_to_openai(
            c2, "m", "r", indexer=indexer).removeprefix("data: ").strip())
        i1 = d1["choices"][0]["delta"]["tool_calls"][0]["index"]
        i2 = d2["choices"][0]["delta"]["tool_calls"][0]["index"]
        assert i1 == 0 and i2 == 1   # 不归零：第二个调用是新 index


class TestMalformedArguments:
    """差集 4：非法 tool arguments 不得让 FunctionCall/FunctionResponse 失配。"""

    def test_bad_json_arguments_falls_back_to_empty(self):
        """assistant.tool_calls.arguments 非法 JSON → 按 {} 构造 FunctionCall，
        不炸、不丢整轮历史。"""
        messages = [
            OpenAIMessage(role="user", content="查天气"),
            OpenAIMessage(role="assistant", content=None, tool_calls=[{
                "id": "call_1", "type": "function",
                "function": {"name": "weather_api", "arguments": "{not-json"}}]),
            OpenAIMessage(role="tool", tool_call_id="call_1", name="weather_api",
                          content='{"temp": 25}'),
        ]
        prompt = create_gemini_prompt(messages, model_name="gemini-3.6-flash")
        fc_args, fr = [], []
        for c in prompt:
            for p in (c.parts or []):
                fc = getattr(p, "function_call", None)
                frp = getattr(p, "function_response", None)
                if fc is not None:
                    fc_args.append((fc.name, fc.args, fc.id))
                if frp is not None:
                    fr.append((frp.name, frp.response, frp.id))
        # FunctionCall 与 FunctionResponse 都在，且 id 配对一致
        assert len(fc_args) == 1 and len(fr) == 1
        assert fc_args[0][0] == "weather_api" and fc_args[0][1] == {}
        assert fr[0][0] == "weather_api" and fr[0][1] == {"temp": 25}
        assert fc_args[0][2] == fr[0][2]

    def test_empty_arguments_string_ok(self):
        messages = [
            OpenAIMessage(role="user", content="查"),
            OpenAIMessage(role="assistant", content=None, tool_calls=[{
                "id": "call_2", "type": "function",
                "function": {"name": "ping", "arguments": ""}}]),
            OpenAIMessage(role="tool", tool_call_id="call_2", name="ping",
                          content="pong"),
        ]
        prompt = create_gemini_prompt(messages, model_name="gemini-3.6-flash")
        fcs = [p.function_call for c in prompt for p in (c.parts or [])
               if getattr(p, "function_call", None) is not None]
        assert len(fcs) == 1 and fcs[0].args == {}
