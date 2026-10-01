"""SDK response conversion contract for Gemini terminal reasons."""
import base64
import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))

from google.genai import types
from message_processing import (
    convert_to_openai_format,
    process_gemini_response_to_openai_dict,
)
from outcome import GenerationEndError
from protocol_conversion import ToolCallIndexer, convert_chunk_to_openai


def _candidate(parts, finish_reason):
    return SimpleNamespace(
        content=SimpleNamespace(parts=parts),
        finish_reason=finish_reason,
    )


def _function_call_part(name="lookup", args=None):
    return SimpleNamespace(
        function_call=SimpleNamespace(name=name, args=args or {}),
        thought_signature=None,
    )


def test_escalation_keeps_content_filter_when_candidate_contains_tool_call():
    response = SimpleNamespace(candidates=[_candidate(
        [_function_call_part()], "ESCALATION")])

    converted = process_gemini_response_to_openai_dict(response, "gemini-test")

    choice = converted["choices"][0]
    assert choice["finish_reason"] == "content_filter"
    assert choice["message"]["tool_calls"][0]["function"]["name"] == "lookup"


def test_synthetic_tool_call_is_preserved_but_does_not_set_tool_calls_finish():
    synthetic_name = "v2o_emit_exact-request-token"
    response = types.GenerateContentResponse(candidates=[types.Candidate(
        content=types.Content(parts=[types.Part(function_call=types.FunctionCall(
            name=synthetic_name, args={"content": "answer"}))]),
        finish_reason=types.FinishReason.STOP,
    )])

    converted = process_gemini_response_to_openai_dict(
        response, "gemini-test", synthetic_tool_name=synthetic_name)

    choice = converted["choices"][0]
    assert choice["finish_reason"] == "stop"
    assert choice["message"]["tool_calls"][0]["function"]["name"] == synthetic_name


def test_stream_function_call_blocks_keep_indices_signatures_and_topology_through_stop():
    indexer = ToolCallIndexer()
    signature_one = b"signature-one"
    signature_two = b"signature-two"
    first = types.GenerateContentResponse(candidates=[types.Candidate(
        content=types.Content(parts=[types.Part(
            function_call=types.FunctionCall(name="first", args={"n": 1}),
            thought_signature=signature_one,
        )]),
    )])
    second = types.GenerateContentResponse(candidates=[types.Candidate(
        content=types.Content(parts=[types.Part(
            function_call=types.FunctionCall(name="second", args={"n": 2}),
            thought_signature=signature_two,
        )]),
    )])
    terminal = types.GenerateContentResponse(candidates=[types.Candidate(
        content=types.Content(parts=[]), finish_reason=types.FinishReason.STOP,
    )])

    chunks = [json.loads(convert_chunk_to_openai(
        response, "gemini-test", "resp-test", indexer=indexer).removeprefix("data: ").strip())
        for response in (first, second, terminal)]

    calls = [chunks[0]["choices"][0]["delta"]["tool_calls"][0],
             chunks[1]["choices"][0]["delta"]["tool_calls"][0]]
    assert [call["index"] for call in calls] == [0, 1]
    encoded_signatures = [call["extra_content"]["google"]["thought_signature"] for call in calls]
    assert encoded_signatures == [
        base64.b64encode(signature_one).decode("ascii"),
        base64.b64encode(signature_two).decode("ascii"),
    ]
    assert chunks[2]["choices"][0]["finish_reason"] == "tool_calls"
    topology = chunks[2]["choices"][0]["delta"]["extra_content"]["google"]
    assert [item["index"] for item in topology["part_order"]] == [0, 1]


def test_max_tokens_is_not_overridden_by_function_call():
    response = SimpleNamespace(candidates=[_candidate(
        [_function_call_part()], "MAX_TOKENS")])

    converted = process_gemini_response_to_openai_dict(response, "gemini-test")

    assert converted["choices"][0]["finish_reason"] == "length"


def test_terminal_protocol_failure_does_not_register_candidate_tool_call():
    indexer = ToolCallIndexer()
    response = SimpleNamespace(candidates=[_candidate(
        [_function_call_part("must-not-be-indexed")], "MALFORMED_RESPONSE")])

    try:
        convert_chunk_to_openai(response, "gemini-test", "resp-test", indexer=indexer)
    except GenerationEndError as exc:
        assert exc.category == "empty_or_protocol"
    else:
        raise AssertionError("malformed terminal response must fail")
    assert indexer.has_tool_calls(0) is False


def test_unknown_final_reason_is_a_protocol_failure():
    response = SimpleNamespace(candidates=[_candidate([], "FUTURE_REASON")])

    try:
        process_gemini_response_to_openai_dict(response, "gemini-test")
    except GenerationEndError as exc:
        assert exc.upstream_finish_reason == "FUTURE_REASON"
    else:
        raise AssertionError("unknown final reason must fail")


def test_candidate_with_no_finish_reason_fails_final_conversion():
    response = types.GenerateContentResponse(candidates=[types.Candidate(
        content=types.Content(parts=[types.Part(text="candidate text")]),
    )])

    try:
        process_gemini_response_to_openai_dict(response, "gemini-test")
    except GenerationEndError as exc:
        assert exc.category == "empty_or_protocol"
    else:
        raise AssertionError("missing final reason must fail")


def test_legacy_text_without_candidates_remains_compatible():
    response = SimpleNamespace(candidates=None, text="legacy answer")

    converted = process_gemini_response_to_openai_dict(response, "gemini-test")

    assert converted["choices"][0]["message"]["content"] == "legacy answer"
    assert converted["choices"][0]["finish_reason"] == "stop"


def test_prompt_block_reason_is_classified_before_empty_response():
    response = SimpleNamespace(
        candidates=None,
        prompt_feedback=SimpleNamespace(block_reason=types.BlockedReason.SAFETY),
    )

    converted = process_gemini_response_to_openai_dict(response, "gemini-test")

    assert converted["choices"][0]["finish_reason"] == "content_filter"


def test_unspecified_prompt_feedback_does_not_hide_missing_final_end():
    response = SimpleNamespace(
        candidates=None,
        prompt_feedback=SimpleNamespace(block_reason=types.BlockedReason.BLOCKED_REASON_UNSPECIFIED),
    )

    try:
        process_gemini_response_to_openai_dict(response, "gemini-test")
    except GenerationEndError as exc:
        assert exc.category == "empty_or_protocol"
    else:
        raise AssertionError("unspecified prompt feedback must not count as a block")


def test_stop_with_synthetic_tool_payload_stays_stop_but_keeps_payload():
    synthetic_name = "v2o_emit_exact-request-token"
    response = types.GenerateContentResponse(candidates=[types.Candidate(
        content=types.Content(parts=[types.Part(function_call=types.FunctionCall(
            name=synthetic_name, args={"content": "answer"}))]),
        finish_reason=types.FinishReason.STOP,
    )])

    converted = process_gemini_response_to_openai_dict(
        response, "gemini-test", synthetic_tool_name=synthetic_name)

    choice = converted["choices"][0]
    assert choice["finish_reason"] == "stop"
    assert choice["message"]["tool_calls"][0]["function"]["name"] == synthetic_name
