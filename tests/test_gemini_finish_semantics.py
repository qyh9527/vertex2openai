import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))

import outcome


def test_normalize_finish_reason_removes_sdk_prefix():
    assert outcome.normalize_finish_reason("FinishReason.STOP") == "STOP"


def test_normalize_finish_reason_uses_sdk_enum_name():
    from enum import Enum

    class FinishReason(Enum):
        STOP = "STOP_VALUE"

    assert outcome.normalize_finish_reason(FinishReason.STOP) == "STOP"


def test_normalize_finish_reason_uses_value_when_name_is_not_finish_reason():
    from enum import Enum

    class ProviderReason(Enum):
        MAX_TOKENS = "MAX_TOKENS"

    assert outcome.normalize_finish_reason(ProviderReason.MAX_TOKENS) == "MAX_TOKENS"


def test_normalize_finish_reason_uses_value_when_name_is_not_a_known_reason():
    from enum import Enum

    class ProviderFinishReason(Enum):
        STATE = "PUP_LIMITED_DISABLED"

    assert outcome.normalize_finish_reason(ProviderFinishReason.STATE) == "PUP_LIMITED_DISABLED"


def test_unknown_sdk_enum_name_is_preserved_when_value_is_not_text():
    from enum import Enum

    class FutureFinishReason(Enum):
        FUTURE_REASON = 88

    assert outcome.normalize_finish_reason(FutureFinishReason.FUTURE_REASON) == "FUTURE_REASON"


def test_prefixed_unspecified_is_middle_state():
    end = outcome.classify_generation_end("FinishReason.FINISH_REASON_UNSPECIFIED")
    assert end.raw_finish_reason == "FINISH_REASON_UNSPECIFIED"
    assert end.terminal is False


def test_stop_is_a_terminal_normal_end():
    end = outcome.classify_generation_end("STOP")
    assert end.raw_finish_reason == "STOP"
    assert end.terminal is True
    assert end.finish_reason == "stop"
    assert end.failure_category is None
    assert end.error_type is None
    assert end.http_status is None


def test_max_tokens_remains_length_even_with_tool_calls():
    end = outcome.classify_generation_end("MAX_TOKENS", has_tool_calls=True)
    assert end.terminal is True
    assert end.finish_reason == "length"
    assert end.failure_category is None


def test_escalation_filter_wins_over_tool_calls():
    end = outcome.classify_generation_end("ESCALATION", has_tool_calls=True)
    assert end.terminal is True
    assert end.finish_reason == "content_filter"
    assert end.failure_category == outcome.POLICY_BLOCKED


def test_unspecified_finish_is_nonterminal_in_middle_and_protocol_error_at_final():
    middle = outcome.classify_generation_end("FINISH_REASON_UNSPECIFIED")
    final = outcome.classify_generation_end(None, is_final=True)
    assert middle.terminal is False
    assert middle.failure_category is None
    assert final.terminal is True
    assert final.failure_category == outcome.EMPTY_OR_PROTOCOL
    assert final.error_type == "upstream_protocol_error"
    assert final.http_status == 502


def test_limited_account_is_credential_failure():
    end = outcome.classify_generation_end("PUP_LIMITED_DISABLED", has_tool_calls=True)
    assert end.terminal is True
    assert end.finish_reason is None
    assert end.failure_category == outcome.CREDENTIAL_PERMANENT
    assert end.error_type == "upstream_account_disabled"
    assert end.http_status == 502


def test_generation_end_error_exposes_stable_fields_without_upstream_text():
    end = outcome.classify_generation_end("PUP_LIMITED_DISABLED")
    error = outcome.GenerationEndError(end)
    assert error.category == outcome.CREDENTIAL_PERMANENT
    assert error.upstream_finish_reason == "PUP_LIMITED_DISABLED"
    assert error.code == 502
    assert error.error_type == "upstream_account_disabled"
    assert "PUP_LIMITED_DISABLED" in str(error)
    assert "stacktrace" not in str(error)


def test_unknown_finish_reason_is_preserved_verbatim():
    assert outcome.normalize_finish_reason("Future_Gemini_Reason") == "Future_Gemini_Reason"


def test_unknown_reason_has_protocol_category_and_original_spelling():
    end = outcome.classify_generation_end("Future_Gemini_Reason")
    assert end.raw_finish_reason == "Future_Gemini_Reason"
    assert end.failure_category == outcome.EMPTY_OR_PROTOCOL


def test_explicit_malformed_function_call_is_protocol_failure_even_with_tools():
    end = outcome.classify_generation_end("MALFORMED_FUNCTION_CALL", has_tool_calls=True)
    assert end.raw_finish_reason == "MALFORMED_FUNCTION_CALL"
    assert end.terminal is True
    assert end.finish_reason is None
    assert end.failure_category == outcome.EMPTY_OR_PROTOCOL
    assert end.error_type == "upstream_protocol_error"
    assert end.http_status == 502
