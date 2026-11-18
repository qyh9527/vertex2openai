"""重试换号与凭证粒度熔断的回归测试。"""
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from google.genai import types

import config as app_config
import runtime_state
import sa_budget
from runtime_state import AppState, _sa_budget_available as _real_budget_available

_SA_A = '{"type":"service_account","client_email":"a@x.iam.gserviceaccount.com","private_key":"k","project_id":"proj-a"}'
_SA_B = '{"type":"service_account","client_email":"b@x.iam.gserviceaccount.com","private_key":"k","project_id":"proj-b"}'


@pytest.fixture
def fresh(tmp_path, monkeypatch):
    monkeypatch.setenv("STATE_DIR", str(tmp_path))
    monkeypatch.setattr(runtime_state, "STATE_FILE", str(tmp_path / "web_state.json"))
    state = AppState()
    monkeypatch.setattr(runtime_state, "app_state", state)
    # 额度账本在测试里不建真实账本：一律视为可用（换号逻辑本身不测账本）。
    monkeypatch.setattr(runtime_state, "_sa_budget_available", lambda *a, **k: True)
    # 候选熔断冷却内存态可能残留：默认不冷却。
    monkeypatch.setattr(runtime_state, "_sa_credential_cooling", lambda *a, **k: False)
    return state


class TestNextSaAccountExcluding:
    def test_skips_already_tried(self, fresh):
        fresh.set_sa_accounts([
            {"project_id": "proj-a", "location": "global", "sa_json": _SA_A},
            {"project_id": "proj-b", "location": "global", "sa_json": _SA_B},
        ])
        project, _location, sa_json = fresh.next_sa_account_excluding({sa_budget.credential_id(_SA_A)})
        assert sa_json == _SA_B and project == "proj-b"

    def test_returns_none_when_all_tried(self, fresh):
        fresh.set_sa_accounts([
            {"project_id": "proj-a", "location": "global", "sa_json": _SA_A},
            {"project_id": "proj-b", "location": "global", "sa_json": _SA_B},
        ])
        assert fresh.next_sa_account_excluding({sa_budget.credential_id(_SA_A), sa_budget.credential_id(_SA_B)}) == (None, None, None)

    def test_single_account_returns_none(self, fresh):
        """只有一个账号时换不了号，调用方保持原 client（行为与改造前一致）。"""
        fresh.set_sa_accounts([{"project_id": "proj-a", "location": "global", "sa_json": _SA_A}])
        assert fresh.next_sa_account_excluding({sa_budget.credential_id(_SA_A)}) == (None, None, None)

    def test_ignores_affinity_project(self, fresh):
        """绑定 project 没容量才走到这里：不受亲和偏好限制，可选另一个 project。"""
        fresh.set_sa_accounts([
            {"project_id": "proj-a", "location": "global", "sa_json": _SA_A},
            {"project_id": "proj-b", "location": "global", "sa_json": _SA_B},
        ])
        project, _location, _sa = fresh.next_sa_account_excluding({sa_budget.credential_id(_SA_A)})
        assert project == "proj-b"


class TestRetryFactoryWiring:
    async def test_factory_switches_and_updates_snapshot(self, fresh, monkeypatch):
        from upstreams import service_account as sa_mod
        fresh.set_sa_accounts([
            {"project_id": "proj-a", "location": "global", "sa_json": _SA_A},
            {"project_id": "proj-b", "location": "global", "sa_json": _SA_B},
        ])
        # Client 构造替换为轻量假对象，避免真实凭证解析。
        monkeypatch.setattr(sa_budget, "BudgetClient",
                            lambda client, key, location, tier: client)
        monkeypatch.setattr(sa_mod, "_get_cached_sa_client",
                            lambda *a, **k: object())

        # service_account 在导入时就绑定了 app_state 引用，必须同时替换该模块里的名字，
        # 否则全量运行时它拿到的是别的测试留下的全局实例。
        monkeypatch.setattr(sa_mod, "app_state", fresh)
        monkeypatch.setattr(sa_mod, "breaker", Mock())
        upstream = sa_mod.ServiceAccountUpstream()
        factory = upstream.make_retry_client_factory(
            dict(app_config.DEFAULT_SETTINGS), "gemini-3.8-flash", _SA_A)
        first = await factory(_RetryableError(429))
        assert first is not None
        from runtime_state import _current_sa_account
        assert _current_sa_account.get()["sa_json"] == _SA_B
        # 两个都试过之后不再有候选
        assert await factory(_RetryableError(429)) is None
        sa_mod.breaker.report_candidate_rate_limited.assert_called_once_with(
            "vertex", runtime_state.sa_breaker_id(_SA_A), str(_RetryableError(429)))


class TestStreamBreakerCredentialGranularity:
    @pytest.mark.parametrize("terminal", ["empty", "limited", "success"])
    async def test_stream_uses_actual_candidate_after_retry(self, terminal, monkeypatch):
        from fastapi.responses import StreamingResponse
        from failover import ChannelBreaker, UpstreamUnstartedError
        from routes import chat_api

        circuit = ChannelBreaker()
        monkeypatch.setattr(circuit, "_threshold", lambda: 1)
        monkeypatch.setattr(circuit, "_cooldown", lambda: 60)
        monkeypatch.setattr(chat_api, "breaker", circuit)
        a, b = map(runtime_state.sa_breaker_id, (_SA_A, _SA_B))
        if terminal == "success":
            circuit.report_failure(("vertex", a))
            circuit.report_failure(("vertex", b))

        async def body():
            yield ": keep-alive\n\n"
            runtime_state._current_sa_account.set({"sa_json": _SA_B})
            if terminal == "limited":
                raise UpstreamUnstartedError("upstream limit") from _RetryableError(429)
            if terminal == "success":
                yield 'data: {"choices":[{"delta":{"content":"ok"}}]}\n\n'
                yield "data: [DONE]\n\n"

        resp = StreamingResponse(body(), media_type="text/event-stream")
        lines = [line async for line in chat_api._stream_with_failover(
            resp, [], _request(True), None, True, "vertex", primary_cred=a)]
        assert lines.count("data: [DONE]\n\n") == 1
        assert circuit.is_cooling(("vertex", a)) == (terminal == "success")
        assert circuit.is_cooling(("vertex", b)) == (terminal != "success")
        assert not circuit.is_cooling("vertex")


def test_rotated_key_does_not_reenter_retry_pool(fresh):
    rotated = _SA_A.replace('"private_key":"k"', '"private_key":"replacement"')
    fresh.set_sa_accounts([{"sa_json": rotated}, {"sa_json": _SA_B}])
    assert fresh.next_sa_account_excluding({sa_budget.credential_id(_SA_A)})[2] == _SA_B


def test_first_selection_skips_cooled_credential_without_affinity(fresh, monkeypatch):
    fresh.set_sa_accounts([{"sa_json": _SA_A}, {"sa_json": _SA_B}])
    monkeypatch.setattr(runtime_state, "_sa_credential_cooling", lambda a: a["sa_json"] == _SA_A)
    assert fresh.get_current_sa_account()[2] == _SA_B


def test_share_one_ledger_without_unifying_credential_identity(fresh, monkeypatch):
    fresh.set_sa_accounts([{"sa_json": _SA_A}, {"sa_json": _SA_B}])
    monkeypatch.setattr(runtime_state, "_sa_budget_available", _real_budget_available)
    a, b = map(sa_budget.credential_id, (_SA_A, _SA_B))
    sa_budget.initialize(a, 294)
    preview = sa_budget.merge_accounts(b, a)
    sa_budget.merge_accounts(b, a, confirmation=preview["confirmation"])
    assert fresh.next_sa_account_excluding({a})[2] == _SA_B
    token = sa_budget.reserve(a, sa_budget.money(1))
    assert fresh.next_sa_account_excluding({a}) == (None, None, None)
    sa_budget.settle(token, 0)
    assert fresh.next_sa_account_excluding({a})[2] == _SA_B


# ---------------------------------------------------------------------------
# execute_gemini_call 换号重试（非流式 / 真流式 / 假流式）
# 契约：retry_max=1；首个凭证抛出 429 时由 async retry_client_factory(error) 换下一个
# 凭证，第二次请求只打到新凭证；已向客户端输出过真流正文后再失败则不换号、不重试，
# 以明确的失败终态收尾。
# ---------------------------------------------------------------------------

def _retry_env(monkeypatch):
    """统一关闭真实退避与统计，并把假流式 keep-alive 间隔压到 0。"""
    import api_helpers as helpers

    monkeypatch.setattr(helpers, "get_retry_settings", lambda *a, **k: (1, 0))
    monkeypatch.setattr(helpers, "stats", Mock())
    original_get_setting = helpers.app_state.get_setting

    def _get_setting(key, default=None):
        if key == "fake_streaming_interval":
            return 0
        return original_get_setting(key, default)

    monkeypatch.setattr(helpers.app_state, "get_setting", _get_setting)
    return helpers


class _RetryableError(Exception):
    """形状对齐 google.genai 的 ClientError：带 code，且文本包含状态码。"""

    def __init__(self, code=429):
        super().__init__(f"{code} Resource exhausted: fake upstream")
        self.code = code


class _FakeCallModels:
    """非流式 / 假流式共用：generate_content 按脚本抛错或返回响应。"""

    def __init__(self, items):
        self.items = list(items)
        self.calls = 0

    async def generate_content(self, **kwargs):
        item = self.items[min(self.calls, len(self.items) - 1)]
        self.calls += 1
        if isinstance(item, BaseException):
            raise item
        return item


class _FakeStreamModels:
    """真流式：每次调用一组事件；首元素为异常时在 await 处抛出（429 的常见形态），
    事件列表内的异常则在迭代中抛出（已建流后中断）。"""

    def __init__(self, attempts):
        self.attempts = list(attempts)
        self.calls = 0

    async def generate_content_stream(self, **kwargs):
        attempt = self.attempts[min(self.calls, len(self.attempts) - 1)]
        self.calls += 1
        if isinstance(attempt, BaseException):
            raise attempt

        async def _events():
            for event in attempt:
                if isinstance(event, BaseException):
                    raise event
                yield event

        return _events()


def _call_client(items):
    return SimpleNamespace(aio=SimpleNamespace(models=_FakeCallModels(items)))


def _stream_client(attempts):
    return SimpleNamespace(aio=SimpleNamespace(models=_FakeStreamModels(attempts)))


def _switching_factory(new_client, seen):
    """新的换号工厂形态：async callable，接收本次失败的 exception。"""

    async def factory(error):
        seen.append(error)
        return new_client

    return factory


def _text_response(text, finish_reason="STOP"):
    return types.GenerateContentResponse(candidates=[types.Candidate(
        content=types.Content(role="model", parts=[types.Part(text=text)]),
        finish_reason=finish_reason)])


def _stream_chunk(text=None, finish_reason=None):
    parts = [types.Part(text=text)] if text is not None else []
    return types.GenerateContentResponse(candidates=[types.Candidate(
        content=types.Content(role="model", parts=parts),
        finish_reason=finish_reason)])


def _sse_payloads(lines):
    return [json.loads(line[6:]) for line in lines
            if line.startswith("data: ") and line.strip() != "data: [DONE]"]


def _sse_text(lines):
    return "".join(choice.get("delta", {}).get("content") or ""
                   for payload in _sse_payloads(lines)
                   for choice in payload.get("choices", []))


def _request(stream):
    from models import OpenAIRequest
    return OpenAIRequest(model="gemini-3.8-flash",
                         messages=[{"role": "user", "content": "hi"}], stream=stream)


class TestExecuteGeminiCallRetrySwitch:
    async def test_nonstream_429_switches_to_next_credential(self, monkeypatch):
        helpers = _retry_env(monkeypatch)
        failure = _RetryableError(429)
        primary = _call_client([failure])
        secondary = _call_client([_text_response("from B")])
        seen = []
        response = await helpers.execute_gemini_call(
            primary, "gemini-3.8-flash", lambda _: [], {}, _request(stream=False),
            channel_name="vertex", retry_client_factory=_switching_factory(secondary, seen))

        assert response.status_code == 200
        assert json.loads(response.body)["choices"][0]["message"]["content"] == "from B"
        # 只按 A→B 各调用一次，不回到 A 重试。
        assert primary.aio.models.calls == 1
        assert secondary.aio.models.calls == 1
        assert len(seen) == 1 and seen[0] is failure and seen[0].code == 429

    async def test_real_stream_429_switches_to_next_credential(self, monkeypatch):
        helpers = _retry_env(monkeypatch)
        failure = _RetryableError(429)
        primary = _stream_client([failure])
        secondary = _stream_client([[_stream_chunk("hello"), _stream_chunk(finish_reason="STOP")]])
        seen = []
        response = await helpers.execute_gemini_call(
            primary, "gemini-3.8-flash", lambda _: [], {}, _request(stream=True),
            channel_name="vertex", retry_client_factory=_switching_factory(secondary, seen))
        lines = [line async for line in response.body_iterator]

        assert primary.aio.models.calls == 1
        assert secondary.aio.models.calls == 1
        assert _sse_text(lines) == "hello"
        assert lines.count("data: [DONE]\n\n") == 1
        assert not any('"error"' in line for line in lines)
        assert len(seen) == 1 and seen[0] is failure and seen[0].code == 429

    async def test_fake_stream_429_switches_to_next_credential(self, monkeypatch):
        helpers = _retry_env(monkeypatch)
        failure = _RetryableError(429)
        primary = _call_client([failure])
        secondary = _call_client([_text_response("hello")])
        seen = []
        response = await helpers.execute_gemini_call(
            primary, "gemini-3.8-flash", lambda _: [], {}, _request(stream=True),
            channel_name="vertex", force_fake_streaming=True,
            retry_client_factory=_switching_factory(secondary, seen))
        lines = [line async for line in response.body_iterator]

        assert primary.aio.models.calls == 1
        assert secondary.aio.models.calls == 1
        assert _sse_text(lines) == "hello"
        assert lines.count("data: [DONE]\n\n") == 1
        assert not any('"error"' in line for line in lines)
        assert len(seen) == 1 and seen[0] is failure and seen[0].code == 429

    async def test_real_stream_after_committed_output_does_not_switch(self, monkeypatch):
        """已输出正文后再遇 429：不换凭证、不重试，单次调用并以失败终态收尾。"""
        helpers = _retry_env(monkeypatch)
        failure = _RetryableError(429)
        primary = _stream_client([[_stream_chunk("partial"), failure]])
        secondary = _stream_client([[_stream_chunk("must not run")]])
        seen = []
        response = await helpers.execute_gemini_call(
            primary, "gemini-3.8-flash", lambda _: [], {}, _request(stream=True),
            channel_name="vertex", retry_client_factory=_switching_factory(secondary, seen))
        lines = [line async for line in response.body_iterator]

        assert primary.aio.models.calls == 1
        assert secondary.aio.models.calls == 0
        assert seen == []
        assert _sse_text(lines) == "partial"
        assert lines.count("data: [DONE]\n\n") == 1
        assert any('"error"' in line for line in lines)


@pytest.mark.parametrize("mode", ["nonstream", "stream", "fake"])
async def test_explicit_403_with_quota_message_does_not_retry(monkeypatch, mode):
    helpers = _retry_env(monkeypatch)
    error = _RetryableError(403)
    primary = _stream_client([error]) if mode == "stream" else _call_client([error])
    seen = []
    kwargs = dict(channel_name="vertex", failover_mode=True,
                  force_fake_streaming=mode == "fake", retry_client_factory=_switching_factory(None, seen))
    if mode == "nonstream":
        with pytest.raises(_RetryableError, match="403"):
            await helpers.execute_gemini_call(primary, "gemini-3.8-flash", lambda _: [], {}, _request(False), **kwargs)
    else:
        response = await helpers.execute_gemini_call(primary, "gemini-3.8-flash", lambda _: [], {}, _request(True), **kwargs)
        errors = [item["error"] for item in _sse_payloads([line async for line in response.body_iterator]) if "error" in item]
        assert len(errors) == 1 and errors[0]["code"] == 403
    assert primary.aio.models.calls == 1 and not seen
