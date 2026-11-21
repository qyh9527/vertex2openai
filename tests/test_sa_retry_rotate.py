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


@pytest.mark.parametrize("project_b,mode,scenario", [
    pytest.param(project, mode, "success", id=f"{mode}-{'same' if project == 'proj-a' else 'other'}-project")
    for project in ("proj-a", "proj-b") for mode in ("nonstream", "stream", "fake-stream")
] + [
    pytest.param("proj-a", "stream", "partial", id="stream-breaks-after-content"),
    pytest.param("proj-a", "stream", "bad-terminal", id="stream-invalid-terminal"),
    pytest.param("proj-a", "nonstream", "failed", id="both-credentials-fail"),
    pytest.param("proj-a", "nonstream", "cooling", id="fallback-before-first-attempt"),
    pytest.param("proj-b", "nonstream", "disabled", id="affinity-disabled"),
])
async def test_sa_affinity_follows_only_successful_credentials(fresh, monkeypatch, project_b, mode, scenario):
    import sa_affinity
    from failover import ChannelBreaker
    from routes import chat_api
    from upstreams import express_sdk, service_account

    _retry_env(monkeypatch)
    accounts = [{"project_id": "proj-a", "location": "global", "sa_json": _SA_A},
                {"project_id": project_b, "location": "global", "sa_json": _SA_B}]
    monkeypatch.setattr(fresh, "get_sa_accounts", lambda: accounts)
    fresh.update_settings({"roundrobin": True})
    for module in (chat_api, express_sdk, service_account):
        monkeypatch.setattr(module, "app_state", fresh)
    circuit = ChannelBreaker()
    monkeypatch.setattr(chat_api, "breaker", circuit)
    monkeypatch.setattr(service_account, "breaker", circuit)
    monkeypatch.setattr(sa_affinity, "enabled", lambda: scenario != "disabled")
    monkeypatch.setattr(sa_budget, "status", lambda key: {"initialized": True, "blocked": False})
    monkeypatch.setattr(sa_budget, "reserve", lambda *a: "tok")
    monkeypatch.setattr(sa_budget, "settle", lambda *a: None)
    if mode == "stream":
        a = _stream_client([[_stream_chunk("first", "STOP")], _RetryableError(429)])
        events = [_stream_chunk("from B", "STOP")]
        if scenario == "partial":
            events = [_stream_chunk("partial from B"), _RetryableError(503)]
        elif scenario == "bad-terminal":
            events = [_stream_chunk("partial from B"), _stream_chunk(finish_reason="MALFORMED_FUNCTION_CALL")]
        b = _stream_client([events])
    else:
        a = _call_client([_text_response("first"), _RetryableError(429)])
        b = _call_client([_RetryableError(503) if scenario == "failed" else _text_response("from B")])

    async def count_a(**kwargs):
        return SimpleNamespace(total_tokens=7000)

    async def count_b(**kwargs):
        return SimpleNamespace(total_tokens=10)

    a.aio.models.count_tokens = count_a
    b.aio.models.count_tokens = count_b
    monkeypatch.setattr(service_account, "_get_cached_sa_client",
                        lambda sa_json, *args: {_SA_A: a, _SA_B: b}[sa_json])
    monkeypatch.setattr(chat_api, "CHANNELS", {"vertex": service_account.ServiceAccountUpstream()})

    session_id = sa_affinity.session_id_for_messages(_request(False).messages)
    key_a, key_b = map(sa_budget.credential_id, (_SA_A, _SA_B))

    async def ask(expect_failure=False):
        runtime_state._current_sa_account.set(None)
        request = _request(mode != "nonstream")
        if mode == "fake-stream":
            request.model = "fake-" + request.model
        response = await chat_api._dispatch(["vertex"], request, None, False, raw_mode=True)
        if request.stream:
            lines = [line async for line in response.body_iterator]
            assert any("error" in payload for payload in _sse_payloads(lines)) == expect_failure
            return _sse_text(lines)
        if expect_failure:
            assert response.status_code >= 400
            return None
        assert response.status_code == 200
        return json.loads(response.body)["choices"][0]["message"]["content"]

    assert await ask() == "first"
    if scenario != "disabled":
        assert sa_affinity.lookup_binding(session_id)[0].credential_id == key_a
    if scenario == "cooling":
        monkeypatch.setattr(runtime_state, "_sa_credential_cooling", lambda candidate: candidate["sa_json"] == _SA_A)
    fresh._sa_rr_index = 0
    if scenario in ("partial", "bad-terminal", "failed"):
        text = await ask(expect_failure=True)
        if mode == "stream":
            assert text == "partial from B"
        assert (a.aio.models.calls, b.aio.models.calls) == (2, 1)
        assert sa_affinity.lookup_binding(session_id)[0].credential_id == key_a
        return
    assert await ask() == "from B"
    calls_a = 1 if scenario == "cooling" else 2
    assert (a.aio.models.calls, b.aio.models.calls) == (calls_a, 1)
    if scenario == "disabled":
        assert sa_affinity.lookup_binding(session_id)[0] is None
        return
    binding, _ = sa_affinity.lookup_binding(session_id)
    assert (binding.project_id, binding.credential_id) == (project_b, key_b)
    # With A healthy and first in the round-robin pool, only affinity keeps this on B.
    monkeypatch.setattr(runtime_state, "_sa_credential_cooling", lambda candidate: False)
    fresh._sa_rr_index = 0
    assert await ask() == "from B"
    assert (a.aio.models.calls, b.aio.models.calls) == (calls_a, 2)

class TestRegionFallback:
    """global 429 时同一凭证换 us/eu 多区域节点：仅一次、不写账号配置、不计入凭证熔断。"""

    def _factory(self, fresh, monkeypatch, model, location="global", accounts=None,
                 allow_region_fallback=True, settings=None):
        from upstreams import service_account as sa_mod
        fresh.set_sa_accounts(accounts or [
            {"project_id": "proj-a", "location": "global", "sa_json": _SA_A},
        ])
        monkeypatch.setattr(
            sa_mod, "_get_cached_sa_client",
            lambda sa_json, project, client_location, headers, timeout:
                SimpleNamespace(sa_json=sa_json, project=project, location=client_location))
        monkeypatch.setattr(sa_budget, "BudgetClient",
                            lambda client, key, client_location, tier: client)
        monkeypatch.setattr(sa_mod, "app_state", fresh)
        monkeypatch.setattr(sa_mod, "breaker", Mock())
        upstream = sa_mod.ServiceAccountUpstream()
        effective_settings = dict(app_config.DEFAULT_SETTINGS)
        if settings:
            effective_settings.update(settings)
        factory = upstream.make_retry_client_factory(
            effective_settings, model, _SA_A, "proj-a", location,
            allow_region_fallback=allow_region_fallback)
        return factory, sa_mod

    async def test_two_attempts_try_both_regions_in_random_order(self, fresh, monkeypatch):
        from api_helpers import RetryStop
        factory, sa_mod = self._factory(
            fresh, monkeypatch, "gemini-4-neon",
            settings={"sa_region_fallback_attempts": 2})
        first = _RetryableError(429)
        client = await factory(first)
        first_location = client.location
        assert first_location in ("us", "eu")
        # 第二个区域必须与第一个不同：2 次 = 完整覆盖 us/eu，不重复同一个区域。
        client = await factory(_RetryableError(404))
        assert client.location in ("us", "eu")
        assert client.location != first_location
        assert runtime_state._pending_region_failure.get() is first
        # 两个区域都失败：保留最初的 global 错误供终态/hybrid 判断。
        with pytest.raises(RetryStop) as raised:
            await factory(_RetryableError(429))
        assert raised.value.original_error is first
        sa_mod.breaker.report_candidate_rate_limited.assert_called_once_with(
            "vertex", runtime_state.sa_breaker_id(_SA_A), str(first))
        sa_mod.breaker.report_failure.assert_not_called()

    async def test_two_attempts_still_requires_global_429(self, fresh, monkeypatch):
        """2 次配置不改变触发条件：global 502/503 仍走凭证轮换，不消耗区域尝试。"""
        accounts = [
            {"project_id": "proj-a", "location": "global", "sa_json": _SA_A},
            {"project_id": "proj-b", "location": "global", "sa_json": _SA_B},
        ]
        factory, _ = self._factory(
            fresh, monkeypatch, "gemini-4-hydrogen", accounts=accounts,
            settings={"sa_region_fallback_attempts": 2})
        client = await factory(_RetryableError(503))
        assert (client.sa_json, client.location) == (_SA_B, "global")

    async def test_two_attempts_budget_failure_skips_second_region(self, fresh, monkeypatch):
        from api_helpers import RetryStop
        factory, _ = self._factory(
            fresh, monkeypatch, "gemini-4-hydrogen",
            settings={"sa_region_fallback_attempts": 2})
        first = _RetryableError(429)
        await factory(first)
        budget_error = sa_budget.BudgetUnavailable("region reservation failed")
        with pytest.raises(RetryStop) as raised:
            await factory(budget_error)
        assert raised.value.original_error is budget_error

    async def test_region_construction_failure_still_tries_remaining_candidate(
            self, fresh, monkeypatch):
        """2 次配置下第一个区域构造失败，不能把第二个区域也一起跳过。"""
        from upstreams import service_account as sa_mod
        # 工厂在创建时就完成洗牌，必须先固定顺序，才能保证构造失败的是第一个区域。
        monkeypatch.setattr(sa_mod.random, "shuffle", lambda items: None)
        seen = []

        def build(sa_json, project, location, headers, timeout):
            seen.append(location)
            if location == "us":
                raise RuntimeError("us unavailable")
            return SimpleNamespace(sa_json=sa_json, project=project, location=location)

        factory, _ = self._factory(
            fresh, monkeypatch, "gemini-4-hydrogen",
            settings={"sa_region_fallback_attempts": 2})
        # _factory 会安装默认 client 构造器，创建工厂后再换成本用例的失败构造器。
        monkeypatch.setattr(sa_mod, "_get_cached_sa_client", build)
        client = await factory(_RetryableError(429))
        assert seen == ["us", "eu"]
        assert client.location == "eu"

    async def test_real_retry_loop_uses_both_regions_after_404(self, fresh, monkeypatch):
        """从真实 execute_gemini_call 入口验证：区域 404 仍会触发第二次区域尝试。"""
        from upstreams import service_account as sa_mod
        helpers = _retry_env(monkeypatch)
        monkeypatch.setattr(helpers, "get_retry_settings", lambda *a, **k: (2, 0))
        monkeypatch.setattr(sa_mod.random, "shuffle", lambda items: None)
        region_clients = {
            "us": _call_client([_RetryableError(404)]),
            "eu": _call_client([_text_response("from eu")]),
        }
        seen_locations = []

        def build(sa_json, project, location, headers, timeout):
            seen_locations.append(location)
            return region_clients[location]

        monkeypatch.setattr(sa_mod, "_get_cached_sa_client", build)
        monkeypatch.setattr(sa_budget, "BudgetClient",
                            lambda client, key, location, tier: client)
        fresh.set_sa_accounts([{"project_id": "proj-a", "location": "global", "sa_json": _SA_A}])
        monkeypatch.setattr(sa_mod, "app_state", fresh)
        monkeypatch.setattr(sa_mod, "breaker", Mock())
        factory = sa_mod.ServiceAccountUpstream().make_retry_client_factory(
            {**app_config.DEFAULT_SETTINGS, "sa_region_fallback_attempts": 2},
            "gemini-4-hydrogen", _SA_A, "proj-a", "global")
        primary = _call_client([_RetryableError(429)])
        response = await helpers.execute_gemini_call(
            primary, "gemini-4-hydrogen", lambda _: [], {}, _request(stream=False),
            channel_name="vertex", retry_client_factory=factory)
        assert response.status_code == 200
        assert json.loads(response.body)["choices"][0]["message"]["content"] == "from eu"
        assert seen_locations == ["us", "eu"]
        assert primary.aio.models.calls == 1
        assert region_clients["us"].aio.models.calls == 1
        assert region_clients["eu"].aio.models.calls == 1

    async def test_rotated_credential_uses_its_region_pin(self, fresh, monkeypatch, tmp_path):
        """通道内换到备用凭证时，也要用该凭证自己的 24 小时固定区域。"""
        import sa_region_pin
        monkeypatch.setenv("STATE_DIR", str(tmp_path))
        accounts = [
            {"project_id": "proj-a", "location": "global", "sa_json": _SA_A},
            {"project_id": "proj-b", "location": "global", "sa_json": _SA_B},
        ]
        factory, _ = self._factory(fresh, monkeypatch, "gemini-4-hydrogen", accounts=accounts)
        sa_region_pin.pin(sa_budget.credential_id(_SA_B), "eu")
        client = await factory(sa_budget.BudgetUnavailable("no budget"))
        assert (client.sa_json, client.location) == (_SA_B, "eu")

    async def test_non_gemini4_rotation_does_not_pin_region(self, fresh, monkeypatch, tmp_path):
        """非 Gemini 4 请求即使换到手动 us/eu 凭证，也不产生区域固定记录。"""
        import sa_region_pin
        monkeypatch.setenv("STATE_DIR", str(tmp_path))
        accounts = [
            {"project_id": "proj-a", "location": "global", "sa_json": _SA_A},
            {"project_id": "proj-b", "location": "us", "sa_json": _SA_B},
        ]
        factory, _ = self._factory(fresh, monkeypatch, "gemini-3.8-flash", accounts=accounts)
        client = await factory(_RetryableError(429))
        assert (client.sa_json, client.location) == (_SA_B, "us")
        await sa_region_pin.commit_success()
        assert sa_region_pin.lookup(sa_budget.credential_id(_SA_B)) is None

    async def test_successful_fallback_pins_credential_for_24h(self, fresh, monkeypatch, tmp_path):
        import sa_region_pin
        monkeypatch.setenv("STATE_DIR", str(tmp_path))
        factory, _ = self._factory(
            fresh, monkeypatch, "gemini-4-hydrogen",
            settings={"sa_region_fallback_attempts": 2})
        client = await factory(_RetryableError(429))
        # 模拟上游成功后才提交（本测试直接调用成功钩子）。
        await sa_region_pin.commit_success()
        assert sa_region_pin.lookup(sa_budget.credential_id(_SA_A)) == client.location

    async def test_global_overload_switches_region_once(self, fresh, monkeypatch):
        from api_helpers import RetryStop
        factory, sa_mod = self._factory(fresh, monkeypatch, "gemini-4-hydrogen")
        first = _RetryableError(429)
        client = await factory(first)
        assert client.location in ("us", "eu")
        assert runtime_state._current_pricing_location.get() == client.location
        sa_mod.breaker.report_candidate_rate_limited.assert_called_once_with(
            "vertex", runtime_state.sa_breaker_id(_SA_A), str(first))
        # 区域尝试失败且没有其它凭证：停止该通道重试，不再重复请求 us/eu。
        with pytest.raises(RetryStop):
            await factory(_RetryableError(429))
        assert sa_mod.breaker.report_candidate_rate_limited.call_count == 1
        sa_mod.breaker.report_failure.assert_not_called()

    async def test_budget_rotation_then_region_uses_active_credential_and_skips_breaker(self, fresh, monkeypatch):
        from api_helpers import RetryStop
        accounts = [
            {"project_id": "proj-a", "location": "global", "sa_json": _SA_A},
            {"project_id": "proj-b", "location": "global", "sa_json": _SA_B},
        ]
        factory, sa_mod = self._factory(fresh, monkeypatch, "gemini-4-hydrogen", accounts=accounts)
        # A 的本地预算不可用 → 先轮换到 B；B 再遇 429 → 应使用 B 的区域回退。
        client = await factory(sa_budget.BudgetUnavailable("no budget"))
        assert (client.sa_json, client.location) == (_SA_B, "global")
        region_error = _RetryableError(429)
        client = await factory(region_error)
        assert client.sa_json == _SA_B and client.location in ("us", "eu")
        sa_mod.breaker.report_candidate_rate_limited.assert_called_once_with(
            "vertex", runtime_state.sa_breaker_id(_SA_B), str(region_error))
        # B 的区域尝试失败后也不把失败记到 B 的 global 端点上；已无其它凭证 → 停止。
        with pytest.raises(RetryStop):
            await factory(_RetryableError(429))
        assert sa_mod.breaker.report_candidate_rate_limited.call_count == 1
        sa_mod.breaker.report_failure.assert_not_called()

    async def test_ineligible_model_or_non_global_rotates_credentials(self, fresh, monkeypatch):
        accounts = [
            {"project_id": "proj-a", "location": "global", "sa_json": _SA_A},
            {"project_id": "proj-b", "location": "global", "sa_json": _SA_B},
        ]
        factory, _ = self._factory(fresh, monkeypatch, "gemini-3.8-flash", accounts=accounts)
        client = await factory(_RetryableError(429))
        assert (client.sa_json, client.location) == (_SA_B, "global")

        factory, _ = self._factory(fresh, monkeypatch, "gemini-4-hydrogen",
                                   location="eu", accounts=accounts)
        client = await factory(_RetryableError(429))
        assert (client.sa_json, client.location) == (_SA_B, "global")

    async def test_region_fallback_only_on_429(self, fresh, monkeypatch):
        """502/503 仍按普通换号重试，不消耗一次性区域回退。"""
        accounts = [
            {"project_id": "proj-a", "location": "global", "sa_json": _SA_A},
            {"project_id": "proj-b", "location": "global", "sa_json": _SA_B},
        ]
        factory, _ = self._factory(fresh, monkeypatch, "gemini-4-hydrogen", accounts=accounts)
        client = await factory(_RetryableError(503))
        assert (client.sa_json, client.location) == (_SA_B, "global")

    async def test_pinned_resource_path_disables_region_fallback(self, fresh, monkeypatch):
        factory, _ = self._factory(fresh, monkeypatch, "gemini-4-hydrogen",
                                   allow_region_fallback=False)
        assert await factory(_RetryableError(429)) is None

    async def test_region_construction_failure_without_alternate_keeps_retrying(self, fresh, monkeypatch):
        factory, sa_mod = self._factory(fresh, monkeypatch, "gemini-4-hydrogen")

        def boom(sa_json, project, location, headers, timeout):
            raise RuntimeError("multi-region unavailable")

        monkeypatch.setattr(sa_mod, "_get_cached_sa_client", boom)
        # 没有真正发出区域请求：保留旧行为（返回 None → 用当前 client 重试），不触发 RetryStop。
        assert await factory(_RetryableError(429)) is None

    async def test_region_failure_reports_original_global_error_once(self, fresh, monkeypatch):
        from api_helpers import RetryStop
        from routes import chat_api
        from runtime_state import _current_region_fallback
        factory, sa_mod = self._factory(fresh, monkeypatch, "gemini-4-hydrogen")
        first = _RetryableError(429)
        client = await factory(first)
        assert client.location in ("us", "eu")
        with pytest.raises(RetryStop):
            await factory(_RetryableError(404))
        # 原始 global 限流在换区时已上报一次；终态不再重复上报区域错误。
        sa_mod.breaker.report_candidate_rate_limited.assert_called_once_with(
            "vertex", runtime_state.sa_breaker_id(_SA_A), str(first))
        monkeypatch.setattr(chat_api, "breaker", Mock())
        chat_api._report_failure("vertex", "cred-x", "rate_limited", "404")
        chat_api._report_success("vertex", "cred-x")
        chat_api.breaker.report_candidate_rate_limited.assert_not_called()
        chat_api.breaker.report_failure.assert_not_called()
        chat_api.breaker.report_success.assert_not_called()
        assert _current_region_fallback.get() is True

    async def test_retry_stop_carries_original_global_error(self, fresh, monkeypatch):
        from api_helpers import RetryStop, _region_terminal_error
        from runtime_state import _current_region_fallback, _pending_region_failure
        factory, _ = self._factory(fresh, monkeypatch, "gemini-4-hydrogen")
        first = _RetryableError(429)
        await factory(first)
        with pytest.raises(RetryStop) as raised:
            await factory(_RetryableError(404))
        assert raised.value.original_error is first
        # 即使最后一次尝试不再调用工厂，终态也应把区域 404 换回原始 global 错误。
        assert _region_terminal_error(_RetryableError(404)) is first
        # 区域 403 是永久错误，不能被换成原始 429 去触发通道切换。
        region_forbidden = _RetryableError(403)
        assert _region_terminal_error(region_forbidden) is region_forbidden
        budget_error = sa_budget.BudgetUnavailable("no budget")
        assert _region_terminal_error(budget_error) is budget_error
        _current_region_fallback.set(False)
        _pending_region_failure.set(None)

    async def test_region_failure_then_rotation_reports_original_global_error_once(self, fresh, monkeypatch):
        accounts = [
            {"project_id": "proj-a", "location": "global", "sa_json": _SA_A},
            {"project_id": "proj-b", "location": "global", "sa_json": _SA_B},
        ]
        factory, sa_mod = self._factory(fresh, monkeypatch, "gemini-4-hydrogen", accounts=accounts)
        first = _RetryableError(429)
        client = await factory(first)
        assert client.sa_json == _SA_A and client.location in ("us", "eu")
        # 区域尝试也失败后换到 B：只把最初的 global 失败上报给 A，区域尝试本身不计熔断。
        client = await factory(_RetryableError(429))
        assert (client.sa_json, client.location) == (_SA_B, "global")
        sa_mod.breaker.report_candidate_rate_limited.assert_called_once_with(
            "vertex", runtime_state.sa_breaker_id(_SA_A), str(first))
        # 已换到 B 后再次失败：不再被误判成「区域尝试」而抛 RetryStop，交回上层处理。
        assert await factory(_RetryableError(429)) is None
        assert sa_mod.breaker.report_candidate_rate_limited.call_count == 1

    async def test_region_budget_failure_without_alternate_keeps_budget_error(self, fresh, monkeypatch):
        from api_helpers import RetryStop
        factory, _ = self._factory(fresh, monkeypatch, "gemini-4-hydrogen")
        await factory(_RetryableError(429))
        budget_error = sa_budget.BudgetUnavailable("region reservation failed")
        with pytest.raises(RetryStop) as raised:
            await factory(budget_error)
        assert raised.value.original_error is budget_error

    async def test_region_client_failure_falls_back_to_rotation(self, fresh, monkeypatch):
        accounts = [
            {"project_id": "proj-a", "location": "global", "sa_json": _SA_A},
            {"project_id": "proj-b", "location": "global", "sa_json": _SA_B},
        ]
        factory, sa_mod = self._factory(fresh, monkeypatch, "gemini-4-hydrogen",
                                        accounts=accounts)

        def boom(sa_json, project, location, headers, timeout):
            if location != "global":
                raise RuntimeError("multi-region unavailable")
            return SimpleNamespace(sa_json=sa_json, project=project, location=location)

        monkeypatch.setattr(sa_mod, "_get_cached_sa_client", boom)
        client = await factory(_RetryableError(429))
        assert (client.sa_json, client.location) == (_SA_B, "global")

    def test_region_attempt_count_only_accepts_1_or_2(self):
        from upstreams.service_account import _region_attempt_count
        assert _region_attempt_count({"sa_region_fallback_attempts": 1}) == 1
        assert _region_attempt_count({"sa_region_fallback_attempts": 2}) == 2
        assert _region_attempt_count({"sa_region_fallback_attempts": "2"}) == 2
        for invalid in (0, 3, 2.5, "2.5", " 2 ", None, True):
            assert _region_attempt_count({"sa_region_fallback_attempts": invalid}) == 1

    def test_region_model_allowlist(self):
        from upstreams.service_account import _supports_region_fallback
        for model in ("gemini-4", "gemini-4-hydrogen", "gemini-4-neon",
                      "gemini-4-carbon-cyber-permissive"):
            assert _supports_region_fallback(model)
        assert _supports_region_fallback(
            "projects/p/locations/global/publishers/google/models/gemini-4-argon")
        assert _supports_region_fallback("gemini-4-carbon-search")
        assert not _supports_region_fallback("gemini-3.8-flash")
        assert not _supports_region_fallback("gemini-3.6-flash-search")


class TestRegionPin:
    @pytest.fixture(autouse=True)
    def store(self, tmp_path, monkeypatch):
        import sa_region_pin
        monkeypatch.setenv("STATE_DIR", str(tmp_path))
        return sa_region_pin

    def test_pin_roundtrip_does_not_refresh(self, store):
        key = sa_budget.credential_id(_SA_A)
        assert store.lookup(key) is None
        assert store.pin(key, "us") is True
        assert store.lookup(key) == "us"
        # 固定期内不因另一个区域成功而改写，也不滑动续期。
        assert store.pin(key, "eu") is False
        assert store.lookup(key) == "us"

    def test_pin_expires_after_exactly_24h(self, store, monkeypatch):
        key = sa_budget.credential_id(_SA_A)
        now = [1_000_000.0]
        monkeypatch.setattr(store, "_now", lambda: now[0])
        assert store.pin(key, "eu") is True
        now[0] += 24 * 60 * 60 - 1
        assert store.lookup(key) == "eu"
        now[0] += 2
        assert store.lookup(key) is None
        assert store.pin(key, "us") is True
        assert store.lookup(key) == "us"

    async def test_commit_success_pins_only_region_fallback(self, store):
        key = sa_budget.credential_id(_SA_A)
        runtime_state._current_sa_account.set({"sa_json": _SA_A, "location": "us"})
        runtime_state._current_pricing_location.set("us")
        runtime_state._current_region_fallback.set(False)
        await store.commit_success()
        assert store.lookup(key) is None
        runtime_state._current_region_fallback.set(True)
        await store.commit_success()
        assert store.lookup(key) == "us"

    async def test_commit_success_ignores_non_region_and_missing_account(self, store):
        key = sa_budget.credential_id(_SA_A)
        runtime_state._current_region_fallback.set(True)
        await store.commit_success()
        assert store.lookup(key) is None
        runtime_state._current_sa_account.set({"sa_json": _SA_A, "location": "us"})
        runtime_state._current_pricing_location.set("global")
        await store.commit_success()
        assert store.lookup(key) is None


async def test_nonstream_success_awaits_region_pin_hook(monkeypatch):
    """非流式成功必须真的调用固定钩子；错误响应不能写固定。"""
    from fastapi.responses import JSONResponse
    from upstreams import express_sdk as es

    upstream = es.ExpressSDKUpstream()
    resolved = {
        "client": object(), "model_to_call": "gemini-3.8-flash",
        "priority_paygo": False, "fallback_model": None,
        "fallback_client_factory": None, "retry_client_factory": None,
    }
    monkeypatch.setattr(upstream, "_resolve_client", lambda *args: resolved)
    calls = []

    async def fake_commit():
        calls.append(True)

    monkeypatch.setattr(es.sa_region_pin, "commit_success", fake_commit)

    async def successful_execute(*args, **kwargs):
        return JSONResponse(content={"choices": [{"message": {"content": "ok"}}]})

    monkeypatch.setattr(es, "execute_gemini_call", successful_execute)
    response = await upstream.chat_completions(_request(stream=False), object())
    assert response.status_code == 200
    assert calls == [True]

    async def failed_execute(*args, **kwargs):
        return JSONResponse(status_code=500, content={"error": {"message": "boom"}})

    monkeypatch.setattr(es, "execute_gemini_call", failed_execute)
    response = await upstream.chat_completions(_request(stream=False), object())
    assert response.status_code == 500
    assert calls == [True]


async def test_retry_client_propagates_retry_stop():
    from api_helpers import RetryStop, _retry_client

    async def factory(_error):
        raise RetryStop("stop")

    with pytest.raises(RetryStop):
        await _retry_client(object(), factory, _RetryableError(429))

def test_pinned_resource_location_parsing():
    from upstreams.express_sdk import pinned_resource_location
    base = "projects/locations/locations/global/publishers/google/models/gemini-4-hydrogen"
    assert pinned_resource_location(base) == "global"
    assert pinned_resource_location(
        "projects/p/locations/us-central1/publishers/google/models/gemini-4-hydrogen") == "us-central1"
    assert pinned_resource_location("gemini-4-hydrogen") is None
    assert pinned_resource_location("projects/p/models/gemini-4-hydrogen") is None
    # 保留原始段（不悄悄修剪），由 is_valid_location 拒绝带空白的路径。
    assert pinned_resource_location(
        "projects/p/locations/global /publishers/google/models/gemini-4-hydrogen") == "global "

class TestPinnedResourceCredentialRotation:
    """完整资源路径换号：沿用路径区域/项目，不换成备用账号的配置区域。"""

    async def test_pinned_path_disables_credential_rotation(self, fresh, monkeypatch):
        from upstreams import service_account as sa_mod
        accounts = [
            {"project_id": "proj-a", "location": "global", "sa_json": _SA_A},
            {"project_id": "proj-a", "location": "global", "sa_json": _SA_B},
        ]
        fresh.set_sa_accounts(accounts)
        monkeypatch.setattr(
            sa_mod, "_get_cached_sa_client",
            lambda sa_json, project, location, headers, timeout:
                SimpleNamespace(sa_json=sa_json, project=project, location=location))
        monkeypatch.setattr(sa_budget, "BudgetClient", lambda client, key, loc, tier: client)
        monkeypatch.setattr(sa_mod, "app_state", fresh)
        monkeypatch.setattr(sa_mod, "breaker", Mock())
        path = ("projects/proj-a/locations/us-central1/publishers/google/models/"
                "gemini-4-hydrogen")
        factory = sa_mod.ServiceAccountUpstream().make_retry_client_factory(
            dict(app_config.DEFAULT_SETTINGS), path, _SA_A, "proj-a", "us-central1",
            allow_region_fallback=False)
        # 路径已钉定项目；换号可能改变项目/权限，因此直接停止内部重试（返回 None）。
        assert await factory(_RetryableError(429)) is None

    async def test_pinned_path_with_different_project_also_does_not_rotate(self, fresh, monkeypatch):
        from upstreams import service_account as sa_mod
        fresh.set_sa_accounts([
            {"project_id": "proj-a", "location": "global", "sa_json": _SA_A},
            {"project_id": "proj-b", "location": "global", "sa_json": _SA_B},
        ])
        monkeypatch.setattr(sa_mod, "_get_cached_sa_client",
                            lambda *a, **k: SimpleNamespace(location=a[2]))
        monkeypatch.setattr(sa_budget, "BudgetClient", lambda client, key, loc, tier: client)
        monkeypatch.setattr(sa_mod, "app_state", fresh)
        monkeypatch.setattr(sa_mod, "breaker", Mock())
        path = ("projects/proj-a/locations/global/publishers/google/models/"
                "gemini-4-hydrogen")
        factory = sa_mod.ServiceAccountUpstream().make_retry_client_factory(
            dict(app_config.DEFAULT_SETTINGS), path, _SA_A, "proj-a", "global",
            allow_region_fallback=False)
        assert await factory(_RetryableError(429)) is None


class TestResolveClientPaths:
    def _prepare(self, monkeypatch, fresh, account_tuple=("proj-a", "global", _SA_A)):
        from upstreams import service_account as sa_mod
        monkeypatch.setattr(sa_mod, "app_state", SimpleNamespace(
            get_current_sa_account=lambda **kwargs: account_tuple))
        monkeypatch.setattr(sa_mod.sa_budget, "status",
                            lambda key: {"expired": False, "initialized": True, "blocked": False})
        monkeypatch.setattr(sa_mod.sa_budget, "price", lambda *args, **kwargs: None)
        monkeypatch.setattr(sa_mod, "_get_cached_sa_client", lambda *args, **kwargs: object())
        monkeypatch.setattr(sa_mod.sa_budget, "BudgetClient",
                            lambda client, key, location, tier: client)
        return sa_mod

    def test_pinned_target_project_keeps_affinity_on_credential_project(self, fresh, monkeypatch):
        import sa_affinity
        sa_mod = self._prepare(monkeypatch, fresh)
        path = "projects/proj-b/locations/global/publishers/google/models/gemini-4-hydrogen"
        token = sa_affinity._session_id.set("session-x")
        try:
            resolved = sa_mod.ServiceAccountUpstream()._resolve_client(
                None, path, dict(app_config.DEFAULT_SETTINGS))
        finally:
            sa_affinity._session_id.reset(token)
        assert resolved["sa_affinity"]["project_id"] == "proj-a"

    def test_invalid_path_returns_400_before_budget(self, fresh, monkeypatch):
        sa_mod = self._prepare(monkeypatch, fresh)
        monkeypatch.setattr(sa_mod.sa_budget, "status",
                            lambda key: (_ for _ in ()).throw(AssertionError("budget called")))
        resolved = sa_mod.ServiceAccountUpstream()._resolve_client(
            None, "projects/p/locations//publishers/google/models/gemini-4-hydrogen",
            dict(app_config.DEFAULT_SETTINGS))
        assert resolved["error"].status_code == 400


def test_resource_location_error_distinguishes_missing_and_invalid():
    from upstreams.express_sdk import resource_location_error
    assert resource_location_error(
        "projects/p/locations/global/publishers/google/models/gemini-4-hydrogen") is None
    assert resource_location_error("projects/p/models/gemini-4-hydrogen") is None
    assert "缺少" in resource_location_error(
        "projects/p/locations//publishers/google/models/gemini-4-hydrogen")
    assert "非法" in resource_location_error(
        "projects/p/locations/global /publishers/google/models/gemini-4-hydrogen")


def test_location_segment_validation():
    from upstreams.express_sdk import is_valid_location
    assert is_valid_location("us-central1")
    assert is_valid_location("eu")
    assert not is_valid_location("US")
    assert not is_valid_location("EU")
    assert not is_valid_location("evil.com/path")
    assert not is_valid_location("us@attacker")
    assert not is_valid_location(" us")
    assert not is_valid_location("us ")
    assert not is_valid_location("")
