"""SA 会话亲和测试：指纹稳定性 / TTL 滑动 / 容量淘汰 / 门槛 / 成功后才写 / 故障语义。"""
import pytest

import sa_affinity
import config as app_config
from models import OpenAIRequest
from runtime_state import app_state


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    """每个测试独立 SQLite + 开关打开 + 清空请求级 ContextVar。"""
    monkeypatch.setattr(sa_affinity, "_db_path",
                        lambda: tmp_path / "sa_affinity.sqlite3")
    monkeypatch.setattr(sa_affinity, "enabled", lambda: True)
    sa_affinity._session_id.set(None)
    sa_affinity._selection.set(None)
    sa_affinity._pending.set(None)
    yield
    sa_affinity._session_id.set(None)
    sa_affinity._selection.set(None)
    sa_affinity._pending.set(None)


def _msgs(*roles, content="hi"):
    return [{"role": role, "content": content} for role in roles]


class TestFingerprint:
    def test_stable_across_appended_turns(self):
        base = _msgs("system", "user")
        first = sa_affinity.session_id_for_messages(base)
        grown = base + _msgs("assistant") + _msgs("user", content="next")
        assert first and first == sa_affinity.session_id_for_messages(grown)

    def test_changed_system_changes_id(self):
        a = sa_affinity.session_id_for_messages(_msgs("system", "user", content="A"))
        b = sa_affinity.session_id_for_messages(_msgs("system", "user", content="B"))
        assert a and b and a != b

    def test_no_system_no_user_is_unstable(self):
        assert sa_affinity.session_id_for_messages(_msgs("assistant")) == ""
        assert sa_affinity.session_id_for_messages([]) == ""

    def test_system_only_is_stable(self):
        assert sa_affinity.session_id_for_messages(_msgs("system")) != ""

    def test_content_part_order_encoding_stable(self):
        a = [{"role": "system", "content": [{"type": "text", "text": "x"}]}, {"role": "user", "content": "hi"}]
        b = [{"role": "system", "content": [{"text": "x", "type": "text"}]}, {"role": "user", "content": "hi"}]
        assert sa_affinity.session_id_for_messages(a) == sa_affinity.session_id_for_messages(b)

    def test_id_is_32_hex(self):
        sid = sa_affinity.session_id_for_messages(_msgs("system", "user"))
        assert len(sid) == 32 and int(sid, 16) >= 0


class TestStorage:
    def test_ttl_expiry(self, monkeypatch):
        now = [1_000_000.0]
        monkeypatch.setattr(sa_affinity.time, "time", lambda: now[0])
        sa_affinity._write("sid", "proj")
        assert sa_affinity._lookup("sid")[0] == "proj"
        now[0] += sa_affinity.TTL_SECONDS + 1
        assert sa_affinity._lookup("sid")[0] is None

    def test_sliding_renewal(self, monkeypatch):
        now = [1_000_000.0]
        monkeypatch.setattr(sa_affinity.time, "time", lambda: now[0])
        sa_affinity._write("sid", "proj")
        now[0] += sa_affinity.TTL_SECONDS - 10
        assert sa_affinity._lookup("sid")[0] == "proj"
        now[0] += sa_affinity.TTL_SECONDS - 10   # 距上次访问 < TTL（距创建已 > TTL）
        assert sa_affinity._lookup("sid")[0] == "proj"

    def test_capacity_evicts_oldest(self, monkeypatch):
        now = [1_000_000.0]
        monkeypatch.setattr(sa_affinity.time, "time", lambda: now[0])
        monkeypatch.setattr(sa_affinity, "MAX_BINDINGS", 2)
        for sid in ("a", "b", "c"):
            sa_affinity._write(sid, "proj")
            now[0] += 1
        assert sa_affinity._lookup("a")[0] is None
        assert sa_affinity._lookup("b")[0] == "proj"
        assert sa_affinity._lookup("c")[0] == "proj"
        assert sa_affinity.stats()["count"] == 2

    def test_clear_all(self, monkeypatch):
        monkeypatch.setattr(sa_affinity.time, "time", lambda: 1_000_000.0)
        sa_affinity._write("a", "p1")
        sa_affinity._write("b", "p2")
        assert sa_affinity.clear_all() == 2
        assert sa_affinity.stats()["count"] == 0


class TestThreshold:
    def test_model_thresholds(self):
        assert sa_affinity.min_tokens_for_model("gemini-3.8-flash") == 6144
        assert sa_affinity.min_tokens_for_model("gemini-3.1-pro-preview") == 6144
        assert sa_affinity.min_tokens_for_model("gemini-3.7-flash") == 6144
        assert sa_affinity.min_tokens_for_model("gemini-3.0-flash-preview") == 6144
        assert sa_affinity.min_tokens_for_model("gemini-3.6-flash") == 4096
        assert sa_affinity.min_tokens_for_model("gemini-4-carbon") == 4096
        assert sa_affinity.min_tokens_for_model("gemini-2.5-flash") == 2048

    def test_projects_path_stripped(self):
        assert sa_affinity.min_tokens_for_model(
            "projects/p/locations/global/publishers/google/models/gemini-3.8-flash") == 6144


def _begin(state="new", project="proj", model="gemini-3.8-flash"):
    sa_affinity.begin_commit({
        "session_id": "sid", "project_id": project, "state": state, "model": model})
    sa_affinity._selection.set({"session_id": "sid", "bound_project": None, "state": state})


class TestCommit:
    def test_new_below_threshold_not_written(self):
        _begin("new")
        sa_affinity.note_prompt_tokens(4095)
        sa_affinity.commit_success()
        assert sa_affinity._lookup("sid")[0] is None

    def test_new_at_threshold_written(self):
        _begin("new")
        sa_affinity.note_prompt_tokens(6144)
        sa_affinity.commit_success()
        assert sa_affinity._lookup("sid")[0] == "proj"

    def test_count_tokens_failure_never_writes(self):
        _begin("new")
        sa_affinity.commit_success()      # 没有 prompt token 数
        assert sa_affinity._lookup("sid")[0] is None

    def test_structural_rebind_ignores_threshold(self):
        _begin("structural")
        sa_affinity.note_prompt_tokens(10)
        sa_affinity.commit_success()
        assert sa_affinity._lookup("sid")[0] == "proj"

    def test_transient_fallback_never_writes(self):
        _begin("transient", project="other")
        sa_affinity.note_prompt_tokens(999999)
        sa_affinity.commit_success()
        assert sa_affinity._lookup("sid")[0] is None

    def test_hit_does_not_rewrite(self, monkeypatch):
        monkeypatch.setattr(sa_affinity.time, "time", lambda: 1_000_000.0)
        sa_affinity._write("sid", "old")
        _begin("hit", project="new")
        sa_affinity.note_prompt_tokens(999999)
        sa_affinity.commit_success()
        assert sa_affinity._lookup("sid")[0] == "old"

    def test_note_request_then_commit_roundtrip(self, monkeypatch):
        now = [1_000_000.0]
        monkeypatch.setattr(sa_affinity.time, "time", lambda: now[0])
        request = OpenAIRequest(model="gemini-3.8-flash",
                                messages=[{"role": "system", "content": "sys"},
                                          {"role": "user", "content": "hi"}])
        assert sa_affinity.note_request(request) is None
        sa_affinity.note_selection("proj", "new")
        decision = sa_affinity.build_decision("proj", "gemini-3.8-flash")
        sa_affinity.begin_commit(decision)
        sa_affinity.note_prompt_tokens(7000)
        sa_affinity.commit_success()
        now[0] += 60
        again = sa_affinity.note_request(request)
        assert again == "proj"
        assert sa_affinity._selection.get()["state"] == "hit"


class TestRuntimeSelection:
    def _accounts(self, *projects):
        return [{"project_id": p, "location": "global", "sa_json": f'{{"project_id":"{p}"}}'}
                for p in projects]

    def test_preferred_project_selected(self, monkeypatch):
        accounts = self._accounts("p1", "p2")
        monkeypatch.setattr(app_state, "get_sa_accounts", lambda: accounts)
        monkeypatch.setattr(app_state, "get_setting", lambda key, default=None: True)
        monkeypatch.setattr("runtime_state._sa_budget_available", lambda account, module: True)
        monkeypatch.setattr("runtime_state._sa_credential_cooling", lambda account: False)
        project, _, _ = app_state.get_current_sa_account(preferred_project="p2")
        assert project == "p2"

    def test_absent_preferred_falls_back(self, monkeypatch):
        accounts = self._accounts("p1")
        monkeypatch.setattr(app_state, "get_sa_accounts", lambda: accounts)
        monkeypatch.setattr(app_state, "get_setting", lambda key, default=None: True)
        monkeypatch.setattr("runtime_state._sa_budget_available", lambda account, module: True)
        monkeypatch.setattr("runtime_state._sa_credential_cooling", lambda account: False)
        assert app_state.get_current_sa_account(preferred_project="gone")[0] == "p1"

    def test_cooling_preferred_uses_other_project(self, monkeypatch):
        accounts = self._accounts("p1", "p2")
        monkeypatch.setattr(app_state, "get_sa_accounts", lambda: accounts)
        monkeypatch.setattr(app_state, "get_setting", lambda key, default=None: True)
        monkeypatch.setattr("runtime_state._sa_budget_available", lambda account, module: True)
        monkeypatch.setattr("runtime_state._sa_credential_cooling",
                            lambda account: account["project_id"] == "p2")
        assert app_state.get_current_sa_account(preferred_project="p2")[0] == "p1"

    def test_project_availability(self, monkeypatch):
        accounts = self._accounts("p1", "p2")
        monkeypatch.setattr(app_state, "get_sa_accounts", lambda: accounts)
        monkeypatch.setattr("runtime_state._sa_budget_available",
                            lambda account, module: account["project_id"] == "p1")
        assert app_state.sa_project_availability("p1") == "eligible"
        assert app_state.sa_project_availability("p2") == "blocked"
        assert app_state.sa_project_availability("gone") == "absent"

class TestDispatchWiring:
    def _request(self):
        return OpenAIRequest(model="gemini-3.8-flash",
                             messages=[{"role": "system", "content": "sys"},
                                       {"role": "user", "content": "hi"}])

    def _fake_sa(self, status=200):
        from fastapi.responses import JSONResponse

        class FakeSA:
            async def chat_completions(self, request, fastapi_request, failover_mode=False,
                                       raw_mode=False):
                return JSONResponse(status_code=status, content={"ok": True})
        return FakeSA()

    def _install(self, monkeypatch, bound, availability="eligible"):
        from routes import chat_api
        from failover import ChannelBreaker
        seen = {}
        monkeypatch.setattr(chat_api, "CHANNELS", {"vertex": self._fake_sa()})
        monkeypatch.setattr(chat_api, "breaker", ChannelBreaker())
        monkeypatch.setattr(sa_affinity, "lookup_binding", lambda session_id: (bound, 0.0))
        monkeypatch.setattr(sa_affinity, "apply_lookup", lambda sid, project, age=0.0: None)
        monkeypatch.setattr(sa_affinity, "preferred_project", lambda: bound)
        monkeypatch.setattr(sa_affinity, "current_selection",
                            lambda: {"session_id": "sid", "bound_project": bound})
        monkeypatch.setattr(sa_affinity, "effective_project", lambda pid, sj: "p1")
        monkeypatch.setattr(sa_affinity, "note_selection",
                            lambda project, state: seen.update(project=project, state=state))

        def fake_account(preferred_project=None):
            from runtime_state import _current_sa_account
            if preferred_project is not None:
                seen["preferred"] = preferred_project
            snapshot = _current_sa_account.get()
            if snapshot:
                return (snapshot["project_id"], snapshot["location"], snapshot["sa_json"])
            return ("p1", "global", '{"project_id":"p1"}')

        monkeypatch.setattr(app_state, "get_current_sa_account", fake_account)
        monkeypatch.setattr(app_state, "sa_project_availability", lambda project: availability)
        return chat_api, seen

    def test_preferred_project_passed_and_hit(self, monkeypatch):
        import asyncio
        chat_api, seen = self._install(monkeypatch, bound="p1")
        asyncio.run(chat_api._dispatch(["vertex"], self._request(), None, False))
        assert seen["preferred"] == "p1"
        assert seen["project"] == "p1" and seen["state"] == "hit"

    def test_new_session_state(self, monkeypatch):
        import asyncio
        chat_api, seen = self._install(monkeypatch, bound=None)
        asyncio.run(chat_api._dispatch(["vertex"], self._request(), None, False))
        assert seen.get("preferred") is None
        assert seen["state"] == "new"

    def test_structural_rebind_state(self, monkeypatch):
        import asyncio
        chat_api, seen = self._install(monkeypatch, bound="p9", availability="blocked")
        asyncio.run(chat_api._dispatch(["vertex"], self._request(), None, False))
        assert seen["preferred"] == "p9"
        assert seen["state"] == "structural"

    def test_transient_fallback_state(self, monkeypatch):
        import asyncio
        chat_api, seen = self._install(monkeypatch, bound="p9", availability="eligible")
        asyncio.run(chat_api._dispatch(["vertex"], self._request(), None, False))
        assert seen["state"] == "transient"

    def test_express_only_does_not_touch_affinity(self, monkeypatch):
        import asyncio
        from routes import chat_api
        from failover import ChannelBreaker
        called = {"note": 0}
        monkeypatch.setattr(sa_affinity, "prepare_request",
                            lambda request: called.__setitem__("note", called["note"] + 1))
        monkeypatch.setattr(chat_api, "breaker", ChannelBreaker())
        monkeypatch.setattr(chat_api, "CHANNELS", {"express": self._fake_sa()})
        monkeypatch.setattr(app_config, "VERTEX_EXPRESS_API_KEY_VAL", ["k"])
        monkeypatch.setattr(app_state, "get_express_keys", lambda: ["k"])
        asyncio.run(chat_api._dispatch(["express"], self._request(), None, False))
        assert called["note"] == 0


class TestUpstreamCommit:
    def _upstream(self, monkeypatch, response_status):
        from fastapi.responses import JSONResponse, StreamingResponse
        from upstreams.express_sdk import ExpressSDKUpstream
        upstream = ExpressSDKUpstream()
        monkeypatch.setattr(upstream, "_resolve_client",
                            lambda fastapi_request, base, settings: {
                                "client": object(), "model_to_call": base,
                                "priority_paygo": False, "fallback_model": None,
                                "fallback_client_factory": None,
                                "sa_affinity": {
                                    "session_id": "sid", "project_id": "p1",
                                    "state": "new", "model": "gemini-3.8-flash"},
                            })
        monkeypatch.setattr("upstreams.express_sdk.validate_request_schemas", lambda request: None)

        async def fake_execute(*args, **kwargs):
            sa_affinity.note_prompt_tokens(7000)
            return JSONResponse(status_code=response_status, content={"ok": True})

        monkeypatch.setattr("upstreams.express_sdk.execute_gemini_call", fake_execute)
        return upstream

    def test_success_writes_binding(self, monkeypatch):
        import asyncio
        upstream = self._upstream(monkeypatch, 200)
        request = OpenAIRequest(model="gemini-3.8-flash",
                                messages=[{"role": "system", "content": "sys"},
                                          {"role": "user", "content": "hi"}])
        asyncio.run(upstream.chat_completions(request, None))
        assert sa_affinity._lookup("sid")[0] == "p1"

    def test_failure_does_not_write(self, monkeypatch):
        import asyncio
        upstream = self._upstream(monkeypatch, 400)
        request = OpenAIRequest(model="gemini-3.8-flash",
                                messages=[{"role": "system", "content": "sys"},
                                          {"role": "user", "content": "hi"}])
        asyncio.run(upstream.chat_completions(request, None))
        assert sa_affinity._lookup("sid")[0] is None


class TestStreamCommit:
    def _budget_models(self, monkeypatch, fail=False, fail_on_iter=False):
        import sa_budget
        monkeypatch.setattr(sa_budget, "reserve", lambda key, amount: "tok")
        monkeypatch.setattr(sa_budget, "settle", lambda token, amount: None)
        monkeypatch.setattr(sa_budget, "price", lambda *a, **k: 1)

        class FakeModels:
            async def count_tokens(self, **kwargs):
                class R:
                    total_tokens = 7000
                return R()

            async def generate_content_stream(self, **kwargs):
                if fail:
                    raise RuntimeError("upstream rejected")
                async def gen():
                    if fail_on_iter:
                        raise RuntimeError("first chunk failed")
                    yield object()
                return gen()

        return sa_budget.BudgetModels(FakeModels(), "key", "global", "standard")

    def test_stream_commit_after_first_chunk(self, monkeypatch):
        import asyncio
        models = self._budget_models(monkeypatch)
        _begin("new", project="p1")

        async def run():
            stream = await models.generate_content_stream(
                model="gemini-3.8-flash", contents=[])
            assert sa_affinity._lookup("sid")[0] is None   # 尚未取到首个 chunk
            async for _ in stream:
                pass

        asyncio.run(run())
        assert sa_affinity._lookup("sid")[0] == "p1"

    def test_stream_first_iteration_failure_does_not_write(self, monkeypatch):
        import asyncio
        models = self._budget_models(monkeypatch, fail_on_iter=True)
        _begin("new", project="p1")

        async def run():
            stream = await models.generate_content_stream(
                model="gemini-3.8-flash", contents=[])
            async for _ in stream:
                pass

        with pytest.raises(RuntimeError):
            asyncio.run(run())
        assert sa_affinity._lookup("sid")[0] is None

    def test_stream_failure_does_not_write(self, monkeypatch):
        import asyncio
        models = self._budget_models(monkeypatch, fail=True)
        _begin("new")
        with pytest.raises(RuntimeError):
            asyncio.run(models.generate_content_stream(model="gemini-3.8-flash", contents=[]))
        assert sa_affinity._lookup("sid")[0] is None

    def test_fake_stream_empty_stop_does_not_write(self, monkeypatch):
        """假流式空 STOP（无正文/工具）与非流式一样判为无效，不得写亲和。"""
        import asyncio
        from google.genai import types
        from api_helpers import gemini_fake_stream_generator

        class FakeModels:
            async def generate_content(self, **kwargs):
                return types.GenerateContentResponse(
                    candidates=[types.Candidate(
                        finish_reason=types.FinishReason.STOP,
                        content=types.Content(parts=[]))],
                    usage_metadata=types.UsageMetadata(
                        prompt_token_count=10, candidates_token_count=0, total_token_count=10))

        class FakeClient:
            class aio:
                models = FakeModels()

        _begin("new", project="p1")
        request = OpenAIRequest(model="gemini-3.8-flash", stream=True,
                                messages=[{"role": "user", "content": "hi"}])

        async def run():
            chunks = []
            async for chunk in gemini_fake_stream_generator(
                    FakeClient(), "gemini-3.8-flash", [], {}, request, False,
                    channel_name="express"):
                chunks.append(chunk)
            return chunks

        chunks = asyncio.run(run())
        assert any('"error"' in chunk for chunk in chunks)
        assert sa_affinity._lookup("sid")[0] is None


class TestOcrRegressions:
    def test_post_user_system_included_in_fingerprint(self):
        a = [{"role": "user", "content": "hi"}, {"role": "system", "content": "A"}]
        b = [{"role": "user", "content": "hi"}, {"role": "system", "content": "B"}]
        assert sa_affinity.session_id_for_messages(a) != sa_affinity.session_id_for_messages(b)

    def test_lookup_failure_is_isolated(self, monkeypatch):
        monkeypatch.setattr(sa_affinity, "_lookup", lambda sid: (_ for _ in ()).throw(RuntimeError("db down")))
        bound, age = sa_affinity.lookup_binding("sid")
        assert bound is None and age == 0.0

    def test_write_failure_is_isolated(self, monkeypatch):
        monkeypatch.setattr(sa_affinity, "_write_decision",
                            lambda decision: (_ for _ in ()).throw(RuntimeError("disk full")))
        sa_affinity.write_decision({"session_id": "sid", "project_id": "p1", "state": "new"})

    def test_new_binding_never_overwrites_existing(self):
        sa_affinity._write("sid", "first")
        sa_affinity._write_decision({
            "session_id": "sid", "project_id": "second", "state": "new",
            "bound_project": None})
        assert sa_affinity._lookup("sid")[0] == "first"

    def test_structural_rebind_only_replaces_observed_binding(self):
        sa_affinity._write("sid", "current")
        sa_affinity._write_decision({
            "session_id": "sid", "project_id": "new", "state": "structural",
            "bound_project": "stale"})
        assert sa_affinity._lookup("sid")[0] == "current"
        sa_affinity._write_decision({
            "session_id": "sid", "project_id": "new", "state": "structural",
            "bound_project": "current"})
        assert sa_affinity._lookup("sid")[0] == "new"

    def test_media_prompt_uses_real_count_tokens(self, monkeypatch):
        import asyncio
        import sa_budget
        monkeypatch.setattr(sa_budget, "reserve", lambda key, amount: "tok")
        monkeypatch.setattr(sa_budget, "price", lambda *a, **k: 1)

        class FakeModels:
            async def count_tokens(self, **kwargs):
                class R:
                    total_tokens = 321
                return R()

        models = sa_budget.BudgetModels(FakeModels(), "key", "global", "standard")
        _begin("new")
        contents = [{"role": "user", "parts": [{"inline_data": {"mime_type": "image/png", "data": "x"}}]}]
        asyncio.run(models._reserve({"model": "gemini-3.8-flash", "contents": contents, "config": {}}))
        assert sa_affinity._pending.get()["prompt_tokens"] == 321
        sa_affinity.commit_success()
        assert sa_affinity._lookup("sid")[0] is None      # 321 < 6144，不建绑
