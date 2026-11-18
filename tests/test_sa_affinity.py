"""SA 会话亲和测试：指纹稳定性 / TTL 滑动 / 容量淘汰 / 门槛 / 成功后才写 / 故障语义。"""
import asyncio
from contextlib import closing
import sqlite3

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
    sa_affinity.reset_request()
    yield
    sa_affinity.reset_request()


def _msgs(*roles, content="hi"):
    return [{"role": role, "content": content} for role in roles]


def _seed(sid, project, credential="key"):
    assert sa_affinity._write_decision({"session_id": sid, "project_id": project,
                                        "credential_id": credential})
    return sa_affinity._lookup(sid)[0]


def _project(sid):
    binding, _ = sa_affinity._lookup(sid)
    return binding.project_id if binding else None


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
        _seed("sid", "proj")
        assert _project("sid") == "proj"
        now[0] += sa_affinity.TTL_SECONDS + 1
        assert _project("sid") is None

    def test_sliding_renewal(self, monkeypatch):
        now = [1_000_000.0]
        monkeypatch.setattr(sa_affinity.time, "time", lambda: now[0])
        _seed("sid", "proj")
        now[0] += sa_affinity.TTL_SECONDS - 10
        assert _project("sid") == "proj"
        now[0] += sa_affinity.TTL_SECONDS - 10   # 距上次访问 < TTL（距创建已 > TTL）
        assert _project("sid") == "proj"

    def test_capacity_evicts_oldest(self, monkeypatch):
        now = [1_000_000.0]
        monkeypatch.setattr(sa_affinity.time, "time", lambda: now[0])
        monkeypatch.setattr(sa_affinity, "MAX_BINDINGS", 2)
        for sid in ("a", "b", "c"):
            _seed(sid, "proj")
            now[0] += 1
        assert _project("a") is None
        assert _project("b") == "proj"
        assert _project("c") == "proj"
        assert sa_affinity.stats()["count"] == 2

    def test_clear_all(self, monkeypatch):
        monkeypatch.setattr(sa_affinity.time, "time", lambda: 1_000_000.0)
        _seed("a", "p1")
        _seed("b", "p2")
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


def _begin(project="proj", credential="key", previous=None):
    sa_affinity.apply_lookup("sid", previous, 0)
    sa_affinity.begin_commit(sa_affinity.build_decision(project, credential, "gemini-3.8-flash"))


class TestCommit:
    @pytest.mark.parametrize("tokens,expected", [(4095, None), (6144, "proj"), (None, None)])
    def test_only_new_bindings_require_count_tokens_threshold(self, tokens, expected):
        _begin()
        sa_affinity.note_prompt_tokens(tokens)
        asyncio.run(sa_affinity.commit_success())
        assert _project("sid") == expected

    def test_switch_ignores_threshold(self):
        original = _seed("sid", "proj", "first")
        _begin(credential="second", previous=original)
        sa_affinity.note_prompt_tokens(10)
        asyncio.run(sa_affinity.commit_success())
        assert sa_affinity._lookup("sid")[0].credential_id == "second"

    def test_hit_does_not_rewrite(self):
        original = _seed("sid", "proj")
        _begin(previous=original)
        assert sa_affinity.take_pending() is None

    def test_legacy_project_binding_is_preserved_and_upgraded(self):
        with closing(sqlite3.connect(sa_affinity._db_path())) as db, db:
            db.execute("CREATE TABLE affinity (session_id TEXT PRIMARY KEY, project_id TEXT NOT NULL, "
                       "created_at REAL NOT NULL, last_seen REAL NOT NULL)")
            now = sa_affinity.time.time()
            db.execute("INSERT INTO affinity VALUES (?, ?, ?, ?)", ("sid", "proj", now, now))
        binding, age = sa_affinity.lookup_binding("sid")
        assert binding == sa_affinity.Binding("proj", "", now)
        sa_affinity.apply_lookup("sid", binding, age)
        assert sa_affinity.selection_preferences() == {"preferred_project": "proj"}
        _begin(previous=binding)
        asyncio.run(sa_affinity.commit_success())
        assert sa_affinity.lookup_binding("sid")[0].credential_id == "key"
        # Older versions only read/write their original columns; the schema remains usable.
        with closing(sqlite3.connect(sa_affinity._db_path())) as db, db:
            db.execute("INSERT INTO affinity(session_id,project_id,created_at,last_seen) "
                       "VALUES (?, ?, ?, ?)", ("old-worker", "proj", now, now))
        assert sa_affinity.lookup_binding("old-worker")[0].credential_id == ""


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

class TestDispatchWiring:
    @pytest.mark.parametrize("bound_credential", ["", "account-1"], ids=["legacy", "credential"])
    def test_passes_affinity_preferences(self, monkeypatch, bound_credential):
        from fastapi.responses import JSONResponse
        from failover import ChannelBreaker
        from routes import chat_api

        class FakeSA:
            async def chat_completions(self, request, fastapi_request, **kwargs):
                return JSONResponse({"ok": True})

        request = OpenAIRequest(model="gemini-3.8-flash", messages=_msgs("system", "user"))
        sid = sa_affinity.session_id_for_messages(request.messages)
        _seed(sid, "p1", bound_credential)
        seen = {}

        def fake_account(**kwargs):
            seen.update(kwargs)
            return "p1", "global", '{"project_id":"p1"}'

        monkeypatch.setattr(chat_api, "CHANNELS", {"vertex": FakeSA()})
        monkeypatch.setattr(chat_api, "breaker", ChannelBreaker())
        monkeypatch.setattr(app_state, "get_current_sa_account", fake_account)
        response = asyncio.run(chat_api._dispatch(["vertex"], request, None, False))
        expected = {"preferred_project": "p1"}
        if bound_credential:
            expected["preferred_credential"] = bound_credential
        assert response.status_code == 200
        assert seen == expected

    def test_express_only_does_not_touch_affinity(self, monkeypatch):
        from fastapi.responses import JSONResponse
        from routes import chat_api
        from failover import ChannelBreaker
        called = {"note": 0}
        monkeypatch.setattr(sa_affinity, "lookup_binding",
                            lambda sid: called.__setitem__("note", called["note"] + 1))
        monkeypatch.setattr(chat_api, "breaker", ChannelBreaker())
        class FakeExpress:
            async def chat_completions(self, request, fastapi_request, **kwargs):
                return JSONResponse({"ok": True})

        monkeypatch.setattr(chat_api, "CHANNELS", {"express": FakeExpress()})
        monkeypatch.setattr(app_config, "VERTEX_EXPRESS_API_KEY_VAL", ["k"])
        monkeypatch.setattr(app_state, "get_express_keys", lambda: ["k"])
        request = OpenAIRequest(model="gemini-3.8-flash", messages=_msgs("user"))
        asyncio.run(chat_api._dispatch(["express"], request, None, False))
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
                                    "credential_id": "key", "model": "gemini-3.8-flash"},
                            })
        monkeypatch.setattr("upstreams.express_sdk.validate_request_schemas", lambda request: None)

        async def fake_execute(*args, **kwargs):
            sa_affinity.note_prompt_tokens(7000)
            return JSONResponse(status_code=response_status, content={"ok": True})

        monkeypatch.setattr("upstreams.express_sdk.execute_gemini_call", fake_execute)
        return upstream

    def test_success_writes_binding(self, monkeypatch):
        upstream = self._upstream(monkeypatch, 200)
        request = OpenAIRequest(model="gemini-3.8-flash",
                                messages=[{"role": "system", "content": "sys"},
                                          {"role": "user", "content": "hi"}])
        asyncio.run(upstream.chat_completions(request, None))
        assert _project("sid") == "p1"

    def test_failure_does_not_write(self, monkeypatch):
        upstream = self._upstream(monkeypatch, 400)
        request = OpenAIRequest(model="gemini-3.8-flash",
                                messages=[{"role": "system", "content": "sys"},
                                          {"role": "user", "content": "hi"}])
        asyncio.run(upstream.chat_completions(request, None))
        assert _project("sid") is None


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

    def test_budget_stream_does_not_commit_unvalidated_chunks(self, monkeypatch):
        models = self._budget_models(monkeypatch)
        _begin(project="p1")

        async def run():
            stream = await models.generate_content_stream(model="gemini-3.8-flash", contents=[])
            async for _ in stream:
                assert _project("sid") is None
            assert _project("sid") is None
            assert sa_affinity._pending.get()["prompt_tokens"] == 7000

        asyncio.run(run())

    def test_stream_first_iteration_failure_does_not_write(self, monkeypatch):
        models = self._budget_models(monkeypatch, fail_on_iter=True)
        _begin(project="p1")

        async def run():
            stream = await models.generate_content_stream(
                model="gemini-3.8-flash", contents=[])
            async for _ in stream:
                pass

        with pytest.raises(RuntimeError):
            asyncio.run(run())
        assert _project("sid") is None

    def test_stream_failure_does_not_write(self, monkeypatch):
        models = self._budget_models(monkeypatch, fail=True)
        _begin()
        with pytest.raises(RuntimeError):
            asyncio.run(models.generate_content_stream(model="gemini-3.8-flash", contents=[]))
        assert _project("sid") is None

    def test_fake_stream_empty_stop_does_not_write(self, monkeypatch):
        """假流式空 STOP（无正文/工具）与非流式一样判为无效，不得写亲和。"""
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

        _begin(project="p1")
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
        assert _project("sid") is None


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
        _seed("sid", "first")
        assert not sa_affinity._write_decision({"session_id": "sid", "project_id": "second",
                                               "credential_id": "second"})
        assert _project("sid") == "first"

    def test_stale_switch_cannot_overwrite_same_project_rebinding(self, monkeypatch):
        monkeypatch.setattr(sa_affinity.time, "time", lambda: 1_000_000.0)
        original = _seed("sid", "p1", "A")
        _begin(project="p1", credential="C", previous=original)
        stale = sa_affinity.take_pending()
        _begin(project="p1", credential="B", previous=original)
        asyncio.run(sa_affinity.commit_success())
        assert not sa_affinity._write_decision(stale)
        rebound = sa_affinity._lookup("sid")[0]
        assert rebound.credential_id == "B"
        # A→B→A doesn't make the old A snapshot valid again.
        _begin(project="p1", credential="A", previous=rebound)
        asyncio.run(sa_affinity.commit_success())
        assert not sa_affinity._write_decision(stale)
        assert sa_affinity._lookup("sid")[0].credential_id == "A"

    def test_inflight_rebind_does_not_restore_a_cleared_binding(self):
        original = _seed("sid", "p1", "A")
        _begin(project="p1", credential="B", previous=original)
        sa_affinity.clear_all()
        asyncio.run(sa_affinity.commit_success())
        assert _project("sid") is None

    def test_media_prompt_uses_real_count_tokens(self, monkeypatch):
        import sa_budget
        monkeypatch.setattr(sa_budget, "reserve", lambda key, amount: "tok")
        monkeypatch.setattr(sa_budget, "price", lambda *a, **k: 1)

        class FakeModels:
            async def count_tokens(self, **kwargs):
                class R:
                    total_tokens = 321
                return R()

        models = sa_budget.BudgetModels(FakeModels(), "key", "global", "standard")
        _begin()
        contents = [{"role": "user", "parts": [{"inline_data": {"mime_type": "image/png", "data": "x"}}]}]
        asyncio.run(models._reserve({"model": "gemini-3.8-flash", "contents": contents, "config": {}}))
        assert sa_affinity._pending.get()["prompt_tokens"] == 321
        asyncio.run(sa_affinity.commit_success())
        assert _project("sid") is None      # 321 < 6144，不建绑
