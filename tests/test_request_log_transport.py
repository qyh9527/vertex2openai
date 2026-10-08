"""结构化日志的落盘、还原与 /stream-logs 传输（事件契约 v1）。"""
import asyncio
import json
import logging
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import logger as logger_mod
from logger import emit_record, read_recent_log_lines, read_recent_log_records, rt_logger


@pytest.fixture
def disk_log(tmp_path, monkeypatch):
    """把落盘文件与读取目录都指到 tmp_path，并清空内存历史。"""
    monkeypatch.setenv("STATE_DIR", str(tmp_path))
    log_path = tmp_path / "vertex2openai.log"
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(message)s", "%Y-%m-%d %H:%M:%S"))
    file_logger = logging.getLogger(f"v2o.test.{tmp_path.name}")
    file_logger.setLevel(logging.INFO)
    file_logger.propagate = False
    file_logger.addHandler(handler)
    monkeypatch.setattr(logger_mod, "file_logger", file_logger)
    monkeypatch.setattr(rt_logger, "history", [])
    logger_mod.REQUEST_CTX.set(None)
    yield log_path
    handler.close()
    file_logger.removeHandler(handler)
    logger_mod.REQUEST_CTX.set(None)


def strip_volatile(record):
    return {k: v for k, v in record.items() if k not in ("replayed",)}


# ------------------------------------------------------------------ 落盘与还原

def test_envelope_required_fields_and_omitted_none(disk_log):
    rec = emit_record("hello", event_type="attempt_start", level="INFO", request_id="rabc",
                      fields={"attempt": 1, "channel": None, "usage": "unknown"}, echo=False)
    assert rec["schema_version"] == 1
    assert rec["event_id"] == f"{logger_mod.BOOT_ID}-{rec['seq']}"
    assert rec["boot_id"] == logger_mod.BOOT_ID and isinstance(rec["timestamp"], int)
    assert len(rec["time"]) == 8 and rec["event_type"] == "attempt_start"
    assert rec["request_id"] == "rabc" and rec["level"] == "INFO"
    assert rec["attempt"] == 1 and rec["usage"] == "unknown"
    assert "channel" not in rec, "值为 None 的字段一律省略"
    second = emit_record("again", echo=False)
    assert second["seq"] == rec["seq"] + 1
    assert "request_id" not in second and "level" not in second


def test_disk_line_format_and_roundtrip(disk_log):
    rec = emit_record("单行消息 \x1b[31m带颜色\x1b[0m", event_type="log", request_id="rabc", echo=False)
    text = disk_log.read_text(encoding="utf-8")
    assert " #v2o:{" in text and "带颜色" in text and "\x1b" not in text
    meta = json.loads(text.rsplit(" #v2o:", 1)[1])
    assert "message" not in meta and "time" not in meta
    assert meta["event_id"] == rec["event_id"] and meta["request_id"] == "rabc"
    assert ", " not in text.rsplit(" #v2o:", 1)[1], "meta 用紧凑分隔符"
    [restored] = read_recent_log_records(10)
    assert restored["message"] == "单行消息 带颜色"
    assert restored["event_id"] == rec["event_id"] and restored["time"] == rec["time"]
    assert restored["event_type"] == "log" and "line_count" not in restored


def test_multiline_message_roundtrip_and_legacy_lines(disk_log):
    disk_log.write_text("2026-01-02 03:04:05 旧格式的一行\n", encoding="utf-8")
    first = emit_record("第一行\n第二行\n第三行", request_id="rabc", echo=False)
    with open(disk_log, "a", encoding="utf-8") as fh:
        fh.write("2026-01-02 03:04:06 另一条旧文本\n")
    single = emit_record("末尾单行", echo=False)

    records = read_recent_log_records(50)
    assert [r["event_type"] for r in records] == ["legacy_text", "log", "legacy_text", "log"]
    legacy = records[0]
    assert legacy["event_id"] is None and legacy["request_id"] is None
    assert legacy["message"] == "2026-01-02 03:04:05 旧格式的一行" and legacy["time"] == "03:04:05"
    assert records[1]["message"] == "第一行\n第二行\n第三行"
    assert records[1]["event_id"] == first["event_id"] and "partial" not in records[1]
    assert records[3]["event_id"] == single["event_id"]
    # 被拼回的行不再作为旧文本重复输出
    assert sum("第二行" in r["message"] for r in records) == 1


def test_truncated_multiline_marks_partial(disk_log):
    emit_record("甲\n乙\n丙", request_id="rabc", echo=False)
    records = read_recent_log_records(2)        # 尾部只读到后两行：缺第一行
    assert len(records) == 1
    assert records[0]["message"] == "乙\n丙" and records[0]["partial"] is True


def test_invalid_or_foreign_marker_is_legacy(disk_log):
    disk_log.write_text(
        '2026-01-02 03:04:05 坏标记 #v2o:{not json\n'
        '2026-01-02 03:04:06 版本不对 #v2o:{"schema_version":2,"event_id":"x-1"}\n',
        encoding="utf-8")
    records = read_recent_log_records(10)
    assert [r["event_type"] for r in records] == ["legacy_text", "legacy_text"]


def test_read_recent_log_lines_unchanged(disk_log):
    emit_record("abc", echo=False)
    lines = read_recent_log_lines(5)
    assert len(lines) == 1 and lines[0].endswith("}") and " abc #v2o:{" in lines[0]


def test_custom_print_single_record_with_request_id(disk_log, monkeypatch):
    got = []
    monkeypatch.setattr(rt_logger, "push", lambda rec: got.append(rec))
    ctx = SimpleNamespace(request_id="rprint1")
    token = logger_mod.REQUEST_CTX.set(ctx)
    try:
        print("带上下文的普通输出")
    finally:
        logger_mod.REQUEST_CTX.reset(token)
    print("无上下文的普通输出")
    assert [(r["event_type"], r.get("request_id"), r["message"]) for r in got] == [
        ("log", "rprint1", "带上下文的普通输出"), ("log", None, "无上下文的普通输出")]
    assert len(read_recent_log_records(10)) == 2, "每个 print 只落一条记录"


def test_emit_record_never_raises_and_does_not_double_record(disk_log, monkeypatch, capsys):
    got = []
    monkeypatch.setattr(rt_logger, "push", lambda rec: got.append(rec))
    emit_record("只出现一次", echo=True)
    assert capsys.readouterr().out.count("只出现一次") == 1
    assert len(got) == 1

    def boom(record):
        raise RuntimeError("push failed")
    monkeypatch.setattr(rt_logger, "push", boom)
    monkeypatch.setattr(logger_mod, "file_logger", SimpleNamespace(info=boom))
    emit_record("日志故障不影响业务", echo=False)


async def test_event_id_consistent_across_disk_history_and_live(disk_log):
    q = rt_logger.subscribe()
    try:
        rec = emit_record("三处一致", request_id="rabc", echo=False)
        live = await asyncio.wait_for(q.get(), 1.0)
    finally:
        rt_logger.unsubscribe(q)
    [from_disk] = read_recent_log_records(10)
    [from_history] = rt_logger.snapshot_history()
    assert rec["event_id"] == live["event_id"] == from_disk["event_id"] == from_history["event_id"]


# ------------------------------------------------------------------ SSELogger

async def test_history_holds_envelopes_and_push_str_is_wrapped(monkeypatch):
    monkeypatch.setattr(rt_logger, "history", [])
    assert rt_logger.max_history == 300
    rt_logger.push("纯文本兼容")
    [rec] = rt_logger.snapshot_history()
    assert rec["event_type"] == "log" and rec["message"] == "纯文本兼容" and rec["schema_version"] == 1


async def test_queue_overflow_drops_oldest_and_counts():
    sse = logger_mod.SSELogger(queue_size=2)
    q = sse.subscribe()
    for i in range(5):
        sse.push({"event_id": f"t-{i}", "message": str(i)})
    await asyncio.sleep(0)
    assert q.qsize() == 2
    assert [q.get_nowait()["event_id"] for _ in range(2)] == ["t-3", "t-4"]
    assert sse.take_dropped(q) == 3
    assert sse.take_dropped(q) == 0
    assert len(sse.snapshot_history()) == 5


# ------------------------------------------------------------------ /stream-logs

class FakeRequest:
    def __init__(self):
        self.disconnected = False

    async def is_disconnected(self):
        return self.disconnected


def parse(frame):
    assert frame.startswith("data: ") and frame.endswith("\n\n")
    return json.loads(frame[6:])


async def read_until(iterator, predicate, limit=100):
    frames = []
    for _ in range(limit):
        frame = await asyncio.wait_for(iterator.__anext__(), 3.0)
        if frame.startswith(":"):
            continue
        frames.append(parse(frame))
        if predicate(frames[-1]):
            return frames
    raise AssertionError("未等到目标帧")


async def stop(request, iterator):
    request.disconnected = True
    await iterator.aclose()


async def test_stream_logs_merges_dedupes_and_orders_replay(disk_log):
    import main as app_main
    on_disk_and_hist = [emit_record(f"补发 {i}", request_id="rabc", echo=False) for i in range(3)]
    request = FakeRequest()
    response = await app_main.stream_logs_endpoint(request, True)
    iterator = response.body_iterator
    frames = await read_until(iterator, lambda f: f.get("control") == "replay_complete")

    replayed = [f for f in frames if f["event_type"] != "stream_control"]
    ids = [f["event_id"] for f in replayed]
    assert ids == [r["event_id"] for r in on_disk_and_hist], "磁盘与内存同一事件只补发一次"
    assert all(f["replayed"] is True for f in replayed)
    assert frames[-1] == {"schema_version": 1, "event_type": "stream_control",
                          "control": "replay_complete", "replayed": False}

    live = emit_record("实时事件", request_id="rabc", echo=False)
    [frame] = await read_until(iterator, lambda f: f.get("event_id") == live["event_id"])
    assert frame["replayed"] is False and frame["message"] == "实时事件"
    assert "replayed" not in live, "不修改 rt_logger 里的原 dict"
    await stop(request, iterator)


async def test_stream_logs_includes_memory_only_and_legacy_records(disk_log, monkeypatch):
    import main as app_main
    disk_log.write_text("2026-01-02 03:04:05 旧文本\n", encoding="utf-8")
    emit_record("落盘的", echo=False)
    monkeypatch.setattr(logger_mod, "file_logger", None)      # 之后的事件只在内存历史里
    memory_only = emit_record("仅内存", echo=False)
    request = FakeRequest()
    response = await app_main.stream_logs_endpoint(request, True)
    iterator = response.body_iterator
    frames = await read_until(iterator, lambda f: f.get("control") == "replay_complete")
    messages = [f.get("message") for f in frames if f["event_type"] != "stream_control"]
    assert messages == ["2026-01-02 03:04:05 旧文本", "落盘的", "仅内存"]
    assert [f["event_type"] for f in frames[:3]] == ["legacy_text", "log", "log"]
    assert frames[2]["event_id"] == memory_only["event_id"]
    await stop(request, iterator)


async def test_stream_logs_event_between_subscribe_and_snapshot_is_sent_once(disk_log):
    import main as app_main
    request = FakeRequest()
    response = await app_main.stream_logs_endpoint(request, True)
    iterator = response.body_iterator
    frames = await read_until(iterator, lambda f: f.get("control") == "replay_complete")
    assert not [f for f in frames if f["event_type"] != "stream_control"]
    for i in range(3):
        emit_record(f"live-{i}", echo=False)
    got = await read_until(iterator, lambda f: f.get("message") == "live-2")
    assert [f["message"] for f in got] == ["live-0", "live-1", "live-2"]
    await stop(request, iterator)


async def test_stream_logs_sends_dropped_control_frame(disk_log, monkeypatch):
    import main as app_main
    monkeypatch.setattr(rt_logger, "queue_size", 2)
    request = FakeRequest()
    response = await app_main.stream_logs_endpoint(request, True)
    iterator = response.body_iterator
    await read_until(iterator, lambda f: f.get("control") == "replay_complete")
    for i in range(6):
        emit_record(f"burst-{i}", echo=False)
    frames = await read_until(iterator, lambda f: f.get("control") == "dropped")
    assert frames[-1]["dropped"] >= 1 and frames[-1]["event_type"] == "stream_control"
    await stop(request, iterator)


def test_stream_logs_requires_auth(tmp_path, monkeypatch):
    import main as app_main
    monkeypatch.setenv("STATE_DIR", str(tmp_path))
    client = TestClient(app_main.app)
    assert client.get("/stream-logs").status_code == 401
