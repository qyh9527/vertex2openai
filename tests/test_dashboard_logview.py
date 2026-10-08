"""控制台「运行日志」按请求分组视图的回归测试。

dashboard.html 是纯 HTML/JS。日志视图的纯逻辑核心被 LOGVIEW-CORE 标记包住，
这里把它抽出来用 Node 真实执行（无 DOM），覆盖去重、清空水位、分组、上限、
展开/折叠和筛选语义；DOM 渲染层只做字符串契约断言，真实浏览器效果另行验证。
"""
import re
import shutil
import subprocess
from pathlib import Path

import pytest

DASHBOARD = Path(__file__).resolve().parents[1] / "app" / "console" / "dashboard.html"

# 改动前 dashboard.html 中 "<!--" 的出现次数；本次改动不得新增 HTML 注释。
BASELINE_HTML_COMMENT_COUNT = 23

BEGIN = "/* LOGVIEW-CORE:BEGIN */"
END = "/* LOGVIEW-CORE:END */"


def _html() -> str:
    return DASHBOARD.read_text(encoding="utf-8")


def _core_source() -> str:
    html = _html()
    start = html.index(BEGIN) + len(BEGIN)
    end = html.index(END)
    return html[start:end]


def _logs_region() -> str:
    html = _html()
    start = html.index("/* ---------- Logs ---------- */")
    end = html.index("/* ---------- Project ID")
    return html[start:end]


# ---------------------------------------------------------------- 字符串契约


def test_logview_core_markers_and_entrypoints():
    html = _html()
    assert html.count(BEGIN) == 1 and html.count(END) == 1
    assert html.index(BEGIN) < html.index(END)
    assert "function createLogStore(" in html
    assert "EventSource('/stream-logs')" in html
    assert "aria-expanded" in html
    assert "aria-controls" in html


def test_logview_adds_no_new_html_comments():
    assert _html().count("<!--") <= BASELINE_HTML_COMMENT_COUNT


def test_logview_core_is_dom_free():
    core = _core_source()
    assert not re.search(r"\b(document|window|navigator|localStorage)\b", core)
    assert "innerHTML" not in core


def test_logview_render_layer_contracts():
    region = _logs_region()
    # message 一律走 textContent / 文本节点，不当 HTML。
    assert "innerHTML" not in region
    assert region.count("setInterval(") == 1
    assert "visibilityState" in region
    assert "requestAnimationFrame" in region
    assert "LV_FRAME_BATCH" in region
    assert "条新日志" in region
    assert "createTextNode" in region


def test_logview_css_accessibility_contracts():
    html = _html()
    assert ":focus-visible" in html
    assert "@media (pointer:coarse)" in html
    assert "prefers-reduced-motion" in html
    assert "min-width:0" in html
    assert "overflow-wrap:anywhere" in html
    assert "white-space:pre-wrap" in html
    assert "tabular-nums" in html
    # 手机端：级别芯片单行横向滚动，不再折成多行
    assert "overflow-x:auto" in html
    assert ".lv-chips" in html


# ---------------------------------------------------------------- Node 行为测试

NODE = shutil.which("node")

PRELUDE = r"""
const assert = require('assert');
const T0 = 1700000000000;
let _seq = 0;
function ev(o) {
  _seq++;
  const boot = (o && o.boot_id) || 'aaaaaa';
  const s = (o && o.seq !== undefined) ? o.seq : _seq;
  return Object.assign({
    schema_version: 1, event_id: boot + '-' + s, boot_id: boot, seq: s,
    timestamp: T0 + s * 10, time: '12:00:00', event_type: 'log',
    message: 'm' + s, replayed: false
  }, o);
}
function rep(o) { return ev(Object.assign({ replayed: true }, o)); }
const REPLAY_DONE = { event_type: 'stream_control', control: 'replay_complete' };
"""

CASES = {}

CASES["dedup_and_legacy"] = r"""
const st = createLogStore();
const a = ev({ request_id: 'r1', event_type: 'request_start', client_model: 'gemini-x', replayed: true });
st.ingest(a, 1000);
st.ingest(Object.assign({}, a, { replayed: false }), 1001);      // 同一 event_id 先补发后实时
assert.strictEqual(st.getGroup('r1').events.length, 1);

// 第一次补发阶段内的 legacy_text 被接受
st.ingest({ event_type: 'legacy_text', event_id: null, request_id: null, message: '旧行 1', replayed: true }, 1002);
assert.strictEqual(st.ungrouped().length, 1);

const cs = st.ingest(REPLAY_DONE, 1003);
assert.strictEqual(cs.control.type, 'replay_complete');
assert.strictEqual(st.replayComplete(), true);

// 重连后再补发同一事件 -> 不重复
st.ingest(Object.assign({}, a, { replayed: true }), 1004);
assert.strictEqual(st.getGroup('r1').events.length, 1);
assert.strictEqual(st.stats().events, 2);   // 1 个分组事件 + 1 条未分组

// replay_complete 之后的 legacy_text 丢弃
const cs2 = st.ingest({ event_type: 'legacy_text', event_id: null, message: '旧行 2', replayed: true }, 1005);
assert.strictEqual(cs2.accepted, false);
assert.strictEqual(st.ungrouped().length, 1);

// 非 JSON 文本按 legacy_text 处理（新 store，补发阶段内）；心跳字符串被忽略
const st2 = createLogStore();
st2.ingest('plain text line', 1);
assert.strictEqual(st2.ungrouped().length, 1);
assert.strictEqual(st2.ungrouped()[0].message, 'plain text line');
assert.strictEqual(st2.ingest(': keep-alive heartbeat', 2).accepted, false);
st2.ingest(JSON.stringify(ev({ request_id: 'rx', event_type: 'request_start' })), 3);
assert.ok(st2.getGroup('rx'));
"""

CASES["ungrouped_and_incomplete"] = r"""
const st = createLogStore();
st.ingest(ev({ message: '系统启动' }), 1);                                  // 无 request_id
st.ingest({ event_type: 'legacy_text', message: '旧文本', replayed: true }, 2);
assert.strictEqual(st.ungrouped().length, 2);
assert.strictEqual(st.groups().length, 0);

// 只有中段事件、没有 request_start
st.ingest(ev({ request_id: 'mid', event_type: 'attempt_start', attempt: 2, max_attempts: 5 }), 3);
let s = st.summary(st.getGroup('mid'), 4);
assert.strictEqual(s.incomplete, true);
assert.ok(s.reasons.includes('no_start'));
assert.strictEqual(s.statusKey, 'in_progress');

// 有 request_start 的完整开头：不算不完整
st.ingest(ev({ request_id: 'ok1', event_type: 'request_start', client_model: 'gemini-a' }), 5);
assert.strictEqual(st.summary(st.getGroup('ok1'), 6).incomplete, false);

// 无 request_end，且之后实时看到新的 boot_id -> 记录不完整，状态不被推成成功/失败
const cs = st.ingest(ev({ boot_id: 'bbbbbb', seq: 1, timestamp: T0 + 10 * 1000 * 1000, request_id: 'new', event_type: 'request_start' }), 7);
assert.ok(cs.updated.includes('ok1'));
s = st.summary(st.getGroup('ok1'), 8);
assert.ok(s.reasons.includes('boot_changed'));
assert.strictEqual(s.incomplete, true);
assert.strictEqual(st.getGroup('ok1').status, 'in_progress');
assert.ok(!['success', 'failed', 'cancelled'].includes(s.statusKey));
assert.strictEqual(s.live, false);
// 新 boot 的组本身不受影响
assert.strictEqual(st.summary(st.getGroup('new'), 9).incomplete, false);

// 补发的旧 boot 事件（时间戳更小）不会把最新 boot 拉回去
st.ingest(rep({ boot_id: 'aaaaaa', seq: 777, request_id: 'old2', event_type: 'request_start' }), 10);
assert.strictEqual(st.latestBootId(), 'bbbbbb');

// 订阅队列溢出提示进未分组区
st.ingest({ event_type: 'stream_control', control: 'dropped', dropped: 7 }, 11);
const last = st.ungrouped().pop();
assert.ok(last.message.includes('7 条'));
assert.strictEqual(last.type, 'client_notice');
"""

CASES["interleaved_and_channel_switch"] = r"""
const st = createLogStore();
const M = 'gemini-3-pro';
st.ingest(ev({ request_id: 'A', event_type: 'request_start', client_model: M, mode: 'fake_stream' }), 1);
st.ingest(ev({ request_id: 'B', event_type: 'request_start', client_model: M, mode: 'fake_stream' }), 2);
st.ingest(ev({ request_id: 'A', event_type: 'attempt_start', attempt: 1, max_attempts: 11, channel: 'express', elapsed_ms: 100 }), 3);
st.ingest(ev({ request_id: 'B', event_type: 'attempt_start', attempt: 1, max_attempts: 11, channel: 'vertex', elapsed_ms: 50 }), 4);
st.ingest(ev({ request_id: 'A', event_type: 'attempt_failed', attempt: 1, retryable: true, error_type: 'Timeout' }), 5);
st.ingest(ev({ request_id: 'B', event_type: 'log', message: 'B 的普通日志' }), 6);

let a = st.getGroup('A'), b = st.getGroup('B');
assert.strictEqual(st.groups().length, 2);
assert.strictEqual(a.events.length, 3);
assert.strictEqual(b.events.length, 3);
assert.ok(a.events.every(e => e.frame.request_id === 'A'));
assert.ok(b.events.every(e => e.frame.request_id === 'B'));
assert.strictEqual(st.summary(a, 10).channel, 'express');
assert.strictEqual(st.summary(b, 10).channel, 'vertex');
assert.strictEqual(st.summary(a, 10).retries, 1);
assert.strictEqual(st.summary(b, 10).retries, 0);

// A 跨 express -> vertex：一组，切换 1，重试计数，channel 为最后通道
st.ingest(ev({ request_id: 'A', event_type: 'attempt_start', attempt: 2, max_attempts: 11, channel: 'express' }), 11);
st.ingest(ev({ request_id: 'A', event_type: 'attempt_failed', attempt: 2, retryable: true }), 12);
st.ingest(ev({ request_id: 'A', event_type: 'channel_switch', source_channel: 'express', target_channel: 'vertex', level: 'WARN' }), 13);
st.ingest(ev({ request_id: 'A', event_type: 'attempt_start', attempt: 1, max_attempts: 11, channel: 'vertex', phase: 'waiting_upstream' }), 14);
assert.strictEqual(st.groups().length, 2);
let s = st.summary(st.getGroup('A'), 15);
assert.strictEqual(s.switches, 1);
assert.strictEqual(s.retries, 2);
assert.strictEqual(s.channel, 'vertex');
assert.strictEqual(s.channelLabel, '服务账号');
assert.strictEqual(s.statusKey, 'in_progress');
assert.strictEqual(s.modeLabel, '假流式');
assert.strictEqual(s.phaseLabel, '等待上游');

st.ingest(ev({ request_id: 'A', event_type: 'request_end', status: 'success', channel: 'vertex', elapsed_ms: 4200, retries: 2, switches: 1, finish_reason: 'stop', usage: { prompt_tokens: 5, completion_tokens: 7 } }), 16);
s = st.summary(st.getGroup('A'), 17);
assert.strictEqual(s.statusKey, 'success');
assert.strictEqual(s.live, false);
assert.strictEqual(s.elapsedMs, 4200);
assert.strictEqual(s.retries, 2);
assert.strictEqual(s.switches, 1);
assert.strictEqual(s.channel, 'vertex');
assert.strictEqual(s.incomplete, false);
// B 不受影响，仍在进行中
assert.strictEqual(st.summary(st.getGroup('B'), 18).statusKey, 'in_progress');

// 耗时：进行中 = 最近 elapsed_ms + (现在 - 收到时间)；没有 elapsed_ms 则为 null
const c = createLogStore();
c.ingest(ev({ request_id: 'C', event_type: 'attempt_start', elapsed_ms: 1000 }), 5000);
assert.strictEqual(c.elapsedOf(c.getGroup('C'), 7000), 3000);
c.ingest(ev({ request_id: 'D', event_type: 'request_start' }), 5000);
assert.strictEqual(c.elapsedOf(c.getGroup('D'), 7000), null);
c.ingest(ev({ request_id: 'C', event_type: 'request_end', status: 'failed', elapsed_ms: 9000 }), 6000);
assert.strictEqual(c.elapsedOf(c.getGroup('C'), 99999), 9000);
assert.strictEqual(c.summary(c.getGroup('C'), 1).statusKey, 'failed');
// request_end.status 缺失/非法 -> unknown，而不是成功
c.ingest(ev({ request_id: 'E', event_type: 'request_end' }), 7000);
assert.strictEqual(c.summary(c.getGroup('E'), 1).statusKey, 'unknown');
// slow 是标记，不是状态
c.ingest(ev({ request_id: 'F', event_type: 'upstream_slow', slow: true, level: 'WARN' }), 8000);
assert.strictEqual(c.summary(c.getGroup('F'), 1).slow, true);
assert.strictEqual(c.summary(c.getGroup('F'), 1).statusKey, 'in_progress');
"""

CASES["clear_watermark"] = r"""
const st = createLogStore();
st.ingest(ev({ request_id: 'r1', event_type: 'request_start' }), 1);
st.ingest(ev({ request_id: 'r1', event_type: 'attempt_start' }), 2);
st.ingest(ev({ message: '系统行' }), 3);
const maxTs = T0 + _seq * 10;
st.clear();
assert.strictEqual(st.groups().length, 0);
assert.strictEqual(st.ungrouped().length, 0);
assert.ok(st.watermark() >= T0);

// 重连后补发：尚未见过但时间戳 <= 水位的旧记录一律丢弃
const oldUnseen = rep({ request_id: 'r1', event_type: 'attempt_failed', seq: 5000, timestamp: maxTs - 5 });
assert.strictEqual(st.ingest(oldUnseen, 4).accepted, false);
assert.strictEqual(st.ingest(rep({ message: '旧系统行', seq: 5001, timestamp: maxTs }), 5).accepted, false);
// 已见过的也不会回来
st.ingest(rep({ request_id: 'r1', event_type: 'request_start', seq: 1, timestamp: T0 + 10 }), 6);
assert.strictEqual(st.groups().length, 0);
assert.strictEqual(st.ungrouped().length, 0);

// 清空之后的实时记录照收（即便本地看起来时间戳较小，只要不是补发）
assert.strictEqual(st.ingest(ev({ request_id: 'r2', event_type: 'request_start', seq: 6000, timestamp: maxTs - 100 }), 7).accepted, true);
assert.strictEqual(st.ingest(ev({ request_id: 'r3', event_type: 'request_start', seq: 6001, timestamp: maxTs + 100 }), 8).accepted, true);
// 水位之后产生的补发记录（时间戳更大）也收
assert.strictEqual(st.ingest(rep({ request_id: 'r4', event_type: 'request_start', seq: 6002, timestamp: maxTs + 200 }), 9).accepted, true);
assert.deepStrictEqual(st.groups().map(g => g.id), ['r2', 'r3', 'r4']);
// 被清空的 request_id 的后续实时事件进入新组，并标记记录不完整
st.ingest(ev({ request_id: 'r1', event_type: 'request_end', status: 'success', seq: 6003 }), 10);
assert.ok(st.summary(st.getGroup('r1'), 11).reasons.includes('no_start'));
"""

CASES["limits"] = r"""
const L = LV_LIMITS;
const st = createLogStore();
const ended = new Set();
const TOTAL = 10000, GROUPS = 300;
for (let i = 0; i < TOTAL; i++) {
  const gi = i % GROUPS;
  const rid = 'req-' + String(gi).padStart(3, '0');
  const isLast = i >= TOTAL - GROUPS;                 // 每组的最后一个事件
  if (isLast && gi % 3 === 0) {
    ended.add(rid);
    st.ingest(ev({ request_id: rid, event_type: 'request_end', status: 'success' }), i);
  } else {
    st.ingest(ev({ request_id: rid, event_type: 'attempt_start', attempt: 1 }), i);
  }
  const s = st.stats();
  assert.ok(s.groups <= L.MAX_GROUPS, 'groups ' + s.groups);
  assert.ok(s.events <= L.MAX_TOTAL_EVENTS, 'events ' + s.events);
  assert.ok(s.seen <= L.MAX_SEEN_IDS, 'seen ' + s.seen);
}
// 一个超长的实时组：单组上限 + 已省略标记 + 记录不完整
for (let i = 0; i < 500; i++) st.ingest(ev({ request_id: 'big', event_type: 'log', message: 'x' + i }), TOTAL + i);
const big = st.getGroup('big');
assert.ok(big.events.length <= L.MAX_EVENTS_PER_GROUP);
assert.strictEqual(big.omitted, 500 - L.MAX_EVENTS_PER_GROUP);
let bs = st.summary(big, 1);
assert.ok(bs.reasons.includes('truncated'));
assert.strictEqual(bs.incomplete, true);
assert.strictEqual(bs.omitted, big.omitted);
// 未分组行上限
for (let i = 0; i < 1500; i++) st.ingest(ev({ message: 'sys ' + i }), TOTAL + i);

const stats = st.stats();
assert.ok(stats.groups <= L.MAX_GROUPS);
assert.ok(stats.events <= L.MAX_TOTAL_EVENTS);
assert.ok(stats.ungrouped <= L.MAX_UNGROUPED);
assert.ok(stats.seen <= L.MAX_SEEN_IDS);
for (const g of st.groups()) assert.ok(g.events.length <= L.MAX_EVENTS_PER_GROUP);
// 没有结束事件的组绝不会被推成终态
for (const g of st.groups()) {
  if (!ended.has(g.id)) {
    assert.strictEqual(g.status, 'in_progress', g.id);
    assert.strictEqual(g.hasEnd, false, g.id);
    assert.ok(!['success', 'failed', 'cancelled'].includes(st.summary(g, 1).statusKey));
  }
}
assert.ok(stats.evictedGroups > 0);

// 淘汰顺序：先淘汰最旧的已结束组；只有没有已结束组可淘汰时才淘汰最旧的进行中组
const s2 = createLogStore({ limits: { MAX_GROUPS: 3 } });
s2.ingest(ev({ request_id: 'A', event_type: 'request_start' }), 1);
s2.ingest(ev({ request_id: 'A', event_type: 'request_end', status: 'failed' }), 2);
s2.ingest(ev({ request_id: 'B', event_type: 'request_start' }), 3);
s2.ingest(ev({ request_id: 'C', event_type: 'request_start' }), 4);
const csD = s2.ingest(ev({ request_id: 'D', event_type: 'request_start' }), 5);
assert.deepStrictEqual(csD.removed, ['A']);
assert.deepStrictEqual(s2.groups().map(g => g.id), ['B', 'C', 'D']);
const csE = s2.ingest(ev({ request_id: 'E', event_type: 'request_start' }), 6);
assert.deepStrictEqual(csE.removed, ['B']);
assert.deepStrictEqual(s2.groups().map(g => g.id), ['C', 'D', 'E']);
assert.ok(s2.groups().every(g => g.status === 'in_progress' && !g.hasEnd));
"""

CASES["expand_collapse"] = r"""
const st = createLogStore();
// 实时新出现的进行中组默认展开
st.ingest(ev({ request_id: 'live1', event_type: 'request_start' }), 1);
assert.strictEqual(st.isExpanded('live1'), true);
// 手动折叠后，后续事件绝不改变状态
st.setUserExpanded('live1', false);
st.ingest(ev({ request_id: 'live1', event_type: 'attempt_start' }), 2);
st.ingest(ev({ request_id: 'live1', event_type: 'upstream_slow', slow: true }), 3);
assert.strictEqual(st.isExpanded('live1'), false);
st.ingest(ev({ request_id: 'live1', event_type: 'request_end', status: 'success' }), 4);
assert.strictEqual(st.isExpanded('live1'), false);

// 手动展开的组，结束后也保持展开
st.ingest(ev({ request_id: 'live2', event_type: 'request_start' }), 5);
st.setUserExpanded('live2', true);
st.ingest(ev({ request_id: 'live2', event_type: 'request_end', status: 'failed' }), 6);
assert.strictEqual(st.isExpanded('live2'), true);

// 未手动操作的组：request_end 后自动折叠
st.ingest(ev({ request_id: 'live3', event_type: 'request_start' }), 7);
assert.strictEqual(st.isExpanded('live3'), true);
st.ingest(ev({ request_id: 'live3', event_type: 'request_end', status: 'success' }), 8);
assert.strictEqual(st.isExpanded('live3'), false);

// 补发的已结束组默认折叠
st.ingest(rep({ request_id: 'old1', event_type: 'request_start' }), 9);
st.ingest(rep({ request_id: 'old1', event_type: 'request_end', status: 'success' }), 10);
assert.strictEqual(st.isExpanded('old1'), false);
// 补发的进行中组默认折叠；之后收到实时事件才展开
st.ingest(rep({ request_id: 'old2', event_type: 'request_start' }), 11);
assert.strictEqual(st.isExpanded('old2'), false);
st.ingest(ev({ request_id: 'old2', event_type: 'attempt_start' }), 12);
assert.strictEqual(st.isExpanded('old2'), true);
// 未知 request_id 不报错
assert.strictEqual(st.setUserExpanded('nope', true), false);
assert.strictEqual(st.isExpanded('nope'), false);
"""

CASES["filtering"] = r"""
const st = createLogStore();
st.ingest(ev({ request_id: 'F1', event_type: 'request_start', client_model: 'gemini-3-pro', message: '收到请求' }), 1);
st.ingest(ev({ request_id: 'F1', event_type: 'attempt_failed', level: 'WARN', message: '上游返回 503 Service Unavailable' }), 2);
st.ingest(ev({ request_id: 'F1', event_type: 'attempt_failed', level: 'WARN', message: '再次失败：503' }), 3);
st.ingest(ev({ request_id: 'F1', event_type: 'log', level: 'INFO', message: '普通信息' }), 4);
st.ingest(ev({ request_id: 'F2', event_type: 'log', level: 'ERROR', message: '致命错误 boom' }), 5);
const f1 = st.getGroup('F1'), f2 = st.getGroup('F2');

// 无筛选
let m = st.matches(f1, { q: '', level: 'all' });
assert.strictEqual(m.match, true);
assert.strictEqual(m.active, false);

// 关键词命中组内事件 message
m = st.matches(f1, { q: '503', level: 'all' });
assert.strictEqual(m.match, true);
assert.strictEqual(m.hits, 2);
assert.strictEqual(st.matches(f2, { q: '503', level: 'all' }).match, false);
// 大小写不敏感，且按去首尾空白处理
assert.strictEqual(st.matches(f1, { q: '  SERVICE unavailable ', level: 'all' }).hits, 1);
// 关键词命中摘要字段（模型名 / 短 ID）也算匹配
assert.strictEqual(st.matches(f1, { q: 'gemini-3', level: 'all' }).match, true);
assert.strictEqual(st.matches(f1, { q: 'f1', level: 'all' }).match, true);
assert.strictEqual(st.matches(f1, { q: 'nope-not-there', level: 'all' }).match, false);

// 级别筛选对组内任一事件生效：WARN -> warn；ERROR -> err
m = st.matches(f1, { q: '', level: 'warn' });
assert.strictEqual(m.match, true);
assert.strictEqual(m.hits, 2);
assert.strictEqual(st.matches(f2, { q: '', level: 'warn' }).match, false);
assert.strictEqual(st.matches(f2, { q: '', level: 'err' }).match, true);
assert.strictEqual(st.matches(f1, { q: '', level: 'err' }).match, false);
// 无 level 的事件靠 logLevel(message) 推断
st.ingest(ev({ request_id: 'F3', event_type: 'log', message: '✅ 请求完成' }), 6);
assert.strictEqual(st.matches(st.getGroup('F3'), { q: '', level: 'ok' }).match, true);
// 关键词 + 级别同时生效
assert.strictEqual(st.matches(f1, { q: '503', level: 'warn' }).hits, 2);
assert.strictEqual(st.matches(f1, { q: '收到', level: 'warn' }).hits, 0);

// 事件级匹配（供展开组内变淡）与未分组行
const ung = createLogStore();
ung.ingest(ev({ message: 'WARN: 磁盘快满了' }), 1);
const row = ung.ungrouped()[0];
assert.strictEqual(ung.eventMatches(row, { q: '磁盘', level: 'warn' }), true);
assert.strictEqual(ung.eventMatches(row, { q: '网络', level: 'all' }), false);
assert.strictEqual(ung.eventMatches(row, { q: '', level: 'err' }), false);
"""


CASES["level_inference"] = r"""
const st = createLogStore();
// 后端显式 INFO：文案里的 🔄 / 重试 不再被推成 warn；ok / cost 保留
st.ingest(ev({ request_id: 'L1', event_type: 'request_start', level: 'INFO', message: '收到请求' }), 1);
st.ingest(ev({ request_id: 'L1', event_type: 'phase', level: 'INFO', phase: 'converting_response', message: '🔄 转换响应中' }), 2);
st.ingest(ev({ request_id: 'L1', event_type: 'request_end', level: 'INFO', status: 'success', message: '请求完成，重试 0 次' }), 3);
const g = st.getGroup('L1');
assert.deepStrictEqual(g.events.map(e => e.lv), ['other', 'other', 'other']);
assert.strictEqual(st.matches(g, { q: '', level: 'warn' }).match, false);
st.ingest(ev({ request_id: 'L1', event_type: 'log', level: 'INFO', message: '✅ 完成' }), 4);
st.ingest(ev({ request_id: 'L1', event_type: 'log', level: 'INFO', message: '💰 计费' }), 5);
assert.deepStrictEqual(g.events.slice(3).map(e => e.lv), ['ok', 'cost']);
st.ingest(ev({ request_id: 'L1', event_type: 'log', level: 'INFO', message: 'ERROR 字样只是文本' }), 6);
assert.strictEqual(g.events[5].lv, 'other');
// WARN / ERROR 仍然按后端标记
st.ingest(ev({ request_id: 'L2', event_type: 'attempt_failed', level: 'WARN', message: '普通文案' }), 7);
st.ingest(ev({ request_id: 'L2', event_type: 'attempt_failed', level: 'ERROR', message: '普通文案' }), 8);
assert.deepStrictEqual(st.getGroup('L2').events.map(e => e.lv), ['warn', 'err']);
assert.strictEqual(st.matches(st.getGroup('L2'), { q: '', level: 'warn' }).hits, 1);
// level 缺失 / null（普通 print、legacy_text）完全沿用文本推断
st.ingest(ev({ request_id: 'L3', event_type: 'log', level: null, message: '⚠️ 某个警告' }), 9);
st.ingest(ev({ request_id: 'L3', event_type: 'log', message: '❌ 某个错误' }), 10);
assert.deepStrictEqual(st.getGroup('L3').events.map(e => e.lv), ['warn', 'err']);
assert.strictEqual(st.matches(st.getGroup('L3'), { q: '', level: 'warn' }).match, true);
st.ingest({ event_type: 'legacy_text', message: '⚠️ 旧文本警告', replayed: true }, 11);
assert.strictEqual(st.ungrouped()[0].lv, 'warn');
"""

CASES["ended_phase_label"] = r"""
const st = createLogStore();
function endGroup(id, status, phase, extra) {
  st.ingest(ev({ request_id: id, event_type: 'request_start' }), 1);
  st.ingest(ev({ request_id: id, event_type: 'phase', phase: 'sending_downstream' }), 2);
  st.ingest(ev(Object.assign({ request_id: id, event_type: 'request_end', status: status }, phase ? { phase: phase } : {}, extra || {})), 3);
  return st.summary(st.getGroup(id), 4);
}
// 进行中：照旧显示当前阶段
st.ingest(ev({ request_id: 'run', event_type: 'phase', phase: 'sending_downstream' }), 1);
assert.strictEqual(st.summary(st.getGroup('run'), 2).phaseLabel, '发送中');
// 成功 -> 已完成（不再显示“发送中”）
assert.strictEqual(endGroup('s', 'success', 'sending_downstream').phaseLabel, '已完成');
// 失败 / 取消 -> “失败于 X” / “取消于 X”，X 来自 request_end.phase
assert.strictEqual(endGroup('f', 'failed', 'waiting_upstream').phaseLabel, '失败于 等待上游');
assert.strictEqual(endGroup('c', 'cancelled', 'converting_response').phaseLabel, '取消于 转换中');
// 没有 phase 时退化
assert.strictEqual(endGroup('f2', 'failed', null).phaseLabel, '失败于 发送中');
// unknown -> 已结束
assert.strictEqual(endGroup('u', 'unknown', 'sending_downstream').phaseLabel, '已结束');
// 状态本身不受影响
assert.strictEqual(st.summary(st.getGroup('s'), 5).statusKey, 'success');
// 服务重启后的无结束组仍显示最后阶段，不伪造结局
const b = createLogStore();
b.ingest(ev({ request_id: 'x', event_type: 'phase', phase: 'sending_downstream' }), 1);
b.ingest(ev({ boot_id: 'bbbbbb', seq: 1, timestamp: T0 + 99999999, request_id: 'y', event_type: 'request_start' }), 2);
assert.strictEqual(b.summary(b.getGroup('x'), 3).phaseLabel, '发送中');
"""


CASES["warn_filter_discrimination"] = r"""
const st = createLogStore();
const W = { q: '', level: 'warn' };
// 只有信息类内容：🔄 [消息转换] 普通 print、INFO phase、成功 request_end -> 不命中 warn
st.ingest(ev({ request_id: 'quiet', event_type: 'request_start', level: 'INFO', message: '收到请求' }), 1);
st.ingest(ev({ request_id: 'quiet', event_type: 'log', level: null, message: '🔄 [消息转换] 转换 3 条消息' }), 2);
st.ingest(ev({ request_id: 'quiet', event_type: 'phase', level: 'INFO', phase: 'sending_downstream', message: '🔄 阶段：发送中' }), 3);
st.ingest(ev({ request_id: 'quiet', event_type: 'request_end', level: 'INFO', status: 'success', message: '请求完成，重试 0 次' }), 4);
assert.strictEqual(logLevel('🔄 [消息转换] x'), 'other');
assert.strictEqual(st.matches(st.getGroup('quiet'), W).match, false);
// 含 upstream_slow / attempt_failed / channel_switch / ⚠️ print 的组命中
st.ingest(ev({ request_id: 'slow', event_type: 'upstream_slow', level: 'WARN', slow: true, message: '上游等待超过 60s' }), 5);
st.ingest(ev({ request_id: 'fail', event_type: 'attempt_failed', level: 'WARN', retryable: true, message: '尝试失败' }), 6);
st.ingest(ev({ request_id: 'sw', event_type: 'channel_switch', level: 'WARN', source_channel: 'express', target_channel: 'vertex', message: '切换通道' }), 7);
st.ingest(ev({ request_id: 'pr', event_type: 'log', level: null, message: '⚠️ 某个警告' }), 8);
for (const id of ['slow', 'fail', 'sw', 'pr']) assert.strictEqual(st.matches(st.getGroup(id), W).match, true, id);
"""

CASES["error_field_scoping"] = r"""
const st = createLogStore();
// hybrid：第一通道失败后切到第二通道最终成功 -> 已结束组不再显示旧错误
st.ingest(ev({ request_id: 'h', event_type: 'request_start', client_model: 'gemini-3.8-hybrid' }), 1);
st.ingest(ev({ request_id: 'h', event_type: 'attempt_start', attempt: 1, channel: 'express' }), 2);
st.ingest(ev({ request_id: 'h', event_type: 'attempt_failed', attempt: 1, retryable: false, level: 'ERROR', error_type: 'ValueError', error_summary: 'express-down boom' }), 3);
// 进行中：显示最近错误，并标明通道和第几次尝试
let e = st.summary(st.getGroup('h'), 4).error;
assert.ok(e);
assert.ok(e.label.includes('最近错误'));
assert.ok(e.label.includes('Express'));
assert.ok(e.label.includes('第 1 次尝试'));
assert.ok(e.text.includes('ValueError') && e.text.includes('express-down boom'));
st.ingest(ev({ request_id: 'h', event_type: 'channel_switch', source_channel: 'express', target_channel: 'vertex', level: 'WARN' }), 5);
st.ingest(ev({ request_id: 'h', event_type: 'attempt_start', attempt: 1, channel: 'vertex' }), 6);
// 切换后新通道失败：最近错误指向新通道
st.ingest(ev({ request_id: 'h', event_type: 'attempt_failed', attempt: 1, error_type: 'Timeout', error_summary: 'vertex slow', channel: 'vertex' }), 7);
e = st.summary(st.getGroup('h'), 8).error;
assert.ok(e.label.includes('服务账号') && e.text.includes('Timeout'));
// 最终成功：request_end 没带错误 -> error 为 null
st.ingest(ev({ request_id: 'h', event_type: 'request_end', status: 'success', channel: 'vertex' }), 9);
assert.strictEqual(st.summary(st.getGroup('h'), 10).error, null);
// 中间 attempt_failed 仍在时间线里
assert.strictEqual(st.getGroup('h').events.filter(x => x.type === 'attempt_failed').length, 2);

// 失败结束：只取 request_end 的错误
st.ingest(ev({ request_id: 'f', event_type: 'attempt_failed', attempt: 1, error_type: 'Old', error_summary: 'old one' }), 11);
st.ingest(ev({ request_id: 'f', event_type: 'request_end', status: 'failed', error_type: 'Final', http_status: 502, error_summary: 'bad gateway' }), 12);
const fe = st.summary(st.getGroup('f'), 13).error;
assert.strictEqual(fe.label, '错误');
assert.ok(fe.text.includes('Final') && fe.text.includes('HTTP 502') && !fe.text.includes('Old'));
"""

CASES["model_short_name"] = r"""
const st = createLogStore();
st.ingest(ev({ request_id: 'a', event_type: 'request_start', client_model: 'fake-gemini-3.8-flash', mode: 'fake_stream' }), 1);
let s = st.summary(st.getGroup('a'), 2);
assert.strictEqual(s.modelShort, 'gemini-3.8-flash');
assert.strictEqual(s.clientModel, 'fake-gemini-3.8-flash');   // 完整名不变
// 非假流式模式、或名称不以 fake- 开头：不缩短
st.ingest(ev({ request_id: 'b', event_type: 'request_start', client_model: 'fake-gemini-3.8-flash', mode: 'stream' }), 3);
assert.strictEqual(st.summary(st.getGroup('b'), 4).modelShort, null);
st.ingest(ev({ request_id: 'c', event_type: 'request_start', client_model: 'gemini-3.8-flash', mode: 'fake_stream' }), 5);
assert.strictEqual(st.summary(st.getGroup('c'), 6).modelShort, null);
"""


@pytest.fixture(scope="module")
def core_js() -> str:
    return _core_source()


@pytest.mark.skipif(NODE is None, reason="node 不可用，跳过日志核心行为测试")
@pytest.mark.parametrize("case", sorted(CASES))
def test_logview_core_behaviour_in_node(case, core_js, tmp_path):
    script = tmp_path / f"logview_{case}.js"
    script.write_text(core_js + "\n" + PRELUDE + "\n" + CASES[case] + "\n", encoding="utf-8")
    result = subprocess.run(
        [NODE, str(script)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
    )
    assert result.returncode == 0, f"{case} 失败:\n{result.stdout}\n{result.stderr}"
