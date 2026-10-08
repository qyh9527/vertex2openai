"""请求级日志上下文：逻辑请求 ID、生命周期事件与假流式诊断（事件契约 v1）。

约定见 .scratch/request-log-grouping/plan.md：
  - 一个客户端请求 = 一个 request_id（路由入口 begin_request 分配），贯穿重试与故障转移；
  - `request_end` 是每个 request_id 唯一的终态事件（finish 幂等）；
  - 所有事件的 message 都是人类可读的安全文本，不含密钥、Cookie、提示词、生成正文；
  - 缺失的数值就不写，绝不把“未知”默认成成功或 0。
本模块的所有日志函数都吞掉自身异常，日志问题不能影响请求。
"""
import asyncio
import json
import re
import secrets
import time
from typing import Any, Callable, Optional

from fastapi.responses import JSONResponse, StreamingResponse

import http_options
from logger import REQUEST_CTX, emit_record
from usage_mapping import map_usage

# 测试可 monkeypatch：整个模块的单调时钟统一从这里取。
_monotonic = time.monotonic

# 等待上游期间每隔多久汇报一次进度 / 单次尝试累计等待多久升级为慢请求告警（秒）。
WAIT_PROGRESS_INTERVAL_SECONDS = 30
SLOW_REQUEST_WARN_SECONDS = 60

_MODE_LABELS = {"fake_stream": "假流式", "stream": "流式", "non_stream": "非流式"}
_STATUS_LABELS = {"success": "成功", "failed": "失败", "cancelled": "已取消", "unknown": "状态未知"}
_STATUS_LEVELS = {"success": "INFO", "failed": "ERROR", "cancelled": "WARN", "unknown": "INFO"}


def now_mono() -> float:
    return _monotonic()


# ---------------------------------------------------------------- 脱敏

_TOKEN = r"[A-Za-z0-9_\-.~+/=%]+"      # 只吃 ASCII 令牌字符，避免把后面的中文标点/文案一起抹掉
_REDACTIONS = [
    (re.compile(r'"private_key"\s*:\s*"[^"]*"'), '"private_key": "[已脱敏]"'),
    (re.compile(r"AIza[0-9A-Za-z_\-]{20,}"), "[已脱敏]"),
    (re.compile(r"Bearer\s+" + _TOKEN), "Bearer [已脱敏]"),
    (re.compile(r"ya29\." + _TOKEN), "[已脱敏]"),
    (re.compile(r"key=" + _TOKEN), "key=[已脱敏]"),
    (re.compile(r"(SAPISID|__Secure-[A-Za-z0-9_\-]+|SID|HSID|SSID|APISID)=" + _TOKEN), r"\1=[已脱敏]"),
]


def scrub(text: Any) -> str:
    """只脱敏凭证特征，不改排版、不截断（用于必须保留原文案的 message）。"""
    out = str(text if text is not None else "")
    for pattern, repl in _REDACTIONS:
        out = pattern.sub(repl, out)
    return out


def redact(text: Any, limit: int = 200) -> str:
    """脱敏凭证特征、压成单行并截断（用于 error_summary 等会进入事件的错误文本）。"""
    try:
        out = re.sub(r"\s+", " ", scrub(text)).strip()
        if len(out) > limit:
            out = out[:limit] + "…"
        return out
    except Exception:
        return ""


def _seconds(ms: Optional[int]) -> str:
    return f"{(ms or 0) / 1000:.1f}s"


def _channel_text(channel: Optional[str]) -> Optional[str]:
    if not channel:
        return None
    try:
        from api_helpers import channel_display_name
        return channel_display_name(channel)
    except Exception:
        return str(channel)


# ---------------------------------------------------------------- 上游响应摘要（只读属性）

def usage_of(response: Any):
    """上游响应的用量摘要：有 usage_metadata 用 map_usage，没有写 "unknown"。"""
    try:
        metadata = getattr(response, "usage_metadata", None)
        if metadata is None:
            return "unknown"
        return map_usage(metadata)
    except Exception:
        return "unknown"


def finish_reason_of(response: Any) -> Optional[str]:
    """第一个候选的原始 finish_reason 名称；取不到返回 None（字段省略）。"""
    try:
        candidates = getattr(response, "candidates", None) or []
        if not candidates:
            return None
        reason = getattr(candidates[0], "finish_reason", None)
        if reason is None:
            return None
        return getattr(reason, "name", None) or str(reason)
    except Exception:
        return None


# ---------------------------------------------------------------- 请求上下文

class RequestLogContext:
    """一个客户端请求的日志上下文（事件循环线程内使用，不加锁）。"""

    def __init__(self, client_model: Optional[str], mode: Optional[str],
                 strategy: Optional[str] = None):
        # 不含任何凭证/用户信息的随机短 ID
        self.request_id = "r" + secrets.token_hex(5)
        self.client_model = client_model
        self.mode = mode
        self.strategy = strategy
        self.started_mono = _monotonic()
        self.channel: Optional[str] = None
        self.phase = "received"
        self.slow = False
        self.retries = 0
        self.switches = 0
        self.paygo_tier: Optional[str] = None
        self.transport_timeout_note: Optional[str] = None
        self.outcome: Optional[dict] = None
        self.terminal_emitted = False
        self.owner = False        # True = 由假流式生成器自己创建（直接调用生成器时），由它负责 finish

    def elapsed_ms(self) -> int:
        return max(0, int((_monotonic() - self.started_mono) * 1000))

    def emit(self, event_type: str, message: str, level: str = "INFO", **fields):
        """发一条属于本请求的事件（自动带 request_id 与累计耗时）；失败只吞不抛。"""
        try:
            if event_type != "request_start":
                fields.setdefault("elapsed_ms", self.elapsed_ms())
            return emit_record(message, event_type=event_type, level=level,
                               request_id=self.request_id, fields=fields)
        except Exception:
            return None

    def set_outcome(self, status: str, **fields) -> None:
        """记录业务结论；后写覆盖先写（故障转移后新通道的结论覆盖旧通道）。"""
        self.outcome = {"status": status, **{k: v for k, v in fields.items() if v is not None}}


def current() -> Optional[RequestLogContext]:
    return REQUEST_CTX.get()


def begin_request(client_model: Optional[str], mode: Optional[str],
                  strategy: Optional[str] = None) -> RequestLogContext:
    """路由入口调用：分配 request_id、设置 ContextVar 并发 request_start。"""
    ctx = RequestLogContext(client_model, mode, strategy)
    REQUEST_CTX.set(ctx)
    parts = [f"📥 [请求开始] {ctx.request_id}", f"模型 {client_model}",
             _MODE_LABELS.get(mode, str(mode))]
    if strategy:
        parts.append(f"策略 {strategy}")
    ctx.emit("request_start", " | ".join(parts), "INFO", client_model=client_model, mode=mode,
             strategy=strategy, phase="received", status="in_progress")
    return ctx


def note_channel(channel: Optional[str]) -> None:
    """记录请求当前所在通道（未接入详细诊断的通道也能在 request_end 里带上通道）。"""
    ctx = current()
    if ctx is not None and channel:
        ctx.channel = channel


def channel_switch(source: str, target: str, message: str,
                   error_summary: Optional[str] = None) -> None:
    """故障转移切换：计数、清掉旧通道的结论（由新通道覆盖）并发 channel_switch。

    message 沿用原 print 文案；无请求上下文时也要落一条日志，不丢信息。
    """
    summary = redact(error_summary) if error_summary else None
    message = scrub(message)      # 原文案里带的异常摘要也可能含凭证特征
    ctx = current()
    if ctx is None:
        emit_record(message, event_type="channel_switch", level="WARN",
                    fields={"source_channel": source, "target_channel": target,
                            "error_summary": summary})
        return
    ctx.switches += 1
    ctx.outcome = None
    ctx.channel = target
    ctx.emit("channel_switch", message, "WARN", source_channel=source, target_channel=target,
             error_summary=summary)


def note_transport(headers: Optional[dict], timeout: Optional[int]) -> None:
    """记录“请求附带的 PayGo 层级头”和传输超时说明（不是上游实际命中的档位）。"""
    ctx = current()
    if ctx is None:
        return
    try:
        lowered = {str(k).lower(): v for k, v in (headers or {}).items()}
        shared = lowered.get("x-vertex-ai-llm-shared-request-type")
        if shared:
            ctx.paygo_tier = str(shared)
        elif str(lowered.get("x-vertex-ai-llm-request-type", "")).lower() == "shared":
            ctx.paygo_tier = "standard"
        elif not lowered:
            ctx.paygo_tier = "off"
        ctx.transport_timeout_note = http_options.describe_transport_timeout(timeout)
    except Exception:
        pass


# ---------------------------------------------------------------- 终态

def _end_message(ctx: RequestLogContext, status: str, merged: dict, elapsed_ms: int) -> str:
    parts = [f"🏁 [请求结束] {ctx.request_id} {_STATUS_LABELS.get(status, status)}",
             _MODE_LABELS.get(ctx.mode, str(ctx.mode))]
    channel = _channel_text(ctx.channel)
    if channel:
        parts.append(channel)
    total = f"总耗时 {_seconds(elapsed_ms)}"
    breakdown = [f"{label} {_seconds(merged[key])}" for key, label in
                 (("upstream_wait_ms", "等待上游"), ("convert_ms", "转换"), ("send_ms", "下游发送"))
                 if isinstance(merged.get(key), (int, float))]
    if breakdown:
        total += "（" + " / ".join(breakdown) + "）"
    parts.append(total)

    if status == "cancelled":
        detail = ["连接断开或任务取消"]
        notes = []
        if merged.get("phase"):
            notes.append(f"阶段 {merged['phase']}")
        if merged.get("upstream_task_done") is True:
            notes.append("上游任务已完成")
        elif merged.get("upstream_task_done") is False:
            notes.append("上游任务未完成")
        if merged.get("cancel_requested") is True:
            notes.append("已请求取消")
        if notes:
            detail[0] += "（" + "，".join(notes) + "）"
        parts.extend(detail)
    elif status == "failed":
        if merged.get("phase"):
            parts.append(f"阶段 {merged['phase']}")
        if merged.get("http_status") is not None:
            parts.append(f"HTTP {merged['http_status']}")
        reason = ": ".join(str(merged[k]) for k in ("error_type", "error_summary") if merged.get(k))
        if reason:
            parts.append(reason)
    elif status == "unknown":
        parts.append("该路径没有详细诊断结论")
    if ctx.slow:
        parts.append("慢请求")
    if ctx.retries or ctx.switches:
        parts.append(f"重试 {ctx.retries} 次 / 切换通道 {ctx.switches} 次")
    return " | ".join(parts)


def finish(ctx: Optional[RequestLogContext], status: Optional[str] = None, **fields) -> None:
    """发唯一的 request_end（幂等）。status 缺省取 ctx.outcome 的 status，再缺省 unknown。"""
    if ctx is None or ctx.terminal_emitted:
        return
    ctx.terminal_emitted = True
    try:
        merged = dict(ctx.outcome or {})
        outcome_status = merged.pop("status", None)
        final_status = status or outcome_status or "unknown"
        merged.update({k: v for k, v in fields.items() if v is not None})
        elapsed_ms = ctx.elapsed_ms()
        event = {
            "status": final_status,
            "phase": merged.pop("phase", None) or ctx.phase,
            "elapsed_ms": elapsed_ms,
            "channel": ctx.channel,
            "client_model": ctx.client_model,
            "mode": ctx.mode,
            "strategy": ctx.strategy,
            "retries": ctx.retries,
            "switches": ctx.switches,
        }
        if ctx.slow:
            event["slow"] = True
        event.update(merged)
        message = _end_message(ctx, final_status, {**merged, "phase": event["phase"]}, elapsed_ms)
        ctx.emit("request_end", message, _STATUS_LEVELS.get(final_status, "WARN"), **event)
    except Exception:
        pass
    finally:
        if REQUEST_CTX.get() is ctx and ctx.owner:
            REQUEST_CTX.set(None)


def _json_error_summary(response: JSONResponse) -> Optional[str]:
    try:
        data = json.loads(response.body)
        error = data.get("error") if isinstance(data, dict) else None
        message = error.get("message") if isinstance(error, dict) else None
        return redact(message) if message else None
    except Exception:
        return None


async def aclose_quietly(iterator) -> None:
    """关闭内层异步生成器（吞掉异常）。

    `async for` 在外层被 aclose/GeneratorExit 打断时不会自动关闭内层生成器，
    内层的清理（取消上游任务、写取消结论）要等 GC 才发生；各包装层在 finally 里调用它，
    让清理沿调用链确定性地向内传递。
    """
    close = getattr(iterator, "aclose", None)
    if close is None:
        return
    try:
        await close()
    except BaseException:
        pass


async def _tracked_stream(ctx: RequestLogContext, inner):
    status = None
    extra: dict = {}
    iterator = inner.__aiter__()
    try:
        while True:
            # 不同任务/上下文迭代同一个流：每次取下一块前重设 ContextVar（不用 token reset，
            # 跨上下文 aclose 时 reset 会抛 ValueError）。
            REQUEST_CTX.set(ctx)
            try:
                chunk = await iterator.__anext__()
            except StopAsyncIteration:
                break
            yield chunk
    except (asyncio.CancelledError, GeneratorExit):
        status = "cancelled"
        raise
    except BaseException as error:
        status = "failed"
        extra = {"error_type": type(error).__name__, "error_summary": redact(error)}
        raise
    finally:
        # 先关内层生成器（让它写入自己的取消/失败细节），再发唯一终态。
        await aclose_quietly(iterator)
        finish(ctx, status, **extra)


def track_response(ctx: Optional[RequestLogContext], response: Any) -> Any:
    """路由返回前包一层：保证每个请求恰好发出一次 request_end。"""
    if ctx is None:
        return response
    try:
        if isinstance(response, StreamingResponse):
            response.body_iterator = _tracked_stream(ctx, response.body_iterator)
            return response
        if isinstance(response, JSONResponse):
            code = response.status_code
            if code < 400:
                finish(ctx, "success", http_status=code)
            else:
                finish(ctx, "failed", http_status=code, error_summary=_json_error_summary(response))
            return response
    except Exception:
        pass
    finish(ctx, "unknown")
    return response


# ---------------------------------------------------------------- 等待上游观察器

class UpstreamWaitObserver:
    """观察一次上游调用任务：定时汇报“仍在等待”，完成时立即发 upstream_received。

    不创建任何 asyncio Task：用 scheduler（默认 loop.call_later）注册定时回调，
    回调里再注册下一次；任务的完成通知走 add_done_callback。所有路径（成功、失败、
    取消、生成器 aclose）都必须调用 close()。内部全部 try/except，不影响请求。
    clock / scheduler 可注入，便于测试驱动。
    """

    def __init__(self, ctx: RequestLogContext, task: "asyncio.Future", attempt: int,
                 max_attempts: int, *, clock: Optional[Callable[[], float]] = None,
                 scheduler: Optional[Callable[..., Any]] = None):
        self.ctx = ctx
        self.task = task
        self.attempt = attempt
        self.max_attempts = max_attempts
        self._clock = clock or now_mono
        self._scheduler = scheduler
        self._handle = None
        self._ticks = 0
        self._warned = False
        self._closed = False
        self._done_handled = False
        self.started_at: Optional[float] = None
        self.done_at: Optional[float] = None

    # ---- 生命周期

    def start(self) -> "UpstreamWaitObserver":
        try:
            self.started_at = self._clock()
            if self._scheduler is None:
                self._scheduler = asyncio.get_running_loop().call_later
            self.task.add_done_callback(self._on_done)
            self._schedule()
        except Exception:
            pass
        return self

    def close(self) -> None:
        """取消定时句柄、移除完成回调；任务已完成但回调还没跑时补发 upstream_received。"""
        if self._closed:
            return
        self._closed = True
        try:
            self._cancel_timer()
            self.task.remove_done_callback(self._on_done)
            if self.task.done():
                self._handle_done()
        except Exception:
            pass

    def attempt_elapsed_ms(self) -> Optional[int]:
        """本次尝试从启动到完成（未完成则到现在）的毫秒数。"""
        if self.started_at is None:
            return None
        end = self.done_at if self.done_at is not None else self._clock()
        return max(0, int((end - self.started_at) * 1000))

    def wait_ms(self) -> Optional[int]:
        """上游实际耗时（需要记录到完成时刻，取消的尝试返回 None）。"""
        if self.started_at is None or self.done_at is None:
            return None
        return max(0, int((self.done_at - self.started_at) * 1000))

    # ---- 内部

    def _schedule(self) -> None:
        if self._closed or self.task.done() or self._scheduler is None:
            return
        self._handle = self._scheduler(WAIT_PROGRESS_INTERVAL_SECONDS, self._tick)

    def _cancel_timer(self) -> None:
        handle, self._handle = self._handle, None
        if handle is not None:
            try:
                handle.cancel()
            except Exception:
                pass

    def _tick(self) -> None:
        self._handle = None
        try:
            if self._closed or self.task.done():
                return
            self._ticks += 1
            elapsed = max(0.0, self._clock() - (self.started_at or 0.0))
            # 以名义间隔数兜底，避免定时器略早触发（时钟分辨率）时漏掉 60 秒告警
            waited = max(elapsed, self._ticks * WAIT_PROGRESS_INTERVAL_SECONDS)
            attempt_ms = int(elapsed * 1000)
            total_ms = self.ctx.elapsed_ms()
            suffix = (f"第 {self.attempt}/{self.max_attempts} 次尝试：等待上游完整响应；"
                      f"心跳不代表生成进度 | 本轮已等待 {_seconds(attempt_ms)}，累计 {_seconds(total_ms)}")
            common = dict(phase="waiting_upstream", attempt=self.attempt,
                          max_attempts=self.max_attempts, attempt_elapsed_ms=attempt_ms,
                          elapsed_ms=total_ms, channel=self.ctx.channel)
            if waited >= SLOW_REQUEST_WARN_SECONDS and not self._warned:
                self._warned = True
                self.ctx.slow = True
                self.ctx.emit("upstream_slow",
                              f"🐢 [上游较慢] {self.ctx.request_id} 本次尝试已超过 "
                              f"{SLOW_REQUEST_WARN_SECONDS} 秒仍未返回 | {suffix}",
                              "WARN", slow=True, **common)
            else:
                self.ctx.emit("upstream_waiting",
                              f"⏳ [等待上游] {self.ctx.request_id} {suffix}", "INFO", **common)
        except Exception:
            pass
        finally:
            try:
                self._schedule()
            except Exception:
                pass

    def _on_done(self, task) -> None:
        self._cancel_timer()
        self._handle_done()

    def _handle_done(self) -> None:
        if self._done_handled:
            return
        self._done_handled = True
        try:
            self.done_at = self._clock()
            task = self.task
            if task.cancelled() or task.exception() is not None:
                return
            result = task.result()
            attempt_ms = self.attempt_elapsed_ms()
            usage = usage_of(result)
            reason = finish_reason_of(result)
            usage_text = "未知" if usage == "unknown" else (
                f"{usage.get('prompt_tokens')}+{usage.get('completion_tokens')} tokens")
            message = (f"📬 [上游已返回] {self.ctx.request_id} 第 {self.attempt}/{self.max_attempts} "
                       f"次尝试已收到上游完整响应 | 本轮耗时 {_seconds(attempt_ms)}，"
                       f"累计 {_seconds(self.ctx.elapsed_ms())} | 用量 {usage_text}")
            self.ctx.emit("upstream_received", message, "INFO", phase="upstream_received",
                          attempt=self.attempt, max_attempts=self.max_attempts,
                          attempt_elapsed_ms=attempt_ms, finish_reason=reason, usage=usage,
                          channel=self.ctx.channel)
        except Exception:
            pass
