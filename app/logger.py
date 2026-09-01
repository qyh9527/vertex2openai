import builtins
import json
import logging
import os
import time
import asyncio
import re
import threading
from logging.handlers import TimedRotatingFileHandler
from typing import List, Optional

original_print = builtins.print
ANSI_ESCAPE = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

# 日志/统计时间统一时区：默认北京（Asia/Shanghai），LOG_TZ 环境变量可覆盖。
# 覆盖范围：文件日志时间戳（logging asctime）、按天轮转（midnight）、SSE 控制台时间戳、
# stats.json 按天聚合的日期 key——全部走 time.localtime，改 TZ 一次全部生效。
# tzset 仅 Unix 存在（Docker 为 Linux，生产环境生效）；Windows 本机开发不生效，
# 仅本地显示差 8 小时，不影响容器行为。
if hasattr(time, "tzset"):
    _tz = os.environ.get("LOG_TZ", "Asia/Shanghai")
    if _tz:
        os.environ["TZ"] = _tz
        time.tzset()


def _setup_file_logger():
    """日志落盘：按天轮转，保留 7 天，与 STATE_DIR 同目录（挂载卷内，重建容器不丢）。

    容器日志（docker logs）随容器重建即清空，排查"昨天发生了什么"无从查起；
    落盘后 VPS 上 `tail -f <STATE_DIR>/vertex2openai.log` 或 1Panel 文件管理直接看。
    与 web_state.json 同目录的代价是同一份权限约束（0600 内含 Cookie 的目录），可接受。
    """
    state_dir = os.environ.get("STATE_DIR", ".")
    try:
        os.makedirs(state_dir, exist_ok=True)
        handler = TimedRotatingFileHandler(
            os.path.join(state_dir, "vertex2openai.log"),
            when="midnight", backupCount=7, encoding="utf-8",
        )
        handler.setFormatter(logging.Formatter("%(asctime)s %(message)s", "%Y-%m-%d %H:%M:%S"))
        logger = logging.getLogger("vertex2openai.file")
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)
        logger.propagate = False
        return logger
    except Exception as e:
        print(f"⚠️ [日志] 文件日志初始化失败（不影响运行）：{e}")
        return None


file_logger = _setup_file_logger()


def classify_error(status=None, message="") -> str:
    """把一次错误归类到固定分类（数据概览「错误/拦截分类」用）。

    分类依据 Google 官方错误码与安全拦截原因（HTTP 429 RESOURCE_EXHAUSTED /
    401 UNAUTHENTICATED / 403 PERMISSION_DENIED / blockReason SAFETY、
    PROHIBITED_CONTENT、RECITATION、BLOCKLIST、SPII、IMAGE_SAFETY 等），
    再叠加本项目的自身错误形态（Cookie 失效、空流、客户端断开）。
    语义消息优先于状态码（同一个 400 可能是安全拦截）。
    """
    s = status if isinstance(status, int) else None
    low = str(message or "").lower()
    if s == 499 or "客户端已断开" in low or "client_closed" in low:
        return "客户端断开"
    if ("安全策略拦截" in low or "prohibited_content" in low or "content_filter" in low
            or any(k in low for k in ("safety", "recitation", "blocklist",
                                      "spii", "image_safety"))):
        return "安全拦截"
    if s in (401, 403) or any(k in low for k in ("cookie", "unauthenticated",
                                                  "permission denied",
                                                  "凭证", "权限")):
        return "凭证/权限失效"
    if (s == 429 or "429" in low or "quota" in low or "too many requests" in low
            or "resource_exhausted" in low):
        return "429 限流/配额"
    if any(k in low for k in ("未返回任何内容", "无有效内容", "空流", "未返回任何响应")):
        return "空流/无有效内容"
    if s is not None:
        if 500 <= s <= 599:
            return "5xx 上游错误"
        if 400 <= s <= 499:
            return f"HTTP {s} 请求错误"
    return "其他错误"


class ProxyStats:
    """算力消耗统计：内存累计 + 持久化到 STATE_DIR/stats.json（重建容器不丢）。

    设计（参考 new-api / one-api / LiteLLM 等网关的用量追踪做法，适配单实例 JSON 落盘）：
      - 每次请求把 prompt/completion tokens 与请求数累加到内存，并写入"当天"的按天聚合；
      - 写盘节流（默认 30s 一次，原子写临时文件 + rename），启动时从磁盘恢复历史；
      - 前端展示累计数字 + 最近 7/30 天趋势柱状图；
      - 按天聚合含 requests/success/error/retries，另有按小时聚合（最近 72 小时桶）
        与错误分类计数 error_categories（供数据概览按小时趋势与分类排行）。
    """

    def __init__(self):
        self.start_time = time.time()
        self.total_requests = 0
        self.success_requests = 0
        self.error_requests = 0
        self.retry_counts = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.cached_prompt_tokens = 0     # 命中上下文缓存的输入 token（隐式缓存 90% 折扣）
        self.cost = 0.0                   # 估算美刀成本（按官方按量价，见 model_pricing）
        self.error_categories = {}        # 分类名 -> 次数（classify_error 的固定分类集）
        self._daily = {}          # "YYYY-MM-DD" -> {requests, success, error, retries, prompt_tokens, completion_tokens, cached_prompt_tokens, cost}
        self._hourly = {}          # "YYYY-MM-DD HH" -> {requests, success, error, retries}（最近 72 桶）
        self._last_save = 0.0
        self.lock = threading.Lock()
        self._load_from_disk()

    # ---------- 持久化 ----------

    def _today_key(self) -> str:
        return time.strftime("%Y-%m-%d")

    def _this_hour_key(self) -> str:
        return time.strftime("%Y-%m-%d %H")

    def _prune_hourly(self):
        """只保留最近 72 个小时桶（3 天），防止无界增长。"""
        if len(self._hourly) > 72:
            for k in sorted(self._hourly)[:len(self._hourly) - 72]:
                self._hourly.pop(k, None)

    def _load_from_disk(self):
        """启动时恢复历史统计（与 web_state.json 同目录、同在挂载卷内）。"""
        try:
            if not os.path.exists(STATS_FILE):
                return
            with open(STATS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            t = data.get("total") or {}
            self.total_requests = int(t.get("requests", 0))
            self.success_requests = int(t.get("success", 0))
            self.error_requests = int(t.get("error", 0))
            self.retry_counts = int(t.get("retries", 0))
            self.prompt_tokens = int(t.get("prompt_tokens", 0))
            self.completion_tokens = int(t.get("completion_tokens", 0))
            self.cached_prompt_tokens = int(t.get("cached_prompt_tokens", 0))
            self.cost = float(t.get("cost", 0) or 0)
            daily = data.get("daily") or {}
            if isinstance(daily, dict):
                self._daily = {k: dict(v) for k, v in daily.items() if isinstance(v, dict)}
            hourly = data.get("hourly") or {}
            if isinstance(hourly, dict):
                self._hourly = {k: dict(v) for k, v in hourly.items() if isinstance(v, dict)}
            cats = data.get("error_categories") or {}
            if isinstance(cats, dict):
                self.error_categories = {str(k): int(v) for k, v in cats.items()
                                          if isinstance(v, (int, float))}
            print(f"📊 [用量统计] 已恢复历史：请求 {self.total_requests} 次，"
                  f"累计 {self.prompt_tokens + self.completion_tokens} tokens，"
                  f"按天记录 {len(self._daily)} 天。")
        except Exception as e:
            print(f"⚠️ [用量统计] 历史统计加载失败（不影响运行）：{e}")

    def _save(self):
        """原子写 stats.json（临时文件 + rename）。"""
        try:
            payload = {
                "version": 1,
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "total": {
                    "requests": self.total_requests,
                    "success": self.success_requests,
                    "error": self.error_requests,
                    "retries": self.retry_counts,
                    "prompt_tokens": self.prompt_tokens,
                    "completion_tokens": self.completion_tokens,
                    "cached_prompt_tokens": self.cached_prompt_tokens,
                    "cost": round(self.cost, 6),
                },
                "daily": self._daily,
                "hourly": self._hourly,
                "error_categories": self.error_categories,
            }
            tmp = STATS_FILE + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            os.replace(tmp, STATS_FILE)
            try:
                os.chmod(STATS_FILE, 0o600)
            except OSError:
                pass
        except Exception as e:
            print(f"⚠️ [用量统计] 落盘失败（不影响运行）：{e}")

    def _touch_daily(self, prompt=0, completion=0, success=0, request=0, error=0,
                     retry=0, cached=0, cost=0.0):
        key = self._today_key()
        day = self._daily.get(key)
        if day is None:
            day = {"requests": 0, "success": 0, "error": 0, "retries": 0,
                   "prompt_tokens": 0, "completion_tokens": 0,
                   "cached_prompt_tokens": 0, "cost": 0.0}
            self._daily[key] = day
        day["requests"] = day.get("requests", 0) + request
        day["success"] = day.get("success", 0) + success
        day["error"] = day.get("error", 0) + error
        day["retries"] = day.get("retries", 0) + retry
        day["prompt_tokens"] = day.get("prompt_tokens", 0) + prompt
        day["completion_tokens"] = day.get("completion_tokens", 0) + completion
        day["cached_prompt_tokens"] = day.get("cached_prompt_tokens", 0) + cached
        day["cost"] = float(day.get("cost", 0) or 0) + cost
        # 只保留最近 180 天，防止无界增长
        if len(self._daily) > 180:
            for k in sorted(self._daily)[:len(self._daily) - 180]:
                self._daily.pop(k, None)

    def _touch_hourly(self, success=0, request=0, error=0, retry=0):
        key = self._this_hour_key()
        hour = self._hourly.get(key)
        if hour is None:
            hour = {"requests": 0, "success": 0, "error": 0, "retries": 0}
            self._hourly[key] = hour
        hour["requests"] = hour.get("requests", 0) + request
        hour["success"] = hour.get("success", 0) + success
        hour["error"] = hour.get("error", 0) + error
        hour["retries"] = hour.get("retries", 0) + retry
        self._prune_hourly()

    def _maybe_save(self):
        """节流写盘（默认 30s 一次），避免每个请求都做一次磁盘 IO。"""
        now = time.time()
        if now - self._last_save < STATS_SAVE_INTERVAL:
            return
        self._last_save = now
        self._save()

    def _flush(self):
        """立即落盘（进程退出/测试用）。"""
        with self.lock:
            self._last_save = time.time()
            self._save()

    # ---------- 累加 ----------

    def increment_total(self):
        with self.lock:
            self.total_requests += 1
            self._touch_daily(request=1)
            self._touch_hourly(request=1)
            self._maybe_save()

    def add_error(self, status=None, message=""):
        """计一次错误；可选带 HTTP 状态与消息用于分类统计。"""
        with self.lock:
            self.error_requests += 1
            self._touch_daily(error=1)
            self._touch_hourly(error=1)
            try:
                cat = classify_error(status, message)
                self.error_categories[cat] = self.error_categories.get(cat, 0) + 1
            except Exception:
                pass
            self._maybe_save()

    def add_retry(self):
        with self.lock:
            self.retry_counts += 1
            self._touch_daily(retry=1)
            self._touch_hourly(retry=1)
            self._maybe_save()

    def add_success(self):
        """直接计一次成功请求（Cookie 通道不产生 token 统计行，成功数单独计入）。"""
        with self.lock:
            self.success_requests += 1
            self._touch_daily(success=1)
            self._touch_hourly(success=1)
            self._maybe_save()

    def add_tokens(self, p_tokens, c_tokens, cached=0, model=None, tier="standard"):
        """累计 token 用量；cached = 命中上下文缓存的输入 token（隐式缓存 90% 折扣）。

        model + tier（standard/priority/flex/auto/off）用于按官方按量价估算美刀成本
        （model_pricing）；未知模型不计费。
        """
        with self.lock:
            self.prompt_tokens += p_tokens
            self.completion_tokens += c_tokens
            cached = max(0, cached or 0)
            self.cached_prompt_tokens += cached
            try:
                from model_pricing import estimate_cost
                cost = estimate_cost(model, p_tokens, c_tokens, cached, tier=tier)
            except Exception:
                cost = None
            cost = cost if cost is not None else 0.0
            self.cost += cost
            self.success_requests += 1
            self._touch_daily(prompt=p_tokens, completion=c_tokens, success=1,
                              cached=cached, cost=cost)
            self._touch_hourly(success=1)
            self._maybe_save()

    def get_json_stats(self):
        with self.lock:
            daily = [{"date": d, **(self._daily[d] or {})} for d in sorted(self._daily)]
            hourly = [{"hour": k, **(self._hourly[k] or {})} for k in sorted(self._hourly)]
            return {
                "uptime": round(time.time() - self.start_time, 2),
                "total": self.total_requests,
                "success": self.success_requests,
                "error": self.error_requests,
                "retries": self.retry_counts,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "cached_prompt_tokens": self.cached_prompt_tokens,
                "cost": round(self.cost, 6),
                "daily": daily,
                "hourly": hourly,
                "error_categories": dict(sorted(self.error_categories.items(),
                                                key=lambda kv: -kv[1])),
            }


# 用量统计持久化文件（与 web_state.json 同目录、同在挂载卷内，重建容器不丢）
STATS_FILE = os.path.join(os.environ.get("STATE_DIR", "."), "stats.json")
STATS_SAVE_INTERVAL = 30   # 秒：统计落盘节流间隔


stats = ProxyStats()


class SSELogger:
    """把运行日志推给控制台的 SSE 订阅者。

    P1-5 修复的三个隐患：
      - `asyncio.Queue.put_nowait` 不是线程安全的，而 push() 可能从任意线程被调用
        （图片压缩线程、to_thread 里的消息转换等）。改为 loop.call_soon_threadsafe。
      - 订阅者列表会被事件循环并发增删，遍历时可能 RuntimeError。改为持锁取快照。
      - 队列原本无界，慢客户端会把内存撑爆。改为有界 + 满则丢最旧。
    """

    def __init__(self, max_history: int = 100, queue_size: int = 500):
        self.max_history = max_history
        self.queue_size = queue_size
        self.history: List[str] = []
        self._subscribers: List[tuple] = []       # [(queue, loop)]
        self._lock = threading.Lock()

    # ---- 订阅管理 ----

    def subscribe(self, loop: Optional[asyncio.AbstractEventLoop] = None) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=self.queue_size)
        loop = loop or asyncio.get_running_loop()
        with self._lock:
            self._subscribers.append((q, loop))
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        with self._lock:
            self._subscribers = [(sq, sl) for (sq, sl) in self._subscribers if sq is not q]

    def snapshot_history(self) -> List[str]:
        with self._lock:
            return list(self.history)

    # ---- 推送 ----

    @staticmethod
    def _offer(q: asyncio.Queue, msg: str) -> None:
        """在事件循环线程里执行：队列满时丢最旧的一条，保证新日志能进来。"""
        try:
            q.put_nowait(msg)
        except asyncio.QueueFull:
            try:
                q.get_nowait()
                q.put_nowait(msg)
            except Exception:
                pass

    def push(self, plain_text: str) -> None:
        timestamp = time.strftime("%H:%M:%S")
        formatted_msg = f"[{timestamp}] {plain_text}"

        with self._lock:
            self.history.append(formatted_msg)
            if len(self.history) > self.max_history:
                self.history.pop(0)
            subscribers = list(self._subscribers)

        for q, loop in subscribers:
            try:
                if loop.is_closed():
                    continue
                loop.call_soon_threadsafe(self._offer, q, formatted_msg)
            except RuntimeError:
                # 事件循环已停止，忽略即可
                pass


rt_logger = SSELogger()


def read_recent_log_lines(n: int = 200) -> List[str]:
    """读取持久化日志文件（STATE_DIR/vertex2openai.log）尾部 n 行。

    前端「运行日志」页初始加载时用它展示历史（重建容器后也能看到此前落盘的日志），
    之后再订阅实时流。文件不存在/读取失败时返回空列表，不影响运行。
    """
    path = os.path.join(os.environ.get("STATE_DIR", "."), "vertex2openai.log")
    try:
        if not os.path.exists(path):
            return []
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            block = min(size, 256 * 1024)
            f.seek(size - block)
            tail = f.read(block)
            lines = tail.splitlines()
            if len(lines) < n and size > block:
                f.seek(max(0, size - 2 * block))
                lines = f.read().splitlines()
            return lines[-n:]
    except Exception:
        return []

# 防止 print 钩子内部再触发 print 导致递归
_in_hook = threading.local()


def custom_print(*args, **kwargs):
    """把业务 print 同时推到控制台日志流。

    说明（P1-5 第一步）：仍保留对 builtins.print 的接管以兼容现有 6000 行代码里的
    print 调用，但已经**移除了用正则从日志文本反解 token 数**的逻辑——
    那与中文文案强耦合，改一个字就静默失效。现在由 api_helpers._record_usage()
    直接调用 stats.add_tokens()。后续可分批把 print 迁移到 logging。
    """
    if getattr(_in_hook, "active", False):
        original_print(*args, **kwargs)
        return

    _in_hook.active = True
    try:
        import io
        buf = io.StringIO()
        kwargs_for_buffer = dict(kwargs)
        kwargs_for_buffer["file"] = buf

        raw_msg = ""
        try:
            original_print(*args, **kwargs_for_buffer)
            raw_msg = buf.getvalue().strip()
        except Exception:
            pass

        try:
            original_print(*args, **kwargs)
        except Exception:
            pass

        if not raw_msg:
            return

        # 文件日志（STATE_DIR/vertex2openai.log，按天轮转保留 7 天）。
        # 用清理 ANSI 后的文本，保证 VPS 上 tail -f 看到的是纯文本。
        if file_logger is not None:
            try:
                file_logger.info(ANSI_ESCAPE.sub('', raw_msg))
            except Exception:
                pass

        try:
            rt_logger.push(ANSI_ESCAPE.sub('', raw_msg))
        except Exception:
            pass
    finally:
        _in_hook.active = False


builtins.print = custom_print
