"""SA 独立预算：SQLite 原子预留、逐尝试结算；未知费用保留预留金额。"""
import asyncio
import hashlib
import json
import os
import sqlite3
import uuid
from decimal import Decimal, ROUND_CEILING
from pathlib import Path
from contextlib import contextmanager

LIMIT = 295_000_000_000  # 纳美元，避免浮点累计误差
_pending_cleanup = set()


class BudgetError(RuntimeError):
    code = 429


def credential_id(sa_json):
    data = json.loads(sa_json)
    identity = data.get("client_email")
    if not identity:
        raise BudgetError("SA 缺少稳定身份，无法建立额度账本。")
    return hashlib.sha256(str(identity).strip().lower().encode()).hexdigest()


@contextmanager
def _connect():
    path = Path(os.getenv("STATE_DIR", ".")) / "sa_budget.sqlite3"
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path, timeout=10)
    db.execute("PRAGMA synchronous=FULL")
    db.execute("CREATE TABLE IF NOT EXISTS accounts (id TEXT PRIMARY KEY, baseline INTEGER NOT NULL, spent INTEGER NOT NULL DEFAULT 0)")
    db.execute("CREATE TABLE IF NOT EXISTS holds (id TEXT PRIMARY KEY, account TEXT NOT NULL, amount INTEGER NOT NULL)")
    db.commit()
    try:
        with db:
            yield db
    finally:
        db.close()


def money(value):
    try:
        value = Decimal(str(value))
    except ArithmeticError as exc:
        raise ValueError("金额格式无效。") from exc
    if not value.is_finite() or value < 0 or value > Decimal("1000000000"):
        raise ValueError("金额必须是有限的非负美元数，且不超过十亿美元。")
    return int((value * 1_000_000_000).to_integral_value(rounding=ROUND_CEILING))


def initialize(key, amount):
    amount = money(amount)
    with _connect() as db:
        db.execute("BEGIN IMMEDIATE")
        if db.execute("SELECT 1 FROM accounts WHERE id=?", (key,)).fetchone():
            raise ValueError("历史花费已确认，不能重复初始化或清零。")
        db.execute("INSERT INTO accounts(id, baseline) VALUES (?, ?)", (key, amount))


def status(key):
    with _connect() as db:
        row = db.execute("SELECT baseline, spent, (SELECT COALESCE(SUM(amount),0) FROM holds WHERE account=?) FROM accounts WHERE id=?", (key, key)).fetchone()
    if row is None:
        return {"initialized": False, "limit": 295, "blocked": True}
    baseline, spent, reserved = row
    return {"initialized": True, "limit": 295, "baseline": baseline / 1e9,
            "spent": spent / 1e9, "reserved": reserved / 1e9,
            "remaining": max(0, LIMIT - baseline - spent - reserved) / 1e9,
            "blocked": baseline + spent + reserved >= LIMIT}


def reserve(key, amount):
    token = uuid.uuid4().hex
    with _connect() as db:
        db.execute("BEGIN IMMEDIATE")
        row = db.execute("SELECT baseline + spent + (SELECT COALESCE(SUM(amount),0) FROM holds WHERE account=?) FROM accounts WHERE id=?", (key, key)).fetchone()
        if row is None:
            raise BudgetError("SA 尚未确认历史花费，请在控制台初始化额度账本。")
        if not isinstance(amount, int) or isinstance(amount, bool) or amount < 0:
            raise ValueError("预留金额必须是非负纳美元整数。")
        if row[0] >= LIMIT or row[0] + amount > LIMIT:
            raise BudgetError("SA 剩余额度不足以预留本次请求费用（独立限额 $295）。")
        db.execute("INSERT INTO holds VALUES (?, ?, ?)", (token, key, amount))
    return token


def settle(token, amount):
    if not isinstance(amount, int) or isinstance(amount, bool) or amount < 0:
        raise ValueError("结算金额必须是非负纳美元整数。")
    with _connect() as db:
        db.execute("BEGIN IMMEDIATE")
        row = db.execute("SELECT account FROM holds WHERE id=?", (token,)).fetchone()
        if row:
            db.execute("UPDATE accounts SET spent=spent+? WHERE id=?", (amount, row[0]))
            db.execute("DELETE FROM holds WHERE id=?", (token,))


# 官方 Vertex 价表（2026-10-24）：输入 / 文本输出 / 图像输出，USD / 1M tokens。
IMAGE_PRICING = {
    "gemini-nano-banana-2.1": (1.5, 7.5, 30),
    "gemini-3.1-flash-lite-image": (0.25, 1.5, 30),
    "gemini-3.1-flash-image": (0.5, 3, 60),
    "gemini-3.1-flash-image-preview": (0.5, 3, 60),
    "gemini-3-pro-image": (2, 12, 120),
    "gemini-3-pro-image-preview": (2, 12, 120),
}


def price(model, prompt, output, cached=0, tier="standard", location="global", image_tokens=0,
          audio_tokens=0, cached_audio_tokens=0):
    """保护额度采用未返还促销抵扣前价格；未覆盖的 SKU 拒绝派发。"""
    from model_pricing import MODEL_PRICING, get_model_price, get_cached_input_ratio
    name = model.rsplit("/", 1)[-1]
    if name not in MODEL_PRICING and name not in IMAGE_PRICING:
        raise BudgetError("该 SA 模型尚无可靠预算价表，已阻止调用：" + name)
    inp, out = IMAGE_PRICING[name][:2] if name in IMAGE_PRICING else get_model_price(name)
    # 官方 introductory 是事后 credits back；额度保护不能提前消费返还。
    if name in ("gemini-4-argon", "gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash"):
        inp, out = max(inp, 4 if name == "gemini-4-argon" else 1.5), max(out, 20 if name == "gemini-4-argon" else 7.5)
    if prompt > 200_000:
        if name in ("gemini-3.1-pro", "gemini-3.1-pro-preview"):
            inp, out = 4, 18
        elif name == "gemini-2.5-pro":
            inp, out = 2.5, 15
    factor = Decimal("1.8" if tier == "priority" else "0.5" if tier == "flex" else "1")
    if location != "global":
        factor *= Decimal("1.1")
    cached = min(max(0, cached), prompt)
    dollars = (Decimal(str(inp)) * (prompt - cached + Decimal(str(get_cached_input_ratio(name))) * cached)
               + Decimal(str(out)) * (output - image_tokens)
               + Decimal(str(IMAGE_PRICING.get(name, (0, 0, out))[2])) * image_tokens) * factor / 1_000_000
    if name in ("gemini-2.5-flash", "gemini-3.1-flash-lite"):
        audio_price = Decimal("1" if name == "gemini-2.5-flash" else "0.5")
        delta = audio_price - Decimal(str(inp))
        dollars += delta * (audio_tokens - cached_audio_tokens + Decimal("0.1") * cached_audio_tokens) * factor / 1_000_000
    return money(dollars)


def has_media(value):
    if hasattr(value, "model_dump"):
        value = value.model_dump(exclude_none=True)
    if isinstance(value, dict):
        return any(k in ("inline_data", "file_data", "inlineData", "fileData") or has_media(v)
                   for k, v in value.items())
    return isinstance(value, (list, tuple)) and any(has_media(v) for v in value)


class BudgetModels:
    def __init__(self, models, key, location, tier):
        self.models, self.key, self.location, self.tier = models, key, location, tier

    async def _reserve(self, kwargs):
        config = kwargs.get("config") or {}
        config = config if isinstance(config, dict) else config.model_dump(exclude_none=True)
        if config.get("tools"):
            for tool in config["tools"]:
                tool = tool if isinstance(tool, dict) else tool.model_dump(exclude_none=True)
                if any(k not in ("function_declarations", "functionDeclarations", "google_search", "googleSearch") for k in tool):
                    raise BudgetError("该内置工具暂无可靠价表，不能按零费用放行。")
        if config.get("cached_content"):
            raise BudgetError("SA 限额保护暂不支持外部显式缓存费用。")
        if config.get("candidate_count", 1) not in (None, 1):
            raise BudgetError("SA 限额保护当前仅支持单候选生成。")
        output = config.get("max_output_tokens") or 65536
        if not isinstance(output, int) or isinstance(output, bool) or not 0 < output <= 65536:
            raise BudgetError("SA 限额保护要求 max_output_tokens 在 1–65536 之间。")
        # 固定输出上限并禁止 SDK 内部自动调用工具，否则可能绕过逐尝试预留。
        config["max_output_tokens"] = output
        config["automatic_function_calling"] = {"disable": True}
        http_options = config.get("http_options") or {}
        http_options = dict(http_options) if isinstance(http_options, dict) else http_options.model_dump(exclude_none=True)
        http_options["retry_options"] = {"attempts": 1}
        config["http_options"] = http_options
        kwargs["config"] = config
        contents = kwargs.get("contents")
        model = kwargs["model"]
        price(model, 1, 1, tier=self.tier, location=self.location)
        # countTokens 含 system/tools，使用与生成相同的配置，禁止本地字符估算。
        from google.genai import types
        generation = {k: v for k, v in config.items() if k in types.GenerationConfig.model_fields}
        result = await self.models.count_tokens(model=model, contents=kwargs.get("contents"),
                                                config={"system_instruction": config.get("system_instruction"),
                                                        "tools": config.get("tools"),
                                                        "generation_config": generation})
        prompt = getattr(result, "total_tokens", None)
        if not isinstance(prompt, int) or prompt < 0:
            raise BudgetError("上游未返回可靠输入 token 数，已停止 SA 调用。")
        output = config.get("max_output_tokens") or 65536
        candidates = config.get("candidate_count") or 1
        # 媒体计数可能低估：预留整个模型上下文，而非把估算视为精确上限。
        if has_media(contents):
            prompt = 1_048_576
        image_output = output * candidates if model.rsplit("/", 1)[-1] in IMAGE_PRICING else 0
        amount = price(model, prompt, output * candidates,
                       tier="standard" if self.tier == "flex" else self.tier, location=self.location,
                       image_tokens=image_output,
                       audio_tokens=prompt if has_media(contents) else 0)
        task = asyncio.create_task(asyncio.to_thread(reserve, self.key, amount))
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            # 取消等待不停止线程：先取得事务结果，再释放尚未派发的预留。
            async def release_unsent():
                try:
                    token = await task
                    await asyncio.to_thread(settle, token, 0)
                except Exception:
                    pass
            cleanup = asyncio.create_task(release_unsent())
            _pending_cleanup.add(cleanup)
            cleanup.add_done_callback(_pending_cleanup.discard)
            raise

    def _settle(self, token, response, model, search_queries=0, media_input=False):
        usage = getattr(response, "usage_metadata", None)
        if usage is None:
            return
        from usage_mapping import map_usage
        if any(getattr(usage, field, None) is None for field in
               ("prompt_token_count", "candidates_token_count")):
            return
        mapped = map_usage(usage)
        traffic = str(getattr(getattr(usage, "traffic_type", None), "value", None) or getattr(usage, "traffic_type", ""))
        tier = "standard" if traffic == "ON_DEMAND" else self.tier
        service = getattr(usage, "service_tier", None)
        if service:
            service = str(getattr(service, "value", service)).lower()
            if service in ("standard", "priority", "flex"):
                tier = service
        image_tokens = 0
        details = getattr(usage, "candidates_tokens_details", None)
        for entry in details or []:
            modality = str(getattr(getattr(entry, "modality", None), "value", None) or getattr(entry, "modality", ""))
            if modality == "IMAGE":
                image_tokens += getattr(entry, "token_count", 0) or 0
        if model.rsplit("/", 1)[-1] in IMAGE_PRICING and details is None:
            # 无模态明细不能按低价文本结算图像，保留原预留等待核账。
            return
        if image_tokens > mapped["completion_tokens"]:
            return
        prompt = mapped["prompt_tokens"]
        # Google Search 的检索上下文官方不收输入费；其它工具的输入需另行定价。
        tool_prompt = getattr(usage, "tool_use_prompt_token_count", None) or 0
        if tool_prompt and not search_queries:
            return
        def modality_count(details, modality):
            return sum((getattr(e, "token_count", None) or 0) for e in details or []
                       if str(getattr(getattr(e, "modality", None), "value", None) or getattr(e, "modality", "")) == modality)
        input_details = getattr(usage, "prompt_tokens_details", None)
        cache_details = getattr(usage, "cache_tokens_details", None)
        audio_tokens = modality_count(input_details, "AUDIO")
        cached_audio = modality_count(cache_details, "AUDIO")
        name = model.rsplit("/", 1)[-1]
        if name in ("gemini-2.5-flash", "gemini-3.1-flash-lite") and media_input and input_details is None:
            return
        if name in ("gemini-2.5-flash", "gemini-3.1-flash-lite") and audio_tokens and mapped["prompt_tokens_details"]["cached_tokens"] and cache_details is None:
            return
        amount = price(model, prompt, mapped["completion_tokens"],
                       mapped["prompt_tokens_details"]["cached_tokens"], tier, self.location,
                       image_tokens=image_tokens, audio_tokens=audio_tokens, cached_audio_tokens=cached_audio)
        # 免费额度按结算账号/月共享，无法由 SA 本地证明，保护账本不预扣免费份额。
        amount += money(Decimal(search_queries) * Decimal("0.014"))
        settle(token, amount)

    @staticmethod
    def _search_queries(response):
        queries = []
        for candidate in getattr(response, "candidates", None) or []:
            grounding = getattr(candidate, "grounding_metadata", None)
            queries.extend(getattr(grounding, "web_search_queries", None) or [])
        return len(queries)

    async def generate_content(self, **kwargs):
        token = await self._reserve(kwargs)
        try:
            response = await self.models.generate_content(**kwargs)
        except Exception as exc:
            if getattr(exc, "code", None) in (400, 401, 403, 404, 429):
                await asyncio.to_thread(settle, token, 0)
            raise
        await asyncio.to_thread(self._settle, token, response, kwargs["model"], self._search_queries(response),
                                has_media(kwargs.get("contents")))
        return response

    async def generate_content_stream(self, **kwargs):
        token = await self._reserve(kwargs)
        try:
            stream = await self.models.generate_content_stream(**kwargs)
        except Exception as exc:
            if getattr(exc, "code", None) in (400, 401, 403, 404, 429):
                await asyncio.to_thread(settle, token, 0)
            raise

        async def tracked():
            last = None
            received = False
            search_queries = 0
            try:
                async for chunk in stream:
                    received = True
                    search_queries = max(search_queries, self._search_queries(chunk))
                    if getattr(chunk, "usage_metadata", None) is not None:
                        last = chunk
                    yield chunk
                if last is not None:
                    await asyncio.to_thread(self._settle, token, last, kwargs["model"], search_queries,
                                            has_media(kwargs.get("contents")))
            except Exception as exc:
                if not received and getattr(exc, "code", None) in (400, 401, 403, 404, 429):
                    await asyncio.to_thread(settle, token, 0)
                raise
            finally:
                close = getattr(stream, "aclose", None)
                if close:
                    await close()
        return tracked()


class BudgetAio:
    def __init__(self, aio, key, location, tier):
        self._aio = aio
        self.models = BudgetModels(aio.models, key, location, tier)

    def __getattr__(self, name):
        return getattr(self._aio, name)


class BudgetClient:
    def __init__(self, client, key, location, tier):
        self._client = client
        self.aio = BudgetAio(client.aio, key, location, tier)

    def __getattr__(self, name):
        return getattr(self._client, name)
