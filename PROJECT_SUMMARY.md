# vertex2openai 项目改造总结

> 本文档总结本项目 fork 以来（2026-08）完成的所有改造、原因、实现方式与部署说明。
> 仓库：`qyh9527/vertex2openai`（fork 自 `bad-woman/vertex2openai`）

---

## 一、项目背景

**vertex2openai** 是一个 OpenAI 兼容 API 代理：把 Google **Agent Platform（原 Vertex AI）** 的 Gemini 模型（Express Mode API Key / 网页端 batchGraphql）包装成标准的 OpenAI `/v1/chat/completions` 与 `/v1/models` 接口，供酒馆（SillyTavern）、RikkaHub 等前端直接使用。

**两条上游通道**：

| 通道 | 凭证 | 原理 | 特点 |
|---|---|---|---|
| **Express API Key（标准）** | `VERTEX_EXPRESS_API_KEY`（可多个） | 官方 SDK 调 Express Mode | 计费走 API Key；429 限流较常见 |
| **Cookie 直连反代** | 控制台 Google Cookie + Project ID | 直连网页端私有 `batchGraphql` 接口 | 走网页端配额，可规避 429；Cookie 会过期 |

**改造前的问题**：两条通道由 `use_web_proxy` 布尔开关硬二选一，互斥、无并发、无负载均衡、无跨通道故障转移；假流式是全局开关，开启后所有模型强制假流式。

---

## 二、改造总览

| # | 改动 | 类型 | 对应 Commit |
|---|---|---|---|
| 1 | 双通道三档策略 + 熔断故障转移 | 功能 | `297a512` |
| 2 | fork + GHCR CI 自动构建 + 部署流程 | 部署 | `3bbc594` / `19edcdd` / `9efb0b4` |
| 3 | upstream 定期同步流程 | 文档 | `9efb0b4` |
| 4 | 带预填充的流式请求 429 不重试 bug 修复 | Bug 修复 | `e1d61c1` |
| 5 | 假流式改为按模型选择（`fake-` 前缀模型） | 功能 | `21c4141` |
| 6 | 上游稳定性：锁 google-genai 2.x + 钉 api_version=v1beta1 + Client 复用 | 功能 | `ab53e48` |
| 7 | 多账号凭证管理：多 Express Key + 多 Cookie 账号（请求级快照不串号） | 功能 | 本批 |
| 8 | 自动化测试体系：pytest 142 用例（`tests/`） | 工程 | 本批 |
| 9 | 日志落盘：按天轮转保留 7 天（`STATE_DIR/vertex2openai.log`） | 运维 | 本批 |
| 10 | CI 镜像双 tag（`latest` + commit sha，可回滚） | 部署 | 本批 |
| 11 | Express Client 复用防护：开关 + 连接级失败自动舍弃 / 硬错误立即舍弃 | 功能 | 本批 |
| 12 | 第三通道：服务账号（Vertex SA）+ 四标签页通道管理 + 混合自动可配（3 通道开关/顺序/每通道重试）+ PayGo 层级融合 | 功能 | 本批 |

---

## 三、改动详情

### 3.1 双通道策略：express / cookie / hybrid（含熔断故障转移）

**现状问题**：`chat_api.py` 里 `if app_state.is_web_proxy_enabled()` 硬分流，两条通道互斥，无法在 Express 被限流时自动切换。

**目标**：三档通道策略，hybrid 模式下 Express 主 + Cookie 兜底。

#### 核心设计

- **策略三档**（控制台「通道与凭证」页 radio 单选）：
  - `express`（默认）：只走 Express API Key，行为与改造前完全一致
  - `cookie`：只走 Cookie 直连反代
  - `hybrid`（推荐）：Express 优先，限流/5xx/未出流失败自动切 Cookie 兜底

- **流式 failover 的关键约束**：SSE 流一旦发出第一个有效 chunk 就不能中途切换上游（客户端已接到半截回复）。因此：
  - **上游内部**：新增 `failover_mode` 参数，"未出流 + 可切换错误 + 内部重试耗尽" → `raise UpstreamUnstartedError`
  - **路由层**：外层包装 generator 捕获该异常 → 报告熔断器 → 调下一通道重新发请求
  - **"出流"判定**：SSE 心跳（`: keep-alive`）与静态预填充前缀**不算**出流；只有正文/工具调用 chunk 才算

- **可切换错误白名单**：429 / 500 / 502 / 503 / 504。400 / 401 / 403（配置、鉴权、权限问题）**不切换**——切换也不会变好

- **Cookie 会话失效（cookie_error）**：hybrid + 未出流时也切换（Cookie 挂了自动走 Express 让请求成功，日志提示刷新 Cookie）

- **熔断器**（`app/failover.py`，内存版）：
  - 某通道连续失败 ≥ `failover_threshold`（默认 3）→ 冷却 `failover_cooldown_seconds`（默认 60s）
  - 冷却期间路由层直接跳过该通道，避免限流风暴反复撞墙
  - 成功后立即清零计数
  - 参数走控制台 settings 持久化，可热调

#### 文件改动

- **新增 `app/failover.py`**：`UpstreamUnstartedError` 异常 + `ChannelBreaker` 熔断器（单例）
- **`app/runtime_state.py`**：`use_web_proxy` 布尔 → `channel_strategy` 三档；磁盘旧数据自动迁移（`true→cookie`、`false→express`），迁移以磁盘原始数据为准；保留旧布尔接口兼容
- **`app/config.py`**：`DEFAULT_SETTINGS` 增加 `failover_threshold: 3`、`failover_cooldown_seconds: 60`
- **`app/routes/chat_api.py`**：路由层重写——通道顺序、可用性预检（无凭证通道直接剔除）、`_dispatch` 按序尝试、`_stream_with_failover` 流式包装
- **`app/api_helpers.py` / `app/upstreams/express_sdk.py`**：Express 通道三路径（真流式/假流式/非流式）加 `failover_mode`，可切换错误重试耗尽后抛异常
- **`app/upstreams/cookie_proxy.py`**：流式 `stream_generator` 的 `retryable_error`（耗尽）与 `cookie_error` 分支、生图假流式失败分支加 failover；`fatal_error` 不切换
- **`app/main.py`**：控制台通道卡片三档 UI；`/api/settings/mode` 接受 `express|cookie|hybrid`（兼容旧值 `api_key`/`web_proxy`）；`/api/settings/runtime` 回显新策略
- **`app/routes/models_api.py`**：模型列表放行条件改为 `strategy != "express"`（cookie/hybrid 都放行）
- **README.md**：新增「双通道策略」章节

#### 顺手修复的原有两个 bug

1. Express 真流式 `is_auto_attempt` 在已出流后仍 `raise`，可能触发整段答案重发 → 改为 `is_auto_attempt and not has_yielded` 才抛
2. `fallback_model`（location 钉定失败回退）重试不计入 attempt 上限，可能死循环 → 挪入带上限的循环内

---

### 3.2 fork + GHCR CI 自动构建 + 持续化部署

**目标**：改代码 → push → 自动出镜像 → 1Panel 一键升级。

**已完成**：

1. fork 到 `qyh9527/vertex2openai`
2. 本地 remote：`origin` → fork；`upstream` → 原作者 `bad-woman/vertex2openai`
3. 沿用上游自带的 GHCR CI（`.github/workflows/docker-image.yml`）：push 到 main 自动 `docker build` → `docker push ghcr.io/qyh9527/vertex2openai:latest`（fork 仓库 public，包公开，VPS 拉取免登录）
   - 期间曾自建 `build.yml` 后删除（避免与上游 CI 重复构建同一 tag）
4. 验证：镜像已产出 `ghcr.io/qyh9527/vertex2openai:latest`，匿名拉取可行（PUBLIC）
5. **双 tag 可回滚**（本批）：build/push 同时打 `latest` 与 `${{ github.sha }}`。VPS 出问题时把
   docker-compose 的 image 改回上一个 `ghcr.io/qyh9527/vertex2openai:<sha>` 即回滚，不用等重新构建

**VPS 操作**（一次性）：

```yaml
# docker-compose.yml
image: ghcr.io/qyh9527/vertex2openai:latest   # 替换原来的 bad-woman 镜像
```

1Panel 里重建容器。**数据不丢**：Cookie、Project ID、全部控制台设置持久化在挂载卷 `./data:/app/data`（`web_state.json`），升级镜像/重建容器都保留，**无需 sqlite**（单实例 JSON 落盘足够，sqlite 只在多实例并发写时才有意义）。

**日常流程**：本地改代码 → `git push` → Actions 自动构建（约 1-2 分钟）→ 1Panel 点「重建/升级」→ 生效。

---

### 3.3 upstream 定期同步

```bash
git remote add upstream https://github.com/bad-woman/vertex2openai.git   # 已配置

# 原作者更新时：
git fetch upstream
git merge upstream/main      # 或 git rebase upstream/main；冲突手动解决
git push origin main         # 推送后 GHCR CI 自动重建镜像
```

**冲突提示**：上游更新容易与本仓库改动冲突的文件：`app/routes/chat_api.py`、`app/api_helpers.py`、`app/main.py`。合并后先跑 `python -m compileall app` 语法检查再推。

---

### 3.4 Bug 修复：带预填充的流式请求 429 不重试

**现象**：自己用（普通对话）遇到 429 正常退避重试；别人用（酒馆预设，消息以 assistant 结尾带预填充）遇到 429 直接放弃不重试。

**根因**：Express 真流式把预填充静态前缀先发给客户端并置 `has_yielded=True`，而重试条件要求 `not has_yielded` → 带预填充的请求被误判为"已出流"而拒绝重试。

**修复**（`app/api_helpers.py`）：拆分为两个标志：

| 标志 | 含义 | 作用 |
|---|---|---|
| `has_yielded` | 已发出正文/工具调用 | 429 重试、故障转移的唯一判断依据 |
| `prefill_sent` | 已发出预填充静态前缀 | 重试时不重发（客户端不会看到重复开头）；预填充已发不触发跨通道 failover（避免切换后新通道重发导致重复） |

**验证**：模拟"预填充 + 前两次 429"——正常退避重试 3 次（1+2），预填充只出现 1 次不重复。

---

### 3.5 假流式改为按模型选择（`fake-` 前缀模型）

**现状问题**：`fake_streaming` 全局开关开启后所有模型强制假流式，无法按模型区分。

**目标**：开关语义改为"**注册 fake- 前缀模型**"——开关开启时 `/v1/models` 列表为每个模型额外生成 `fake-<模型名>` 条目；客户端选择 `fake-gemini-3.7-flash` 走假流式，选普通模型名走真流式。

**设计**：统一前缀常量 `FAKE_PREFIX = "fake-"`（定义在 `app/api_helpers.py` 供三处共用）。解析顺序：**先剥 `fake-`，再走现有 `-search` / legacy 逻辑**（循环剥除，支持 `fake-[EXPRESS] x` 任意顺序组合）。请求的 `request_obj.model` 保持原样（SSE chunk 回显 `fake-xxx`），仅"上游调用名 / 能力判定名"用剥完前缀的 base。

**文件改动**：

- **`app/routes/models_api.py`**：`add_model` 在开关开启时对每个 `(id, root)` 额外生成 `fake-` 变体，与 `-search` 变体正交（`fake-gemini-x-search` 也生成）
- **`app/upstreams/express_sdk.py`**：`_normalize_model_name` 返回 4 元组（新增 `is_fake`）；`chat_completions` 透传 `force_fake_streaming`
- **`app/api_helpers.py`**：`execute_gemini_call` 加 `force_fake_streaming`；假流式判定改为 `force_fake_streaming or is_image_request`（移除全局开关的强制语义）
- **`app/upstreams/cookie_proxy.py`**：模型名解析先剥 `fake-`（Cookie 通道无假流式实现，剥前缀当普通模型处理）
- **`app/main.py`**：控制台开关文案改为「注册 fake- 前缀模型」+ 说明（ⓘ 折叠）
- **`app/config.py`**：`fake_streaming` 注释更新

**不改的点**：生图模型强制假流式保留（`fake-gemini-x-image` 剥前缀后仍命中）；控制台「模型参数」下拉仍只列真实模型（fake- 变体参数与其基础模型一致）；`model_capabilities` / `resolve_express_model_path` / 预填充均基于剥完前缀的 base 名，无需动。

**宽容设计**：`fake-` 前缀请求**恒生效**（即使开关关闭），便于手动测试；开关只控制列表暴露。

---

### 3.6 上游稳定性：锁 SDK 2.x + 钉 api_version=v1beta1 + Client 复用

**起因**：研读官方 Express Mode REST 参考（`reference/express-mode/api-reference` 与 `.../rest/v1beta1/publishers.models/{generateContent,streamGenerateContent}`），确认三件事：

1. **Express Mode 是 Pre-GA**（文档明示 "as is"，接口可能变）→ 不能放任 SDK 版本漂移，否则重建镜像后行为静默变化
2. **端点只有三个方法**：`countTokens` / `generateContent` / `streamGenerateContent`（v1 与 v1beta1 各一套）——项目已全覆盖，无缺能力
3. 请求体另有 `labels`（仅用于计费与报告）与 `cachedContent`（显式上下文缓存，可省成本）字段，当前均未使用（见"已知边界"）

**改动**：

| 文件 | 改动 | 原因 |
|---|---|---|
| `app/requirements.txt` | `google-genai>=2.0.0` → `>=2.0.0,<3.0.0` | 允许 2.x 补丁更新，挡住 3.0 破坏性变更 |
| `app/http_options.py` | `HttpOptions` 显式 `api_version="v1beta1"` | `thinking_level`/`thinking_budget` 只在 v1beta1 提供；不钉会随 SDK 升级的默认值漂移 |
| `app/upstreams/express_sdk.py` | 新增 `_CLIENT_CACHE` + `_get_cached_client()` | 按 `(api_key, base_url, priority_paygo)` 复用 genai.Client（httpx 连接池惰性创建，复用即省 TLS 握手） |

**Client 复用的三个设计约束**（改缓存时别破坏）：

- 缓存键必须含 `priority_paygo`——Priority PayGo 请求头只能用于钉定到 global 的资源路径，串给普通请求会标错流量等级
- dict 的 get/set 在 GIL 下原子，异步协程并发安全；极端并发下重复创建只是多费一个对象
- **额度按 API Key 与请求计费，与连接是否复用无关**；429 限流也按请求判定，复用不会更严——单号用户放心复用
- 主路径与 location 钉定回退路径共用同一缓存（`fallback_client_factory` 也走 `_get_cached_client(key, False)`）

**踩坑记录（重要）**：

- ⚠️ **`gh run list` 显示的是旧记录**（列表有缓存/同步问题），判断 CI 是否触发、是否成功**必须用 API**：
  ```bash
  gh api repos/qyh9527/vertex2openai/actions/runs --jq '.workflow_runs[0:5][] | {id, status, conclusion, head_sha}'
  ```
- ⚠️ **系统 Python 3.14 的依赖环境是坏的**（pydantic 与 annotated_types 冲突、fastapi 缺 `__version__`），全量 import 冒烟在 Windows 本机跑不了；验证手段是 `python -m compileall app` + 直接 grep 已安装 google-genai 的 `types.py` 确认字段存在（2.19.0 确认 `HttpOptions.api_version` 合法）
- 文档核对结论：Express Mode 的**额度不是无限的**——官方专门有 [Error code 429](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/deploy/error-code-429) 与[吞吐量配额](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/resources/throughput-quota) 文档，429 是常态（项目熔断/failover 正是为此存在）

---

### 3.7 多账号凭证管理：多 Express Key + 多 Cookie 账号（控制台可管）

**现状问题**：Express Key 只能从环境变量启动时加载（增删 Key 得改 compose 重启）；Cookie 只有单一份，多账号分摊配额/规避 429 无从谈起。

**目标**：控制台「通道与凭证」页直接管理多 Key 与多 Cookie 账号，热生效，持久化在挂载卷（重建容器不丢）。

#### 核心设计

- **来源优先级**：控制台持久化列表（`web_state.json` 的 `express_keys` / `cookie_accounts`）> 环境变量（`VERTEX_EXPRESS_API_KEY` / `GOOGLE_COOKIE` 作初始值/兜底）。控制台保存后 `refresh_keys()` 热生效；清空列表自动回落环境变量。
- **多 Cookie 选择 = 请求级快照**：`get_current_cookie_account()` 用 **contextvars** 按 asyncio task（＝一次请求）隔离——`_get_cookie_string()` 与 `_get_project_id()` 在重试/流式路径会被调用多次，若每次都重新选号就会**串号**（cookie 是 A 账号的、project 是 B 的）；快照保证同一请求内（含重试、流式、failover 重发）恒用同一账号。多账号时按现有 `roundrobin` 开关轮询或随机；单账号行为与改造前完全一致。
- **选择粒度 = 账号**：location 钉定用的 project 取"第一个账号"（保存时 `google_cookie`/`google_project_id` 同步为首项），保持稳定不随轮询漂移。
- **兼容迁移**：旧 `google_cookie`/`google_project_id` 字段自动视为单账号列表视图；保存新列表时同步旧字段，旧读取接口与 `/api/cookie` 端点全部继续有效。
- **掩码回显**：Express Key 与 Cookie 在控制台一律只回显掩码（Key 露前后 4 位；Cookie 只报字段数/长度），不回填输入框（留空 = 保持原值），完整凭证永不进入前端 JS / 浏览器缓存。
- **Cookie 校验**：新增账号时校验 SAPISID 族字段（`cookie_auth.validate_cookie`），无效直接拒绝并提示。

#### 文件改动

- `app/runtime_state.py`：`express_keys` / `cookie_accounts` 持久化 + `get_current_cookie_account()`（contextvar 快照 + 轮询/随机，index 内存态）
- `app/express_key_manager.py`：来源改为「控制台列表优先，环境变量兜底」，`refresh_keys()` 热生效
- `app/upstreams/cookie_proxy.py`：`_get_cookie_string` / `_get_project_id` 改走请求级账号快照
- `app/routes/chat_api.py`：Express 预检改为「控制台列表 **或** 环境变量」——否则控制台配的 key 会被误剔除直接 503（真 bug，冒烟发现）
- `app/main.py`：通道页新增 Express Key 列表编辑器（多行文本域整表覆盖 + 清空回落）+ 多 Cookie 账号卡片（逐行增删改）；新端点 `POST /api/express-keys`（整表覆盖）、`POST /api/cookie-account`（单账号增改删）

**顺手修掉的 bug**：`set_express_keys` 曾把 `None` 存成字符串 `"None"`（已过滤非字符串类型）。

---

### 3.8 自动化测试体系（pytest，本机 3.11 venv）

**背景**：7000+ 行 Python 此前零自动化测试，"测试记录"全为手动验证；upstream merge 后只有 `compileall` 语法检查，行为被改坏看不出来。

**现状**：`tests/` 9 个文件、**142 个用例**全绿（`pytest.ini` 配 `asyncio_mode=auto`）。覆盖：

- runtime_state：迁移、旧布尔接口、非法值拒绝、深拷贝隔离、原子写、损坏文件降级
- ChannelBreaker 熔断：阈值、冷却到期、成功清零、通道独立、status
- ExpressKeyManager：轮询/随机/无 Key/刷新
- 模型名解析（`fake-`/`-search`/`[PAY]`/任意前缀组合）与 location 钉定、Client 缓存复用约束
- **路由层故障转移全路径**：429/5xx 切换、400/401/403 不切、熔断冷却跳过、流式未出流切换、无兜底错误流、非 hybrid 零行为变化
- Cookie 错误分类：项目级（Project ID/计费）vs Cookie 过期——README 专门纠正过的逻辑
- 预填充去重：流式（PrefillDeduper）与非流式（strip_prefill_overlap）
- 多账号凭证：迁移、轮询/随机、请求级快照不串号、来源优先级

**跑法**（Windows 本机，Python 3.11 venv，与 Docker `python:3.11-slim` 一致；系统 3.14 环境是坏的，别用）：

```bash
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install -r app\requirements.txt pytest pytest-asyncio
.venv\Scripts\python.exe -m pytest tests -q
```

conftest 把 `STATE_DIR` 指向独立临时目录（模块导入时即生效），**绝不碰真实 `web_state.json`**；contextvar 账号快照有 autouse fixture 清理。

**踩坑**：pytest-asyncio 需 `asyncio_mode=auto`；流式测试的"未出流失败"generator 必须带一个不可达的 `yield`（否则函数是普通协程而非 async generator，`'coroutine' object is not iterable`）。

---

### 3.9 日志落盘（按天轮转保留 7 天）

`docker logs` 随容器重建即清空，排查"昨天发生了什么"无从查起。`app/logger.py` 落盘到 `STATE_DIR/vertex2openai.log`（与 `web_state.json` 同目录、同在挂载卷内，重建容器不丢）：

- `TimedRotatingFileHandler` 按天轮转、保留 7 天、utf-8
- `custom_print` 同时写文件（去 ANSI，VPS 上 `tail -f` 看到纯文本）与推 SSE 控制台流
- 初始化失败不影响运行（打印警告降级为纯容器日志）；日志文件与 web_state.json 同一份 0600 权限约束

---

### 3.10 Express Client 复用防护：开关 + 自动舍弃

**现状问题**：3.6 引入按 `(api_key, base_url, priority_paygo)` 复用 genai.Client（省 TLS 握手）后，长驻进程里可能残留**失效的 keep-alive 连接**（上游空闲超时关闭、VPS 网络/网关变化等）。复用死连接会持续抛 `httpx.TransportError` 类错误（`RemoteProtocolError` / `ConnectError` / 超时）——连接本身已坏，重试 429/503 的逻辑救不了它。

**目标**：给复用层加两重保护——① 控制台可一键关掉复用（彻底新建 Client）；② 自动识别"连接级失败"，连续达到阈值后舍弃缓存 Client，下次请求重建连接池。

#### 核心设计

- **两个新设置**（控制台「Express Client 复用」卡片）：
  - `client_reuse`（默认 true）：关闭后 `_get_cached_client` 每请求新建 Client 不入缓存（排查"复用后持续连接错误"时用；代价是失去连接池、延迟增大）
  - `client_reuse_evict_threshold`（默认 5，0=不自动舍弃）：缓存 Client **连续连接级失败** ≥ 阈值即自动舍弃，下次请求重建
- **失败分类上报**（`api_helpers.report_client_failure(client, kind, reason)` → Client 上挂 `_vertex_on_failure` 回调）：
  - `kind="conn"`：**连接级失败计数**（`httpx.TransportError` 类：`ConnectError`/`RemoteProtocolError`/超时等）。**429 限流不算**——连接本身健康，误舍弃会在限流最狠的时候频繁重建握手
  - `kind="evict"`：**立即舍弃**，不等阈值——安全策略拦截（`PROHIBITED_CONTENT` 等 `promptFeedback.block_reason`）等"这条连接/会话状态不对"的硬错误
- **接入点**（api_helpers 三条路径）：假流式 / 真流式 / 非流式异常处理中识别 `httpx.TransportError` → `kind="conn"` 上报；假流式与非流式的 `block_reason` 检测点 → `kind="evict"` 上报。非复用 Client 无回调，静默跳过
- **计数不清零**（成功不归零）：stale 池会连续失败直到被舍弃，计数模型匹配；零星几次失败后正常回收也不痛（重建仅多一次握手）

**踩坑**：线程锁不可重入——舍弃逻辑在持有 `_CLIENT_CACHE_LOCK` 时内联 pop，不能调用外层再加锁的函数。

#### 文件改动

- `app/config.py`：新增 `client_reuse` / `client_reuse_evict_threshold` 默认值
- `app/upstreams/express_sdk.py`：缓存重构为 `_new_express_client` + `_on_client_failure(kind)`，`_get_cached_client` 支持开关
- `app/api_helpers.py`：新增 `report_client_failure`，三条异常路径 / 两处 block 检测接线
- `app/main.py`：控制台「Express Client 复用」卡片 + loadParams/saveSettings 接线
- `tests/test_express_sdk.py`：新增 4 条（开关 / 阈值舍弃 / 硬错误立即舍弃 / 阈值 0）

**研究依据**（联网 + 已装 SDK 源码核验）：

- 共享 httpx.AsyncClient 复用失效 keep-alive 连接 → `RemoteProtocolError` 的实战案例（"Ghost 503s"）；httpx 连接池参数（`max_connections` / `keepalive_expiry`）说明；google-genai [issue #516](https://github.com/googleapis/python-genai/issues/516)（并发下每请求新建 Client 明显变慢 → 复用有必要，应保留默认复用、靠自动舍弃兜底）
- 已装 SDK 源码核验：`genai.Client` 持有单个 `AsyncHttpxClient` 跨请求复用；`HttpOptions.async_client_args` 可传 httpx 池参数；SDK 自带 `_CONNECTION_ERRORS = (httpx.TimeoutException, httpx.ConnectError)`；`types.py` 确认 `PROHIBITED_CONTENT` 枚举与 `block_reason_message` 字段

**已知边界**：

- 舍弃只救"下一请求"：正在失败的请求已经失败；且 Express 通道连接级错误当前**不参与内部重试**（仅 429/503/quota 会重试），是"连续 N 次请求失败后恢复"
- 安全拦截舍弃**不改变拦截结果**：客户端收到的仍是拦截错误，仅强制后续请求重建连接池
- 真流式路径的 `prompt_feedback` block 未接（流式拦截以上抛异常形态出现，需字符串匹配，留待有实际报错样本再加）

---

### 3.11 第三通道：服务账号（Vertex SA）+ 四标签页通道管理 + 混合自动可配 + PayGo 层级融合

**需求**：① 新增"服务账号 JSON 凭证"的第三条上游通道（区域/项目方案由调研定案）；② 上游调用通道改四个标签页切换；③ 混合自动可配置 3 个渠道的开关、优先级顺序与每通道独立重试次数；④ 评估 ST-Vertex-PayGo 系列能否融合。

**调研结论（定案依据）**：
- **服务账号认证走官方库，不重写 gproxy 的手动 JWT**：gproxy（Rust 反代）与 ST-Vertex-PayGo 的服务账号模式都是标准 OAuth2 JWT-bearer（RS256 断言 → `oauth2.googleapis.com/token` → Bearer token）。本项目直接复用 `google-genai` 的 `credentials=` 参数：`service_account.Credentials.from_service_account_info(sa_json)` → `genai.Client(vertexai=True, project=, location=, credentials=)`，SDK 内部 `get_token_from_credentials` 在 token 过期时自动刷新（约 1h，异步锁防并发重刷）——即 gproxy `needs_refresh + exchange_token` 的官方等价物。**整条请求管线（真流式/假流式/非流式、预填充、思考、工具、生图、重试、failover、Client 复用防护）全部复用**，第三通道 = 新 upstream 类 + 凭证管理。佐证：原仓库 commit `0b5f9df` 前就有同方案的 `credentials_manager.py`，后被上游 "Express Mode only" 重构删除，本次=现代化恢复。
- **区域默认 `global`**（多数 Gemini 模型只在 global 提供，与 Express 钉定实测一致）；project 取 SA JSON 自带 `project_id`（可控制台覆盖）。SDK 原生按 Client 的 project/location 拼路径，无需 Express 那套模型路径透传 hack。
- **ST-Vertex-PayGo 不是新通道，是"PayGo 流量等级"开关**：它俩是 SillyTavern 插件对，给标准 Vertex 请求注入官方"按量共享容量"层级头（Priority/Flex/PayGo-only），认证借宿主。本项目已有 Priority 头，净增量仅 **Flex 档**（`X-Vertex-AI-LLM-Shared-Request-Type: flex` + `X-Server-Timeout: 1800`，需放大 httpx 超时）、**paygo_only 单独开关**、模型名单提示。回环票据机制是浏览器插件特有需求，不搬。

**核心改动**：

| 文件 | 改动 |
|---|---|
| `app/upstreams/service_account.py` | **新增**：SA Client 复用缓存（按 sa_json/project/location/层级头）、`validate_sa_credentials`、`ServiceAccountUpstream`（继承 ExpressSDKUpstream 仅覆写 `_resolve_client`，含 `[PAY]` 旧前缀兼容入口） |
| `app/routes/chat_api.py` | `CHANNELS` 注册 `vertex`；`_channel_order` 支持 hybrid 可配顺序（`get_hybrid_channels()`）；`_available_channels` 加 SA 预检 |
| `app/runtime_state.py` | `CHANNEL_STRATEGIES` 加 `vertex`；`sa_accounts` 持久化 + `get_current_sa_account()`（contextvar 请求级快照，复刻 cookie 模式）+ `_sa_env_fallback()`（VERTEX_SA_JSON/VERTEX_SA_FILE）；`get_hybrid_channels()` / `get_channel_retry()` |
| `app/config.py` | `DEFAULT_SETTINGS` 加 `hybrid_channels`、`channel_retry_overrides`、`paygo_tier`、`paygo_only`；AppSettings 加 `VERTEX_SA_JSON`/`VERTEX_SA_FILE` |
| `app/api_helpers.py` | `get_retry_settings(channel)` 支持每通道独立重试覆盖；`execute_gemini_call` / `gemini_fake_stream_generator` 加 `channel_name` 透传 |
| `app/http_options.py` | `resolve_paygo_headers(tier, paygo_only, is_global)` 头矩阵 + `paygo_timeout`（flex=1800）+ `resolve_paygo_bundle`；`is_flex_supported` **自动化黑名单**（gemini-2.x 系不支持 flex，真机 400，降级告警）；`get_http_options` 支持 headers/timeout；**PROXY_URL/SSL_CERT_FILE 改走预构建 httpx client**（修复 genai 2.19 `client_args['proxy']` 被 mTLS SSLContext 污染导致代理隧道 ConnectTimeout 的真机问题） |
| `app/upstreams/express_sdk.py` | `_resolve_client` 抽象点（通道客户端来源可覆写）；PayGo 头接入层级设置；Client 缓存键含层级头 |
| `app/upstreams/cookie_proxy.py` | 重试读取改 `get_retry_settings("cookie")` |
| `app/main.py` | 通道页改**四标签页**（Express/Cookie/服务账号/混合自动）；服务账号多账号编辑器（掩码回显，不回填 JSON）；混合自动页（3 通道开关+排序+每通道重试+PayGo 层级下拉）；`/api/sa-account` 增改删端点；`/api/settings/runtime` 回显新字段；mode 映射加 `vertex` |
| `app/routes/models_api.py` | 放行条件加 `has_sa_account` |
| `app/requirements.txt` | 显式加 `google-auth` |
| `tests/` | 新增 `test_service_account.py` / `test_http_options.py`；扩展 `test_route_dispatch.py`（vertex 路由/3 通道 failover/每通道重试）；conftest 清 SA 快照 |

**测试**：全量 **203 用例全绿**（原 146 + 新 57）+ `compileall` 绿 + 真实 uvicorn 冒烟 14/14
（SA 增删改/掩码/vertex 策略/模型列表/hybrid 配置持久化）+ **真实服务账号端到端验证**：
本机代理下 token 换发（JWT-bearer）→ 非流式/流式真实出文 → PayGo Priority 头被接受 → Flex 头对
`gemini-2.5-flash` 返回 400（`Flex API is not supported for model`，对 `gemini-3.6-flash` 正常）——
由此实现 `is_flex_supported` **自动化正则黑名单**（gemini-2.x 系打 flex 头自动降级+告警）。

**已知边界**：
- 服务账号需项目**开计费** + SA 有 `roles/aiplatform.user`，否则 403 `requires billing`（错误文案已规划，同 Cookie 通道的项目级排查指引）。
- **Flex 头只支持 3.x 及更新模型**（2.5 返回 400）：已用 `is_flex_supported` 自动化正则黑名单处理——`gemini-2.x` 系请求自动降级为普通请求并告警，3.x/未来模型前向放行（无需维护静态名单）。
- **Flex 档 1800s 排队超时可能拖长单请求**；默认 `auto` 不影响既有行为。
- SA Client 复用走独立的 `_SA_CLIENT_CACHE`（与 Express `_CLIENT_CACHE` 同款防护语义：连接级失败阈值舍弃/硬错误立即舍弃）。
- `[PAY]` 前缀只在 `vertex`/`hybrid` 策略下被剥离；`express` 策略下仍报"已移除"。
- **PROXY_URL 真机修复**：genai 2.19 的 `client_args['proxy']` 会被 `_ensure_httpx_ssl_ctx` 注入 mTLS
  SSLContext，走 HTTP 代理建 CONNECT 隧道时 `start_tls` ConnectTimeout；`get_http_options` 已改为预构建
  httpx client（`httpx_client`/`httpx_async_client` 字段），VPS 直连不受影响。

---

## 四、测试记录

所有测试在本机 Python 3.11 venv（与项目 Docker 环境 `python:3.11-slim` 一致）中进行：

| 测试项 | 结果 |
|---|---|
| 旧 `web_state.json`（`use_web_proxy: true`）迁移 → `channel_strategy=cookie` | ✅ |
| 旧布尔接口 `enable_web_proxy` / `is_web_proxy_enabled` 兼容 | ✅ |
| `set_channel_strategy` 非法值拒绝 | ✅ |
| 熔断器：连续 3 次失败进冷却、成功清零、通道间互不影响 | ✅ |
| 非流式：express 抛 UpstreamUnstartedError → 切 cookie 成功 | ✅ |
| 非流式：express 返回 429 → 切 cookie 成功 | ✅ |
| 非流式：400 不切换（如实报错） | ✅ |
| 双通道全失败 → 兜底错误响应 | ✅ |
| 流式：express 心跳后抛异常 → 切换后客户端只收到一份完整 SSE | ✅ |
| 熔断：冷却期 express 被跳过，请求仍成功 | ✅ |
| **非 hybrid 零行为变化**：不包装、不切换、异常透传 | ✅ |
| 预填充 + 429：正常退避重试 3 次，预填充不重复 | ✅ |
| 假流式：`fake-` 前缀模型解析（含 `-search`、任意前缀组合） | ✅ |
| 假流式：开关关无 fake- 条目 / 开关开每模型 +2 条目（含 `-search`） | ✅ |
| 假流式：fake- 请求走 `🌊 [假流式]`，普通请求走真流式 | ✅ |
| 假流式：root 字段、Cookie 通道前缀剥离 | ✅ |
| 3.6 改动：`python -m compileall app` 全绿 | ✅ |
| 3.6 改动：google-genai 2.19.0 的 `HttpOptions.api_version` 字段存在（grep 已装 SDK 源码确认） | ✅ |
| 3.6 改动：GHCR CI 构建 `ab53e48` 成功（run 32828775055） | ✅ |
| **自动化测试（3.8）：pytest 142 用例全绿** | ✅ |
| 多账号：控制台 key 列表保存/清空/环境变量回落（热生效） | ✅ |
| 多账号：cookie 旧字段迁移、轮询/随机、请求级快照不串号 | ✅ |
| 多账号：通道预检认控制台 key（修复的 bug） | ✅ |
| 日志落盘：custom_print → 文件 + SSE 双写 | ✅ |
| 真实 uvicorn 冒烟：保存 key → 增删 Cookie 账号 → 重启磁盘读回 → 请求路径 401/流式错误流（非 503 无通道） | ✅ |
| Client 复用：开关关 → 每请求新建 Client（无回调、不入缓存） | ✅ |
| Client 复用：连接级失败达阈值 → 自动舍弃重建 | ✅ |
| Client 复用：硬错误（安全拦截）→ 立即舍弃不等阈值 | ✅ |
| Client 复用：阈值 0 = 不自动舍弃 | ✅ |
| **回归：全量 pytest 146 用例全绿（原 142 + 新 4）** | ✅ |
| 3.11 服务账号：SA JSON 校验（缺字段/坏 JSON/type 拒绝）、环境变量兜底（内联/文件/缺失） | ✅ |
| 3.11 服务账号：多账号请求级快照不串号、轮询/随机、控制台清空回落环境变量 | ✅ |
| 3.11 服务账号：SA Client 缓存键（sa_json/project/location/层级头）、无账号返回 401 | ✅ |
| 3.11 服务账号：`[PAY]` 前缀在 vertex 通道被剥除 | ✅ |
| 3.11 PayGo：头矩阵（auto/off/standard/flex/priority × paygo_only × global/非 global）、非 global 降级告警 | ✅ |
| 3.11 路由：strategy=vertex 只走 SA；hybrid 三通道顺序可配；每通道重试覆盖生效 | ✅ |
| 3.11 冒烟：真实 uvicorn 14/14（SA 增删改/掩码/vertex 策略/模型列表/hybrid 配置持久化） | ✅ |
| 3.11 真机：真实服务账号 token 换发 + 非流式/流式出文 + PayGo Priority 头 + Flex 头(3.6-flash) 成功；Flex 头(2.5) 400 证实模型名单必要 | ✅ |
| 3.11 Flex 黑名单：`is_flex_supported` 判定矩阵（2.x 拒 / 3.x+/未来放行 / 大小写）+ flex 自动降级告警 | ✅ |
| **回归：全量 pytest 203 用例全绿（原 146 + 新 57）** | ✅ |

---

## 五、当前仓库结构（改动相关）

```
app/
├── failover.py                  # 新增：UpstreamUnstartedError + ChannelBreaker 熔断器
├── runtime_state.py             # channel_strategy 四档 + 旧数据迁移；sa_accounts / 混合自动可配
├── config.py                    # failover_threshold / failover_cooldown_seconds / hybrid_channels / paygo_tier
├── api_helpers.py               # FAKE_PREFIX；execute_gemini_call(failover_mode, force_fake_streaming)；
│                                #   get_retry_settings(channel) 每通道重试；真流式 has_yielded/prefill_sent
├── http_options.py              # PayGo 层级头矩阵 resolve_paygo_headers / resolve_paygo_bundle / paygo_timeout
├── main.py                      # 控制台四标签页通道 UI、mode API、/api/sa-account
├── routes/
│   ├── chat_api.py              # 多通道路由、_dispatch、_stream_with_failover（三通道 + 可配顺序）
│   └── models_api.py            # fake- 变体注册、放行条件（含服务账号）
└── upstreams/
    ├── express_sdk.py           # _normalize_model_name 4 元组、_resolve_client 抽象点、
    │                            #   Client 复用防护（开关 + 连接级失败舍弃/硬错误立即舍弃）
    ├── cookie_proxy.py          # failover_mode 分支、fake- 前缀剥离
    └── service_account.py       # 新增：SA Client 复用缓存、凭证校验、ServiceAccountUpstream（[PAY] 兼容）
.github/workflows/docker-image.yml  # 上游自带 GHCR CI（自动构建 latest）
```

---

## 六、部署与日常使用

### 部署

```bash
# VPS docker-compose.yml 改一行 image，1Panel 重建即可
image: ghcr.io/qyh9527/vertex2openai:latest
```

### 日常使用要点

1. **通道策略**：控制台「通道与凭证」选「混合自动」最省心——Express 被 429 限流自动切 Cookie 兜底
2. **假流式**：需要假流式的模型在前端选 `fake-<模型名>`（需控制台开关开启）；普通模型名是真流式
3. **Cookie 过期**：hybrid 模式下 Cookie 失效会自动切 Express 不中断服务，日志提示刷新 Cookie（控制台「通道与凭证」重新粘贴）
4. **429 多 Key**：`VERTEX_EXPRESS_API_KEY` 配多个 + 控制台开「多 Key 轮询」可进一步摊薄限流

### 升级与同步

```bash
git push origin main                      # 改完代码直接推，CI 自动出镜像
git fetch upstream && git merge upstream/main && git push   # 合并原作者更新
```

---

## 七、已知边界与后续建议

> 本文件是**自包含的项目状态文档**：开新会话时直接 @ 本文件即可获得完整背景、改造历史、当前能力、测试与部署说明。

- **流式 failover 是"响应头之前"级**：已出流的失败只能如实收尾，无法切换。对 429（请求未开始就被拒）完全够用
- **Cookie 通道假流式**：目前没有实现，`fake-` 前缀在 Cookie 通道被剥掉当普通模型处理（生图除外）。若需要可为 Cookie 通道补假流式实现
- **熔断器是进程内存态**：重启后计数清零，属正常（冷却期最长 60s，影响极小）
- **Client 复用自动舍弃是"下一请求级"**：正在失败的请求不会因舍弃而重放成功；且 Express 通道连接级错误不参与内部重试（仅 429/503/quota 重试）。若想"一次请求内连接错误立即重连"，需额外把 `httpx.TransportError` 类并入可重试判定（当前刻意没做，避免改变既有失败语义）
- **多 Cookie 轮询 index 是内存态**：重启后从头轮，属正常（轮换本身无状态要求）
- **账号快照的一致性优先**：保存账号列表后，正在进行的请求仍用旧快照账号，下一请求才用新列表（可接受）
- **多实例部署**：若未来多副本，需把熔断状态/会话/签名缓存迁移到 Redis 等共享存储（当前单实例够用）
- **Express Mode 是 Pre-GA**：SDK 已锁 2.x、api_version 已钉 v1beta1，行为可复现；升级镜像/同步 upstream 前先看 google-genai CHANGELOG，升级后开「出站调试」核对 thinking 参数是否照旧
- **「钉定 location」是未文档化用法**：官方 Express 端点格式不含 projects/locations，实测可用但 Pre-GA 下 Google 可能收紧；已有 `is_location_pin_failure` 兜底（失败自动退回裸模型名），真失效时不会更糟，只是回到后端自选路由
- **Flex 层级名单是自动化黑名单**（`is_flex_supported`）：只挡真机确认不支持的 `gemini-2.x`，3.x 及未来模型前向放行（不维护静态名单）；若某未来模型真不支持 flex 会收到明确 400 如实报错
- **后续可选**：① `labels` 请求体字段（仅计费报告用，一行级改动，可给请求打 `channel=express` 等标签方便看账单）；② `cachedContent` 显式上下文缓存（官方称可保证成本节省，酒馆长预设场景有价值，但需自己管缓存创建/过期/删除生命周期，工作量中）；③ 若需把 Flex 名单升级为「只放行已知模型」的保守白名单（对照 ST 的 KNOWN_FLEX_MODELS），可加一个控制台可配置的名单覆盖项

---

## 八、本批次 TODO 收尾记录（原 TODO.md 已整合进本文档后删除）

**任务**：第三通道（服务账号）+ 四标签页 + 混合自动可配 + PayGo 融合。状态 **✅ 完成并真机验证**。

| 阶段 | 内容 | 状态 |
|---|---|---|
| A | 配置/状态层：config 新设置键 + runtime_state 的 sa_accounts / hybrid 可配 / 请求级快照 | ✅ |
| B | 服务账号通道：`service_account.py`（SDK credentials= 认证、多账号、[PAY] 兼容） | ✅ |
| C | 路由层：vertex 注册、hybrid 可配顺序、每通道重试、模型列表放行 | ✅ |
| D | PayGo 融合：层级头矩阵、flex 超时、flex 模型黑名单、PROXY_URL 预构建 client 修复 | ✅ |
| E | 控制台 UI：四标签页 + 服务账号编辑器 + 混合自动页 + PayGo 下拉 | ✅ |
| F | 测试：`test_service_account.py` / `test_http_options.py` + 路由扩展，**203 用例全绿** | ✅ |
| G | 文档：README + 本文件 + requirements 显式 google-auth | ✅ |

**真机验证结论**：服务账号 OAuth2 JWT-bearer 认证、非流式/流式出文、PayGo Priority/Flex 头全部真实可用；Flex 对 2.x 模型 400 已由黑名单自动化降级；PROXY_URL 代理隧道 bug 已修复。VPS 部署后建议：控制台配好 SA JSON → 切 `vertex` 或 `hybrid` → 用 `debug_outbound` 核对出站头。**剩余可选增强**：`labels` 计费标签、`cachedContent` 上下文缓存、Flex 保守白名单（见上文"后续可选"）。
