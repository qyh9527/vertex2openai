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

---

## 五、当前仓库结构（改动相关）

```
app/
├── failover.py                  # 新增：UpstreamUnstartedError + ChannelBreaker 熔断器
├── runtime_state.py             # channel_strategy 三档 + 旧数据迁移
├── config.py                    # failover_threshold / failover_cooldown_seconds
├── api_helpers.py               # FAKE_PREFIX；execute_gemini_call(failover_mode, force_fake_streaming)；
│                                #   真流式 has_yielded / prefill_sent 双标志修复
├── main.py                      # 控制台三档通道 UI、假流式开关文案、mode API
├── routes/
│   ├── chat_api.py              # 多通道路由、_dispatch、_stream_with_failover
│   └── models_api.py            # fake- 变体注册、放行条件
└── upstreams/
    ├── express_sdk.py           # _normalize_model_name 4 元组、force_fake_streaming 透传
    └── cookie_proxy.py          # failover_mode 分支、fake- 前缀剥离
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

- **流式 failover 是"响应头之前"级**：已出流的失败只能如实收尾，无法切换。对 429（请求未开始就被拒）完全够用
- **Cookie 通道假流式**：目前没有实现，`fake-` 前缀在 Cookie 通道被剥掉当普通模型处理（生图除外）。若需要可为 Cookie 通道补假流式实现
- **熔断器是进程内存态**：重启后计数清零，属正常（冷却期最长 60s，影响极小）
- **多实例部署**：若未来多副本，需把熔断状态/会话/签名缓存迁移到 Redis 等共享存储（当前单实例够用）
