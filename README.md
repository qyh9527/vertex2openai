---
title: Vertex2OpenAI Express Adapter
emoji: 🔄
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
---

# Vertex2OpenAI Express Adapter

Vertex2OpenAI 是一个 **OpenAI API 兼容代理**。它对外提供 OpenAI 风格的 `/v1/chat/completions` 与 `/v1/models` 接口，对内支持三条上游通道调用 **Google Agent Platform（原 Vertex AI）的 Gemini 模型**：

- **Express API Key（标准模式）**：用官方 `google-genai` SDK + `VERTEX_EXPRESS_API_KEY` 调用。
- **Cookie 直连反代模式**：用 Google Cloud 控制台 Cookie + Project ID 直连控制台私有 `batchGraphql` 接口（无需浏览器）。
- **服务账号（Vertex SA）**：用 Google Cloud **服务账号 JSON** 走标准 Vertex AI 认证（Bearer token），由 `google-genai` SDK 内部完成 OAuth2 JWT-bearer 换 token。

> 管理控制台 UI 名为 **agentplatform2api**；仓库/镜像名仍沿用 `Vertex2OpenAI`。
>
> 📌 **说明**：Google 已将 Vertex AI 更名为 **Agent Platform**，文档表述已同步（历史标识符如环境变量 `VERTEX_EXPRESS_API_KEY`、SDK 参数 `vertexai=True`、文件名 `vertexModels.json` 为兼容性保留）。**无头浏览器（Playwright）模式已移除**，由更轻量的 Cookie 直连模式完全替代。

## 功能特性

- **三上游通道，一键切换**
  - Express API Key：多 Key 随机或轮询调用官方 SDK。**控制台可在线增删 Key 列表**（无需改 compose 重启），保存后热生效；未在控制台配置时回落环境变量。
  - Cookie 直连反代：Cookie + SAPISIDHASH 直连 `batchGraphql`，走网页端配额（含最新预览模型），真流式、防 60s 超时。**支持多 Cookie 账号**（每账号 = 一份 Cookie + 一个 Project ID），按「多 Key 轮询」开关轮询或随机选号，同一请求内恒用同一账号（快照隔离不串号）。**注意：走的是私有接口，见下方风险提示。**
  - **服务账号（Vertex SA）**：Google Cloud 服务账号 JSON 认证，走标准 Vertex AI 端点（与 Express 同一套 generateContent，用 Bearer token）。**支持多服务账号**，按轮询/随机选号、请求级快照不串号；区域默认 `global`。适合"有项目配额、想按项目维度管理"的场景。
  - **混合自动（四标签页可配）**：通道页分 **Express / Cookie / 服务账号 / 混合自动** 四个标签页；「混合自动」页可**开关 3 个参与通道、调整优先级顺序、选择固定优先级或三渠道随机调度、设置每通道独立重试次数**（熔断仍按通道独立计数）。
- **OpenAI 兼容接口**：`GET /v1/models`、`POST /v1/chat/completions`。
  也支持按渠道固定入口（适合把 OpenAI 客户端的 `base_url` 直接设为对应路径）：
  `/<channel>/v1`，其中三个简短渠道名为 `express`（Express API Key）、`cookie`（Cookie 直连）、`vertex`（服务账号 SA）。
  例如 `http://127.0.0.1:8050/vertex/v1`；该前缀下同时提供 `/models` 与 `/chat/completions`。
  不使用前缀时，`/v1` 仍完全按控制台当前通道策略执行。
- **管理控制台（浅色风格，单文件，免构建）**
  - **仅密码登录**：打开根路径 `/`，输入密码（即 `API_KEY`）即可，无需账号。
  - 标准模式 / Cookie 直连 / 服务账号 / **混合自动（三通道故障转移）** 在线一键切换。
  - 在线热更新并保存 Google Cookie 与 Project ID；智能解析 `Cookie-Editor` 导出的 JSON / Header String；自动从整条控制台 URL 提取 Project ID。
  - **模型参数面板**：按所选模型显示其支持能力，并在线调整思考强度、生图分辨率与比例、采样默认值、输入图压缩、重试、假流式/轮询/安全分显示、预填充兼容模式（详见下文）。
  - **可配置输入搬运**：从客户端用自定义 XML 包裹的最新输入中提取载荷，追加到前一条 assistant 消息尾部，并替换 user 消息；支持“不开 / 只在假流开 / 无论真假流都开”三态。标签与占位语都留空默认，必须由控制台显式配置。
  - **实时监控**：运行日志推流（含**持久化历史回放** + 「📌 自动滚动」开关）、健康度图表（成功/错误环形占比，重试单列展示，避免把“额外尝试”误画成请求结果）、**Token 用量统计已持久化**（`STATE_DIR/stats.json`，重建容器不丢）+ **每日趋势堆叠柱状图**（7/30 天切换；稀疏日期自动补齐窗口、Prompt/Completion 双色分层、悬停/聚焦查看每日明细）+ **缓存命中率**（隐式上下文缓存 90% 折扣是否生效，命中即省钱）+ **估算美刀花费**（按官方按量价，见下方「用量与花费统计」）。Token 比例条按真实数值动态渲染；Cookie 私有接口通常不返回可靠用量，token/缓存/花费统计仅标准 Express / 服务账号通道计入。
  - **防截断（Anti-Truncation）**：下游请求体加 `"anti_truncation": true`（字段名可在控制台自定义）即对**该请求**启用"合成传输工具"包装——指示模型把最终回答放进工具参数输出，绕开重提示词场景（酒馆复杂预设/长历史）下的回答截断，代理透明还原为 `assistant.content`，真实工具调用不受影响（仅文本/Chat 模型适用）。
  - **模型列表手动管理**：模型列表**不再自动拉取**远程配置（曾启动时 + `/v1/models` 每 3600s 自动刷新）；控制台「🌐 获取远程模型」手动刷新（结果持久化到磁盘）、「📝 编辑模型」弹窗添加/删除**自定义模型**（持久化，合并进 `/v1/models`）。
- **Gemini 能力与适配**
  - 文本对话、流式（SSE）与非流式。
  - OpenAI tools / function calling ↔ Gemini function calling 适配（含 Gemini 3.x 多轮所需的 thought signature 编解码）；Express 与 Cookie 直连通道均支持自定义函数声明、调用与结果回传。
  - **安全分类对齐**：三条通道下发同一套安全设置（`HARM_CATEGORY_HATE_SPEECH`、`DANGEROUS_CONTENT`、`SEXUALLY_EXPLICIT`、`HARASSMENT`、`JAILBREAK`），阈值最宽松，避免通道间行为不一致。
    > 📌 **更正**：早期版本曾把"有思考、正文为空"归因于 Cookie 通道缺少 `HARM_CATEGORY_JAILBREAK`。这个解释是错的——按官方文档，**越狱分类器默认就是关闭的**，要打开必须显式把该分类的阈值设成具体的拦截值；不下发它不会启用任何过滤，下发 `OFF` 也只是空操作。该现象的真实成因见下方"3.6-flash 只返回思考"一节（前端恒发 `reasoning_effort=xhigh` 导致原生思考在 HIGH 档跑飞/被截断）。
  - **上游错误如实透传**：模型在当前项目/区域不可用（404）、权限不足（403）、参数非法（400）等，会以对应 HTTP 状态码 + OpenAI 错误格式返回，而非笼统的 500。
  - Google Search 增强：**文本模型**在模型名后加 `-search` 后缀按需开启。
  - 自动保留 Gemini 思考过程（Thinking），以 `reasoning_content` 返回。
  - 生图模型：输入图压缩、按模型的比例白名单校验、4K 等分辨率、图片输出转 Markdown data URL；生图强制"假流式"整块输出，避免超大 base64 卡死前端。
  - **按模型能力自动裁剪参数**：不同模型支持的参数不同，代理会自动移除目标模型不支持的参数以避免 400（详见"控制台与模型参数"）。
  - **自动退避重试**：三条通道均内置 429/拥堵自动退避重试，**次数与间隔可在控制台调整**（默认约 10 次；混合自动里还可按通道独立设重试次数）。流式模式下，连接建立即发送 SSE 心跳、且退避等待期间持续发送心跳，避免前端因长时间无数据（如 3.1-pro 频繁 429）而**超时断开**。
  - **预填充智能兼容**：自动处理"以 model 轮次结尾"被新模型拒绝（400）的问题（详见下文）。
  - **断连即停**：客户端断开后立即停止上游调用与重试。
- **中文运行日志**：密钥轮询、上游调用、重试退避、权限报错、Token 统计等均为中文实时说明。

## 快速开始（本地 Docker）

编辑 `docker-compose.yml`，设置初始环境变量：

```yaml
environment:
  - API_KEY=your_adapter_api_key
  - VERTEX_EXPRESS_API_KEY=your_vertex_express_api_key
```

启动：

```bash
docker compose up -d
```

默认将宿主机 `8050` 映射到容器 `7860`。浏览器打开控制台并用密码（`API_KEY`）登录：

```text
http://localhost:8050
```

## 环境变量

| 变量 | 必填 | 默认值 | 说明 |
|---|---:|---|---|
| `API_KEY` | 是 | `123456` | 保护本代理的 Key，同时也是**控制台登录密码**。客户端请求用 `Authorization: Bearer <API_KEY>`。 |
| `VERTEX_EXPRESS_API_KEY` | 否 | 空 | Gemini Express Mode API Key，多个用英文逗号分隔。标准模式使用。**控制台可在线增删 Key 列表并持久化（`web_state.json`），保存后覆盖环境变量；未在控制台配置时环境变量作为兜底。** |
| `ROUNDROBIN` | 否 | `false` | 多 Express Key 轮询(`true`)或随机(`false`)。可在控制台热改。 |
| `FAKE_STREAMING` | 否 | `false` | 假流式开关：开启后 `/v1/models` 为每个模型注册 `fake-<模型名>` 条目，客户端选中即对该请求强制假流式（其余模型保持真实流式）；生图模型始终强制假流式。可在控制台热改。 |
| `FAKE_STREAMING_INTERVAL` | 否 | `1.0` | 假流式等待期间 keep-alive 间隔秒数。可在控制台热改。 |
| `MODELS_CONFIG_URL` | 否 | 仓库 `vertexModels.json` | 远程模型列表地址；改远程文件即可刷新，无需重部署。 |
| `SAFETY_SCORE` | 否 | `false` | 是否把 Gemini safety ratings 附加到输出。可在控制台热改。 |
| `PROXY_URL` | 否 | 空 | 上游 HTTP/HTTPS/SOCKS 代理。 |
| `SSL_CERT_FILE` | 否 | 空 | 自定义证书路径。 |
| `VERTEX_BASE_URL` | 否 | 空 | **高级**：覆盖标准模式的上游 `base_url`，正常无需设置。要钉定区域请用控制台的「标准模式 location」（详见下方「标准模式的 location」）。 |
| `GOOGLE_COOKIE` | 否 | 空 | Cookie 直连模式的 Google Cookie（初始值，后续可在控制台更新）。 |
| `GOOGLE_PROJECT_ID` | 否 | 空 | Cookie 直连模式的 Project ID（初始值，后续可在控制台更新）。 |
| `VERTEX_SA_JSON` | 否 | 空 | 服务账号通道的 SA JSON（内联字符串；控制台未配账号时作兜底，也可直接在控制台粘贴）。 |
| `VERTEX_SA_FILE` | 否 | 空 | 服务账号通道的 SA JSON **文件路径**（docker 挂载常用；优先级低于 `VERTEX_SA_JSON`，两者都有时用内联）。 |
| `EXPERIMENT_FLAGS` | 否 | 空 | 可选：batchGraphql 的 `experimentFlagsBinary`，一般无需设置。 |
| `STATE_DIR` | 否 | `.` | `web_state.json` 的存放目录。用 Docker 时请指向挂载卷，否则重建容器会丢失全部设置与 Cookie。 |
| `ALLOW_DEFAULT_KEY` | 否 | 空 | 仅用于在公开托管环境（如 HF Space）临时放行默认 `API_KEY`，正常部署不要设置。 |

> 提示：`ROUNDROBIN`、`FAKE_STREAMING(_INTERVAL)`、`SAFETY_SCORE` 等环境变量仅作为**初始值**，运行时以控制台设置为准。

---

## 行为变更说明（整改后）

按 `REFACTOR_PLAN.md` 完成整改后，以下行为与旧版本不同，升级时请注意：

| 项目 | 旧行为 | 新行为 |
|---|---|---|
| **思考档位兜底** | 非法档位一律抬到 `high`（Pro 上选 minimal 反而变 high） | 就近**向下**夹取：Pro 选 minimal → `low` |
| **`retry_max` 语义** | Express 通道当作总次数，设 0 时一次请求都不发 | 统一为「重试次数」，总请求数 = `retry_max + 1`，设 0 仍请求一次；取值钳到 0–50 |
| **并行函数调用（流式）** | 只发第一个，`index` 恒为 0 | 全部下发，`index` 跨 chunk 稳定递增 |
| **思考签名** | base64 拼进 `tool_call_id`（上千字符，易被前端截断） | 主要通过 OpenAI 扩展 `extra_content.google.thought_signature` 原样回传；进程内短期缓存、旧 id 格式与官方 `skip_thought_signature_validator` 哨兵仅作兼容兜底 |
| **Cookie 通道 + 函数调用** | 静默把 `role=tool` 折成 model，发出错乱历史 | 原生下发函数声明，按 `functionCall` / `functionResponse` 回放历史并保留 thought signature |
| **Cookie 通道输入图** | 不压缩、不支持 http(s) 图片、不解析正文内联图 | 与标准通道一致（压缩开关对三条通道都生效） |
| **`stop` 字段** | 只接受数组，传字符串 422 | 字符串/数组都接受 |
| **`logprobs` 字段** | 按 Gemini 语义当整数 | 兼容 OpenAI 的 `logprobs: bool` + `top_logprobs: int` |
| **Express 流式 usage** | 从不下发，客户端显示 0 | 支持 `stream_options.include_usage` |
| **控制台 Cookie 回显** | 明文返回完整 Cookie | 仅返回“已配置/字段数/总长度”，不返回任何值的前后缀；输入框留空＝保持原 Cookie 不变 |
| **登录** | 无限速、会话 token 为确定值 | 失败 3 次后指数退避；随机会话 token，可单独失效 |
| **状态文件** | 每次读设置都同步读盘、非原子写 | 内存优先 + 原子写 + 权限 0600 + 支持 `STATE_DIR` |
| **文本保真** | 所有消息的多空格/缩进被压平 | 仅在确实抽走内联图片时才压平 |
| **2.5 Flash-Lite 思考预算** | 下限按 0 处理 | 下限 512（`0` 仍表示关闭） |
| **生图比例白名单** | 所有生图模型共用同一套比例 | 按模型分别校验；Flash Image 支持含 `9:21` 在内的扩展比例，Pro Image 使用较小白名单 |

思考签名优先使用请求/响应中的 `extra_content.google` 携带，因此跨进程、多副本部署不依赖本地缓存；旧格式、进程内缓存与跳过校验哨兵仅用于兼容缺失显式签名的客户端。

---

## 控制台与模型参数

控制台"模型参数"面板可在线调整全局默认值，并按所选模型显示其支持情况。可调项：思考强度（3.x 档位 / 2.5 预算）、生图分辨率与默认比例、采样默认值（temperature/top_p/max_tokens）、输入图压缩（开关/边长/质量）、重试次数与退避、假流式/轮询/安全分显示、预填充兼容模式与"预填充时压制原生思考"开关、续写指令模板、Cookie 调试日志，以及全局的可配置输入搬运。

### 参数优先级

**单次请求 > 该模型专属 > 控制台全局默认 > 内置默认。**

- 有请求级写法的参数：请求体字段优先。标准字段 `temperature`/`top_p`/`max_tokens` 等，扩展字段 `reasoning_effort`、`thinking_budget`、`image_size`、`aspect_ratio`/`ar`。客户端不传时依次回退到"该模型专属 → 控制台全局 → 模型默认"。
- 两个例外：
  1. **"模型不支持"优先级最高**（最后一步裁剪）：目标模型不支持的参数，无论来自请求、专属还是全局都会被移除（避免 400）。
  2. **全局项无请求级/专属写法**：图压缩、重试、假流式、轮询、安全分显示、预填充模式与压制开关、续写模板、Cookie 调试仅由控制台全局决定。

### 按模型单独保存参数（per-model overrides）

模型参数面板顶部选择模型后，可为**当前所选模型**单独保存专属值——支持覆盖的三类：**思考（档位/预算）、生图（分辨率/比例）、采样默认（temperature/top_p/max_tokens）**。

- **保存为该模型专属**：只把上述三类字段存成该模型的专属配置；下拉框中该模型名后会显示 `★`，并出现"已有专属参数"徽章。
- **清除该模型专属**：删除该模型专属配置，回退到全局默认。
- 底部"保存设置"按钮保存**全局默认 + 基础设施项**；当所选模型已有专属配置时，它**不会**用面板里显示的专属值覆盖全局（避免误操作），只保存基础设施项。
- 基础设施项（图压缩/重试/假流式/预填充模式与压制/安全分/Cookie 调试）不支持按模型覆盖，始终全局唯一。

### 按模型能力自动裁剪（`app/model_capabilities.py`，依据官方文档）

- **采样参数弃用**：自 **Gemini 3.6 Flash / 3.5 Flash-Lite 起（及所有更新/未来模型）**，`temperature`/`top_p`/`top_k` 已废弃（现被忽略、未来返回 400），代理会**自动移除**；更早的 3.x（如 3.0–3.5 非 lite）仍可用，但官方建议保持默认。
- **`candidate_count`**：所有 Gemini 3.x 不支持，自动移除。
- **思考**：Gemini 3.x 用 `thinking_level`（`minimal`/`low`/`medium`/`high`，不可完全关闭；各模型默认不同，如 3.6-flash=medium、pro=high、flash-lite=minimal）；
  - **`gemini-3.7-flash`（2026-08 新增）**：官方默认档位 **`medium`**（取代旧 3.x 的 `high`），档位表只提供 **`low`/`medium`/`high`——没有 `minimal`**（出处：`intro_gemini_3_7_flash.ipynb`）。因此在 3.7 上选「关闭原生思考」会压到 `low`（就近合法档），而不是发一个不支持的枚举去冒 400 风险。同理，**3.7 及更新/未知型号一律不提供 `minimal`**，等真机验证后再逐个放开；`thinking_level` 与旧的 `thinking_budget` **不可同时出现**（官方明确会 400），本代理只会二选一。Gemini 2.5 用 `thinking_budget`（`-1` 动态；2.5-flash 可设 `0` 关闭，2.5-pro 最低 128）。
  - **原生思考控制（`native_thinking_mode`）**——控制台"思考强度"卡片的下拉，支持"保存为该模型专属"：
    - **跟随请求（默认）**：用前端发来的 `reasoning_effort`。⚠️ SillyTavern 等前端常在**每次请求都发 `reasoning_effort`（如 `xhigh`）**，会覆盖你在控制台设的档位。
    - **关闭原生思考（角色扮演推荐）**：忽略前端 effort，把档位压到该模型最低（3.x=`minimal`、2.5-flash 预算 `0`、2.5-pro `128`），并**不返回思考**。
    - **强制用上方档位**：忽略前端 effort，用你在卡片里选的档位（返回思考）。
  - 🎭 **酒馆预设“卡原生思维链”一键配置**：许多预设把思维链写在 **system 提示**里（不是预填充），并恒发 `reasoning_effort=xhigh`。把"原生思考控制"选 **“关闭原生思考”** 即可（可用"保存为该模型专属"只对 3.6-flash 生效），让预设自己的思维链接管。
  - ⚠️ **重要（Studio/Cookie 通道实测）**：`batchGraphql` 私有接口**会忽略 `includeThoughts=false`**——即使设了也照样回传思考。因此 Cookie 通道在"关闭原生思考"时会**在响应侧主动剥离思考块**，并**把档位压到 `minimal`**（这才是真正减少原生思考、避免重预设在思考阶段被截断/无正文的关键）。标准（Express）通道由 SDK 原生支持不返回思考。
  - 🩺 **3.6-flash 在 Studio 只返回思考、无正文（`FINISH_REASON_UNSPECIFIED`）怎么办**：真机定案——这是**原生思考在 HIGH 档跑飞/被截断**（SillyTavern 恒发 `reasoning_effort=xhigh` 覆盖了控制台档位），**不是**安全策略/`HARM_CATEGORY_JAILBREAK` 的问题（已用含/不含 jailbreak 的多组对照验证）。**解决：把"原生思考控制"设为"关闭原生思考"**（对 3.6-flash 用"保存为该模型专属"），即忽略前端 `xhigh`、压到 `minimal` 并剥离原生思考——真机验证可稳定输出正文与预设自带的思维链。仅设"思考档位=minimal"无效，因为会被前端 `xhigh` 覆盖。
- **生图**：剥离全部采样参数；`response_modalities=["TEXT","IMAGE"]`；**两个生图模型比例白名单不同**（pro-image 10 种；flash-image 15 种，含 `1:4/4:1/1:8/8:1/9:21`），控制台按所选模型过滤，后端也会校验，选到不支持的比例会**自动回退为"由模型决定"（不报错）**。
- **预填充（重要，专为 SillyTavern 等酒馆预设优化）**：Gemini 3.x 拒绝以 `model` 轮次结尾的请求（返回 `Requests ending with a model turn are not supported.`）。代理内置"预填充智能兼容"（`smart`/`minimal`/`off`，默认 `smart`，控制台可切）：
  - **按模型能力自动选策略**：2.5 及更早模型允许以 model 轮次结尾 → **原生透传**，模型直接续写你的预填充，最忠实；3.x 拒绝 → 自动把末尾 assistant 预填充转成末尾 user 的"续写指令"（模板可在控制台自定义）。两种情况都会把预填充文本**拼回输出开头**，并对模型复述的重叠部分**自动去重**。
  - **预填充时压制原生思考（"卡思维链"，默认开启）**：酒馆预设通常自带思维链，靠预填充卡掉模型原生思考、让预设的思维链接管。开启后，检测到预填充即把思考压到该模型最低并**不回传思考**：3.x 压到 `minimal`（`pro` 无 minimal 则 `low`，官方规定 3.x 无法完全关闭思考）；2.5-flash 预算设 `0` **完全关闭**、2.5-pro 降到最低 `128`。**单次请求显式传 `reasoning_effort` / `thinking_budget` 时不压制**（请求优先）。可在控制台关闭此开关恢复模型原生思考。
  - **与模型名无关，新模型自动生效**。
- **可选输入处理**：控制台可配置输入搬运与顶部输入注入；均支持不开、仅假流、全请求三态，并可按需使用持久化的内容方案。默认关闭，不配置标签、占位语或方案时不改变请求。
- **新增/未来模型**：按家族/版本模式自动归类；未知/未来型号按"最新代"前向安全处理（自动移除已废弃采样参数、走预填充兼容）。

---

## 标准模式的 location（能指定，且能修好"偶发 404 / 随机区域"）

**实测结论（2026-08-14，真机验证）**：留空时 location 由 Google 后端自选，**可能落到该模型不提供服务的区域直接 404**；显式钉定后同一个 Key、同一个模型即正常。

| 请求形态 | `gemini-2.5-pro` | 说明 |
|---|---|---|
| 裸模型名（旧行为）→ express 端点 `https://aiplatform.googleapis.com/v1/publishers/google/models/{model}:generateContent` | **404** `projects/…/locations/asia-southeast1/…was not found` | 区域由后端路由，落到没有该模型的区域 |
| `projects/{project}/locations/global/publishers/google/models/{model}` | **200 正常出文** | 区域由我们钉定 ✅ |

**默认已经是 `global`**（多数 Gemini 模型只在 global 提供）。控制台「模型参数 → ② 全局设置 → 标准模式 location」下拉里除 `global` 外还列出了 Agent Platform 提供 Gemini 的各区域（美国 / 欧洲 / 亚太 / 其它，共 30 项）可按需切换；选「默认（后端自选）」＝回到旧行为。

**项目 ID 不需要单独配**：自动用「通道与凭证」页填的那个（或环境变量 `GOOGLE_PROJECT_ID`）——一个人通常只有一个 Express 项目。两者都没有时自动退回裸模型名（不会拼出半截路径）。

**钉定失败会自动兜底**：若钉定路径返回"模型不存在 / 需要计费"，代理会**自动退回裸模型名重试一次**（＝旧行为）并在日志说明怎么修。实机验证：故意把 Project ID 换成一个未开计费的项目，非流式与流式都仍能正常出文。所以把默认值设成 `global` 不会让任何既有配置变糟。

几个必须知道的边界（都是实测）：

- **项目必须是该 API Key 有权、且已开启计费的项目**。换成别的项目会 `403 requires billing to be enabled`。
- **区域端点主机不通**：`https://{location}-aiplatform.googleapis.com/...` 配 Express Key 会 404（该模型在那个区域没有）。因此钉定走的是**全局主机 + 路径里带 location**，不是换主机。
- **有的模型只在 `global` 提供**（如 `gemini-2.5-pro`），所以推荐 `global`。
- SDK 层面**不能**给 `Client` 传 `location`：`api_key` 与 `project/location` 互斥，硬传会 `ValueError`。钉定是靠**模型资源路径**实现的（`google-genai` 的 `t_model()` 对 `projects/` 开头的 model 原样透传）。
- 启动后第一次调用会打印一行端点解析结果，便于核对；钉定生效时另有一行 `🌐 [上游端点] 已钉定 location: projects/...`。

> 📌 **勘误**：本文件上一版写的"Express 模式无法指定 location、报错里的区域无从干预"**是错的**——那个结论只验证了"不能给 Client 传 location"，漏了"可以用完整资源路径钉定"这条路。已按真机结果更正。

---

## Studio(Cookie) 通道：免登录 与 非 Express 项目

两种设想都做了真机验证（2026-08-14）：

| 场景 | 结果 |
|---|---|
| **不带 Cookie（免登录）** + Express 项目 | ❌ `Permission 'aiplatform.endpoints.predict' denied` |
| **不带 Cookie（免登录）** + 非 Express 项目 | ❌ `requires billing to be enabled` |
| 不带 Cookie + 空项目 | ❌ `Request contains an invalid argument` |
| 带 Cookie + Express 项目 | ✅ 正常出文 |
| 带 Cookie + 非 Express 项目（未开计费） | ❌ `requires billing to be enabled` |

结论与代码现状：

- **本通道用的 `batchGraphql` 控制台接口不接受匿名调用**，必须有登录态（SAPISIDHASH）。浏览器里"不登录也能用 Studio"走的是**另一条**面向未登录用户的接口/额度，本项目当前不支持。若你能抓到那条请求（F12 → Network → 复制 URL、请求头与 payload），加一条上游并不难。
- **代理里并没有"检测是不是 Express 项目就拒绝"的逻辑**——能不能用只取决于：① 有登录态 Cookie；② Project ID 填的项目**该账号有权限且已开启计费**。是否 Express 项目本身不是门槛。
- 但旧版有个**误导人的 bug**：`Permission ... denied on resource //…/projects/xxx` 这类**项目级**错误会命中"Cookie 过期"关键词，于是提示你反复重取 Cookie，怎么弄都好不了。现在已按错误内容分流：点名了具体项目/计费的错误会给**项目层面**的排查指引（检查 Project ID / 开启计费 / 账号权限），只有纯会话失效才提示重取 Cookie。

---

## Cookie(Studio) 通道的原生函数调用

Cookie 直连通道会把 OpenAI `tools` 转成 batchGraphql 的原生 `functionDeclarations`。`tool_choice=none` 会直接省略全部上游工具；`auto` / `required` / 指定自定义函数分别使用 Studio 的 `AUTO` / `ANY` 模式。嵌套对象与数组参数会递归转换成 Studio 私有 UI Schema。内建 `googleSearch` 没有可放进 `allowedFunctionNames` 的声明名，因此搜索请求不附加自定义函数的 `toolConfig`。

| 情况 | 行为 |
|---|---|
| 自定义函数声明 | 原生下发；流式与非流式响应都返回 OpenAI `tool_calls` |
| 声明里含搜索类工具（`google_search` / `web_search` 等） | 映射为 Studio 内建 `googleSearch`。实机确认 batchGraphql 不允许它与自定义函数混用：混合 AUTO/required 请求优先自定义函数并忽略搜索；显式强制搜索时只发送 `googleSearch` |
| 历史里的 assistant `tool_calls` | 回放为 role=`model` 的 `functionCall` Parts，并从 `extra_content` 或短期缓存恢复 thought signature |
| 连续的 role=`tool` 结果 | 按 `tool_call_id` 关联函数名，合并为 role=`user` 的 `functionResponse` Parts；并行顺序为 FC1, FC2, FR1, FR2 |
| 模型或协议明确拒绝原生函数 Schema | 仅本次固定降级为文本工具观测并重试一次；没有用户可见策略开关，也不会恢复旧版严格拒绝路径 |

---

## 思维链守卫（`prefill_cot_guard`，默认开）

针对「**预填充确实卡掉了原生思维链，但预设自己的思维链也经常不写、直接出正文**」：

- **成因**：预填充只是把话头停在你预设自己的**未闭合思维链开标签**上，请求里**没有任何一句话**告诉模型"接下来必须先完成思考再写正文"。模型于是常常跨过思考直接写正文，输出里只剩一个孤立开标签，前端正则自然抓不到思维链。
- **做法**：自动识别预填充里未闭合的标签名（**不预设任何具体标签**，只按"开了没闭合"这一形状匹配，`<thinking>`、`<CoT>`、`<plan_1>`、`<分析~>` 之类自定义标签都能识别），在续写指令**末尾**（即模型最后读到的位置）追加一条硬性要求：先逐条写完该标签内的思考 → 用对应闭合标签收尾 → 然后才写正文，且不允许空标签。`smart` 与 `keep_turn` 两种模式都生效。
- **无副作用**：预填充里没有未闭合标签时（普通句子），本项什么都不做；生图模型不适用；可按模型单独开关。
- **仍不稳定时的对照实验**：把「原生思考控制」从 **关闭原生思考** 改成 **强制用上方档位 + `low`** 再测一轮。把档位压到 `minimal` 会把模型推向"直接给答案"的行为，**有可能连带跳过预设要求的长思维链**；`low` 保留一点推理惯性、同时仍显著少于默认档。这一条是机理推断，请以你自己的 A/B 结果为准。

---

## Cookie 直连模式配置指引（支持手机与电脑）

> ⚠️ **使用前请先读这段风险提示**
>
> Cookie 直连模式的原理是：用硬编码的网页客户端 key 和固定的 `querySignature`，
> 冒充 Google Cloud 控制台前端去调用其**私有** `batchGraphql` 接口。因此：
>
> - **无兼容性承诺**：这是内部接口，Google 改一次 `querySignature` 或参数结构就会全线失效，且不会有任何公告。
> - **条款风险**：以自动化方式访问非公开接口，可能与 Google Cloud 的使用条款相冲突，存在账号被限制或处置的风险。请仅用自有账号、自担风险。
> - **凭证敏感度极高**：配置的 Cookie 含 `__Secure-1PSID` 等完整会话凭证，等价于该 Google 账号的完整访问权。请勿把本服务部署到公开可访问的地方，务必设置强 `API_KEY`。
>
> 如果你需要的是稳定、可长期依赖的方案，请使用标准模式（Express API Key）。

在控制台切换到 **Agent Platform Studio (Cookie 直连反代)**，需配置 **Cookie** 与 **Project ID**：

### 1. 获取完整 Google Cookie
关键会话凭证（如 `__Secure-1PSIDTS`、`__Secure-1PSID`）带 `HttpOnly`，无法用书签脚本提取，需按下述方式获取：
- **电脑端**：登录 [Google Cloud Console](https://console.cloud.google.com) → 按 **F12** → **Network** → 刷新页面 → 点任意成功请求 → 复制 **Request Headers** 里的 `Cookie:` 整段，粘贴到控制台。
- **手机端**：iOS(Safari) 或 Android(Kiwi) 安装 `Cookie-Editor` → 登录控制台 → 插件 **Export** 为 **Header String** 或 **JSON** → 整段粘贴，系统自动解析。

### 2. 获取 Project ID
- 从控制台顶部项目选择器复制，或直接把含 `?project=xxx` 的整条 URL 粘贴到输入框，系统自动提取。

> ⚠️ **关于 Cookie 有效期**：通常较为持久——只要不退出登录、不修改密码、Google 未主动失效会话，一般可维持**数周甚至更久**（实测可用一个月以上），**并非只有 1~2 小时**。仅当接口确实报 `Permission Denied` / `predict denied` 时，重新获取并到控制台保存激活即可。

---

## 调用示例

### 查询模型

```bash
curl http://localhost:8050/v1/models \
  -H "Authorization: Bearer your_adapter_api_key"
```

### 非流式对话

```bash
curl http://localhost:8050/v1/chat/completions \
  -H "Authorization: Bearer your_adapter_api_key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemini-2.5-flash",
    "messages": [
      {"role": "user", "content": "用一句话介绍 Gemini。"}
    ],
    "stream": false
  }'
```

### 流式对话

```bash
curl http://localhost:8050/v1/chat/completions \
  -H "Authorization: Bearer your_adapter_api_key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemini-2.5-flash",
    "messages": [
      {"role": "user", "content": "写一首短诗。"}
    ],
    "stream": true
  }'
```

### Google Search 搜索增强（文本模型）

在模型名后加 `-search` 后缀：

```json
{
  "model": "gemini-2.5-flash-search",
  "messages": [
    {"role": "user", "content": "今天有哪些 Gemini API 相关更新？"}
  ]
}
```

### 单次请求覆盖参数（任意前端可用）

通过请求体额外字段按需覆盖控制台默认：

```json
{
  "model": "gemini-3-pro-image",
  "messages": [{"role": "user", "content": "生成一只赛博朋克猫"}],
  "image_size": "2K",
  "aspect_ratio": "16:9"
}
```

（文本模型可用 `reasoning_effort`：`low`/`medium`/`high`，或 2.5 用 `thinking_budget` 整数。）

> **注**：`temperature`/`top_p`/`max_tokens` 等标准字段、以及上述扩展字段，其优先级恒高于"该模型专属"与"控制台全局"设置。若要为某模型设持久默认值又不想每次请求都带，请用控制台的"保存为该模型专属"。

---

## 模型列表配置

> ⚠️ **行为变更（2026-08 起）**：模型列表**不再自动从远程拉取**（曾于启动时与 `/v1/models` 每 3600s 自动刷新）。加载顺序为 **内存缓存 → 磁盘缓存 `STATE_DIR/models.json` → 本地 `vertexModels.json` → 空**。
>
> - **升级后首次使用**：在控制台「选择模型」区点 **🌐 获取远程模型**（拉取远程配置并持久化到磁盘），或点 **📝 编辑模型** 手动添加自定义模型。
> - **自定义模型**持久化在 `web_state.json`，重建容器不丢，会合并进 `/v1/models` 与模型下拉（`fake-` / `-search` 变体自动生成）。
> - 远程地址仍为 `https://raw.githubusercontent.com/bad-woman/vertex2openai/main/vertexModels.json`（可设环境变量 `MODELS_CONFIG_URL` 覆盖）；**只有手动点击「获取远程模型」时才会访问远程**。

默认模型列表（本地 `vertexModels.json` 或远程配置）：

```json
{
  "models": [
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-pro-preview",
    "gemini-3.1-flash-lite",
    "gemini-3.1-flash-image",
    "gemini-3-pro-image",
    "gemini-2.5-pro",
    "gemini-2.5-flash"
  ]
}
```

> ⏰ **停用时间线（核对于 2026-07-26，请自行复核官方 deprecations 页）**
> - `gemini-2.5-pro` / `gemini-2.5-flash` / `gemini-2.5-flash-lite`：**2026-10-16 停用**。届时 `thinking_budget` 分支与"2.5 原生预填充透传"路径将没有活模型，可整体删除。
> - `gemini-3-flash-preview`：官方推荐替代为 `gemini-3.5-flash`，但**该模型目前仍可正常调用**（2026-07-26 用 Express Key 实机验证通过），因此**保留在默认清单中**；待官方正式停用后再移除。
> - `gemini-3.1-flash-image` / `gemini-3-pro-image`：GA 版本（不带 `-preview`），对应的 `-preview` 版本已于 2026-06-25 停用。

`/v1/models` 会自动为**非生图**的 Gemini 模型生成带 `-search` 后缀的别名。新增模型在控制台「📝 编辑模型」添加即可（能力自动归类，无需改代码）。

---

## 用量与花费统计

- **Token 用量**：标准 Express / 服务账号通道请求后由上游回传的 `usageMetadata` 累计（Cookie 私有接口通常不回传，不计入）。持久化在 `STATE_DIR/stats.json`，重建容器不丢；首页展示累计数字、**按实际输入/输出比例动态更新的 Token 条**，以及最近 7/30 天每日趋势。趋势图无论实际只积累了几天数据，都会补齐完整日期窗口，避免单日数据拉成整块柱；每一天以 **Prompt（蓝）/ Completion（橙）** 堆叠柱呈现，鼠标悬停或键盘聚焦可查看 token 与请求明细。
- **缓存命中率**：`usageMetadata.cachedContentTokenCount` 报告命中上下文缓存的输入 token。**服务账号（标准 Vertex）通道默认开启隐式缓存（90% 折扣）**，命中率 = 缓存命中 / 输入 token——命中率越高越省钱。
- **估算美刀花费**：按官方 [Agent Platform 按量价](https://cloud.google.com/gemini-enterprise-agent-platform/generative-ai/pricing)（`app/model_pricing.py`）估算：未命中输入全价 + **命中输入 10%**（90% 折扣）+ 输出全价。**按 PayGo 档位计费**：Standard ×1（基准） / **Priority ×1.8** / **Flex ×0.5**（半价），由控制台 `paygo_tier` 设置决定（`auto` = Priority；2.x 模型不支持 flex 会自动降级 standard）。未知/未来模型不计费（避免误报）；生图模型图片输出按张计费、token 统计无法覆盖，仅文本部分计入。价格为 2026-08 核实值，官方调价后请更新 `MODEL_PRICING`（3.7/3.6-flash 的 introductory 价 2027-01 到期）。
- **服务健康度**：圆环只表达互斥的**成功 / 错误**结果，中心显示已完成请求的成功率；“拥堵重试”是一次额外尝试、不属于请求最终结果，单列为独立指标，避免三者相加大于总请求时产生错误的部分-整体图示。

---

## 三通道策略与混合自动（Express API Key / Cookie 直连 / 服务账号）

控制台「通道与凭证」页以**四个标签页**切换与管理，`web_state.json` 中旧版 `use_web_proxy` 布尔开关会自动迁移：

| 标签页 / 策略 | 行为 |
|---|---|
| `express`（默认） | 只走 Express API Key 标准通道 |
| `cookie` | 只走 Cookie 直连反代（走网页端配额，规避 429） |
| `vertex` | 只走服务账号（标准 Vertex AI 认证） |
| `hybrid`（推荐） | 按「混合自动」页配置的顺序尝试，限流/5xx/未出流失败自动切换下一通道 |

**混合自动可配置（「混合自动」标签页）**：
- **参与通道开关**：3 个通道各自勾选是否参与（至少要勾一个）。
- **调度方式**：`priority` 按配置顺序尝试（默认，保持旧行为）；`random` 每次新请求都对已勾选通道做等概率随机排列，再按该排列故障转移。要启用三渠道随机，勾选 Express、Cookie、服务账号三个通道。随机是概率意义上的负载均衡，不承诺严格轮流；熔断中的通道仍会被跳过。
- **参与通道顺序**：固定优先级模式下可拖动/调整顺序；随机模式下该顺序仅作为参与列表，不决定首选通道。
- **优先级顺序**：↑↓ 调整尝试顺序（越靠上越先尝试）。
- **每通道独立重试次数**：各通道可单独设"失败后的内部重试上限"（留空 = 用全局 `retry_max`）——例如 Express 只重试 2 次就切走、Cookie 可重试 8 次扛限流。
- **PayGo 流量等级**：`auto / off / standard / flex / priority` + `paygo_only` 开关（作用于 Express 与服务账号两通道，见下方「PayGo 流量等级」）。

**故障转移规则**：
- 仅对 429 / 500 / 502 / 503 / 504（限流、上游繁忙）类错误切换通道；400 / 401 / 403（配置、鉴权、权限问题）如实报错不切换——切换也不会变好。
- Cookie 会话失效（Permission Denied）在混合模式下也会自动切其它通道，日志会提示你刷新 Cookie。
- **流式响应**：只有「尚未向客户端发出任何内容」时才会切换（SSE 心跳不计入）。一旦正文开始输出，错误只能如实收尾——SSE 流中途无法切换上游。
- **熔断保护**：任一通道连续失败 `failover_threshold` 次（默认 3）后冷却 `failover_cooldown_seconds` 秒（默认 60），冷却期间请求自动走另一条通道，避免限流风暴反复撞墙；成功后立即恢复。熔断按通道独立计数。
- 通道未配置凭证（无 Express Key / 无 Cookie / 无服务账号）会被路由层自动剔除，不会进入失败重试循环。

### 多账号凭证管理（Express Key 列表 / 多 Cookie 账号 / 多服务账号）

控制台「通道与凭证」页可在线管理（持久化在挂载卷 `web_state.json`，重建容器不丢）：

- **Express Key 列表**：多行文本框整表覆盖（每行一个 Key），可一键清除回落环境变量；只回显掩码（前后 4 位 + 长度），保存后 `refresh_keys()` 热生效。
- **Cookie 账号**：每行 = 一份 Cookie + 一个 Project ID，支持添加/更新/删除；新增时校验 SAPISID 族字段；Cookie 输入框留空 = 保持该账号原值。多账号时按「多 Key 轮询」开关轮询或随机选号；单账号行为与原来完全一致。
- **服务账号**：每行 = 一份 SA JSON（+ 可选 Project ID 覆盖 + 区域，区域默认 `global`），支持添加/更新/删除；新增时校验 JSON 结构与必需字段。只回显 `client_email` + project + location 掩码，SA JSON 永不回填（输入框留空 = 保持原值）。
- **同一请求恒用同一账号**：重试、流式、故障转移重发都不会串号（凭证取自请求级快照）。
- **凭证永不回显明文**：前端只显示掩码，完整 Cookie / Key / SA JSON 不会进入浏览器缓存或截图。

---

## 服务账号（Vertex SA）通道配置指引

**认证原理**：`google-genai` SDK + `google-auth` 内部完成标准 OAuth2 JWT-bearer 授权——用服务账号私钥签 RS256 断言 JWT → `oauth2.googleapis.com/token` 换 `access_token`（约 1 小时）→ 请求带 `Authorization: Bearer`，过期自动刷新。**无需手写认证代码。**

**准备步骤**：
1. 在 Google Cloud 控制台创建服务账号（或直接用现有账号），授予该项目 `roles/aiplatform.user` 角色，并确保项目**已开启计费**（否则报 `403 requires billing to be enabled`）。
2. 为该服务账号创建 **Key（JSON）** 并下载（安全提示：服务账号 Key 是长期静态凭证，泄漏等于该账号全部权限，请妥善保管、只用于本项目、必要时轮换）。
3. 在控制台「通道与凭证 → 服务账号」标签页粘贴整段 SA JSON（`client_email` / `private_key` / `project_id` 三个字段必须有），选区域（默认 `global`），保存。或设置环境变量 `VERTEX_SA_FILE` 指向 key 文件路径（docker 挂载场景）。

**旧配置兼容**：带 `[PAY]` 前缀的模型名（旧版服务账号模式的入口）在 `vertex` / `hybrid` 策略下会被自动剥掉前缀走服务账号通道。**注意（hybrid 顺序相关）**：`[PAY]` 前缀是在服务账号（vertex）通道内被剥除的；混合自动下若把 Express 排在 vertex 之前，请求会先打到 Express——Express 不认 `[PAY]`，会按"该前缀仅在服务账号通道剥除"返回 400 且不切换（400 属不可切换错误）。需要用 `[PAY]` 时请把策略设为 `vertex`，或在「混合自动」里把服务账号通道排到 Express 之前。

**常见错误**：
- `403 Permission 'aiplatform.endpoints.predict' denied` → 服务账号缺角色或项目没开计费（见上）。
- `401 Unauthorized` / token 类错误 → SA Key 无效或已删除，重新生成。

---

## PayGo 流量等级（ST-Vertex-PayGo 方案融合）

融合自 `ST-Vertex-PayGo(-Server)` 的官方「按量共享容量」层级头机制，作用于 **Express 与服务账号**两通道（Cookie 通道走 batchGraphql 不适用）：

| 等级 | 请求头 | 说明 |
|---|---|---|
| `auto`（默认） | global 请求自动打 Priority 头 | 保持改造前行为 |
| `off` | 无 | 不打任何层级头 |
| `standard` | 仅 `paygo_only=true` 时打 `X-Vertex-AI-LLM-Request-Type: shared` | 绕过预配吞吐、纯按量 |
| `flex` | shared + `X-Vertex-AI-LLM-Shared-Request-Type: flex` + `X-Server-Timeout: 1800` | 允许排队至 30 分钟（同步放大 httpx 超时） |
| `priority` | shared + `X-Vertex-AI-LLM-Shared-Request-Type: priority` | 优先调度 |

**约束**：Flex/Priority 仅对 `location=global` 的请求有效，非 global 自动降级为普通请求并在日志告警。

## 关于 429 报错与并发控制

429（Resource Exhausted）常因上游限额不足或请求频率过高。项目已内置退避重试，另建议：
- **优先启用「混合自动」通道策略**：Express 被限流时自动切 Cookie 直连兜底，无需人工干预。
- 控制客户端并发频率。
- 适当减小最大输出 Token。
- 配置多个有效 Express Key 并开启轮询。
- 及时更新失效或权限受限的 Google Cookie。

---

## 自建镜像持续化部署（fork 后一键升级）

不依赖原作者的镜像，自己改代码、自己构建、1Panel 一键升级：

1. **Fork 本仓库**到你的 GitHub 账号，clone 后改代码 push（可加 `git remote add upstream https://github.com/bad-woman/vertex2openai` 定期合并原作者的更新）。
2. **无需自建 CI**：fork 仓库已自带上游的 GitHub Actions（`.github/workflows/docker-image.yml`，GHCR CI），push 到 main 即自动构建并推送 `ghcr.io/你的用户名/vertex2openai:latest`。fork 仓库保持 public 则包公开，VPS 拉取免登录。
3. **VPS 上改一次**：把 `docker-compose.yml` 的 `image` 改为 `ghcr.io/你的用户名/vertex2openai:latest`，用 1Panel 重建容器。
4. **以后日常**：改代码 → push → Actions 自动出镜像 → 1Panel 点「重建/升级」拉取新 latest → 完成。

**数据不丢**：Cookie、Project ID 与全部控制台设置持久化在挂载卷 `./data:/app/data`（`web_state.json`），升级镜像/重建容器都保留，无需 sqlite。

### 定期合并原作者更新（upstream 同步）

Fork 后的仓库默认配置：

```bash
git remote add upstream https://github.com/bad-woman/vertex2openai.git   # 一次即可
```

原作者更新时，拉取并合并：

```bash
git fetch upstream
git merge upstream/main      # 或 git rebase upstream/main
git push origin main         # 推送后 GHCR CI 自动重新构建镜像
```

⚠️ **冲突提示**：上游更新可能与你改过的文件冲突（本项目常见于 `app/routes/chat_api.py`、`app/api_helpers.py`、`app/main.py` 等）。遇到冲突时手动解决后重新提交即可；合并后跑一遍「本地开发与检查」的语法检查再推。

---

## 后续升级与扩展

- **新增模型**：控制台「📝 编辑模型」弹窗直接添加（持久化、即时生效），或改本地 `vertexModels.json` / 远程 `MODELS_CONFIG_URL` 后在控制台「🌐 获取远程模型」刷新。`model_capabilities.py` 按**家族/版本模式**自动归类（思考方式、采样裁剪、生图比例/分辨率、预填充），**未知/未来型号按"最新代"前向安全处理**。基本即插即用。
- **迁移到 Interactions API**：代码按上游通道解耦（`app/upstreams/` 下各类实现 `BaseUpstream`；能力判定、消息转换、参数构建均可复用），新增一个 `InteractionsUpstream` 并在路由层接入即可。

  现状（核对于 2026-07-26）：
  - **Gemini Developer API 侧**：Interactions API 已于 2026-06 **GA**，官方推荐新项目使用；`generateContent` 被标为 legacy 但继续完整支持。
  - **Agent Platform 侧**：Interactions API 仍标注为 **experimental**。
  - **Express 模式**：REST 面只有 `countTokens` / `generateContent` / `streamGenerateContent` —— **这才是本项目暂不迁移的直接原因**（本项目走的正是 Express 通道）。

---

## 本地开发与检查

```bash
# 语法检查
python -m compileall app

# 自动化测试（Python 3.11 venv，与 Docker 环境一致；系统 Python 3.14 环境是坏的别用）
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install -r app\requirements.txt pytest pytest-asyncio
.venv\Scripts\python.exe -m pytest tests -q

# 本地启动
cd app
uvicorn main:app --host 0.0.0.0 --port 7860
```
