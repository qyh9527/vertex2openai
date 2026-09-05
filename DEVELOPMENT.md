# 开发说明

## 可配置输入处理的边界

本项目有两个独立的全局请求改写能力。二者共用同一条上游调用管线，但**不得互相读取、推导或改写对方的配置和结果**。

| 能力 | 触发条件 | 改写结果 | 与另一能力的关系 |
| --- | --- | --- | --- |
| 输入搬运 | 已启用，且最新 user 纯文本完整匹配控制台配置的 XML 标签 | 把标签载荷追加到前一条普通 assistant 消息尾部，并以占位语替换该 user 消息 | 不读取顶部方案 |
| 顶部注入 | 已启用且有有效的控制台方案 | 将所选方案**原始正文**以 system / assistant / user 身份放在 `messages` 第 1 条；同角色首条消息按“方案正文 + 换行 + 原正文”融合 | 不读取或改写 user 输入，也不依赖输入搬运 |

顶部注入的实现入口是 `app/top_input_injection.py`，输入搬运的实现入口是 `app/input_relay.py`。

## 顶部注入契约

- 方案正文按原样注入；网关不渲染任何顶部注入宏或占位符。
- `{{input}}` 不是顶部注入变量，出现在方案中会作为普通文本发送给上游。
- 顶部注入不要求请求中存在 user 消息；最新 user 为多模态内容时同样照常注入。
- 主开关 `top_input_injection_mode` 仅 `off` / `always`：关闭或打开注入。
- 随机选择使用原持久化键 `top_input_injection_random`，现为三态字符串：`off` 全部固定、`always` 全部随机、`non_vertex_only` 仅实际 Express/Cookie 随机，实际 Vertex SA 仍注入固定方案。
- Express/SA 共享执行路径传入 `self.channel_name`，Cookie 传入 `cookie`。不读取控制台的 express/cookie/vertex/hybrid 四选一策略；hybrid 转到 SA 时须重新按固定方案处理原请求。
- 旧随机布尔值 true/false 兼容为 always/off；旧主开关 non_vertex_only 或 fake_stream_only 兼容为 always + 随机 non_vertex_only。前端回显与后端读取同步转换，保存后写入新值。
- 随机选择可能降低隐式缓存命中率，SA 固定方案可保持前缀稳定。

上游管线当前先执行顶部注入，再独立执行输入搬运，随后处理控制台 system / prefill 注入与预填充兼容。这个执行顺序仅是消息变换的顺序，**不构成两项功能的逻辑依赖**。

## 修改检查清单

修改任一能力时：

1. 保持顶部注入不读取最新 user 输入，且不重新引入 `{{input}}` 渲染或自动追加输入。
2. 保持输入搬运的严格护栏：标签、占位语、纯文本输入和前一条普通 assistant 消息缺一不可时必须空操作。
3. 同步更新控制台帮助与 `README.md`，避免 UI 与请求语义不一致。
4. 为正常路径和护栏路径补充或更新 `tests/test_top_input_injection.py`、`tests/test_input_relay.py`；涉及 UI 文案时更新 `tests/test_dashboard_visualization.py`。
5. 运行：

   ```powershell
   .\.venv\Scripts\python.exe -m compileall -q app tests
   .\.venv\Scripts\python.exe -m pytest tests -q
   ```
