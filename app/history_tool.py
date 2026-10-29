"""可选的纯文本历史承载；保留首条 system、末轮与真实工具交互。"""
def history_tool_active(settings, is_fake):
    mode = settings.get("history_tool_mode", "off")
    return mode == "all" or mode == "fake_stream_only" and is_fake


def validate_history_tool(request):
    for message in request.messages:
        content = message.content
        if content is None or isinstance(content, str):
            continue
        if not isinstance(content, list):
            raise ValueError("历史工具承载仅支持文字；请移除图片或文件，或关闭历史承载。")
        for part in content:
            if not isinstance(part, dict) or part.get("type") != "text" or not isinstance(part.get("text"), str):
                raise ValueError("历史工具承载不支持图片、文件或其他非文字内容，请移除后重试。")


def _text(content):
    if isinstance(content, str):
        return content
    return "\n".join(part["text"] for part in (content or []))


def carry_history(request):
    """调用方先做输入搬运；本函数不修改原请求，不拆散真实工具记录。"""
    validate_history_tool(request)
    messages = request.messages
    last_user = next((i for i in range(len(messages) - 1, -1, -1)
                      if messages[i].role == "user"), None)
    if last_user is None:
        raise ValueError("历史工具承载需要至少一条 user 消息。")
    start = 1 if messages and messages[0].role == "system" else 0
    stop = last_user
    for i, message in enumerate(messages):
        if message.role in ("tool", "function") or message.tool_calls or getattr(message, "function_call", None):
            stop = min(stop, i)
            break
    if stop <= start:
        return request
    history = messages[start:stop]
    if not any(_text(message.content).strip() for message in history):
        return request
    description = "Conversation history in chronological order. Role labels identify the original speakers.\n\n" + "\n\n".join(
        f"[{message.role}]\n{_text(message.content)}" for message in history)
    names = {tool.get("function", {}).get("name") or tool.get("name")
             for tool in (request.tools or []) if isinstance(tool, dict)}
    name = "context"
    suffix = 1
    while name in names:
        name = f"context_{suffix}"
        suffix += 1
    tool = {"type": "function", "function": {"name": name, "description": description,
            "parameters": {"type": "object", "properties": {}}}}
    # 指令放在最后一条 user 内，不打乱其后的预填充。
    user = messages[last_user]
    instruction = f"Read the conversation history in the {name} tool description and continue from the latest user input. Do not call {name}; its description already contains the history."
    content = _text(user.content)
    updated_user = user.model_copy(update={"content": instruction + "\n\n" + content})
    kept = list(messages[:start]) + list(messages[stop:])
    kept[last_user - (stop - start)] = updated_user
    return request.model_copy(update={"messages": kept, "tools": list(request.tools or []) + [tool]})
