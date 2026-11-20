"""上下文分层压缩（book §2.7.4 五层分层压缩的 L1+L2 落地）。

原则（书 §2.7.4/2.7.5）：
- 压缩发生在两次模型调用之间；system 前缀永不触碰。
- 接近阈值才批量压缩，不逐轮压缩。
- 保留四类高价值信息：决策、约束、失败路径、引用；噪声直接丢（不摘要）。
- 全文不丢：完整内容已在 runs/ 轨迹里（trace = 持久层），压缩只是上下文内的
  预算控制，占位符注明可回查。

L1 工具结果预算：早于最近 2 条的 tool 消息截断为预览 + [COMPRESSED] 标记。
L2 总量阈值：字符总量超限时对更早的 assistant/tool 消息做占位压缩。
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, AnyMessage, ToolMessage

TOOL_PREVIEW_CHARS = 220
TOTAL_BUDGET_CHARS = 6000
KEEP_RECENT_TOOLS = 2


def _preview(msg: AnyMessage) -> str:
    head = (msg.content or "")[:TOOL_PREVIEW_CHARS]
    more = "…" if len(msg.content or "") > TOOL_PREVIEW_CHARS else ""
    return f"{head}{more}[COMPRESSED: 全文见 run 轨迹]"


def compress_messages(
    messages: list[AnyMessage],
    *,
    total_budget: int = TOTAL_BUDGET_CHARS,
) -> list[AnyMessage]:
    """Return a compressed COPY for model input; graph state stays intact."""
    total = sum(len(m.content or "") for m in messages)
    if total <= total_budget:
        return messages

    tool_indices = [i for i, m in enumerate(messages) if isinstance(m, ToolMessage)]
    protected = set(tool_indices[-KEEP_RECENT_TOOLS:]) if tool_indices else set()
    # never touch: first user message (任务锚点), last message (当前轮)
    protected.update({0, len(messages) - 1})

    out: list[AnyMessage] = []
    for i, m in enumerate(messages):
        if i in protected or not isinstance(m, (ToolMessage, AIMessage)):
            out.append(m)
            continue
        if isinstance(m, ToolMessage) and len(m.content or "") > TOOL_PREVIEW_CHARS:
            out.append(ToolMessage(content=_preview(m), tool_call_id=m.tool_call_id))
        elif (
            isinstance(m, AIMessage)
            and not getattr(m, "tool_calls", None)
            and i < len(messages) - 2
            and len(m.content or "") > TOOL_PREVIEW_CHARS
        ):
            out.append(AIMessage(content=_preview(m), id=m.id))
        else:
            out.append(m)

    # L2: still over budget → drop middle assistant/tool previews entirely
    total2 = sum(len(m.content or "") for m in out)
    if total2 > total_budget:
        mid = [
            i
            for i, m in enumerate(out)
            if 0 < i < len(out) - 1
            and m.type in {"tool", "ai"}
            and "[COMPRESSED" not in (m.content or "")
        ]
        for i in mid:
            if sum(len(x.content or "") for x in out) <= total_budget:
                break
            placeholder = (
                ToolMessage(
                    content="[COMPRESSED: 早前工具结果已省略，全文见 run 轨迹]",
                    tool_call_id=out[i].tool_call_id,
                )
                if out[i].type == "tool"
                else AIMessage(
                    content="[COMPRESSED: 早前回复已省略，全文见 run 轨迹]", id=out[i].id
                )
            )
            out[i] = placeholder
    return out
