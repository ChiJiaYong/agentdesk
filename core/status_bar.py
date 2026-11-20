"""Agent 状态栏（book §2.6）— code-maintained environment summary.

为什么需要（书 §2.6.1）：上下文学习本质是"检索而非推理"，窗口没有提炼层——
模型数不清自己调过几次工具、手里有哪些已确认的订单号。状态栏把这类跨轮状态
预计算成一小段文本，注入到上下文**末尾**（紧邻生成位，注意力最高）。

实现要点（书 §2.6.4）：
- **由代码维护，绝不让 LLM 统计**——模型会无条件相信状态栏（也因此它是投毒面）。
- 更新策略：本项目的状态每轮都变 → 用**替换式**（同 id 覆盖上一条），缓存仅失效
  上一条状态之后的短后缀。状态消息不持久化进图状态，只在模型调用时构造。
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any


def build_status_bar(
    *,
    turn: int,
    known_order_ids: set[str],
    call_fingerprints: dict[str, int],
    risk_flags: list[str],
    needs_human: bool,
    pending_user_info: str | None = None,
) -> str:
    """Render the `<agent_status>` block (injected as a trailing user message)."""
    repeated = {k: v for k, v in call_fingerprints.items() if v >= 2}
    lines = [
        "<agent_status>",
        f"time: {datetime.now(UTC).isoformat(timespec='seconds')} | turn: {turn}",
        f"known_order_ids: {sorted(known_order_ids) if known_order_ids else '(none — 需向用户询问)'}",
    ]
    if repeated:
        lines.append(
            "repeat_call_warning: "
            + json.dumps(repeated, ensure_ascii=False)
            + " （同一调用已重复；≥3 次将熔断转人工，请改写策略）"
        )
    if risk_flags:
        lines.append(f"risk_flags: {risk_flags}")
    if pending_user_info:
        lines.append(f"pending_from_user: {pending_user_info}")
    if needs_human:
        lines.append("escalated: 本轮含写操作/熔断，回复仅供人工审核队列参考")
    lines.append("</agent_status>")
    return "\n".join(lines)


def fingerprint(tool_name: str, args: dict[str, Any]) -> str:
    """Stable per-call fingerprint for repeat detection (§5.1.5 调用指纹)."""
    return f"{tool_name}:{json.dumps(args, sort_keys=True, ensure_ascii=False)}"
