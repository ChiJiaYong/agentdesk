"""工具调用格式漂移修复（响应侧 Harness，book §5.1.4/§2.3.2）。

实测故障（本地 llama-server --jinja + Qwen2.5 GGUF）：模型输出正确的工具调用意图，
但以转义 JSON 文本形式出现在 content 里（`{{"name": ...}}`），原生 tool_calls 为空——
图看不到调用，把 JSON 字符串当最终答案返回给客户。

书的处方：模型工具调用不稳定是常见故障，**Harness 负责规范化**（§5.1.4 Harness 实践、
§2.3.2 参数保真性的响应侧对应物——输入侧禁止静默修正，响应侧修复必须可见可审计）。
修复后的调用在轨迹里标注 `repaired: true`。
"""

from __future__ import annotations

import json
import re

from langchain_core.messages import AIMessage

_VALID_NAMES: set[str] = set()


def set_valid_tools(names: list[str]) -> None:
    global _VALID_NAMES
    _VALID_NAMES = set(names)


def _looks_like_tool_json(text: str) -> dict | None:
    """Parse `{{"name":...}}` / `{"name":...}` / fenced JSON with name+arguments."""
    candidates: list[str] = []
    stripped = text.strip()
    # double-brace escaped variant (observed with llama-server --jinja + Qwen2.5)
    if stripped.startswith("{{") and stripped.endswith("}}"):
        candidates.append(stripped[1:-1])
    candidates.append(stripped)
    fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    candidates.extend(fenced)
    for cand in candidates:
        try:
            obj = json.loads(cand)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "name" in obj and ("arguments" in obj or "parameters" in obj):
            args = obj.get("arguments") or obj.get("parameters") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            if isinstance(args, dict) and (not _VALID_NAMES or obj["name"] in _VALID_NAMES):
                return {"name": obj["name"], "arguments": args}
    return None


def repair_tool_calls(response: AIMessage) -> AIMessage:
    """If the model emitted a tool call as plain JSON text, promote it to a
    native tool_call. Returns the original message when nothing needs repair."""
    if getattr(response, "tool_calls", None):
        return response
    content = str(response.content or "").strip()
    if not content.startswith("{") and "```" not in content:
        return response
    repaired = _looks_like_tool_json(content)
    if repaired is None:
        return response
    return AIMessage(
        content="",
        tool_calls=[{"name": repaired["name"], "args": repaired["arguments"], "id": "repaired-1"}],
        additional_kwargs={**response.additional_kwargs, "repaired": True},
        response_metadata=response.response_metadata,
    )
