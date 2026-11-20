r"""Model tier resolution — 三档模型层，给使用者友好的两种选择 + 零配置演示层。

使用者两条主动路径（UI 设置窗口）：
  ① 本地模型   — llama-server 加载本机 GGUF（如 D:\AI\models\llm\*.gguf），
                 OpenAI 兼容接口，零 API 成本、数据不出本机（书 §3.1.8 立场）。
  ② API Key   — 使用者在设置窗口自行粘贴 key（DeepSeek/GLM/任意 OpenAI 兼容端点）；
                 key 仅存服务端内存，不落盘、不入 git。

外加一个零配置演示层（规则驱动），保证无任何配置也能完整体验图与工具流。

选择规则（UI 显式选择 > 自动探测）：
    auto: 环境变量/UI 提供 key → api；本地 server 健康 → local；否则 demo。
（书 §1.2.3 模型选型 / §7.6 成本分析：层间切换不改动图、工具、护栏与轨迹——
 这是"Harness 与模型解耦"的直接体现。）
"""

from __future__ import annotations

import os
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LOCAL_BASE_URL = os.getenv("LLAMA_SERVER_URL", "http://localhost:8100/v1")
DEFAULT_LOCAL_MODEL = os.getenv("LLAMA_SERVER_MODEL", "qwen2.5-7b-local")

TIERS = ("demo", "local", "api")


def local_server_healthy(base_url: str | None = None, timeout: float = 2.0) -> bool:
    url = (base_url or DEFAULT_LOCAL_BASE_URL).removesuffix("/v1") + "/health"
    try:
        r = httpx.get(url, timeout=timeout)
        return r.status_code == 200 and r.json().get("status") == "ok"
    except (httpx.HTTPError, ValueError):
        return False


def api_key_available() -> bool:
    return bool(os.getenv("DEEPSEEK_API_KEY"))


def resolve_tier(choice: str | None = None) -> str:
    """Return one of TIERS. Explicit user choice is validated, not blindly trusted."""
    if choice == "api":
        if api_key_available():
            return "api"
        raise ValueError("api tier selected but no API key configured (set it in Settings)")
    if choice == "local":
        if local_server_healthy():
            return "local"
        raise ValueError(
            "local tier selected but llama-server is not reachable at "
            f"{DEFAULT_LOCAL_BASE_URL} (start it with the command in README)"
        )
    if choice in TIERS:
        return choice  # demo
    # auto
    if api_key_available():
        return "api"
    if local_server_healthy():
        return "local"
    return "demo"


def build_model(
    tier: str,
    api_key: str | None = None,
    base_url: str | None = None,
    model_name: str | None = None,
):
    """Build the LangChain chat model for a tier. Returns (model, tier_info)."""
    if tier == "demo":
        from service.demo_model import DemoSupportModel

        return DemoSupportModel(), {"tier": "demo", "detail": "规则演示模型（零成本零配置）"}

    from langchain_openai import ChatOpenAI

    if tier == "local":
        url = base_url or DEFAULT_LOCAL_BASE_URL
        return ChatOpenAI(
            model=model_name or DEFAULT_LOCAL_MODEL,
            api_key="local",  # llama-server ignores keys
            base_url=url,
            temperature=0.2,
        ), {"tier": "local", "detail": f"本地模型 @ {url}（数据不出本机）"}

    if tier == "api":
        key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if not key:
            raise ValueError("api tier requires a key")
        url = base_url or os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        return ChatOpenAI(
            model=model_name or os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
            api_key=key,
            base_url=url,
            temperature=0.2,
        ), {"tier": "api", "detail": f"API 模型 @ {url}"}

    raise ValueError(f"unknown tier: {tier}")
