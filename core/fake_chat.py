"""Offline scripted chat models for tests, demos and the CI eval smoke.

Two seams, one purpose: run the *real* agent code path with zero network.

- ``FakeChatClient`` (in ``core.mini_loop``): the raw protocol seam used by the
  hand-rolled loop.
- ``ScriptedToolChatModel`` (here): a LangChain ``BaseChatModel`` for LangGraph.
  The interesting bit is ``bind_tools`` — in a real provider this is where tool
  schemas get serialized into the request. A scripted model has its replies
  pre-authored, so binding is a no-op returning ``self``; overriding it is also
  what makes the model usable by ``create_agent``, which requires ``bind_tools``.
"""

from __future__ import annotations

from typing import Any

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel


class ScriptedToolChatModel(GenericFakeChatModel):
    """Plays back a scripted list of AIMessages; supports the bind_tools seam."""

    def bind_tools(self, tools: Any, **kwargs: Any) -> ScriptedToolChatModel:
        # Replies are pre-scripted (tool_calls included), so "binding" schemas is
        # a no-op. Real providers translate `tools` into their request format here.
        return self
