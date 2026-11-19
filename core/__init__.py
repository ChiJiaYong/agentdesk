"""AgentDesk core: reusable building blocks shared by both agents.

W1 scope: a minimal hand-rolled agent loop (`core.mini_loop`) kept as the
reference implementation for understanding what frameworks do under the hood
(《深入理解AI Agent》v2.0 §2.2.4). Later weeks replace the toy tools here with
the real tool framework (MCP, idempotency, three-state results) while the loop
semantics stay the same.
"""
