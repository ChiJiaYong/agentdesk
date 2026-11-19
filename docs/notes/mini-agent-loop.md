# 理解笔记：手写循环 vs LangGraph（W1 面试防线）

> 目标：被问"Function Calling 底层发生了什么 / 你理解 Agent Loop 吗"时，
> 能从裸 API 一路讲到框架，而不是只背 LangGraph 文档。

## 1. 一轮工具调用的完整生命周期（裸 API 视角）

```
messages[] ──HTTP──▶ 推理服务
                        │ chat template 拼接（系统提示/工具 schema 注入）
                        │ 模型生成
              ◀──────────┘
assistant 消息（含 tool_calls: name + arguments JSON + id）
      │
      ▼ 我们的代码执行工具（模型从不自己执行任何东西）
tool 消息（role=tool, tool_call_id=..., content=结果字符串）
      │
      ▼ 整个 messages[] 再发回去
循环，直到 assistant 不再带 tool_calls（→ 最终回答）或触发步数预算
```

对应实现：`core/mini_loop.py::run_agent_loop`（约 120 行，零依赖）。

三个必背细节：
1. **assistant 的 tool_calls 消息必须原样留在上下文里**，后面的 tool 消息靠
   `tool_call_id` 与之配对——丢了这条，下一次请求直接 400 或模型混乱
2. 工具结果是**字符串**进上下文，不是结构化对象
3. 循环终止条件有两个：无 tool_calls（正常）、步数预算（护栏，书 §1.2.5）

## 2. 逐点对照表（面试时直接引用）

| 关注点 | core/mini_loop.py（手写） | LangGraph |
|---|---|---|
| 状态 | 一个 `messages` 列表 | TypedDict `AgentState`，显式类型化 |
| 控制流 | for + if，一行行可见 | StateGraph 节点+边，图可导出 mermaid |
| 循环上限 | `max_steps` 参数 | `recursion_limit`（config） |
| 中断/恢复 | 无（要自己实现持久化） | checkpointer：状态落库，线程可恢复 → HITL 免费获得 |
| 工具执行 | 自己写 dict 分发 | `ToolNode` 统一处理 |
| 可测试性 | 注入 `ChatClient` Protocol（FakeChatClient） | 注入 BaseChatModel（GenericFakeChatModel） |

## 3. 本项目两条循环的分工

- `core/mini_loop.py`：**教学参考实现**——理解用、测试用、面试讲原理用
- LangGraph `create_react_agent` → W3 换成自绘 StateGraph：**产品实现**——
  需要 checkpoint（HITL 人审中断恢复）、节点级 Trace、可控路由

## 4. 自测题（能脱稿回答才算过关）

1. 为什么 tool 结果要以 `tool` role + `tool_call_id` 追加，而不是拼进 user 消息？
2. `max_steps` 爆掉时应该返回什么？（看 `run_agent_loop` 的 aborted 分支——为什么失败要"响亮"？）
3. LangGraph 的 checkpointer 解决了手写版的哪个缺陷？HITL 审批为什么依赖它？
4. 如果两个工具名字和描述很像，模型容易选错——在循环层（不是提示词层）能做什么？（预演 W3 工具治理：按前置条件收窄本轮暴露的工具集）
