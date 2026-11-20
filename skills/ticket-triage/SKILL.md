---
name: ticket-triage
description: 工单分类规范——把客户诉求映射到标准 issue 类别与优先级，供 create_ticket 的 issue 字段与人工审核队列使用。
---

# 工单分类规范（create_ticket 必读）

## 类别映射
| 客户描述关键词 | issue 类别前缀 | 默认优先级 |
|---|---|---|
| broken / damaged / 破损 / 坏了 | [damaged] | high |
| not received / 没收到 / lost | [non-delivery] | high |
| wrong item / 发错货 / 与描述不符 | [wrong-item] | normal |
| refund / 退款 / money back | [refund-request] | normal |
| late / 太慢 / 物流 / where is my order | [logistics] | low |

## issue 字段写法
- 格式：`[类别] 客户原话摘要（≤15 词）`
- 示例：`[damaged] screen cracked on arrival, wants refund`
- 禁止：在 issue 中添加客户没有说过的推断（如"可能是运输问题"）。

## 幂等键规则
- 使用 `ticket:{order_id}:{类别}` 形式派生稳定键，同一订单同一类别永不重复开单。
