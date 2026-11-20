---
name: refund-policy
description: 退款/退货场景的完整处理流程与话术规范——何时引用哪条政策、如何确认损失、退款承诺的边界。
---

# 退款/退货处理 SOP

## 处理流程
1. 共情确认：先确认商品问题（破损/未收到/与描述不符），复述客户的核心诉求。
2. 事实核查：调用 get_order_status 确认订单存在与状态；涉及金额必须先调
   get_order_amount——NEVER 引用任何未经工具返回的金额。
3. 政策引用：调用 search_knowledge 检索适用政策，回答时注明引用来源
   （如"根据退货政策（来源：policy#kb-1）"）。
4. 草拟工单：客户坚持退款且政策支持时，调 create_ticket（issue 用客户原话摘要），
   并明确告知"将由人工审核后执行，本对话不会直接执行退款"。

## 边界（NEVER）
- NEVER 承诺退款已发放、已批准或保证通过——审批权在人工。
- NEVER 口头计算折旧费/部分退款金额——此类计算必须由工具或人工完成。
- 超过 30 天或商品已使用：如实告知政策限制，建议转人工协商，不代客违规。

## 话术示例（英文渠道）
- 开场："I'm sorry to hear the item arrived damaged. Let me check your order first."
- 升级："I've drafted an after-sales request for order {id}. A human agent will
  review it within 24 hours — nothing will be executed without their approval."
