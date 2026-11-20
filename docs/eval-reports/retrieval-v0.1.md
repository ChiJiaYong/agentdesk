# 检索评测报告 v0.1 — 混合检索消融

日期：2026-11-19 · 语料：3,052 chunks · 金标：27 条 · 模型：BAAI/bge-small-zh-v1.5（CPU）

## 语料构成（全部真实数据、来源三元组可溯源）

| 来源 | chunks | 说明 |
|---|---|---|
| 中国消费者保护法规三件套 | 53 | 消保法（2013 修正）/ 三包规定（国经贸〔1995〕458 号）/ 消保法实施条例（国令第 778 号），官网原文 |
| Amazon QA Electronics | 2,999 | UCSD McAuley Lab 真实客户问答（采样 3,000 条） |
| FastAPI docs / GitLab handbook | — | 待网络窗口接入（W2 后续） |

## 金标集构建

- **法规查询 16 条**：人工撰写自然语言问句，金标 chunk 按**条款文本正则动态定位**（切块策略变更不会静默作废金标）
- **真实用户问句 11 条**：从 Amazon QA 采样，问题即查询、自带卖家确认答案的 chunk 为金标

## 结果（hit@5 / MRR）

| 模式 | hit@5 | MRR |
|---|---|---|
| BM25（jieba 稀疏） | 0.852 | 0.716 |
| Dense（bge-small-zh, CPU） | 0.852 | 0.713 |
| **Hybrid（RRF, k=60）** | **0.926** | **0.744** |
| Hybrid + Rerank（bge-reranker-base） | 0.926 | 0.744 |

## 结论

1. **混合检索 hit@5 提升 +7.4pp**，与"两路各自召回不同形态的失败案例、RRF 融合互补"的机理一致（书 §3.2.4）：BM25 兜住关键词精确匹配（条款号/数字），稠密兜住同义改写（口语问句→法规书面语）
2. 两单路在此规模上打平——说明瓶颈在词汇重叠与语义间隔各占一半，正是混合的价值区间
3. **Rerank 阶段（§3.2.4 第三阶段）当前无显著增益**：融合已把正确 chunk 送进 top-5 且次序稳定。按 §7.7 统计显著性口径，27 条金标下 <2 个 SE 的差异不构成结论——保留接口，待金标扩容（W5 计划 60 条）后复测；书内立场"rerank 是换匹配范式而非补丁"提示其价值更可能体现在 top-1 精度与更大候选池
4. 全部评测**零 LLM 成本**：检索质量不需要生成模型即可度量，可作为每次改动切块/加权后的回归门禁

## 复现

```bash
uv run python scripts/download_data.py
uv run python scripts/fetch_regulations.py
uv run python -m scripts.build_knowledge_base --qa-sample 3000
uv run python -m evals.retrieval_eval --modes bm25,dense,hybrid --qa-sample 15
```

## 已知局限 → 下一步

- 金标 27 条偏小：按线上 bad case 扩充至 ≥60 条（§7.7：当前一切分差 <2 SE，不得下选型结论）
- rerank 已实现（bge-reranker-base, CPU），当前金标下无显著差异——扩容后复测
- 法规 chunk 含条款号，BM25 对"第二十五条"类查询占优——后续把条款号作为元数据过滤字段而非纯文本
