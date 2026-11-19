# ADR-0003: 数据不入 git — 下载脚本 + manifest 溯源

日期：2026-11-19 · 状态：已接受

## 约束

- 简历与面试要求"数据真实可溯源"；同时第三方数据集许可（CC BY-NC-SA 等）禁止随意再分发
- 仓库需保持轻量（GitHub 规范），Olist 原始数据 126MB+

## 决策

1. `data/raw/`、`data/processed/` 全部 .gitignore
2. `scripts/download_data.py` 统一拉取：UCSD Amazon QA（免注册直链）+ Olist（Kaggle CLI）+ 后续文档源（git clone）
3. `data/manifest.json` 记录每个数据集：来源 URL、许可证、抓取时间、大小、sha256
4. `data/README.md` 面向读者写明用途与合规边界；清洗后文档保留来源三元组（URL/许可证/日期）

## 代价

- 复现依赖外源可用性：以 checksum 固定版本缓解；源失效时 manifest 中的存档说明兜底
- 多一步 setup：README 快速开始已写明
