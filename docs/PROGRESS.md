# PROGRESS — 每次开发会话收工必更新（防丢失规程，见根计划 §0）

> 冷启动顺序：读根计划 → 本文件 → `git log --oneline -15` → Milestone open issues
> 热退出顺序：commit+push → 更新本文件 → 未决报错转 Issue

## 当前状态

- **Milestone**: W1 — 仓库初始化 + 理解练习
- **进行中**: GitHub 远程仓库待创建（需要账号操作）
- **下一步**: 创建 GitHub repo 并 push → 配置 main 分支保护 → Kaggle token

## W1 清单

- [x] 工程骨架：pyproject(uv)/.gitignore/.env.example/pre-commit/CI(3 jobs)/issue+PR 模板
- [x] core/mini_loop.py — 手写 ReAct 循环（书 §2.2.4）+ FakeChatClient 离线注入
- [x] examples/langgraph_quickstart.py — 同任务 LangGraph 对照版（离线可跑）
- [x] tests/ — 循环行为 4 测 + LangGraph 1 测（全离线）
- [x] evals/smoke.py — 5 条 canned 任务冒烟（CI 无 key 可跑，失败退出码 1）
- [x] scripts/download_data.py — Amazon QA(UCSD 直链)+Olist(Kaggle CLI)+manifest 溯源
- [x] docs/adr/0001 编排选型、0002 模型路由、0003 数据不入库
- [ ] uv sync + 本地全绿验证（待本机跑）
- [ ] git init + 首批 commit（等 git 身份配置）
- [ ] GitHub repo 创建 + push + main 分支保护（用户操作）
- [ ] Kaggle token 配置（用户操作）
- [ ] Amazon QA URL HEAD 探测通过（脚本 --check）

## 已知问题 / 未决

- UCSD qa 文件名大小写不确定（qa_electronics vs qa_Electronics），脚本已做双候选 HEAD 探测
- Windows 无 make：构建命令以 `uv run` 为准（README 已写），不引入 Makefile

## 会话日志

### 2026-11-19 会话 1（骨架日）
- 完成：上表前 7 项全部文件就绪
- 决策：见 docs/adr/0001~0003；构建命令走 uv run（Windows 无 make）
- 遗留：上表后 4 项依赖用户账号操作
