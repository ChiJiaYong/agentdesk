# PROGRESS — 每次开发会话收工必更新（防丢失规程，见根计划 §0）

> 冷启动顺序：读根计划 → 本文件 → `git log --oneline -15` → Milestone open issues
> 热退出顺序：commit+push → 更新本文件 → 未决报错转 Issue

## 当前状态

- **Milestone**: W1 — 仓库初始化 + 理解练习
- **进行中**: GitHub 远程仓库待创建（需要账号操作）
- **下一步**: 创建 GitHub repo 并 push → 配置 main 分支保护 → Kaggle token

## W1 清单

- [x] 工程骨架：pyproject(uv)/.gitignore/.env.example/pre-commit/CI(3 jobs)/issue+PR 模板/.gitattributes
- [x] core/mini_loop.py — 手写 ReAct 循环（书 §2.2.4）+ FakeChatClient 离线注入
- [x] core/fake_chat.py — ScriptedToolChatModel（bind_tools 接缝，供 create_agent 离线跑）
- [x] examples/ — 两个 demo（手写循环 / langchain 1.x create_agent）均离线跑通
- [x] tests/ — 5 测全绿（循环行为 4 + create_agent 1）
- [x] evals/smoke.py — 5/5 通过（CI 无 key 可跑，失败退出码 1）
- [x] scripts/download_data.py — Amazon QA(UCSD 直链)+Olist(Kaggle CLI)+manifest 溯源
- [x] Amazon QA URL HEAD 探测通过：qa_Electronics.json.gz（大写 E）
- [x] 本地验证全绿：ruff check/format + pytest 5/5 + smoke 5/5 + 双 demo
- [x] git init + 9 个规范 commit（身份：ChiJiaYong，noreply 邮箱，repo 级配置；uv.lock 已入库）
- [x] pre-commit 钩子已验证可安装（见已知问题：Windows hook 环境取不到 venv 可执行文件，暂卸载，以 CI 为 lint 门禁）
- [ ] GitHub repo 创建 + push + main 分支保护（用户操作，指引见会话记录 2026-11-19）
- [ ] Kaggle token 配置（用户操作）
- [ ] W1 收尾：v0.1.0 tag + Release（push 后）

## 已知问题 / 未决

- 依赖版本注意：langchain 1.5.2 + langgraph 1.x，`create_react_agent` 已迁移为 `langchain.agents.create_agent`（项目已用新 API）
- pyproject 用 `[tool.uv] package = false`（应用仓库非库），导入依赖根目录 conftest.py 与 `python -m` 调用方式
- pre-commit：Windows 下 `uv run pre-commit install` 写出的 hook 在 git 触发时找不到 venv 内的 pre-commit 可执行文件；
  处置：暂不启用本地钩子，ruff 门禁由 CI 承担；push 后若需本地钩子，改用全局安装 pre-commit 的方案
- LICENSE 版权行当前为 "AgentDesk Contributors"，push 前可改为真实姓名（可选）

## 会话日志

### 2026-11-19 会话 1（骨架日）✅ 已收工
- 完成：W1 除账号操作外全部完成；本地验证全绿；8 commits；pre-commit 就绪
- 注：公开提交统一使用 GitHub noreply 邮箱（272977235+ChiJiaYong@users.noreply.github.com），私人邮箱不入历史
- 决策：应用仓库用 `[tool.uv] package=false`；LangChain 1.5.2 `create_agent` 新 API；
  `bind_tools` 接缝用 ScriptedToolChatModel 离线覆盖
- 踩坑记录：uv_build 要求 src 布局→改 package=false；
  GenericFakeChatModel 无 bind_tools→NotImplementedError→子类覆写；
  evals 读 trace 里的 tool_calls 要用 wire 格式 `tc["function"]["name"]`
- 遗留（用户操作）：
  1. GitHub 建 public 仓库 `agentdesk`（不要勾选任何初始化文件），
     然后：`git remote add origin https://github.com/<用户名>/agentdesk.git`
     `git push -u origin main`
  2. push 后 Settings→Branches→Add rule：main 设 Require a pull request + status checks
  3. Kaggle：kaggle.com→Account→Create New API Token→kaggle.json 放 ~/.kaggle/
     （或把 KAGGLE_USERNAME/KAGGLE_KEY 填进 .env）
  4. 下次会话冷启动：读根计划 §0 checklist → 本文件 → `git log --oneline`
