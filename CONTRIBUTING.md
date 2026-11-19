# Contributing（单人项目也走全流程——这是本仓库的工程规范声明）

## 工作流（Issue 驱动）

1. 每个功能/缺陷先开 Issue（用模板：背景 + 验收标准），归入对应 Milestone
2. 从 `main` 拉分支：`feat/<topic>`、`fix/<topic>`、`eval/<topic>`、`docs/<topic>`
3. 开发 → 本地验证：
   ```bash
   uv run ruff check . && uv run ruff format --check .
   uv run pytest -m "not slow"
   uv run python -m evals.smoke
   ```
4. 提 PR（用模板，`Closes #N` 关联 issue）→ CI 全绿 → 自审 diff → squash merge → 删分支

## 提交信息

Conventional Commits：`feat:` `fix:` `docs:` `refactor:` `test:` `chore:`

## 版本与发布

SemVer。里程碑完成打 tag（v0.x.y）并写 GitHub Release + CHANGELOG。

## 错误日志规程（防丢失）

- 任何 bug：先立 Issue（复现步骤 + 完整堆栈 + 环境/commit），修复 PR 用 `Fixes #N`
- 运行时日志：`logs/`（结构化 JSON）；失败运行快照：`runs/`
- 每次开发会话收工：commit+push → 更新 `docs/PROGRESS.md` → 未决问题转 Issue

## 设计决策

影响架构或选型的变更必须新增 ADR（`docs/adr/NNNN-*.md`，模板照抄现有文件：
约束 → 候选 → 决策 → 代价 → 替代方案复活条件）。
