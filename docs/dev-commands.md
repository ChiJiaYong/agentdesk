# 开发命令速查（Windows）

## 日常验证（提交前必跑）

```powershell
uv run ruff check .
uv run ruff format --check .
uv run pytest -m "not slow"
uv run python -m evals.smoke
```

## 推送（HTTP 被重置时的两条备用路）

```powershell
# A. 有代理：先给 git 指定代理端口（按你的代理软件实际端口改）
git config --global http.proxy http://127.0.0.1:7890
git push -u origin main
# 用完可取消：git config --global --unset http.proxy

# B. SSH over 443（无代理时的稳定替代）
# 1) 生成密钥：ssh-keygen -t ed25519 -C "ChiJiaYong"
# 2) 公钥(~/.ssh/id_ed25519.pub)内容粘贴到 GitHub → Settings → SSH and GPG keys
# 3) ~/.ssh/config 追加：
#    Host github.com
#      HostName ssh.github.com
#      Port 443
#      User git
# 4) 改用 SSH 远程：git remote set-url origin git@github.com:ChiJiaYong/agentdesk.git
#    git push -u origin main
```

## Git 身份（本仓库已配置，勿用私人邮箱）

```
user.name  = ChiJiaYong
user.email = 272977235+ChiJiaYong@users.noreply.github.com
```

## 会话规程（防丢失）

- 收工：commit+push → 更新 docs/PROGRESS.md → 未决报错转 Issue
- 开工：根计划 §0 checklist → docs/PROGRESS.md → git log → 领 issue
