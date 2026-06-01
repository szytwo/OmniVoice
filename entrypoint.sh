#!/usr/bin/env bash
set -euo pipefail

# 环境变量校验
: "${REPO_URL:?请通过环境变量 REPO_URL 注入仓库地址（如 https://github.com/you/your-repo.git）}"
: "${BRANCH:=main}"            # 默认分支，可通过环境变量覆盖
: "${CODE_DIR:=/code}"         # 代码目录，可通过环境变量覆盖
: "${MAX_RETRY:=3}"            # 最大重试次数
: "${RETRY_INTERVAL:=3}"       # 重试间隔（秒）
: "${GITHUB_TOKEN:?请通过环境变量 GITHUB_TOKEN 注入访问 Token（需 Read-only Contents 权限）}"
: "${ENABLE_GIT_PULL:=true}"   # ✅ 是否启用 Git 拉取，默认启用

# 将 REPO_URL 改写为带 Token 的 HTTPS 链接
REPO_URL="https://${GITHUB_TOKEN}@${REPO_URL#https://}"

# 带重试的命令函数
retry_cmd() {
  local n=1
  until "$@"; do
    if [ "$n" -lt "$MAX_RETRY" ]; then
      echo "Retry $((n+1))/$MAX_RETRY: $*"
      n=$((n+1))
      sleep "$RETRY_INTERVAL"
    else
      echo "Command '$*' failed after $MAX_RETRY attempts. Exiting."
      return 1
    fi
  done
  return 0
}

# ✅ 判断是否启用 Git 拉取逻辑
if [ "$ENABLE_GIT_PULL" = "true" ]; then
  # 切换到代码目录并更新仓库（假设 .git 已存在）
  echo "Switching to code directory: ${CODE_DIR}"
  cd "${CODE_DIR}"

  # 禁止 Git 交互式提示，开启网络层面的调试信息输出
  export GIT_TERMINAL_PROMPT=0
  #export GIT_CURL_VERBOSE=1

  # 重写 origin URL，让后续 fetch 用 Token 认证
  git remote set-url origin "${REPO_URL}"
  echo "origin URL updated to use Token-based HTTPS address"

  echo "Fetching latest commits from remote branch '${BRANCH}'..."
  # 拉取远程更新（带超时保护），并强制覆盖本地代码
  if retry_cmd timeout 30s git fetch --no-progress origin "${BRANCH}" && git reset --hard "origin/${BRANCH}"; then
    # 更新子模块
    if [ -f .gitmodules ]; then
      echo "Initializing and updating submodules..."
      
      # 同步子模块 URL，不包含网络操作
      sed -i "s|https://github.com/szytwo/|https://${GITHUB_TOKEN}@codeup.aliyun.com/682f5f99690eab70119dc36d/szytwo/|g" .gitmodules
      git submodule sync --recursive
      # 拉取子模块更新（带超时保护）
      if retry_cmd timeout 30s git submodule foreach --recursive git fetch --no-progress origin; then
        # 强制覆盖本地代码
        git submodule foreach --recursive '
          branch=$(git symbolic-ref --short refs/remotes/origin/HEAD)
          echo \"Updating submodule $name on branch $branch\"
          git reset --hard $branch
        '
      fi
    fi
    # 输出当前版本
    CURRENT_SHA=$(git rev-parse HEAD)
    echo "Code updated to commit: ${CURRENT_SHA}"
  else
    echo "Code updated failed"
  fi
else
  echo "Git update skipped due to ENABLE_GIT_PULL=${ENABLE_GIT_PULL}"
fi

# 最后执行传入的命令（如 python api.py）
exec -- "$@"
