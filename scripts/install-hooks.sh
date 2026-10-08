#!/usr/bin/env bash
# 安装本地 git pre-commit 钩子以防止意外提交密钥
set -e

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
HOOK_FILE="${REPO_ROOT}/.git/hooks/pre-commit"

if [ ! -d "${REPO_ROOT}/.git/hooks" ]; then
    mkdir -p "${REPO_ROOT}/.git/hooks"
fi

cat << 'EOF' > "${HOOK_FILE}"
#!/usr/bin/env bash
# Git pre-commit secret checking hook
REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
CHECK_SCRIPT="${REPO_ROOT}/scripts/check_secrets.py"

if [ -f "${CHECK_SCRIPT}" ]; then
    python3 "${CHECK_SCRIPT}"
    EXIT_CODE=$?
    if [ $EXIT_CODE -ne 0 ]; then
        exit $EXIT_CODE
    fi
fi
EOF

chmod +x "${HOOK_FILE}"
chmod +x "${REPO_ROOT}/scripts/check_secrets.py"
echo "✅ Git pre-commit 安全拦截钩子已成功安装至 .git/hooks/pre-commit"
