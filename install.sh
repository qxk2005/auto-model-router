#!/usr/bin/env bash
# ======================================================================
#          AMRA (Auto Model Router) - macOS / Linux 一键安装程序
# ======================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "======================================================================"
echo "         AMRA (Auto Model Router) - macOS / Linux 一键安装"
echo "======================================================================"
echo ""

# 探测虚拟环境
PY_EXE=""
if [ -f "AMRA/bin/python" ]; then
    PY_EXE="AMRA/bin/python"
elif [ -f ".venv/bin/python" ]; then
    PY_EXE=".venv/bin/python"
elif [ -f "venv/bin/python" ]; then
    PY_EXE="venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PY_EXE="python3"
elif command -v python >/dev/null 2>&1; then
    PY_EXE="python"
else
    echo "❌ 错误: 未检测到 Python 解释器，请先安装 Python 3.10+"
    exit 1
fi

echo "🚀 使用 Python 解释器: $PY_EXE"
"$PY_EXE" install.py "$@"
