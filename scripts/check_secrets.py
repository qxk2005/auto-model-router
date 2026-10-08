#!/usr/bin/env python3
"""Pre-commit security checker: Prevent committing hardcoded secrets or API keys."""

import re
import subprocess
import sys
from pathlib import Path

# 正则匹配常见的敏感密钥特征
SECRET_PATTERNS = [
    (re.compile(r"sk-[a-zA-Z0-9]{20,}"), "OpenAI / 第三方大模型 API Key (sk-...)"),
    (re.compile(r"quotio-local-[a-zA-Z0-9\-]{10,}"), "本地代理/平台鉴权 Token"),
    (re.compile(r"""["']api_key["']\s*:\s*["'](?!\$\{)[a-zA-Z0-9_-]{24,}["']"""), "JSON配置中未通过环境变量占位符的硬编码 api_key"),
]

# 允许跳过的文件（如测试用例、规则模板、文档）
ALLOWLIST_PATHS = [
    re.compile(r"^tests/"),
    re.compile(r"^\.env\.example$"),
    re.compile(r"^scratch/"),
    re.compile(r"^scripts/check_secrets\.py$"),
    re.compile(r"\.local\."),
]


def is_path_allowlisted(path_str: str) -> bool:
    for pat in ALLOWLIST_PATHS:
        if pat.search(path_str):
            return True
    return False


def get_staged_files():
    try:
        res = subprocess.run(
            ["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"],
            capture_output=True,
            text=True,
            check=True
        )
        return [line.strip() for line in res.stdout.splitlines() if line.strip()]
    except Exception:
        return []


def check_file(path_str: str) -> list[str]:
    if is_path_allowlisted(path_str):
        return []

    p = Path(path_str)
    if not p.exists() or p.is_dir():
        return []

    findings = []
    try:
        text = p.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return []

    lines = text.splitlines()
    for idx, line in enumerate(lines, 1):
        # 忽略注释或纯模板行
        stripped = line.strip()
        if stripped.startswith("#") or stripped.startswith("//"):
            continue

        for pattern, desc in SECRET_PATTERNS:
            match = pattern.search(line)
            if match:
                # 排除 ${VAR} 格式
                matched_str = match.group(0)
                if "${" in matched_str or "$(" in matched_str or "placeholder" in matched_str or "your-" in matched_str:
                    continue
                findings.append(f"  [{desc}] {path_str}:{idx} -> {line.strip()[:60]}...")
                break

    return findings


def main():
    files = sys.argv[1:] if len(sys.argv) > 1 else get_staged_files()
    if not files:
        sys.exit(0)

    all_findings = []
    for f in files:
        f_findings = check_file(f)
        all_findings.extend(f_findings)

    if all_findings:
        print("\n\033[1;31m[SECURITY ALERT] 提交被阻止：检测到可能存在硬编码密钥或敏感凭据！\033[0m")
        print("\033[33m在以下文件中发现了疑似明文密钥：\033[0m")
        for finding in all_findings:
            print(f"\033[31m{finding}\033[0m")
        print("\n\033[1;32m安全修复建议：\033[0m")
        print("  1. 请使用环境变量占位符，如 \"${JUSHENG_API_KEY}\" 代替真实密钥。")
        print("  2. 私有密钥请存放于本地 .env 或 config/router_config.local.json 中（两者均已被 .gitignore 保护）。")
        print("  3. 严禁将包含真实凭据的公共配置文件（如 config/router_config.json）提交至 Git。\n")
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
