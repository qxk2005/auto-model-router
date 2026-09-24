#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AMRA (Auto Model Router) - 跨平台环境与依赖一键安装与自检脚本
支持操作系统: Windows 10/11, macOS (Apple Silicon / Intel), Linux
支持加速后端: NVIDIA CUDA (Windows/Linux), Apple Metal MPS (macOS), CPU Fallback
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
import time
import urllib.request

# 颜色与样式（兼容 Windows 终端与 POSIX）
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RED = "\033[91m"
BOLD = "\033[1m"
RESET = "\033[0m"

# 确保 Windows 终端 UTF-8 编码与 ANSI 彩色输出
if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
    os.system("")

DOMESTIC_PYPI_MIRROR = "https://pypi.tuna.tsinghua.edu.cn/simple"
PYTORCH_CUDA_INDEX = "https://download.pytorch.org/whl/cu124"


def print_banner():
    banner = f"""
{CYAN}{BOLD}======================================================================
     AMRA (Auto Model Router) - 跨平台环境一键安装与自检程序
======================================================================{RESET}
"""
    print(banner)


def check_python_version() -> bool:
    print(f"[*] 检查 Python 运行版本... ", end="")
    v = sys.version_info
    ver_str = f"{v.major}.{v.minor}.{v.micro}"
    if v < (3, 10):
        print(f"{RED}[不符合]{RESET}")
        print(f"    {RED}错误: AMRA 要求 Python 3.10 或更高版本，当前版本为 {ver_str}{RESET}")
        return False
    print(f"{GREEN}[通过]{RESET} (当前 Python {ver_str} @ {sys.executable})")
    return True


def detect_network_mirror() -> bool:
    """探测国内网络连通性，若访问官方源延迟高则自动推荐国内镜像源"""
    print(f"[*] 探测网络连通性与镜像源适配... ", end="", flush=True)
    start = time.time()
    try:
        req = urllib.request.Request("https://pypi.org", headers={"User-Agent": "AMRA-Installer"})
        with urllib.request.urlopen(req, timeout=2.5) as resp:
            elapsed = time.time() - start
            if elapsed < 1.5:
                print(f"{GREEN}[官方源顺畅]{RESET} ({int(elapsed * 1000)}ms)")
                return False
            else:
                print(f"{YELLOW}[官方源延迟较高]{RESET} ({int(elapsed * 1000)}ms, 推荐国内镜像源)")
                return True
    except Exception:
        print(f"{YELLOW}[官方源访问受限]{RESET} (自动切换为国内清华镜像源加速)")
        return True


def detect_hardware() -> tuple[str, str, str | None]:
    """
    返回: (platform_name, accelerator_type, details)
    accelerator_type: "cuda" | "mps" | "cpu"
    """
    system = platform.system()
    machine = platform.machine().lower()

    if system == "Darwin":
        # macOS 检查 Apple Silicon
        if "arm" in machine or "aarch" in machine:
            return "macOS (Apple Silicon)", "mps", f"架构: {machine}，支持 Apple Metal (MPS) 原生 GPU 加速"
        return "macOS (Intel)", "cpu", f"架构: {machine}，运行于 CPU 模式"

    # Windows / Linux 检查 NVIDIA GPU
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        try:
            res = subprocess.run(
                [nvidia_smi, "--query-gpu=name,driver_version", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=3
            )
            if res.returncode == 0 and res.stdout.strip():
                gpu_info = res.stdout.strip().split("\n")[0]
                return f"{system} ({machine})", "cuda", f"检测到 NVIDIA 显卡: {gpu_info} (启用 CUDA 12.4 轮子)"
        except Exception:
            pass

    return f"{system} ({machine})", "cpu", f"未检测到 NVIDIA 独显或不支持 GPU，将采用高效 CPU 模式 (~10ms)"


def run_pip(args: list[str], use_mirror: bool = False, extra_index: str | None = None) -> bool:
    cmd = [sys.executable, "-m", "pip"] + args
    if use_mirror:
        cmd.extend(["-i", DOMESTIC_PYPI_MIRROR])
    if extra_index:
        cmd.extend(["--extra-index-url", extra_index])

    cmd_display = " ".join(cmd)
    print(f"{CYAN}>>> 执行命令:{RESET} {cmd_display}")
    ret = subprocess.run(cmd)
    return ret.returncode == 0


def verify_installation():
    print(f"\n{BOLD}{CYAN}======================================================================")
    print(f"                     AMRA 依赖模块自检与就绪验证")
    print(f"======================================================================{RESET}")

    modules = [
        ("fastapi", "FastAPI Web 框架"),
        ("uvicorn", "Uvicorn ASGI 服务器"),
        ("httpx", "HTTPX 异步客户端"),
        ("yaml", "PyYAML 配置解析"),
        ("dotenv", "python-dotenv 环境变量支持"),
        ("psutil", "psutil 进程与内存监控"),
        ("torch", "PyTorch 深度学习运行时"),
        ("laya", "Laya 本地快速分类引擎"),
    ]

    all_ok = True
    for mod_name, label in modules:
        try:
            m = __import__(mod_name)
            ver = getattr(m, "__version__", "已就绪")
            print(f"  {GREEN}[OK]{RESET}   {label:32s} : {ver}")
        except ImportError as e:
            print(f"  {RED}[FAIL]{RESET} {label:32s} : 未找到或导入失败 ({e})")
            all_ok = False

    # 深度检测硬件加速状态
    print("-" * 70)
    try:
        import torch
        cuda_avail = torch.cuda.is_available()
        mps_avail = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

        if cuda_avail:
            gpu_name = torch.cuda.get_device_name(0)
            print(f"  {GREEN}[OK]{RESET}   硬件加速状态                  : {GREEN}CUDA 可用{RESET} (设备: {gpu_name})")
        elif mps_avail:
            print(f"  {GREEN}[OK]{RESET}   硬件加速状态                  : {GREEN}Apple Metal (MPS) 可用{RESET}")
        else:
            print(f"  {YELLOW}[INFO]{RESET} 硬件加速状态                  : {YELLOW}CPU 模式{RESET} (Laya 本地分类推理稳定运行)")
    except Exception as e:
        print(f"  {RED}[FAIL]{RESET} 硬件加速检测异常              : {e}")

    print("=" * 70)
    if all_ok:
        print(f"\n{GREEN}{BOLD}恭喜！所有依赖均已正确安装并验证通过！{RESET}")
        print(f"您可以通过以下命令立即启动服务：")
        print(f"    {CYAN}python main.py{RESET}")
        print(f"然后在浏览器中访问控制台：")
        print(f"    {CYAN}http://localhost:8765{RESET}\n")
    else:
        print(f"\n{RED}{BOLD}注意：部分依赖未正确就绪，请根据上方红标错误排查。{RESET}\n")


def main():
    parser = argparse.ArgumentParser(description="AMRA 跨平台环境与依赖一键安装与自检程序")
    parser.add_argument("--check-only", action="store_true", help="仅执行环境与依赖自检，不执行 pip 安装")
    parser.add_argument("--mirror", action="store_true", help="强制使用清华/国内 PyPI 镜像加速源")
    parser.add_argument("--no-mirror", action="store_true", help="强制使用官方 PyPI 源")
    parser.add_argument("--cpu", action="store_true", help="强制安装 CPU 版本 PyTorch")
    parser.add_argument("--cuda", action="store_true", help="强制安装 CUDA 版本 PyTorch (Windows/Linux)")
    args = parser.parse_args()

    print_banner()

    if not check_python_version():
        sys.exit(1)

    if args.check_only:
        verify_installation()
        return

    # 1. 硬件与加速器检测
    os_name, accel_type, details = detect_hardware()
    if args.cpu:
        accel_type = "cpu"
        details = "用户指定 --cpu，强制使用 CPU 模式"
    elif args.cuda:
        accel_type = "cuda"
        details = "用户指定 --cuda，强制使用 CUDA 模式"

    print(f"[*] 检测到操作系统环境: {BOLD}{os_name}{RESET}")
    print(f"[*] 硬件加速配置判定  : {BOLD}{accel_type.upper()}{RESET} ({details})")

    # 2. 网络源检测
    if args.no_mirror:
        use_mirror = False
        print("[*] 镜像配置: 用户指定 --no-mirror，使用官方源")
    elif args.mirror:
        use_mirror = True
        print("[*] 镜像配置: 用户指定 --mirror，使用国内镜像加速源")
    else:
        use_mirror = detect_network_mirror()

    if use_mirror:
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        print(f"[*] 已预配置国内 HuggingFace 镜像: {CYAN}HF_ENDPOINT=https://hf-mirror.com{RESET}")

    # 3. 升级 pip
    print(f"\n{BOLD}[步骤 1/3] 升级基础 pip 安装工具...{RESET}")
    run_pip(["install", "--upgrade", "pip"], use_mirror=use_mirror)

    # 4. 根据硬件安装 PyTorch
    print(f"\n{BOLD}[步骤 2/3] 安装 PyTorch 深度学习运行时 ({accel_type.upper()})...{RESET}")
    if accel_type == "cuda":
        # Windows / Linux CUDA 12.4
        print(f"[*] 正在从 PyTorch CUDA 轮子源安装 GPU 加速包...")
        success = run_pip(
            ["install", "torch>=2.0.0"],
            use_mirror=False,
            extra_index=PYTORCH_CUDA_INDEX
        )
    else:
        # macOS MPS 或通用 CPU
        success = run_pip(["install", "torch>=2.0.0"], use_mirror=use_mirror)

    if not success:
        print(f"{YELLOW}[!] PyTorch 专用安装遇到问题，尝试使用标准兼容安装...{RESET}")
        run_pip(["install", "torch>=2.0.0"], use_mirror=use_mirror)

    # 5. 安装 requirements.txt
    req_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "requirements.txt")
    if os.path.exists(req_file):
        print(f"\n{BOLD}[步骤 3/3] 安装 requirements.txt 中的核心业务依赖...{RESET}")
        run_pip(["install", "-r", req_file], use_mirror=use_mirror)
    else:
        print(f"{RED}[!] 未找到 requirements.txt，跳过核心业务包批量安装{RESET}")

    # 6. 安装后自检
    verify_installation()


if __name__ == "__main__":
    main()
