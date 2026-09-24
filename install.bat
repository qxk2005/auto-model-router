@echo off
chcp 65001 >nul
title AMRA 依赖与环境一键安装程序

echo ======================================================================
echo          AMRA (Auto Model Router) - Windows 一键安装程序
echo ======================================================================
echo.

REM 优先检测现有虚拟环境
set "PY_EXE="
if exist "AMRA\Scripts\python.exe" (
    set "PY_EXE=AMRA\Scripts\python.exe"
) else if exist ".venv\Scripts\python.exe" (
    set "PY_EXE=.venv\Scripts\python.exe"
) else if exist "venv\Scripts\python.exe" (
    set "PY_EXE=venv\Scripts\python.exe"
)

if defined PY_EXE (
    echo [*] 检测到虚拟环境: %PY_EXE%
    "%PY_EXE%" install.py %*
) else (
    REM 尝试系统 Python
    where python >nul 2>nul
    if %errorlevel% equ 0 (
        echo [*] 使用系统 Python 执行安装与自检...
        python install.py %*
    ) else (
        where py >nul 2>nul
        if %errorlevel% equ 0 (
            echo [*] 使用 Python 启动器 py 执行安装与自检...
            py -3 install.py %*
        ) else (
            echo [错误] 未在系统 PATH 中检测到 Python 3.10+，请先安装 Python 并勾选 "Add to PATH"。
            echo 下载地址: https://www.python.org/downloads/
        )
    )
)

echo.
echo 按任意键退出...
pause >nul
