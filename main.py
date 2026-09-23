"""Entry point for auto-llm-router-laya.

Run:
    python main.py
    python main.py --port 8765 --host 0.0.0.0
"""

import argparse
import sys
import uvicorn
import torch

BANNER = """
====================================================================
           {title}
====================================================================
  * WebUI 控制台:           http://localhost:{port}
  * OpenAI 兼容代理网关:    http://localhost:{port}/v1/chat/completions
  * Anthropic 兼容端点:     http://localhost:{port}/v1/messages
  * 模型列表端点:           http://localhost:{port}/v1/models
  * 硬件加速状态:           {hardware}
====================================================================
"""

def main():
    parser = argparse.ArgumentParser(description="Auto-LLM-Router with Laya multi-hardware acceleration")
    parser.add_argument("--port", type=int, default=8765, help="Port to listen on (default: 8765)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host interface (default: 0.0.0.0)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    args = parser.parse_args()

    cuda_ok = torch.cuda.is_available()
    mps_ok = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

    if cuda_ok:
        try:
            gpu_name = torch.cuda.get_device_name(0)
        except Exception:
            gpu_name = "NVIDIA CUDA GPU"
        gpu_clean = (gpu_name or "CUDA GPU").strip()
        if not gpu_clean.upper().startswith("NVIDIA"):
            gpu_clean = f"NVIDIA {gpu_clean}"
        hw_desc = f"{gpu_clean} (CUDA) [ACTIVE]"
        title = f"Auto-LLM-Router (Laya on CUDA: {gpu_clean})"
    elif mps_ok:
        hw_desc = "Apple Silicon (Metal MPS) [ACTIVE]"
        title = "Auto-LLM-Router (Laya on Apple Silicon MPS)"
    else:
        import platform
        hw_desc = f"CPU Multi-threading ({platform.machine()})"
        title = "Auto-LLM-Router (Laya on CPU Multi-threading)"

    print(BANNER.format(title=title, port=args.port, hardware=hw_desc), flush=True)

    uvicorn.run(
        "server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )

if __name__ == "__main__":
    main()
