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
           Auto-LLM-Router (Laya on Apple Silicon M4 Max)           
====================================================================
  * WebUI 控制台:           http://localhost:{port}
  * OpenAI 兼容代理网关:    http://localhost:{port}/v1/chat/completions
  * Anthropic 兼容端点:     http://localhost:{port}/v1/messages
  * 模型列表端点:           http://localhost:{port}/v1/models
  * 硬件加速状态:           {hardware}
====================================================================
"""

def main():
    parser = argparse.ArgumentParser(description="Auto-LLM-Router with Laya on M4 Max")
    parser.add_argument("--port", type=int, default=8765, help="Port to listen on (default: 8765)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host interface (default: 0.0.0.0)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    args = parser.parse_args()

    mps_ok = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    hw_desc = "Apple Silicon M4 Max (Metal MPS) [ACTIVE]" if mps_ok else "CPU Multi-threading"

    print(BANNER.format(port=args.port, hardware=hw_desc), flush=True)

    uvicorn.run(
        "server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )

if __name__ == "__main__":
    main()
