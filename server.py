"""Unified FastAPI Server for auto-llm-router-laya.

Combines:
1. OpenAI-compatible / Anthropic-compatible routing proxy (/v1/chat/completions, /v1/messages, /v1/models)
2. Management & Configuration REST APIs (/api/config, /api/status, /api/provider/test)
3. Benchmark Runner & Evaluation Engine (/api/benchmark/run, /api/cases)
4. Interactive HTML Report Server (/api/reports)
5. Modern Single Page WebUI Dashboard (served at /)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import httpx
import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import auto_router.server as ar_server
from auto_router.config import for_http, load_config, save_config
from auto_router.jev import LocalLayaClassifier
from auto_router.router import Router
from evaluator import BenchmarkEvaluator, BenchmarkSummary
from reporter import ReportGenerator

# Ensure logging is configured
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("auto_router.unified")

# Set Hugging Face mirror by default for fast domestic downloads
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# Initialize app
app = ar_server.app
app.title = "Auto-LLM-Router (Laya on Apple M4 Max)"
app.version = "1.0.0"

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

START_TIME = time.time()
CASES_FILE = Path("benchmarks/typical_cost_cases.json")
REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
WEB_DIR = Path("web")
WEB_DIR.mkdir(parents=True, exist_ok=True)


def get_current_raw_config() -> dict:
    cfg_path = Path(os.environ.get("AUTO_ROUTER_CONFIG") or "config/router_config.json")
    if cfg_path.exists():
        with open(cfg_path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def reload_router_system():
    """Reload configuration and re-initialize router instance."""
    raw = get_current_raw_config()
    loaded = for_http(load_config())
    ar_server.config = loaded
    ar_server.router = Router(loaded)
    log.info("Router system reloaded with %d models and %d providers", len(loaded.catalog.models), len(loaded.providers))
    return ar_server.router


@app.on_event("startup")
async def on_startup():
    """Warm up Laya on Apple Silicon MPS on server startup in background."""
    log.info("Starting Auto-LLM-Router unified service on port 8765...")
    try:
        reload_router_system()
        clf = getattr(ar_server.router, "classifier", None)
        if isinstance(clf, LocalLayaClassifier):
            log.info("Pre-warming local Laya classifier on Apple Silicon (MPS) in background...")
            loop = asyncio.get_event_loop()
            asyncio.create_task(loop.run_in_executor(None, clf.warmup))
    except Exception as exc:
        log.warning("Warmup warning: %s", exc)



# ---------------------------------------------------------------------------
# Status & System Info API
# ---------------------------------------------------------------------------
@app.get("/api/status")
async def get_system_status():
    mps_ok = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    device_name = "Apple Silicon M4 Max (Metal MPS)" if mps_ok else ("CUDA GPU" if torch.cuda.is_available() else "CPU")
    
    cfg = get_current_raw_config()
    pol_cfg = cfg.get("policy", {})
    clf_cfg = pol_cfg.get("classifier", {})
    
    clf = getattr(ar_server.router, "classifier", None)
    actual_dev = getattr(clf, "actual_device", "mps" if mps_ok else "cpu")
    
    # Process memory
    rss_mb = 0.0
    try:
        import resource
        rusage = resource.getrusage(resource.RUSAGE_SELF)
        # On macOS ru_maxrss is in bytes
        rss_mb = round(rusage.ru_maxrss / (1024 * 1024), 1)
    except Exception:
        pass

    return {
        "status": "online",
        "uptime_seconds": round(time.time() - START_TIME, 1),
        "device": actual_dev,
        "device_hardware": device_name,
        "mps_available": mps_ok,
        "classifier_backend": clf_cfg.get("backend", "local"),
        "classifier_model": clf_cfg.get("model", "convaiinnovations/laya"),
        "classifier_subfolder": clf_cfg.get("subfolder", "multilingual"),
        "active_policy": pol_cfg.get("name", "F_expected"),
        "providers_count": len(cfg.get("providers", {})),
        "models_count": len(cfg.get("models", [])),
        "memory_rss_mb": rss_mb,
    }


# ---------------------------------------------------------------------------
# Configuration CRUD APIs
# ---------------------------------------------------------------------------
@app.get("/api/config")
async def get_config():
    return get_current_raw_config()


class ConfigUpdateRequest(BaseModel):
    providers: dict[str, Any]
    models: list[dict[str, Any]]
    policy: dict[str, Any]
    subscriptions: dict[str, Any] = Field(default_factory=dict)


@app.post("/api/config")
async def update_config(payload: ConfigUpdateRequest):
    try:
        raw_dict = payload.dict()
        save_config(raw_dict)
        reload_router_system()
        return {"status": "ok", "message": "配置已保存并实时生效"}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"保存配置失败: {exc}")


# ---------------------------------------------------------------------------
# Provider Testing API (Supports local LM Studio & Cloud APIs)
# ---------------------------------------------------------------------------
class ProviderTestRequest(BaseModel):
    base_url: str
    api_key: str | None = None
    model: str | None = None


@app.post("/api/provider/test")
async def test_provider_endpoint(req: ProviderTestRequest):
    base_url = req.base_url.rstrip("/")
    headers = {}
    if req.api_key:
        headers["Authorization"] = f"Bearer {req.api_key}"

    t0 = time.perf_counter()
    async with httpx.AsyncClient(timeout=10.0) as client:
        # First attempt: GET /models
        try:
            r = await client.get(f"{base_url}/models", headers=headers)
            lat_ms = round((time.perf_counter() - t0) * 1000.0, 1)
            if r.status_code == 200:
                data = r.json()
                models_list = [m.get("id") for m in data.get("data", []) if isinstance(m, dict)]
                return {
                    "status": "ok",
                    "latency_ms": lat_ms,
                    "available_models": models_list,
                    "message": f"连接成功！检测到 {len(models_list)} 个可用模型",
                }
        except Exception:
            pass

        # Second attempt: simple chat completion probe
        try:
            probe_body = {
                "model": req.model or "default",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 5,
            }
            r = await client.post(f"{base_url}/chat/completions", json=probe_body, headers=headers)
            lat_ms = round((time.perf_counter() - t0) * 1000.0, 1)
            if r.status_code == 200:
                return {
                    "status": "ok",
                    "latency_ms": lat_ms,
                    "available_models": [req.model or "default"],
                    "message": "端点响应正常！",
                }
            return {
                "status": "error",
                "latency_ms": lat_ms,
                "error": f"HTTP {r.status_code}: {r.text[:200]}",
            }
        except Exception as exc:
            lat_ms = round((time.perf_counter() - t0) * 1000.0, 1)
            return {"status": "error", "latency_ms": lat_ms, "error": str(exc)}


class ProviderModelsRequest(BaseModel):
    base_url: str
    api_key: str | None = None


@app.post("/api/provider/models")
async def fetch_provider_models(req: ProviderModelsRequest):
    """Fetch complete list of available models from an OpenAI-compatible provider."""
    base_url = req.base_url.rstrip("/")
    headers = {}
    if req.api_key:
        headers["Authorization"] = f"Bearer {req.api_key}"

    t0 = time.perf_counter()
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            r = await client.get(f"{base_url}/models", headers=headers)
            lat_ms = round((time.perf_counter() - t0) * 1000.0, 1)
            if r.status_code == 200:
                data = r.json()
                raw_models = data.get("data", [])
                models = []
                for item in raw_models:
                    if isinstance(item, dict) and "id" in item:
                        models.append({
                            "id": item["id"],
                            "created": item.get("created"),
                            "owned_by": item.get("owned_by", "system"),
                        })
                    elif isinstance(item, str):
                        models.append({"id": item, "owned_by": "system"})
                return {
                    "status": "ok",
                    "latency_ms": lat_ms,
                    "count": len(models),
                    "models": models,
                    "message": f"成功获取到 {len(models)} 个可用模型",
                }
            return {
                "status": "error",
                "latency_ms": lat_ms,
                "error": f"上游接口返回 HTTP {r.status_code}: {r.text[:200]}",
            }
        except Exception as exc:
            lat_ms = round((time.perf_counter() - t0) * 1000.0, 1)
            return {"status": "error", "latency_ms": lat_ms, "error": f"无法连接提供商端点: {exc}"}


class ModelProbeRequest(BaseModel):
    provider_id: str | None = None
    base_url: str | None = None
    api_key: str | None = None
    upstream_id: str
    prompt: str = "请用Python写一个快速排序函数，并用一句话解释其原理。"


@app.post("/api/model/probe")
async def probe_model_capability(req: ModelProbeRequest):
    """Live probe model availability, time-to-first-token latency, and output capability."""
    base_url = req.base_url
    api_key = req.api_key

    if not base_url and req.provider_id:
        cfg = get_current_raw_config()
        prov_info = cfg.get("providers", {}).get(req.provider_id)
        if prov_info:
            base_url = prov_info.get("base_url")
            api_key = prov_info.get("api_key")

    if not base_url:
        raise HTTPException(status_code=400, detail="缺少上游端点 base_url 或有效的 provider_id")

    base_url = base_url.rstrip("/")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    t0 = time.perf_counter()
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            body = {
                "model": req.upstream_id,
                "messages": [
                    {"role": "system", "content": "You are a helpful and concise AI assistant."},
                    {"role": "user", "content": req.prompt},
                ],
                "max_tokens": 160,
                "temperature": 0.5,
            }
            r = await client.post(f"{base_url}/chat/completions", json=body, headers=headers)
            lat_ms = round((time.perf_counter() - t0) * 1000.0, 1)

            if r.status_code == 200:
                data = r.json()
                reply = ""
                choices = data.get("choices", [])
                if choices and isinstance(choices[0], dict):
                    msg = choices[0].get("message", {})
                    reply = msg.get("content", "")
                usage = data.get("usage", {})

                return {
                    "status": "ok",
                    "latency_ms": lat_ms,
                    "model": req.upstream_id,
                    "reply_snippet": reply[:400],
                    "usage": usage,
                    "message": "模型可用且响应正常！",
                }
            return {
                "status": "error",
                "latency_ms": lat_ms,
                "model": req.upstream_id,
                "error": f"HTTP {r.status_code}: {r.text[:300]}",
            }
        except Exception as exc:
            lat_ms = round((time.perf_counter() - t0) * 1000.0, 1)
            return {
                "status": "error",
                "latency_ms": lat_ms,
                "model": req.upstream_id,
                "error": f"调用失败: {exc}",
            }


# ---------------------------------------------------------------------------
# Benchmark Test Cases API
# ---------------------------------------------------------------------------
@app.get("/api/cases")
async def get_test_cases():
    if CASES_FILE.exists():
        with open(CASES_FILE, encoding="utf-8") as f:
            return json.load(f)
    return []


@app.post("/api/cases")
async def save_test_cases(cases: list[dict[str, Any]]):
    CASES_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CASES_FILE, "w", encoding="utf-8") as f:
        json.dump(cases, f, indent=2, ensure_ascii=False)
    return {"status": "ok", "count": len(cases), "message": "测试用例集已保存"}


# ---------------------------------------------------------------------------
# Benchmark Execution & HTML Report Generation API
# ---------------------------------------------------------------------------
class BenchmarkRunRequest(BaseModel):
    mode: str = "simulation"  # "simulation" or "real"
    case_ids: list[str] | None = None


@app.post("/api/benchmark/run")
async def run_benchmark(req: BenchmarkRunRequest):
    # Load cases
    cases: list[dict] = []
    if CASES_FILE.exists():
        with open(CASES_FILE, encoding="utf-8") as f:
            cases = json.load(f)
    
    if req.case_ids:
        selected_set = set(req.case_ids)
        cases = [c for c in cases if c.get("id") in selected_set]

    if not cases:
        raise HTTPException(status_code=400, detail="没有可执行的测试用例")

    raw_cfg = get_current_raw_config()
    evaluator = BenchmarkEvaluator(
        router=ar_server.router,
        catalog=ar_server.config.catalog,
        policy_engine=ar_server.router.policy,
        config_raw=raw_cfg,
    )

    if req.mode == "real":
        summary = await evaluator.run_real_benchmark(cases)
    else:
        loop = asyncio.get_event_loop()
        summary = await loop.run_in_executor(None, evaluator.run_simulation, cases)

    # Generate HTML report
    reporter = ReportGenerator(output_dir="reports")
    report_path = reporter.generate(summary)
    report_filename = Path(report_path).name

    return {
        "status": "completed",
        "mode": summary.mode,
        "summary": {
            "timestamp": summary.timestamp,
            "total_cases": summary.total_cases,
            "total_tokens": summary.total_tokens,
            "cost_router_total": summary.cost_router_total,
            "cost_expensive_total": summary.cost_expensive_total,
            "cost_cheap_total": summary.cost_cheap_total,
            "total_savings_usd": summary.total_savings_usd,
            "total_savings_pct": summary.total_savings_pct,
            "avg_classifier_latency_ms": summary.avg_classifier_latency_ms,
            "alignment_rate": summary.alignment_rate,
            "model_distribution": summary.model_distribution,
        },
        "report_filename": report_filename,
        "report_url": f"/api/reports/{report_filename}",
    }


# ---------------------------------------------------------------------------
# Reports Management API
# ---------------------------------------------------------------------------
@app.get("/api/reports")
async def list_reports():
    reports = []
    for p in sorted(REPORTS_DIR.glob("*.html"), key=os.path.getmtime, reverse=True):
        stat = p.stat()
        reports.append({
            "filename": p.name,
            "size_kb": round(stat.st_size / 1024, 1),
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stat.st_mtime)),
            "url": f"/api/reports/{p.name}",
        })
    return reports


@app.get("/api/reports/{filename}")
async def view_report(filename: str, download: bool = False):
    p = REPORTS_DIR / filename
    if not p.exists() or not p.is_file():
        raise HTTPException(status_code=404, detail="报告未找到")
    
    if download:
        return FileResponse(p, media_type="text/html", filename=filename)
    return HTMLResponse(p.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# WebUI Static Frontend
# ---------------------------------------------------------------------------
@app.api_route("/", methods=["GET", "HEAD"], response_class=HTMLResponse)
async def serve_index():
    index_path = WEB_DIR / "index.html"
    if index_path.exists():
        return HTMLResponse(index_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Auto-LLM-Router is running. WebUI index.html not found.</h1>")


# Mount static assets if web dir has assets
if WEB_DIR.exists():
    app.mount("/web", StaticFiles(directory=str(WEB_DIR)), name="web")


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"])
async def catch_all(path: str, request: Request) -> Response:
    """Fallback passthrough for Claude Code telemetry / count_tokens."""
    if not path or path == "favicon.ico" or path.startswith(("api/", "web/", "reports/")):
        raise HTTPException(status_code=404, detail="Not found")
    from auto_router import shim
    body = await request.body()
    return await shim.passthrough(request, body, f"/{path}")

