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
from leaderboard import leaderboard_mgr

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
# Single Prompt Live Diagnostic & Routing Test API
# ---------------------------------------------------------------------------
class SingleTestRequest(BaseModel):
    prompt: str
    mode: str = "predict_only"  # "predict_only" | "end_to_end"
    max_tokens: int = 128


@app.post("/api/router/test-single")
async def test_single_prompt(req: SingleTestRequest):
    """Run a single prompt through routing with full trace logs and stage latency breakdown."""
    prompt = req.prompt.strip()
    if not prompt:
        raise HTTPException(400, "prompt cannot be empty")

    mode = req.mode or "predict_only"
    max_tokens = max(1, min(req.max_tokens or 128, 4096))

    logs: list[dict] = []
    def add_log(level: str, msg: str, stage: str = "general"):
        ts = time.strftime("%H:%M:%S") + f".{int((time.time() % 1) * 1000):03d}"
        logs.append({
            "timestamp": ts,
            "level": level,
            "stage": stage,
            "message": msg
        })

    t_start = time.perf_counter()
    add_log("INFO", f"接收到单次提示词测试请求 (模式: {'⚡ 纯路由预测' if mode == 'predict_only' else '🌐 端到端真实生成'}), 字符数: {len(prompt)}", "init")

    router = ar_server.router
    if router is None:
        router = reload_router_system()

    try:
        t_route_0 = time.perf_counter()
        route_res = router.route(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
        )
        t_route_1 = time.perf_counter()
    except Exception as exc:
        add_log("ERROR", f"路由决策过程发生异常: {exc}", "router")
        raise HTTPException(500, f"Router evaluation failed: {exc}")

    clf = route_res.classification
    laya_ms = round(route_res.classification_ms or (clf.latency_s * 1000.0 if clf else 0.0), 2)
    scoring_ms = max(0.1, round((t_route_1 - t_route_0) * 1000.0 - laya_ms, 2))

    chosen_model = route_res.model
    provider_name = chosen_model.provider
    chosen_name = chosen_model.name
    upstream_id = getattr(chosen_model, "upstream_id", chosen_name)
    decision_reason = route_res.reason

    clf_dev = getattr(router.classifier, "actual_device", "mps") if hasattr(router, "classifier") else "mps"
    add_log("INFO", f"Laya 本地模型 ({clf_dev.upper()} 加速) 特征分类完成: 类别={clf.category if clf else 'general'}, 难度={clf.difficulty if clf else 0.5:.3f}, 错误代价={clf.stakes if clf else 0.2:.3f}, 耗时={laya_ms}ms", "laya")
    add_log("INFO", f"策略算法仲裁完成: 选定目标模型 [{chosen_name}] (提供商: {provider_name}, 算法排序耗时: {scoring_ms}ms)", "decision")
    add_log("INFO", f"决策理由: {decision_reason}", "decision")

    candidates_data = []
    if route_res.explanation and getattr(route_res.explanation, "candidates", None):
        for c in route_res.explanation.candidates:
            candidates_data.append({
                "model": c.model,
                "cost_usd": getattr(c, "cost_usd", None),
                "p_success": getattr(c, "p_success", None),
                "expected_total_cost": getattr(c, "expected_total_cost", None),
            })

    upstream_data = None
    upstream_ms = 0.0
    verify_ms = 0.0

    if mode == "end_to_end":
        provider = ar_server.config.providers.get(provider_name)
        if not provider:
            add_log("ERROR", f"未找到提供商 [{provider_name}] 配置，无法进行端到端调用", "upstream")
            upstream_data = {"status": "error", "error": f"未配置提供商 [{provider_name}]"}
        else:
            base_url = provider.base_url.rstrip("/")
            headers = {"Content-Type": "application/json"}
            api_key = provider.api_key
            if api_key and api_key != "none":
                headers["Authorization"] = f"Bearer {api_key}"
            for k, v in provider.extra_headers.items():
                headers[k] = v

            payload = {
                "model": upstream_id,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
            }

            add_log("INFO", f"开始向上游提供商 [{provider_name}] 发起 HTTP POST 请求: {base_url}/chat/completions (目标模型: {upstream_id})", "upstream")
            t_up_0 = time.perf_counter()
            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    resp = await client.post(f"{base_url}/chat/completions", headers=headers, json=payload)
                    upstream_ms = round((time.perf_counter() - t_up_0) * 1000.0, 2)
                    if resp.status_code == 200:
                        data = resp.json()
                        choices = data.get("choices") or []
                        content = choices[0].get("message", {}).get("content", "") if choices else ""
                        usage = data.get("usage") or {}
                        add_log("INFO", f"上游响应成功: 状态码 200 OK, 网络往返及生成耗时 {upstream_ms}ms, 消耗 Token: prompt={usage.get('prompt_tokens', 0)}, completion={usage.get('completion_tokens', 0)}", "upstream")
                        upstream_data = {
                            "status": "ok",
                            "http_status": 200,
                            "latency_ms": upstream_ms,
                            "reply_snippet": content[:300] + ("..." if len(content) > 300 else ""),
                            "full_reply": content,
                            "usage": usage,
                        }

                        if content and hasattr(router, "check"):
                            t_chk_0 = time.perf_counter()
                            add_log("INFO", "触发 Laya 质量验收判定 (Adequacy Verification)...", "verify")
                            try:
                                verdict = await asyncio.to_thread(router.check, route_res, prompt, content)
                                verify_ms = round((time.perf_counter() - t_chk_0) * 1000.0, 2)
                                add_log("INFO", f"Laya 质量判定完成: 满意度预估={getattr(verdict, 'p_adequate', 'N/A')}, 是否建议升级={getattr(verdict, 'escalate', False)}, 耗时={verify_ms}ms", "verify")
                            except Exception as chk_e:
                                add_log("WARN", f"Laya 质量判定跳过或异常: {chk_e}", "verify")

                    else:
                        err_text = resp.text[:200]
                        add_log("ERROR", f"上游响应异常: HTTP {resp.status_code} - {err_text} (耗时 {upstream_ms}ms)", "upstream")
                        upstream_data = {
                            "status": "error",
                            "http_status": resp.status_code,
                            "latency_ms": upstream_ms,
                            "error": f"HTTP {resp.status_code}: {err_text}",
                        }
            except httpx.TimeoutException:
                upstream_ms = round((time.perf_counter() - t_up_0) * 1000.0, 2)
                add_log("ERROR", f"上游请求超时 (超过 30s)！端点地址: {base_url} 可能未就绪或正在排队", "upstream")
                upstream_data = {
                    "status": "timeout",
                    "latency_ms": upstream_ms,
                    "error": f"上游端点 {base_url} 连接超时 (30s)",
                }
            except Exception as e:
                upstream_ms = round((time.perf_counter() - t_up_0) * 1000.0, 2)
                add_log("ERROR", f"连接上游提供商发生网络异常: {str(e)}", "upstream")
                upstream_data = {
                    "status": "error",
                    "latency_ms": upstream_ms,
                    "error": str(e),
                }

    total_latency_ms = round((time.perf_counter() - t_start) * 1000.0, 2)
    add_log("INFO", f"全路径测试流程执行完毕, 全程总耗时: {total_latency_ms}ms", "done")

    return {
        "status": "ok",
        "mode": mode,
        "timing": {
            "classifier_ms": laya_ms,
            "route_scoring_ms": scoring_ms,
            "upstream_request_ms": upstream_ms,
            "verification_ms": verify_ms,
            "total_latency_ms": total_latency_ms,
        },
        "classification": {
            "category": clf.category if clf else "general",
            "difficulty": round(clf.difficulty, 3) if clf else 0.5,
            "stakes": round(clf.stakes, 3) if clf else 0.2,
            "device": clf_dev,
            "classifier_model": getattr(router.classifier, "model", "convaiinnovations/laya") if hasattr(router, "classifier") else "laya",
        },
        "decision": {
            "chosen_model": chosen_name,
            "provider": provider_name,
            "upstream_id": upstream_id,
            "policy": route_res.explanation.selection.policy if (route_res.explanation and hasattr(route_res.explanation, 'selection')) else "F_expected",
            "reason": decision_reason,
            "candidates": candidates_data,
        },
        "upstream_response": upstream_data,
        "logs": logs,
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
    try:
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
                "avg_total_latency_s": getattr(summary, "avg_total_latency_s", 0.0),
                "alignment_rate": summary.alignment_rate,
                "model_distribution": summary.model_distribution,
                "currency_symbol": getattr(summary, "currency_symbol", "¥"),
                "usd_cny_rate": getattr(summary, "usd_cny_rate", 7.20),
            },
            "report_filename": report_filename,
            "report_url": f"/api/reports/{report_filename}",
        }
    except Exception as e:
        logger.error(f"基准压测运行异常: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"压测运行异常: {str(e)}")


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


@app.delete("/api/reports/{filename}")
async def delete_report(filename: str):
    """Delete a single HTML benchmark report."""
    if "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(status_code=400, detail="非法文件名")
    if not filename.endswith(".html"):
        raise HTTPException(status_code=400, detail="仅允许删除 HTML 格式报告")

    p = REPORTS_DIR / filename
    if not p.exists() or not p.is_file():
        raise HTTPException(status_code=404, detail="报告文件不存在或已被删除")

    try:
        p.unlink()
        return {"status": "ok", "message": f"报告 {filename} 已成功删除"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除报告文件失败: {e}")


class BatchDeleteReportsRequest(BaseModel):
    filenames: list[str]


@app.post("/api/reports/batch-delete")
async def batch_delete_reports(payload: BatchDeleteReportsRequest):
    """Batch delete selected HTML benchmark reports."""
    if not payload.filenames:
        raise HTTPException(status_code=400, detail="未提供需要删除的文件列表")

    deleted = []
    failed = []

    for fname in payload.filenames:
        if "/" in fname or "\\" in fname or ".." in fname or not fname.endswith(".html"):
            raise HTTPException(status_code=400, detail=f"检测到非法文件名: {fname}，仅允许删除 .html 报告文件")

    for fname in payload.filenames:
        p = REPORTS_DIR / fname
        if p.exists() and p.is_file():
            try:
                p.unlink()
                deleted.append(fname)
            except Exception as e:
                failed.append({"filename": fname, "reason": str(e)})
        else:
            failed.append({"filename": fname, "reason": "文件不存在"})

    return {
        "status": "ok",
        "deleted_count": len(deleted),
        "failed_count": len(failed),
        "deleted_files": deleted,
        "failed_files": failed,
        "message": f"成功删除 {len(deleted)} 个报告" + (f"，{len(failed)} 个处理失败" if failed else ""),
    }


# ---------------------------------------------------------------------------
# LMSYS Arena Leaderboard & Model Capability Reference APIs
# ---------------------------------------------------------------------------
@app.get("/api/leaderboard")
async def get_leaderboard(
    search: str | None = None,
    category: str | None = None,
    open_source_only: bool = False,
    candidate_only: bool = False,
):
    """Fetch LMSYS Chatbot Arena leaderboard entries, tagged with candidate router models."""
    cfg = get_current_raw_config()
    configured_models = cfg.get("models", [])

    all_data = leaderboard_mgr.get_all()
    entries = all_data.get("models", [])

    # Filter by search
    if search and search.strip():
        q = search.strip().lower()
        entries = [
            m for m in entries
            if q in m.get("model_name", "").lower()
            or q in m.get("display_name", "").lower()
            or q in m.get("organization", "").lower()
        ]

    # Filter by open source
    if open_source_only:
        entries = [
            m for m in entries
            if str(m.get("license", "")).lower() not in ("proprietary", "closed", "commercial")
        ]

    # Sort by category rating if requested
    if category in ("coding", "math", "hard"):
        cat_key = f"rating_{category}"
        entries = sorted(entries, key=lambda x: x.get(cat_key, 0.0), reverse=True)
    else:
        entries = sorted(entries, key=lambda x: x.get("rank_overall", 999))

    results = []
    for item in entries:
        m_copy = dict(item)
        matched_candidates = []
        for c in configured_models:
            c_name = str(c.get("name", "")).lower()
            c_upstream = str(c.get("upstream_id", "")).lower()
            m_name = str(item.get("model_name", "")).lower()
            d_name = str(item.get("display_name", "")).lower()

            if (c_upstream and (c_upstream in m_name or m_name in c_upstream or c_upstream in d_name)) or \
               (c_name and (c_name in m_name or m_name in c_name or c_name in d_name)):
                matched_candidates.append(c)

        m_copy["is_candidate"] = len(matched_candidates) > 0
        m_copy["matched_candidate"] = matched_candidates[0] if matched_candidates else None

        if candidate_only and not m_copy["is_candidate"]:
            continue
        results.append(m_copy)

    # Cross-comparison data for all currently configured candidate models
    candidate_comparison_list = []
    for c in configured_models:
        c_name = c.get("name", "")
        c_up = c.get("upstream_id", "")
        matches = leaderboard_mgr.match_model(c_up or c_name)
        candidate_comparison_list.append({
            "configured_model": c,
            "matched_leaderboard": matches[0] if matches else None,
        })

    return {
        "status": "ok",
        "total": len(results),
        "last_updated": all_data.get("last_updated", ""),
        "source": all_data.get("source", ""),
        "data": results,
        "candidate_comparison": candidate_comparison_list,
    }


@app.get("/api/leaderboard/match")
async def match_leaderboard_model(query: str):
    """Fuzzy match a model ID/Name against the authoritative LMSYS Chatbot Arena leaderboard."""
    if not query:
        raise HTTPException(status_code=400, detail="query 参数不能为空")

    matches = leaderboard_mgr.match_model(query)
    if not matches:
        return {"status": "not_found", "query": query, "model": None, "candidates": []}

    best = matches[0]
    prof = best.get("normalized_profile", {})
    return {
        "status": "ok",
        "query": query,
        "matched": True,
        "model": best,
        "candidates": matches[:6],
        "normalized_capabilities": {
            "coding": prof.get("coding", 0.85),
            "math": prof.get("math", 0.85),
            "reasoning": prof.get("reasoning", 0.85),
            "general": prof.get("general", 0.85),
        },
    }


@app.post("/api/leaderboard/sync")
async def sync_leaderboard_data():
    """Trigger an online sync from HuggingFace lmarena-ai/leaderboard-dataset."""
    try:
        res = await leaderboard_mgr.sync_from_huggingface()
        return res
    except Exception as exc:
        return {
            "status": "error",
            "message": f"同步失败: {str(exc)}",
            "last_updated": leaderboard_mgr._last_updated,
        }


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

