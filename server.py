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
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import httpx
import torch
try:
    import dotenv
    dotenv.load_dotenv()
except ImportError:
    pass

# Ensure .env is read if present
_env_p = Path(".env")
if _env_p.exists():
    try:
        with open(_env_p, "r", encoding="utf-8") as _ef:
            for _line in _ef:
                _line = _line.strip()
                if _line and not _line.startswith("#") and "=" in _line:
                    _k, _v = _line.split("=", 1)
                    os.environ.setdefault(_k.strip(), _v.strip())
    except Exception:
        pass

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import auto_router.server as ar_server
from auto_router.config import (
    for_http, load_config, save_config, expand_env_vars, normalize_provider_url,
    sanitize_config_for_public_repo, update_local_env_file
)
import auto_router.jev as jev
from auto_router.jev import LocalLayaClassifier, Judgement, JevClassifier, normalize_jev_url
from auto_router.clef import ClefClassifier, DEFAULT_CLEF_MODEL
from auto_router.verify import VerifyPolicy
from auto_router.router import Router
from evaluator import BenchmarkEvaluator, BenchmarkSummary
from reporter import ReportGenerator
from leaderboard import leaderboard_mgr

# Ensure logging is configured
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("auto_router.unified")
logger = log

# Set Hugging Face mirror by default for fast domestic downloads
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# Initialize app
app = ar_server.app
app.title = "Auto-LLM-Router (Laya Multi-Hardware Acceleration)"
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
    cfg_path = os.environ.get("AUTO_ROUTER_CONFIG")
    if not cfg_path:
        local_p = Path("config/router_config.local.json")
        if local_p.exists():
            cfg_path = local_p
        else:
            cfg_path = Path("config/router_config.json")
    p = Path(cfg_path)
    if p.exists():
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return {}


def mask_secret(val: str | None) -> str | None:
    """Mask sensitive API keys for safe display in WebUI."""
    if not val or not isinstance(val, str):
        return val
    val = val.strip()
    # 环境变量占位符原样展示
    if val.startswith("${") or val.startswith("$"):
        return val
    if len(val) <= 8:
        return "********"
    return f"{val[:6]}********{val[-4:]}"


def create_llm_judge(target_model, provider):
    """Create an LLM-as-a-Judge function that synchronously evaluates answer adequacy."""
    raw_api_key = expand_env_vars(provider.api_key) if provider.api_key else ""
    headers = {"Content-Type": "application/json"}
    if raw_api_key:
        headers["Authorization"] = f"Bearer {raw_api_key}"
    base_url = expand_env_vars(provider.base_url)
    resolved_url = normalize_provider_url(base_url)
    endpoint = f"{resolved_url}/chat/completions"
    upstream_model = target_model.upstream_id or target_model.name

    def judge_fn(request_text: str, answer: str, category: str = "general") -> Judgement:
        t0 = time.perf_counter()
        system_prompt = (
            "You are an expert AI quality inspector and adequacy judge. "
            "Your task is to objectively evaluate whether the assistant response fully, correctly, and accurately "
            "addresses the user request. Output STRICTLY a single JSON object with no markdown formatting or extra text."
        )
        user_prompt = (
            f"Task Category: {category}\n\n"
            f"[User Request]:\n{request_text[:4000]}\n\n"
            f"[Assistant Response to Evaluate]:\n{answer[:6000]}\n\n"
            "Criteria:\n"
            "- p_adequate: float from 0.0 to 1.0 (1.0 = completely correct & helpful, 0.0 = completely failed/wrong/empty)\n"
            "- failure: 'fine' if adequate (p_adequate >= 0.3), else 'wrong', 'incomplete', or 'off_topic'\n"
            "- reason: short concise explanation under 25 words in Chinese\n\n"
            "Format:\n"
            "{\n"
            '  "p_adequate": 0.95,\n'
            '  "failure": "fine",\n'
            '  "reason": "解答完整正确，符合全部指令要求"\n'
            "}"
        )
        payload = {
            "model": upstream_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.0,
            "max_tokens": 150,
        }
        try:
            with httpx.Client(timeout=25.0) as client:
                resp = client.post(endpoint, headers=headers, json=payload)
                lat = time.perf_counter() - t0
                if resp.status_code == 200:
                    data = resp.json()
                    choices = data.get("choices") or []
                    if choices:
                        raw_text = choices[0].get("message", {}).get("content", "").strip()
                        import re
                        m = re.search(r"\{[\s\S]*\}", raw_text)
                        content = m.group(0) if m else raw_text
                        parsed = json.loads(content)
                        p_adeq = float(parsed.get("p_adequate", 0.5))
                        p_adeq = max(0.0, min(1.0, p_adeq))
                        failure = str(parsed.get("failure") or "fine").lower()
                        if failure not in ("fine", "wrong", "incomplete", "off_topic"):
                            failure = "wrong" if p_adeq < 0.3 else "fine"
                        reason = str(parsed.get("reason") or "LLM 质检完成")
                        usage = data.get("usage") or {}
                        log.info("LLM Judge [%s] verdict: p_adequate=%.2f, failure=%s, reason=%s (took %.2fs)",
                                 target_model.name, p_adeq, failure, reason, lat)
                        return Judgement(
                            p_adequate=p_adeq,
                            latency_s=lat,
                            failed=False,
                            failure=failure,
                            model=target_model.name,
                            input_tokens=int(usage.get("prompt_tokens") or 0),
                            output_tokens=int(usage.get("completion_tokens") or 0)
                        )
                log.warning("LLM Judge call returned HTTP %s: %s", resp.status_code, resp.text[:200])
        except Exception as e:
            log.warning("LLM Judge call exception: %s", e)
        return Judgement(p_adequate=0.5, latency_s=time.perf_counter() - t0, failed=True)

    judge_fn.__name__ = f"llm_judge[{target_model.name}]"
    return judge_fn


def _find_default_llm_judge(config):
    models = getattr(config.catalog, "models", [])
    for pref in ("flash", "deepseek", "pro", "qwen", "llama", "gemma"):
        for m_info in models:
            if pref in m_info.name.lower():
                provider = config.providers.get(m_info.provider)
                if provider:
                    return create_llm_judge(m_info, provider)
    for m_info in models:
        provider = config.providers.get(m_info.provider)
        if provider:
            return create_llm_judge(m_info, provider)
    return None


def build_judge(config, raw_policy: dict, classifier=None):
    verify_cfg = raw_policy.get("verify") or {}
    if not verify_cfg.get("enabled", False):
        return None

    judge_choice = str(verify_cfg.get("judge_model") or "deepseek-v4-flash").strip()

    # 1. Local Laya engine
    if judge_choice.lower() in ("laya", "local"):
        clf = classifier or getattr(ar_server.router, "classifier", None) or jev.classifier_from_config(raw_policy)
        if hasattr(clf, "judge"):
            def safe_laya_judge(req: str, ans: str, category: str = "general") -> Judgement:
                try:
                    res = clf.judge(req, ans, category=category)
                    if not res.failed:
                        return res
                except Exception as ex:
                    log.warning("Laya local judge failed: %s; falling back to LLM judge", ex)
                fb = _find_default_llm_judge(config)
                if fb:
                    return fb(req, ans, category=category)
                return Judgement(p_adequate=0.5, latency_s=0.0, failed=True)
            safe_laya_judge.__name__ = "laya_judge[local]"
            return safe_laya_judge

    # 2. Remote Jev AI / TypeSafe System One Judge
    if judge_choice.lower() in ("jev", "typesafe", "hosted"):
        clf = classifier or getattr(ar_server.router, "classifier", None)
        if isinstance(clf, JevClassifier):
            def safe_active_jev_judge(req: str, ans: str, category: str = "general") -> Judgement:
                try:
                    res = clf.judge(req, ans, category=category)
                    if not res.failed:
                        return res
                except Exception as ex:
                    log.warning("Active Jev judge failed: %s; falling back to LLM judge", ex)
                fb = _find_default_llm_judge(config)
                if fb:
                    return fb(req, ans, category=category)
                return Judgement(p_adequate=0.5, latency_s=0.0, failed=True)
            safe_active_jev_judge.__name__ = f"jev_judge[{getattr(clf, 'model', 'remote')}]"
            return safe_active_jev_judge

        # 若当前分类器不是 Jev，依据配置独立构建 JevClassifier
        try:
            jev_inst = JevClassifier.from_config(raw_policy.get("classifier", {}))
            def standalone_jev_judge(req: str, ans: str, category: str = "general") -> Judgement:
                try:
                    res = jev_inst.judge(req, ans, category=category)
                    if not res.failed:
                        return res
                except Exception as ex:
                    log.warning("Standalone Jev judge failed: %s; falling back to LLM judge", ex)
                fb = _find_default_llm_judge(config)
                if fb:
                    return fb(req, ans, category=category)
                return Judgement(p_adequate=0.5, latency_s=0.0, failed=True)
            standalone_jev_judge.__name__ = f"jev_judge[{jev_inst.model}]"
            return standalone_jev_judge
        except Exception as exc:
            log.warning("Failed to initialize standalone Jev judge: %s", exc)

    # 3. Configured model in catalog
    models = getattr(config.catalog, "models", [])
    target_model = next((m for m in models if m.name == judge_choice), None)
    if target_model:
        provider = config.providers.get(target_model.provider)
        if provider:
            return create_llm_judge(target_model, provider)

    # 4. Fallback to Laya local engine if configured model was disabled or missing
    clf = classifier or getattr(ar_server.router, "classifier", None) or jev.classifier_from_config(raw_policy)
    if hasattr(clf, "judge"):
        log.warning("Judge model '%s' is disabled or missing; safely falling back to classifier engine", judge_choice)
        def fallback_laya_judge(req: str, ans: str, category: str = "general") -> Judgement:
            try:
                res = clf.judge(req, ans, category=category)
                if not res.failed:
                    return res
            except Exception as ex:
                log.warning("Fallback classifier judge failed: %s", ex)
            return Judgement(p_adequate=0.5, latency_s=0.0, failed=True)
        fallback_laya_judge.__name__ = "classifier_judge[fallback]"
        return fallback_laya_judge

    # 5. Remote TypeSafe Jev API
    if os.environ.get("TYPESAFE_API_KEY") or os.environ.get("JEV_AI_API_KEY"):
        return jev.judge

    # 6. Fallback to default LLM in catalog
    return _find_default_llm_judge(config)


def reload_router_system():
    """Reload configuration and re-initialize router instance."""
    raw = get_current_raw_config()
    loaded = for_http(load_config())
    ar_server.config = loaded
    
    # 优先复用现有本地分类器实例，避免重复占用 GPU 显存
    existing_clf = getattr(ar_server.router, "classifier", None)
    clf = existing_clf or jev.classifier_from_config(raw.get("policy", {}))
    
    judge_fn = build_judge(loaded, raw.get("policy", {}), classifier=clf)
    ar_server.router = Router(loaded, classifier=clf, judge=judge_fn)
    log.info("Router system reloaded with %d models, %d providers, judge=%s, verify.enabled=%s",
             len(loaded.catalog.models), len(loaded.providers),
             getattr(judge_fn, "__name__", str(judge_fn)),
             ar_server.router.verify.enabled)
    return ar_server.router


@app.on_event("startup")
async def on_startup():
    """Warm up Laya on Apple Silicon MPS on server startup in background."""
    log.info("Starting Auto-LLM-Router unified service on port 8765...")
    try:
        reload_router_system()
        clf = getattr(ar_server.router, "classifier", None)
        if isinstance(clf, LocalLayaClassifier):
            target_dev = clf._resolve_device() if hasattr(clf, "_resolve_device") else "auto"
            log.info("Pre-warming local Laya classifier on %s in background...", target_dev)
            loop = asyncio.get_event_loop()
            async def _bg_warmup():
                try:
                    await loop.run_in_executor(None, clf.warmup)
                except Exception as ex:
                    log.warning("Warmup failed: %s", ex)
            asyncio.create_task(_bg_warmup())
        elif isinstance(clf, JevClassifier):
            log.info("Active decision classifier: Jev AI (Endpoint: %s, Model: %s)", clf.systemone_url, clf.model)
    except Exception as exc:
        log.warning("Warmup warning: %s", exc)



# ---------------------------------------------------------------------------
# Status & System Info API
# ---------------------------------------------------------------------------
@app.get("/api/status")
async def get_system_status():
    cuda_ok = torch.cuda.is_available()
    mps_ok = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    
    cuda_device_name = None
    cuda_vram_mb = 0.0
    if cuda_ok:
        try:
            cuda_device_name = torch.cuda.get_device_name(0)
            cuda_vram_mb = round(torch.cuda.get_device_properties(0).total_memory / (1024 * 1024), 1)
        except Exception:
            cuda_device_name = "NVIDIA CUDA GPU"

    if cuda_ok:
        d_name = (cuda_device_name or "CUDA GPU").strip()
        if not d_name.upper().startswith("NVIDIA"):
            d_name = f"NVIDIA {d_name}"
        device_name = f"{d_name} (CUDA)"
        accelerator = "cuda"
    elif mps_ok:
        device_name = "Apple Silicon (Metal MPS)"
        accelerator = "mps"
    else:
        import platform
        device_name = f"CPU 多线程 ({platform.machine()})"
        accelerator = "cpu"
    
    cfg = get_current_raw_config()
    pol_cfg = cfg.get("policy", {})
    clf_cfg = pol_cfg.get("classifier", {})
    
    clf = getattr(ar_server.router, "classifier", None)
    actual_dev = getattr(clf, "actual_device", accelerator)
    
    # Process memory with multi-tier fallback (psutil -> Windows ctypes -> resource)
    rss_mb = 0.0
    try:
        import psutil
        rss_mb = round(psutil.Process().memory_info().rss / (1024 * 1024), 1)
    except Exception:
        if sys.platform == "win32":
            try:
                import ctypes
                from ctypes import wintypes
                class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
                    _fields_ = [
                        ("cb", wintypes.DWORD),
                        ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t),
                        ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t),
                        ("PeakPagefileUsage", ctypes.c_size_t),
                    ]
                counters = PROCESS_MEMORY_COUNTERS()
                counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
                handle = ctypes.windll.kernel32.GetCurrentProcess()
                if ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
                    rss_mb = round(counters.WorkingSetSize / (1024 * 1024), 1)
            except Exception:
                pass
        else:
            try:
                import resource
                rusage = resource.getrusage(resource.RUSAGE_SELF)
                divisor = (1024 * 1024) if sys.platform == "darwin" else 1024
                rss_mb = round(rusage.ru_maxrss / divisor, 1)
            except Exception:
                pass

    # Detailed CUDA VRAM tracking
    cuda_vram_used_mb = 0.0
    cuda_vram_free_mb = 0.0
    if cuda_ok:
        try:
            free_b, total_b = torch.cuda.mem_get_info(0)
            cuda_vram_mb = round(total_b / (1024 * 1024), 1)
            cuda_vram_free_mb = round(free_b / (1024 * 1024), 1)
            cuda_vram_used_mb = round((total_b - free_b) / (1024 * 1024), 1)
        except Exception:
            pass

    active_clf = getattr(ar_server.router, "classifier", None)
    configured_backend = str(clf_cfg.get("backend", "local")).lower()
    is_clef = isinstance(active_clf, ClefClassifier) or configured_backend in ("clef", "cloudflare", "cf")
    is_jev = (not is_clef) and (isinstance(active_clf, JevClassifier) or configured_backend in ("jev", "hosted"))

    if is_clef:
        clf_backend = "clef"
        clf_name = "Cloudflare Clef"
        clef_mod = getattr(active_clf, "model", None) or (clf_cfg.get("clef", {}) or {}).get("model") or (clf_cfg.get("cloudflare", {}) or {}).get("model") or DEFAULT_CLEF_MODEL
        clef_ep = getattr(active_clf, "endpoint", None) or "Cloudflare Workers AI"
        engine_display = f"Cloudflare Clef 极速决策模型 ({clef_mod.replace('@cf/cloudflare/', '')})"
        endpoint_display = clef_ep
    elif is_jev:
        clf_backend = "jev"
        clf_name = "Jev AI"
        jev_mod = getattr(active_clf, "model", None) or (clf_cfg.get("jev", {}) or {}).get("model") or "jev-latest"
        jev_url = getattr(active_clf, "base_url", None) or (clf_cfg.get("jev", {}) or {}).get("base_url") or "https://jev-ai.pro/api/v1/"
        engine_display = f"Jev AI 云端决策模型 ({jev_mod})"
        endpoint_display = jev_url
    elif configured_backend in ("local", "laya"):
        clf_backend = "laya"
        clf_name = "Laya"
        engine_display = f"Laya 本地模型 ({accelerator})"
        endpoint_display = "本地硬件运算"
    else:
        clf_backend = "heuristic"
        clf_name = "启发式规则"
        engine_display = "启发式静态规则"
        endpoint_display = "规则仲裁"

    return {
        "status": "online",
        "uptime_seconds": round(time.time() - START_TIME, 1),
        "device": actual_dev,
        "device_hardware": device_name,
        "accelerator_type": accelerator,
        "cuda_available": cuda_ok,
        "cuda_device_name": cuda_device_name,
        "cuda_vram_total_mb": cuda_vram_mb,
        "cuda_vram_used_mb": cuda_vram_used_mb,
        "cuda_vram_free_mb": cuda_vram_free_mb,
        "mps_available": mps_ok,
        "classifier_backend": clf_backend,
        "classifier_name": clf_name,
        "is_clef": is_clef,
        "is_jev": is_jev,
        "engine_display": engine_display,
        "endpoint_display": endpoint_display,
        "classifier_model": (clf_cfg.get("clef", {}) or {}).get("model", DEFAULT_CLEF_MODEL) if is_clef else ((clf_cfg.get("jev", {}) or {}).get("model", "jev-latest") if is_jev else clf_cfg.get("model", "convaiinnovations/laya")),
        "classifier_subfolder": clf_cfg.get("subfolder", "multilingual"),
        "clef_model": (clf_cfg.get("clef", {}) or {}).get("model") or (clf_cfg.get("cloudflare", {}) or {}).get("model") or DEFAULT_CLEF_MODEL,
        "jev_base_url": (clf_cfg.get("jev", {}) or {}).get("base_url") if isinstance(clf_cfg.get("jev"), dict) else clf_cfg.get("base_url", "https://jev-ai.pro/api/v1/"),
        "jev_model": (clf_cfg.get("jev", {}) or {}).get("model") if isinstance(clf_cfg.get("jev"), dict) else clf_cfg.get("model", "jev-latest"),
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
    """Return raw configuration with sensitive API keys masked for safety."""
    import copy
    cfg = copy.deepcopy(get_current_raw_config())
    for prov_name, prov in (cfg.get("providers") or {}).items():
        if isinstance(prov, dict) and "api_key" in prov:
            prov["api_key"] = mask_secret(prov["api_key"])
    # 敏感信息脱敏：Jev API Key 与 Cloudflare Clef Token
    clf = (cfg.get("policy") or {}).get("classifier") or {}
    if isinstance(clf.get("jev"), dict) and "api_key" in clf["jev"]:
        clf["jev"]["api_key"] = mask_secret(clf["jev"]["api_key"])
    if isinstance(clf.get("clef"), dict) and "api_token" in clf["clef"]:
        clf["clef"]["api_token"] = mask_secret(clf["clef"]["api_token"])
    if isinstance(clf.get("cloudflare"), dict) and "api_token" in clf["cloudflare"]:
        clf["cloudflare"]["api_token"] = mask_secret(clf["cloudflare"]["api_token"])
    if "api_key" in clf:
        clf["api_key"] = mask_secret(clf["api_key"])
    if "api_token" in clf:
        clf["api_token"] = mask_secret(clf["api_token"])
    return cfg


class ConfigUpdateRequest(BaseModel):
    providers: dict[str, Any]
    models: list[dict[str, Any]]
    policy: dict[str, Any]
    subscriptions: dict[str, Any] = Field(default_factory=dict)


@app.post("/api/config")
async def update_config(payload: ConfigUpdateRequest):
    try:
        raw_dict = payload.dict()
        existing_cfg = get_current_raw_config()
        # 若传入的 api_key 为脱敏字符串 (包含 ***)，自动恢复使用库中原有真实密钥
        for prov_name, prov in (raw_dict.get("providers") or {}).items():
            if isinstance(prov, dict):
                in_key = str(prov.get("api_key") or "")
                if "***" in in_key:
                    orig_prov = (existing_cfg.get("providers") or {}).get(prov_name, {})
                    prov["api_key"] = orig_prov.get("api_key", in_key)

        # 恢复 Jev API Key 与 Cloudflare Clef Token
        in_clf = (raw_dict.get("policy") or {}).get("classifier") or {}
        orig_clf = (existing_cfg.get("policy") or {}).get("classifier") or {}
        if isinstance(in_clf.get("jev"), dict):
            in_jev_key = str(in_clf["jev"].get("api_key") or "")
            if "***" in in_jev_key:
                orig_jev = orig_clf.get("jev") if isinstance(orig_clf.get("jev"), dict) else {}
                in_clf["jev"]["api_key"] = orig_jev.get("api_key", in_jev_key)
        if isinstance(in_clf.get("clef"), dict):
            in_clef_token = str(in_clf["clef"].get("api_token") or "")
            if "***" in in_clef_token:
                orig_clef = orig_clf.get("clef") if isinstance(orig_clf.get("clef"), dict) else {}
                in_clf["clef"]["api_token"] = orig_clef.get("api_token", in_clef_token)
        if isinstance(in_clf.get("cloudflare"), dict):
            in_cf_token = str(in_clf["cloudflare"].get("api_token") or "")
            if "***" in in_cf_token:
                orig_cf = orig_clf.get("cloudflare") if isinstance(orig_clf.get("cloudflare"), dict) else {}
                in_clf["cloudflare"]["api_token"] = orig_cf.get("api_token", in_cf_token)
        if "***" in str(in_clf.get("api_key") or ""):
            in_clf["api_key"] = orig_clf.get("api_key", in_clf.get("api_key"))
        if "***" in str(in_clf.get("api_token") or ""):
            in_clf["api_token"] = orig_clf.get("api_token", in_clf.get("api_token"))

        # 双轨自动隔离保护：识别明文凭据，自动写入本地 .env 并转化为占位符
        sanitized_cfg, extracted_secrets = sanitize_config_for_public_repo(raw_dict)
        for var_name, secret_val in extracted_secrets.items():
            update_local_env_file(var_name, secret_val)

        # 公共文件始终保存脱敏后的占位符版本，确保任何 git commit 绝无泄露风险
        save_config(sanitized_cfg, path=Path("config/router_config.json"))

        # 本地私有文件同步保存生效版本（无论直接从 .env 解析还是 local.json 读取均完美兼容）
        local_p = Path("config/router_config.local.json")
        if extracted_secrets or local_p.exists():
            save_config(raw_dict, path=local_p)
            saved_filename = "router_config.local.json (私有) & .env"
        else:
            saved_filename = "router_config.json"

        reload_router_system()
        msg = "配置已保存并实时生效（🔐 凭据已安全隔离至本地 .env，公开配置仅引用占位符）" if extracted_secrets else "配置已保存并实时生效"
        return {"status": "ok", "message": msg, "saved_file": saved_filename}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"保存配置失败: {exc}")


# ---------------------------------------------------------------------------
# Jev Server Endpoint Health & Connectivity Testing API
# ---------------------------------------------------------------------------
class JevTestRequest(BaseModel):
    base_url: str | None = None
    api_key: str | None = None
    model: str | None = None


@app.post("/api/jev/test")
async def test_jev_endpoint(req: JevTestRequest):
    """Test connection to user's Jev endpoint without running inference."""
    raw_url = expand_env_vars(req.base_url) if req.base_url else None
    raw_key = expand_env_vars(req.api_key) if req.api_key else None

    # 如果传入的 key 是掩码值或未输入，尝试从已存配置或环境变量中读取真实密钥
    if not raw_key or "***" in raw_key:
        current_cfg = get_current_raw_config()
        clf_cfg = (current_cfg.get("policy") or {}).get("classifier") or {}
        jev_cfg = clf_cfg.get("jev") if isinstance(clf_cfg.get("jev"), dict) else {}
        raw_key = jev_cfg.get("api_key") or clf_cfg.get("api_key") or os.environ.get("JEV_AI_API_KEY") or os.environ.get("TYPESAFE_API_KEY") or raw_key

    try:
        client = JevClassifier(base_url=raw_url, api_key=raw_key, model=req.model or "jev-latest")
        loop = asyncio.get_event_loop()
        res = await loop.run_in_executor(None, client.test_connection)
        return res
    except Exception as exc:
        return {"status": "error", "error": f"测试执行异常: {exc}"}


class JevDecisionTestRequest(BaseModel):
    base_url: str | None = None
    api_key: str | None = None
    model: str | None = None
    prompt: str | None = None
    test_judge: bool = True


@app.post("/api/jev/test-decision")
async def test_jev_decision(req: JevDecisionTestRequest):
    """Run a live decision/classification and optional adequacy test using Jev."""
    raw_url = expand_env_vars(req.base_url) if req.base_url else None
    raw_key = expand_env_vars(req.api_key) if req.api_key else None

    # 如果传入的 key 是掩码值或未输入，尝试从已存配置或环境变量中读取真实密钥
    if not raw_key or "***" in raw_key:
        current_cfg = get_current_raw_config()
        clf_cfg = (current_cfg.get("policy") or {}).get("classifier") or {}
        jev_cfg = clf_cfg.get("jev") if isinstance(clf_cfg.get("jev"), dict) else {}
        raw_key = jev_cfg.get("api_key") or clf_cfg.get("api_key") or os.environ.get("JEV_AI_API_KEY") or os.environ.get("TYPESAFE_API_KEY") or raw_key

    client = JevClassifier(base_url=raw_url, api_key=raw_key, model=req.model or "jev-latest")
    prompt = (req.prompt or "帮我写一个Python快速排序算法，要求带注释并处理边界情况").strip()

    loop = asyncio.get_event_loop()
    def _run():
        if not client.api_key:
            return {"status": "error", "error": "未配置 Jev API Key，请在上方输入框填入或设置环境变量 JEV_AI_API_KEY"}

        t0 = time.perf_counter()
        try:
            clf_res = client(prompt)
            latency_clf_ms = round((time.perf_counter() - t0) * 1000.0, 1)
            if clf_res.failed:
                return {
                    "status": "error",
                    "error": "Jev 分类调用失败，请检查服务器地址是否可达、API Key 是否有效或账户余额是否充足",
                    "url": client.systemone_url,
                }

            judge_res = None
            latency_judge_ms = 0.0
            if req.test_judge:
                t1 = time.perf_counter()
                sample_ans = "def quicksort(arr): return arr if len(arr) <= 1 else quicksort([x for x in arr[1:] if x < arr[0]]) + [arr[0]] + quicksort([x for x in arr[1:] if x >= arr[0]])"
                judge_res = client.judge(prompt, sample_ans, category=clf_res.category)
                latency_judge_ms = round((time.perf_counter() - t1) * 1000.0, 1)

            return {
                "status": "ok",
                "message": "Jev 决策测试成功，模型已正常生效并完成分析！",
                "classification": {
                    "category": clf_res.category,
                    "category_confidence": round(clf_res.category_confidence, 3),
                    "difficulty": round(clf_res.difficulty, 3),
                    "difficulty_confidence": round(clf_res.difficulty_confidence, 3),
                    "needs_tools": round(clf_res.needs_tools, 3),
                    "needs_vision": round(clf_res.needs_vision, 3),
                    "needs_long_context": round(clf_res.needs_long_context, 3),
                    "stakes": round(clf_res.stakes, 3),
                    "latency_ms": latency_clf_ms,
                    "model": clf_res.model or client.model,
                    "input_tokens": clf_res.input_tokens,
                    "output_tokens": clf_res.output_tokens,
                },
                "judge": {
                    "p_adequate": round(judge_res.p_adequate, 3) if judge_res else None,
                    "failure": judge_res.failure if judge_res else None,
                    "latency_ms": latency_judge_ms,
                } if judge_res else None,
                "latency_total_ms": round(latency_clf_ms + latency_judge_ms, 1),
                "endpoint": client.systemone_url,
            }
        except Exception as exc:
            return {"status": "error", "error": f"Jev 决策测试异常: {exc}", "url": client.systemone_url}

    return await loop.run_in_executor(None, _run)


# ---------------------------------------------------------------------------
# Cloudflare Clef (Workers AI) Testing API
# ---------------------------------------------------------------------------
class ClefTestRequest(BaseModel):
    account_id: str | None = None
    api_token: str | None = None
    model: str | None = None
    base_url: str | None = None


@app.post("/api/clef/test")
async def test_clef_endpoint(req: ClefTestRequest):
    """Test connection to Cloudflare Workers AI Clef endpoint without running inference."""
    raw_acc = expand_env_vars(req.account_id) if req.account_id else None
    raw_token = expand_env_vars(req.api_token) if req.api_token else None
    raw_url = expand_env_vars(req.base_url) if req.base_url else None

    # 如果传入的 token 是掩码值或未输入，尝试从已存配置或环境变量中读取真实密钥
    if not raw_token or "***" in raw_token:
        current_cfg = get_current_raw_config()
        clf_cfg = (current_cfg.get("policy") or {}).get("classifier") or {}
        clef_cfg = clf_cfg.get("clef") if isinstance(clf_cfg.get("clef"), dict) else (clf_cfg.get("cloudflare") if isinstance(clf_cfg.get("cloudflare"), dict) else {})
        raw_token = clef_cfg.get("api_token") or clef_cfg.get("api_key") or os.environ.get("CLOUDFLARE_API_TOKEN") or os.environ.get("CLOUDFLARE_TOKEN") or raw_token

    if not raw_acc:
        current_cfg = get_current_raw_config()
        clf_cfg = (current_cfg.get("policy") or {}).get("classifier") or {}
        clef_cfg = clf_cfg.get("clef") if isinstance(clf_cfg.get("clef"), dict) else (clf_cfg.get("cloudflare") if isinstance(clf_cfg.get("cloudflare"), dict) else {})
        raw_acc = clef_cfg.get("account_id") or os.environ.get("CLOUDFLARE_ACCOUNT_ID") or raw_acc

    try:
        client = ClefClassifier(account_id=raw_acc, api_token=raw_token, model=req.model or DEFAULT_CLEF_MODEL, base_url=raw_url)
        loop = asyncio.get_event_loop()
        res = await loop.run_in_executor(None, client.test_connection)
        return res
    except Exception as exc:
        return {"status": "error", "error": f"Cloudflare 测试异常: {exc}"}


class ClefDecisionTestRequest(BaseModel):
    account_id: str | None = None
    api_token: str | None = None
    model: str | None = None
    base_url: str | None = None
    prompt: str | None = None
    test_judge: bool = True


@app.post("/api/clef/test-decision")
async def test_clef_decision(req: ClefDecisionTestRequest):
    """Run a live decision/classification and optional adequacy test using Cloudflare Clef."""
    raw_acc = expand_env_vars(req.account_id) if req.account_id else None
    raw_token = expand_env_vars(req.api_token) if req.api_token else None
    raw_url = expand_env_vars(req.base_url) if req.base_url else None

    if not raw_token or "***" in raw_token:
        current_cfg = get_current_raw_config()
        clf_cfg = (current_cfg.get("policy") or {}).get("classifier") or {}
        clef_cfg = clf_cfg.get("clef") if isinstance(clf_cfg.get("clef"), dict) else (clf_cfg.get("cloudflare") if isinstance(clf_cfg.get("cloudflare"), dict) else {})
        raw_token = clef_cfg.get("api_token") or clef_cfg.get("api_key") or os.environ.get("CLOUDFLARE_API_TOKEN") or os.environ.get("CLOUDFLARE_TOKEN") or raw_token

    if not raw_acc:
        current_cfg = get_current_raw_config()
        clf_cfg = (current_cfg.get("policy") or {}).get("classifier") or {}
        clef_cfg = clf_cfg.get("clef") if isinstance(clf_cfg.get("clef"), dict) else (clf_cfg.get("cloudflare") if isinstance(clf_cfg.get("cloudflare"), dict) else {})
        raw_acc = clef_cfg.get("account_id") or os.environ.get("CLOUDFLARE_ACCOUNT_ID") or raw_acc

    client = ClefClassifier(account_id=raw_acc, api_token=raw_token, model=req.model or DEFAULT_CLEF_MODEL, base_url=raw_url)
    prompt = (req.prompt or "帮我写一个Python快速排序算法，要求带注释并处理边界情况").strip()

    loop = asyncio.get_event_loop()
    def _run():
        if not client.api_token:
            return {"status": "error", "error": "未配置 Cloudflare API Token，请在上方输入框填入或设置环境变量 CLOUDFLARE_API_TOKEN"}
        if not client.account_id and not client.base_url:
            return {"status": "error", "error": "未配置 Cloudflare Account ID，请填入或设置环境变量 CLOUDFLARE_ACCOUNT_ID"}

        t0 = time.perf_counter()
        try:
            clf_res = client(prompt)
            latency_clf_ms = round((time.perf_counter() - t0) * 1000.0, 1)
            if clf_res.failed:
                return {
                    "status": "error",
                    "error": "Cloudflare Clef 分类调用失败，请检查 Account ID 与 API Token 是否正确、端点是否可达或是否超出配额",
                    "endpoint": client.endpoint,
                }

            judge_res = None
            latency_judge_ms = 0.0
            if req.test_judge:
                t1 = time.perf_counter()
                sample_ans = "def quicksort(arr): return arr if len(arr) <= 1 else quicksort([x for x in arr[1:] if x < arr[0]]) + [arr[0]] + quicksort([x for x in arr[1:] if x >= arr[0]])"
                judge_res = client.judge(prompt, sample_ans, category=clf_res.category)
                latency_judge_ms = round((time.perf_counter() - t1) * 1000.0, 1)

            return {
                "status": "ok",
                "message": f"Cloudflare Clef ({client.model}) 决策测试成功，已极速完成分析！",
                "classification": {
                    "category": clf_res.category,
                    "category_confidence": round(clf_res.category_confidence, 3),
                    "difficulty": round(clf_res.difficulty, 3),
                    "difficulty_confidence": round(clf_res.difficulty_confidence, 3),
                    "needs_tools": round(clf_res.needs_tools, 3),
                    "needs_vision": round(clf_res.needs_vision, 3),
                    "needs_long_context": round(clf_res.needs_long_context, 3),
                    "stakes": round(clf_res.stakes, 3),
                    "latency_ms": latency_clf_ms,
                    "model": clf_res.model or client.model,
                    "input_tokens": clf_res.input_tokens,
                    "output_tokens": clf_res.output_tokens,
                },
                "judge": {
                    "p_adequate": round(judge_res.p_adequate, 3) if judge_res else None,
                    "failure": judge_res.failure if judge_res else None,
                    "latency_ms": latency_judge_ms,
                } if judge_res else None,
                "latency_total_ms": round(latency_clf_ms + latency_judge_ms, 1),
                "endpoint": client.endpoint,
            }
        except Exception as exc:
            return {"status": "error", "error": f"Cloudflare Clef 决策测试异常: {exc}", "endpoint": client.endpoint}

    return await loop.run_in_executor(None, _run)


# ---------------------------------------------------------------------------
# Provider Testing API (Supports local LM Studio & Cloud APIs)
# ---------------------------------------------------------------------------
class ProviderTestRequest(BaseModel):
    base_url: str
    api_key: str | None = None
    model: str | None = None


@app.post("/api/provider/test")
async def test_provider_endpoint(req: ProviderTestRequest):
    raw_base_url = expand_env_vars(req.base_url) or req.base_url or ""
    if not raw_base_url:
        return {"status": "error", "error": "缺少端点 Base URL"}
    base_url = normalize_provider_url(raw_base_url)
    api_key = expand_env_vars(req.api_key) or req.api_key
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    t0 = time.perf_counter()
    async with httpx.AsyncClient(timeout=10.0) as client:
        # Candidate URLs to try: normalized URL (with /v1 by default for OpenAI), and non-v1 alternative as fallback
        urls_to_try = [base_url]
        if base_url.endswith("/v1"):
            urls_to_try.append(base_url[:-3].rstrip("/"))
        elif not re.search(r"/v\d+$", base_url):
            urls_to_try.append(f"{base_url}/v1")

        # First attempt: GET /models
        for try_url in urls_to_try:
            try:
                r = await client.get(f"{try_url}/models", headers=headers)
                lat_ms = round((time.perf_counter() - t0) * 1000.0, 1)
                if r.status_code == 200:
                    data = r.json()
                    models_list = [m.get("id") for m in data.get("data", []) if isinstance(m, dict)]
                    return {
                        "status": "ok",
                        "latency_ms": lat_ms,
                        "available_models": models_list,
                        "resolved_base_url": try_url,
                        "message": f"连接成功！检测到 {len(models_list)} 个可用模型",
                    }
            except Exception:
                pass

        # Second attempt: simple chat completion probe
        for try_url in urls_to_try:
            try:
                probe_body = {
                    "model": req.model or "default",
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 5,
                }
                r = await client.post(f"{try_url}/chat/completions", json=probe_body, headers=headers)
                lat_ms = round((time.perf_counter() - t0) * 1000.0, 1)
                if r.status_code == 200:
                    return {
                        "status": "ok",
                        "latency_ms": lat_ms,
                        "available_models": [req.model or "default"],
                        "resolved_base_url": try_url,
                        "message": "端点响应正常！",
                    }
                if r.status_code not in (404, 405):
                    return {
                        "status": "error",
                        "latency_ms": lat_ms,
                        "resolved_base_url": try_url,
                        "error": f"HTTP {r.status_code}: {r.text[:200]}",
                    }
            except Exception:
                pass

        lat_ms = round((time.perf_counter() - t0) * 1000.0, 1)
        return {"status": "error", "latency_ms": lat_ms, "resolved_base_url": base_url, "error": f"无法连接提供商端点 ({base_url})"}


class ProviderModelsRequest(BaseModel):
    base_url: str
    api_key: str | None = None


@app.post("/api/provider/models")
async def fetch_provider_models(req: ProviderModelsRequest):
    """Fetch complete list of available models from an OpenAI-compatible provider."""
    t0 = time.perf_counter()
    try:
        raw_base_url = expand_env_vars(req.base_url) or req.base_url or ""
        if not raw_base_url:
            return {"status": "error", "error": "缺少端点 Base URL"}
        base_url = normalize_provider_url(raw_base_url)
        api_key = expand_env_vars(req.api_key) or req.api_key

        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        async with httpx.AsyncClient(timeout=10.0) as client:
            urls_to_try = [base_url]
            if base_url.endswith("/v1"):
                urls_to_try.append(base_url[:-3].rstrip("/"))
            elif not re.search(r"/v\d+$", base_url):
                urls_to_try.append(f"{base_url}/v1")

            r = None
            success_url = base_url
            for try_url in urls_to_try:
                try:
                    res = await client.get(f"{try_url}/models", headers=headers)
                    if res.status_code == 200:
                        r = res
                        success_url = try_url
                        break
                    elif r is None:
                        r = res
                except Exception:
                    pass

            lat_ms = round((time.perf_counter() - t0) * 1000.0, 1)
            if r is not None and r.status_code == 200:
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
                    "resolved_base_url": success_url,
                    "message": f"成功获取到 {len(models)} 个可用模型",
                }
            err_msg = f"上游接口返回 HTTP {r.status_code}: {r.text[:200]}" if r is not None else f"无法连接提供商端点 ({base_url})"
            return {
                "status": "error",
                "latency_ms": lat_ms,
                "resolved_base_url": base_url,
                "error": err_msg,
            }
    except Exception as exc:
        log.error("遍历端点可用模型失败: %s", exc, exc_info=True)
        return {"status": "error", "error": f"获取模型异常: {str(exc)}"}


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
    prov_api = "openai"

    if not base_url and req.provider_id:
        cfg = get_current_raw_config()
        prov_info = cfg.get("providers", {}).get(req.provider_id)
        if prov_info:
            base_url = prov_info.get("base_url")
            api_key = prov_info.get("api_key")
            prov_api = prov_info.get("api", "openai")

    base_url = expand_env_vars(base_url) or base_url
    api_key = expand_env_vars(api_key) or api_key

    if not base_url:
        raise HTTPException(status_code=400, detail="缺少上游端点 base_url 或有效的 provider_id")

    base_url = normalize_provider_url(base_url, api=prov_api)
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    # 读取模型超时配置或使用默认 60s
    model_timeout = 60.0
    cfg = get_current_raw_config()
    for m in cfg.get("models", []):
        if (m.get("upstream_id") == req.upstream_id or m.get("name") == req.upstream_id) and m.get("timeout_seconds") is not None:
            try:
                model_timeout = float(m["timeout_seconds"])
            except (ValueError, TypeError):
                pass
            break
    if model_timeout < 10.0:
        model_timeout = 60.0

    t0 = time.perf_counter()
    async with httpx.AsyncClient(timeout=model_timeout) as client:
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

            # If 404 or 405, fallback to alternate URL (e.g. without /v1 or with /v1)
            if r.status_code in (404, 405):
                alt_base = ""
                if base_url.endswith("/v1"):
                    alt_base = base_url[:-3].rstrip("/")
                elif not re.search(r"/v\d+$", base_url):
                    alt_base = f"{base_url}/v1"
                if alt_base:
                    try:
                        r_alt = await client.post(f"{alt_base}/chat/completions", json=body, headers=headers)
                        if r_alt.status_code == 200:
                            r = r_alt
                            base_url = alt_base
                    except Exception:
                        pass

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
                    "resolved_base_url": base_url,
                    "reply_snippet": reply[:400],
                    "usage": usage,
                    "message": "模型可用且响应正常！",
                }
            return {
                "status": "error",
                "latency_ms": lat_ms,
                "model": req.upstream_id,
                "resolved_base_url": base_url,
                "error": f"HTTP {r.status_code}: {r.text[:300]}",
            }
        except Exception as exc:
            lat_ms = round((time.perf_counter() - t0) * 1000.0, 1)
            return {
                "status": "error",
                "latency_ms": lat_ms,
                "model": req.upstream_id,
                "resolved_base_url": base_url,
                "error": f"调用失败: {exc}",
            }



# ---------------------------------------------------------------------------
# Dynamic Decision-Model Role Evaluation & Dispatch Baselines API
# ---------------------------------------------------------------------------
class EvaluateRolesRequest(BaseModel):
    classifier_override: dict[str, Any] | None = None


class ApplyRolesRequest(BaseModel):
    baselines: dict[str, str]
    model_roles: dict[str, list[str]] = Field(default_factory=dict)
    role_reasons: dict[str, str] = Field(default_factory=dict)


@app.post("/api/models/evaluate-roles")
async def evaluate_model_roles(req: EvaluateRolesRequest):
    """Dynamically evaluate active models catalog using current decision engine (Jev / Laya),
    recommending optimal role mappings with multi-dimensional scoring and detailed reasoning.
    Allows a single high-capability model to take multiple roles (e.g. frontier + fallback)."""
    raw_cfg = get_current_raw_config()
    policy = raw_cfg.get("policy") or {}
    clf_cfg = req.classifier_override or policy.get("classifier") or {}
    backend = str(clf_cfg.get("backend") or "local").lower()

    # 1. 识别当前生效的决策模型大脑
    engine_name = "本地 Laya 神经分类器"
    engine_type = "local"
    engine_model = str(clf_cfg.get("model") or "convaiinnovations/laya")
    is_jev = backend in ("jev", "hosted", "remote") or bool(clf_cfg.get("jev"))
    
    jev_key = None
    if is_jev:
        engine_name = "Jev AI 云端决策大模型"
        engine_type = "jev"
        jev_info = clf_cfg.get("jev") if isinstance(clf_cfg.get("jev"), dict) else {}
        engine_model = jev_info.get("model") or clf_cfg.get("model") or "jev-latest"
        jev_key = jev_info.get("api_key") or clf_cfg.get("api_key") or os.environ.get("JEV_AI_API_KEY") or os.environ.get("TYPESAFE_API_KEY")

    # 2. 收集启用的候选模型
    all_models = raw_cfg.get("models") or []
    enabled_models = [m for m in all_models if not m.get("launch_only", False) and m.get("enabled") is not False]
    if not enabled_models:
        raise HTTPException(status_code=400, detail="当前模型目录中没有启用的可用模型，请至少启用一个模型后再执行评判。")

    t0 = time.perf_counter()
    decision_probe_ms = 0.0

    # 3. 若为 Jev 且配置了 Key，尝试执行微探针记录延迟
    if is_jev and jev_key and "***" not in str(jev_key):
        try:
            from auto_router.jev import JevClassifier
            jev_client = JevClassifier.from_config(clf_cfg)
            res = await asyncio.get_event_loop().run_in_executor(None, jev_client.test_connection)
            if res.get("status") == "ok":
                decision_probe_ms = float(res.get("latency_ms") or 0.0)
        except Exception as ex:
            log.warning("Jev decision ping probe warning: %s", ex)

    # 4. 全景多维评判与打分算法
    model_scores = {}
    analysis_list = []

    high_end_keys = ["r1", "pro", "max", "opus", "sonnet", "o1", "o3", "gpt-4", "v4-pro", "70b", "72b", "deepseek-v3", "deepseek-r1"]
    economy_keys = ["flash", "mini", "turbo", "haiku", "8b", "7b", "lite", "small", "speed", "v4-flash"]

    prices_list = []
    for m in enabled_models:
        p = m.get("prices") or {}
        p_in = float(p.get("input", 0.0))
        p_out = float(p.get("output", 0.0))
        prices_list.append(p_in + p_out)
    max_price = max(prices_list) if prices_list else 1.0

    for m in enabled_models:
        name = m.get("name", "unknown")
        upstream = str(m.get("upstream_id") or name).lower()
        provider = str(m.get("provider", "generic")).lower()
        p = m.get("prices") or {}
        p_in = float(p.get("input", 0.0))
        p_out = float(p.get("output", 0.0))
        tot_price = p_in + p_out
        is_free = bool(m.get("free")) or tot_price == 0.0 or provider in ("lm-studio", "ollama", "local")
        ctx = int(m.get("context_tokens") or 128000)
        cap = m.get("capability") or {}
        avg_cap = sum(float(v) for v in cap.values()) / max(len(cap), 1) if cap else 72.0
        code_cap = float(cap.get("coding") or avg_cap)
        reason_cap = float(cap.get("reasoning") or avg_cap)

        is_high_signal = any(k in name.lower() or k in upstream for k in high_end_keys)
        is_econ_signal = any(k in name.lower() or k in upstream for k in economy_keys)

        # -- Frontier Score (0 - 100) --
        s_frontier = 50.0 + (avg_cap - 70.0) * 1.2 + (code_cap - 70.0) * 0.8 + (reason_cap - 70.0) * 0.8
        if is_high_signal:
            s_frontier += 16.0
        if max_price > 0 and tot_price >= max_price * 0.7:
            s_frontier += 12.0
        if ctx >= 128000:
            s_frontier += 6.0
        elif ctx >= 64000:
            s_frontier += 3.0
        if is_free and not is_high_signal:
            s_frontier -= 15.0
        s_frontier = max(20.0, min(99.0, round(s_frontier, 1)))

        # -- Economy Score (0 - 100) --
        if is_free:
            s_economy = 80.0 + (avg_cap - 70.0) * 0.4
        else:
            price_factor = max(0.1, tot_price)
            s_economy = 75.0 + (avg_cap - 70.0) * 0.8 - math.log10(price_factor + 0.5) * 28.0
            if is_econ_signal:
                s_economy += 14.0
            if is_high_signal:
                s_economy -= 18.0
        s_economy = max(10.0, min(99.0, round(s_economy, 1)))

        # -- Judge Score (0 - 100) --
        s_judge = 45.0 + (avg_cap - 70.0) * 1.0 + (reason_cap - 70.0) * 1.5
        if is_high_signal:
            s_judge += 12.0
        if provider in ("lm-studio", "ollama", "local"):
            s_judge -= 10.0
        s_judge = max(20.0, min(98.0, round(s_judge, 1)))

        # -- Fallback Score (0 - 100) --
        s_fallback = 48.0 + (avg_cap - 70.0) * 1.0 + (min(ctx, 131072) / 131072.0) * 18.0
        if is_high_signal:
            s_fallback += 14.0
        if s_frontier >= 85.0:
            s_fallback += 10.0
        s_fallback = max(20.0, min(99.0, round(s_fallback, 1)))

        model_scores[name] = {
            "frontier": s_frontier,
            "economy": s_economy,
            "judge": s_judge,
            "fallback": s_fallback,
            "is_free": is_free,
            "tot_price": tot_price,
            "ctx": ctx,
            "code_cap": code_cap,
            "reason_cap": reason_cap,
            "avg_cap": avg_cap,
        }

    # 5. 综合评选最优角色归属（支持单模型身兼多角色）
    best_frontier_name = max(enabled_models, key=lambda m: model_scores[m["name"]]["frontier"])["name"]

    paid_models = [m for m in enabled_models if not model_scores[m["name"]]["is_free"]]
    if paid_models:
        best_economy_name = max(paid_models, key=lambda m: model_scores[m["name"]]["economy"])["name"]
    else:
        best_economy_name = max(enabled_models, key=lambda m: model_scores[m["name"]]["economy"])["name"]

    if is_jev:
        best_judge_name = "jev"
    else:
        best_judge_name = max(enabled_models, key=lambda m: model_scores[m["name"]]["judge"])["name"]

    best_fallback_name = max(enabled_models, key=lambda m: model_scores[m["name"]]["fallback"])["name"]
    free_names = [m["name"] for m in enabled_models if model_scores[m["name"]]["is_free"]]

    # 6. 生成角色理由与 AI 综合评述
    f_info = model_scores[best_frontier_name]
    e_info = model_scores[best_economy_name]
    fb_info = model_scores[best_fallback_name]

    frontier_reason = f"高阶综合得分 {f_info['frontier']} 分。代码/推理能力评分达 {f_info['code_cap']}/{f_info['reason_cap']}，上下文窗口 {f_info['ctx'] // 1000}k，能够胜任高难、长链条与高价值决策，作为基线可衡量路由优化成效。"
    economy_reason = f"能效比得分 {e_info['economy']} 分。计费单价 ¥{e_info['tot_price']}/1M，综合能力 {e_info['avg_cap']} 分，在常规问答、批量处理与轻量化调用中展现出卓越的成本效益比。"
    
    if best_judge_name == "jev":
        judge_reason = f"采用当前在线激活的 Jev AI 云端决策大模型（{engine_model}）担任质检验收官。无本地显存开销，提供极度客观、校准的通过概率与失误分类判断。"
    else:
        j_info = model_scores[best_judge_name]
        judge_reason = f"逻辑审校评分 {j_info['judge']} 分。推理严谨度 {j_info['reason_cap']} 分，具备优良的指令遵循度，可独立对生成结果进行质检验收。"

    if best_fallback_name == best_frontier_name:
        fallback_reason = f"身兼主力旗舰与降级兜底双重职责。凭借其 {fb_info['ctx'] // 1000}k 的充沛上下文容量与顶尖鲁棒性，在发生超时或质量退化时提供绝对可靠的最终安全护栏。"
    else:
        fallback_reason = f"容错稳定性评分 {fb_info['fallback']} 分。上下文容量 {fb_info['ctx'] // 1000}k，具备优良的可用性与容灾承载能力，保障系统服务平稳降级。"

    role_assignments: dict[str, list[str]] = {m["name"]: [] for m in enabled_models}
    role_reasons: dict[str, str] = {}

    role_assignments[best_frontier_name].append("frontier")
    role_reasons[best_frontier_name] = frontier_reason

    role_assignments[best_economy_name].append("economy")
    if best_economy_name not in role_reasons or best_economy_name != best_frontier_name:
        role_reasons[best_economy_name] = economy_reason

    if best_judge_name != "jev" and best_judge_name in role_assignments:
        role_assignments[best_judge_name].append("judge")
        if best_judge_name not in role_reasons:
            role_reasons[best_judge_name] = judge_reason

    role_assignments[best_fallback_name].append("fallback")
    if best_fallback_name == best_frontier_name:
        role_reasons[best_fallback_name] = f"{frontier_reason} 同时兼任高可用故障降级兜底。"
    elif best_fallback_name not in role_reasons:
        role_reasons[best_fallback_name] = fallback_reason

    for fn in free_names:
        if fn in role_assignments and "free" not in role_assignments[fn]:
            role_assignments[fn].append("free")

    for m in enabled_models:
        m_name = m["name"]
        assigned = role_assignments.get(m_name, [])
        if not assigned:
            assigned = ["balanced"]
        analysis_list.append({
            "name": m_name,
            "provider": m.get("provider", "generic"),
            "assigned_roles": assigned,
            "scores": {
                "frontier": model_scores[m_name]["frontier"],
                "economy": model_scores[m_name]["economy"],
                "judge": model_scores[m_name]["judge"],
                "fallback": model_scores[m_name]["fallback"],
            },
            "reasoning": role_reasons.get(m_name, "综合性价比良好，作为通用动态调度候选模型。"),
        })

    eval_latency_ms = round((time.perf_counter() - t0) * 1000.0, 1)

    summary_text = (
        f"调度决策大脑（{engine_name}，代号 {engine_model}）对当前生效的 {len(enabled_models)} 个候选模型进行了全景多维评判。"
        f"在多维权衡中，推荐「{best_frontier_name}」承担主力旗舰基准（Baseline）；"
        f"推荐「{best_economy_name}」负责经济低成本调度，以大幅削减高频请求开支；"
        f"质检验收由「{'Jev AI 云端裁决模型' if best_judge_name == 'jev' else best_judge_name}」专职裁量；"
        f"高可用故障兜底由「{best_fallback_name}」护航"
        f"{'（实现旗舰与兜底身兼多职，兼顾架构精简与高容灾度）' if best_fallback_name == best_frontier_name else ''}。"
    )

    return {
        "status": "ok",
        "engine": {
            "name": engine_name,
            "type": engine_type,
            "model": engine_model,
            "latency_ms": eval_latency_ms,
            "decision_probe_ms": decision_probe_ms,
        },
        "summary_reasoning": summary_text,
        "recommendations": {
            "frontier": {"model": best_frontier_name, "reason": frontier_reason},
            "economy": {"model": best_economy_name, "reason": economy_reason},
            "judge": {"model": best_judge_name, "reason": judge_reason},
            "fallback": {"model": best_fallback_name, "reason": fallback_reason},
            "free": free_names,
        },
        "role_assignments": role_assignments,
        "role_reasons": role_reasons,
        "model_analysis": analysis_list,
        "all_enabled_models": [m["name"] for m in enabled_models],
    }


@app.post("/api/models/apply-roles")
async def apply_model_roles(req: ApplyRolesRequest):
    """Persist evaluated or user-customized baseline roles into policy.baselines and model metadata,
    then automatically trigger hot-reload of router system."""
    try:
        raw_cfg = get_current_raw_config()
        if not raw_cfg.get("policy"):
            raw_cfg["policy"] = {}
        
        # 1. 写入基线角色映射
        raw_cfg["policy"]["baselines"] = {
            "frontier": req.baselines.get("frontier", ""),
            "economy": req.baselines.get("economy", ""),
            "judge": req.baselines.get("judge", ""),
            "fallback": req.baselines.get("fallback", ""),
        }

        # 2. 同步写入 policy.verify.judge_model
        judge_val = req.baselines.get("judge")
        if judge_val:
            if not raw_cfg["policy"].get("verify"):
                raw_cfg["policy"]["verify"] = {}
            raw_cfg["policy"]["verify"]["judge_model"] = judge_val

        # 3. 沉淀至各模型的 roles 与 role_reason
        for m in raw_cfg.get("models", []):
            m_name = m.get("name")
            if m_name in req.model_roles:
                m["roles"] = req.model_roles[m_name]
            if m_name in req.role_reasons:
                m["role_reason"] = req.role_reasons[m_name]

        # 4. 持久化并热重载
        save_config(raw_cfg)
        reload_router_system()

        return {
            "status": "ok",
            "message": "模型调度角色定义已成功保存并实时热重载生效！",
            "baselines": raw_cfg["policy"]["baselines"],
        }
    except Exception as exc:
        log.error("Failed to apply model roles: %s", exc, exc_info=True)
        raise HTTPException(status_code=400, detail=f"保存模型角色定义失败: {exc}")


# ---------------------------------------------------------------------------
# Single Prompt Live Diagnostic & Routing Test API
# ---------------------------------------------------------------------------
class SingleTestRequest(BaseModel):
    prompt: str
    mode: str = "predict_only"  # "predict_only" | "end_to_end"
    max_tokens: int = 128
    stream: bool = False


@app.post("/api/router/test-single")
async def test_single_prompt(req: SingleTestRequest):
    """Run a single prompt through routing with full trace logs and stage latency breakdown."""
    prompt = req.prompt.strip()
    if not prompt:
        raise HTTPException(400, "prompt cannot be empty")

    mode = req.mode or "end_to_end"
    max_tokens = max(1, min(req.max_tokens or 1024, 4096))

    # --- 🌊 实时流式打字输出分支 (Stream Mode) ---
    if req.stream and mode == "end_to_end":
        async def stream_generator():
            t_start = time.perf_counter()
            logs: list[dict] = []

            def add_log(level: str, msg: str, stage: str = "general"):
                ts = time.strftime("%H:%M:%S") + f".{int((time.time() % 1) * 1000):03d}"
                entry = {
                    "timestamp": ts,
                    "level": level,
                    "stage": stage,
                    "message": msg,
                }
                logs.append(entry)
                return entry

            init_log = add_log("INFO", f"接收到单次提示词流式测试请求 (模式: 🌊 端到端实时打字生成), 字符数: {len(prompt)}", "init")
            yield f"data: {json.dumps({'type': 'init', 'log': init_log}, ensure_ascii=False)}\n\n"

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
                err_log = add_log("ERROR", f"路由决策过程发生异常: {exc}", "router")
                yield f"data: {json.dumps({'type': 'error', 'error': str(exc), 'log': err_log}, ensure_ascii=False)}\n\n"
                return

            clf = route_res.classification
            laya_ms = round(route_res.classification_ms or (clf.latency_s * 1000.0 if clf else 0.0), 2)
            scoring_ms = max(0.1, round((t_route_1 - t_route_0) * 1000.0 - laya_ms, 2))

            chosen_model = route_res.model
            provider_name = chosen_model.provider
            chosen_name = chosen_model.name
            upstream_id = getattr(chosen_model, "upstream_id", chosen_name)
            decision_reason = route_res.reason

            clf_dev = getattr(router.classifier, "actual_device", "mps") if hasattr(router, "classifier") else "mps"
            is_jev_clf = isinstance(getattr(router, "classifier", None), JevClassifier) or hasattr(getattr(router, "classifier", None), "systemone_url")
            clf_engine_name = f"Jev AI 云端模型 ({getattr(router.classifier, 'model', 'jev-latest')})" if is_jev_clf else f"Laya 本地模型 ({clf_dev.upper()} 加速)"
            laya_log = add_log("INFO", f"{clf_engine_name} 特征分类完成: 类别={clf.category if clf else 'general'}, 难度={clf.difficulty if clf else 0.5:.3f}, 错误代价={clf.stakes if clf else 0.2:.3f}, 耗时={laya_ms}ms", "classifier")
            dec_log = add_log("INFO", f"策略算法仲裁完成: 选定目标模型 [{chosen_name}] (提供商: {provider_name}, 算法排序耗时: {scoring_ms}ms)", "decision")
            reason_log = add_log("INFO", f"决策理由: {decision_reason}", "decision")

            raw_cfg = get_current_raw_config()
            global_timeout = float(raw_cfg.get("policy", {}).get("request_timeout_seconds") or 60.0)
            chosen_timeout_val = getattr(chosen_model, "timeout_s", None)
            if chosen_timeout_val is None:
                for rm in raw_cfg.get("models", []):
                    if (rm.get("name") == chosen_name or rm.get("upstream_id") == upstream_id) and rm.get("timeout_seconds") is not None:
                        try:
                            chosen_timeout_val = float(rm["timeout_seconds"])
                        except (ValueError, TypeError):
                            pass
                        break
            chosen_timeout = float(chosen_timeout_val or global_timeout)

            # 推送路由仲裁完成事件
            route_event = {
                "type": "route_result",
                "status": "ok",
                "classification": {
                    "category": clf.category if clf else "general",
                    "difficulty": round(clf.difficulty, 3) if clf else 0.5,
                    "stakes": round(clf.stakes, 3) if clf else 0.2,
                    "device": clf_dev,
                    "engine_type": "jev" if is_jev_clf else "laya",
                    "engine_name": clf_engine_name,
                    "classifier_model": getattr(router.classifier, "model", "convaiinnovations/laya") if hasattr(router, "classifier") else "laya",
                },
                "decision": {
                    "chosen_model": chosen_name,
                    "provider": provider_name,
                    "upstream_id": upstream_id,
                    "policy": route_res.explanation.selection.policy if (route_res.explanation and hasattr(route_res.explanation, 'selection')) else "F_expected",
                    "reason": decision_reason,
                    "timeout_seconds": chosen_timeout,
                },
                "timing": {
                    "classifier_ms": laya_ms,
                    "route_scoring_ms": scoring_ms,
                },
                "logs": [init_log, laya_log, dec_log, reason_log],
            }
            yield f"data: {json.dumps(route_event, ensure_ascii=False)}\n\n"

            # 构建候选调度序列：首选模型优先，随后从 candidates 及 catalog 中挑选备选模型
            candidate_models_to_try = [chosen_model]
            if route_res.explanation and getattr(route_res.explanation, "candidates", None):
                for c in route_res.explanation.candidates:
                    c_model = ar_server.config.catalog.get(c.model)
                    if c_model and c_model.name not in [m.name for m in candidate_models_to_try]:
                        candidate_models_to_try.append(c_model)

            for m in ar_server.config.catalog.all():
                if m.name not in [x.name for x in candidate_models_to_try]:
                    candidate_models_to_try.append(m)

            max_attempts = min(3, len(candidate_models_to_try))
            upstream_ms = 0.0
            verify_ms = 0.0
            dispatched_models: list[dict] = []
            success = False

            for attempt_idx in range(max_attempts):
                cur_target = candidate_models_to_try[attempt_idx]
                cur_provider_name = cur_target.provider
                cur_model_name = cur_target.name
                cur_upstream_id = getattr(cur_target, "upstream_id", cur_model_name)
                is_first_attempt = (attempt_idx == 0)

                cur_provider = ar_server.config.providers.get(cur_provider_name)
                if not cur_provider:
                    err_msg = f"未找到提供商 [{cur_provider_name}] 配置"
                    err_l = add_log("ERROR", f"调度尝试 #{attempt_idx + 1}: {err_msg}", "upstream")
                    err_item = {
                        "attempt": attempt_idx + 1,
                        "model_name": cur_model_name,
                        "provider": cur_provider_name,
                        "upstream_id": cur_upstream_id,
                        "status": "error",
                        "latency_ms": 0.0,
                        "output": err_msg,
                        "error": err_msg,
                        "is_primary": is_first_attempt,
                    }
                    dispatched_models.append(err_item)
                    next_m = candidate_models_to_try[attempt_idx + 1].name if attempt_idx + 1 < max_attempts else None
                    yield f"data: {json.dumps({'type': 'attempt_error', 'item': err_item, 'next_model': next_m, 'log': err_l}, ensure_ascii=False)}\n\n"
                    continue

                base_url = normalize_provider_url(cur_provider.resolved_base_url or cur_provider.base_url, api=getattr(cur_provider, "api", "openai"))
                headers = {"Content-Type": "application/json"}
                api_key = cur_provider.api_key
                if api_key and api_key != "none":
                    headers["Authorization"] = f"Bearer {api_key}"
                for k, v in cur_provider.extra_headers.items():
                    headers[k] = v

                # 读取当前模型专属超时时间
                model_timeout_s = getattr(cur_target, "timeout_s", None)
                if model_timeout_s is None:
                    for rm in raw_cfg.get("models", []):
                        if (rm.get("name") == cur_model_name or rm.get("upstream_id") == cur_upstream_id) and rm.get("timeout_seconds") is not None:
                            try:
                                model_timeout_s = float(rm["timeout_seconds"])
                            except (ValueError, TypeError):
                                pass
                            break
                effective_timeout = float(model_timeout_s or global_timeout)

                dispatch_label = "首选模型" if is_first_attempt else f"自动降级备选模型 #{attempt_idx}"
                start_l = add_log("INFO", f"[{dispatch_label}] 开始向上游端点发起流式调用: {base_url}/chat/completions (模型: {cur_upstream_id}, 超时设定: {effective_timeout:.0f}s)", "upstream")
                
                # 通知前端开始调度此模型
                yield f"data: {json.dumps({'type': 'start_model', 'attempt': attempt_idx + 1, 'model_name': cur_model_name, 'provider': cur_provider_name, 'upstream_id': cur_upstream_id, 'is_primary': is_first_attempt, 'timeout_seconds': effective_timeout, 'log': start_l}, ensure_ascii=False)}\n\n"

                payload = {
                    "model": cur_upstream_id,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "stream": True,
                }

                t_up_0 = time.perf_counter()
                accumulated_content = []
                accumulated_reasoning = []
                cur_up_ms = 0.0

                try:
                    async with httpx.AsyncClient(timeout=effective_timeout) as client:
                        async with client.stream("POST", f"{base_url}/chat/completions", headers=headers, json=payload) as resp:
                            if resp.status_code == 200:
                                async for raw_line in resp.aiter_lines():
                                    line = raw_line.strip()
                                    if not line or not line.startswith("data:"):
                                        continue
                                    data_body = line[len("data:"):].strip()
                                    if data_body == "[DONE]":
                                        break
                                    try:
                                        chunk_json = json.loads(data_body)
                                    except Exception:
                                        continue
                                    choices = chunk_json.get("choices") or []
                                    if not choices:
                                        continue
                                    delta = choices[0].get("delta") or {}
                                    chunk_c = delta.get("content") or ""
                                    chunk_r = delta.get("reasoning_content") or delta.get("reasoning") or ""
                                    if chunk_c:
                                        accumulated_content.append(chunk_c)
                                    if chunk_r:
                                        accumulated_reasoning.append(chunk_r)

                                    if chunk_c or chunk_r:
                                        yield f"data: {json.dumps({'type': 'delta', 'attempt': attempt_idx + 1, 'content': chunk_c, 'reasoning': chunk_r}, ensure_ascii=False)}\n\n"

                                cur_up_ms = round((time.perf_counter() - t_up_0) * 1000.0, 2)
                                upstream_ms += cur_up_ms

                                full_content = "".join(accumulated_content)
                                full_reasoning = "".join(accumulated_reasoning)
                                full_display_output = ""
                                if full_content and full_reasoning:
                                    full_display_output = f"【深度思考推导】\n{full_reasoning}\n\n【生成回答】\n{full_content}"
                                elif full_content:
                                    full_display_output = full_content
                                elif full_reasoning:
                                    full_display_output = full_reasoning
                                else:
                                    full_display_output = "（生成完成，上游返回空白文本）"

                                succ_l = add_log("INFO", f"上游流式传输完成: 耗时 {cur_up_ms}ms, 累计吐出 {len(full_display_output)} 字符", "upstream")
                                dispatch_entry = {
                                    "attempt": attempt_idx + 1,
                                    "model_name": cur_model_name,
                                    "provider": cur_provider_name,
                                    "upstream_id": cur_upstream_id,
                                    "status": "success",
                                    "http_status": 200,
                                    "latency_ms": cur_up_ms,
                                    "timeout_seconds": effective_timeout,
                                    "reply_snippet": full_display_output[:300] + ("..." if len(full_display_output) > 300 else ""),
                                    "output": full_display_output,
                                    "full_reply": full_display_output,
                                    "is_primary": is_first_attempt,
                                }
                                dispatched_models.append(dispatch_entry)

                                # 触发 Laya 质量验收判定 (如果有)
                                ans_to_check = full_content or full_display_output
                                if ans_to_check and hasattr(router, "check"):
                                    t_chk_0 = time.perf_counter()
                                    chk_start_l = add_log("INFO", "触发 Laya 质量验收判定 (Adequacy Verification)...", "verify")
                                    yield f"data: {json.dumps({'type': 'verify_start', 'log': chk_start_l}, ensure_ascii=False)}\n\n"
                                    try:
                                        verdict = await asyncio.to_thread(router.check, route_res, prompt, ans_to_check)
                                        verify_ms = round((time.perf_counter() - t_chk_0) * 1000.0, 2)
                                        chk_done_l = add_log("INFO", f"Laya 质量判定完成: 满意度预估={getattr(verdict, 'p_adequate', 'N/A')}, 是否建议升级={getattr(verdict, 'escalate', False)}, 耗时={verify_ms}ms", "verify")
                                        dispatch_entry["verification"] = {
                                            "verified": True,
                                            "score": getattr(verdict, "p_adequate", 0.5),
                                            "escalate": getattr(verdict, "escalate", False),
                                            "failure": getattr(verdict, "failure", "fine"),
                                            "reason": getattr(verdict, "reason", ""),
                                            "latency_ms": verify_ms,
                                        }
                                        yield f"data: {json.dumps({'type': 'verify_done', 'log': chk_done_l, 'verification_ms': verify_ms, 'verdict': {'escalate': getattr(verdict, 'escalate', False), 'p_adequate': getattr(verdict, 'p_adequate', 0.5), 'failure': getattr(verdict, 'failure', 'unknown'), 'reason': getattr(verdict, 'reason', '')}}, ensure_ascii=False)}\n\n"

                                        # 若判定需要升级 (escalate == True)，触发第二跳高阶模型升级调用
                                        if getattr(verdict, "escalate", False):
                                            try:
                                                retry_res, _, _ = await asyncio.to_thread(
                                                    router.escalate_after_verdict, route_res, verdict, messages, full_content
                                                )
                                            except Exception as esc_err:
                                                log.warning("escalate_after_verdict error: %s", esc_err)
                                                retry_res = None
                                            
                                            esc_model = retry_res.model if (retry_res and retry_res.model) else None
                                            if not esc_model or esc_model.name == cur_model_name:
                                                pro_model = None
                                                for m in getattr(router.config.catalog, "models", []):
                                                    if m.name != cur_model_name and (not pro_model or m.cap("general") > pro_model.cap("general")):
                                                        pro_model = m
                                                esc_model = pro_model

                                            if esc_model and esc_model.name != cur_model_name:
                                                esc_prov = router.config.providers.get(esc_model.provider)
                                                if esc_prov:
                                                    esc_warn_l = add_log("WARN", f"首选模型 [{cur_model_name}] 质量未通过验收 (满意度评分={getattr(verdict, 'p_adequate', 'N/A')})，触发两跳升级，转向调度高阶旗舰模型 [{esc_model.name}]...", "verify")
                                                    yield f"data: {json.dumps({'type': 'escalate_start', 'target_model': esc_model.name, 'reason': getattr(verdict, 'reason', ''), 'log': esc_warn_l}, ensure_ascii=False)}\n\n"
                                                    
                                                    sep_txt = f"\n\n---\n> ⚠️ **质检验收判定未通过**（满意度评分: {getattr(verdict, 'p_adequate', 0.0):.2f}，缺陷类型: {getattr(verdict, 'failure', 'inadequate')}）。已自动级联升级至 **{esc_model.name}** 重新生成优质解答：\n\n"
                                                    yield f"data: {json.dumps({'type': 'chunk', 'content': sep_txt}, ensure_ascii=False)}\n\n"
                                                    
                                                    t_esc_0 = time.perf_counter()
                                                    esc_payload = {**upstream_body, "model": esc_model.upstream_id or esc_model.name, "stream": True}
                                                    esc_headers = {"Content-Type": "application/json"}
                                                    if esc_prov.api_key:
                                                        esc_headers["Authorization"] = f"Bearer {expand_env_vars(esc_prov.api_key)}"
                                                    esc_url = f"{normalize_provider_url(expand_env_vars(esc_prov.base_url))}/chat/completions"
                                                    esc_full_content = ""
                                                    try:
                                                        async with httpx.AsyncClient(timeout=effective_timeout) as esc_client:
                                                            async with esc_client.stream("POST", esc_url, headers=esc_headers, json=esc_payload) as esc_resp:
                                                                if esc_resp.status_code == 200:
                                                                    async for raw_line in esc_resp.aiter_lines():
                                                                        line = raw_line.strip()
                                                                        if not line:
                                                                            continue
                                                                        if line.startswith("data: "):
                                                                            chunk_data = line[6:].strip()
                                                                            if chunk_data == "[DONE]":
                                                                                break
                                                                            try:
                                                                                c_json = json.loads(chunk_data)
                                                                                delta = c_json.get("choices", [{}])[0].get("delta", {})
                                                                                chunk_content = delta.get("content", "")
                                                                                if chunk_content:
                                                                                    esc_full_content += chunk_content
                                                                                    yield f"data: {json.dumps({'type': 'chunk', 'content': chunk_content}, ensure_ascii=False)}\n\n"
                                                                            except Exception:
                                                                                pass
                                                                    esc_up_ms = round((time.perf_counter() - t_esc_0) * 1000.0, 2)
                                                                    upstream_ms += esc_up_ms
                                                                    esc_succ_l = add_log("INFO", f"升级模型 [{esc_model.name}] 成功生成优质答案 (耗时 {esc_up_ms}ms)", "upstream")
                                                                    esc_entry = {
                                                                        "attempt": attempt_idx + 2,
                                                                        "model_name": esc_model.name,
                                                                        "provider": esc_prov.name,
                                                                        "upstream_id": esc_model.upstream_id,
                                                                        "status": "success",
                                                                        "http_status": 200,
                                                                        "latency_ms": esc_up_ms,
                                                                        "timeout_seconds": effective_timeout,
                                                                        "output": esc_full_content,
                                                                        "full_reply": esc_full_content,
                                                                        "is_primary": False,
                                                                        "is_escalation": True,
                                                                    }
                                                                    dispatched_models.append(esc_entry)
                                                                    yield f"data: {json.dumps({'type': 'model_success', 'attempt': attempt_idx + 2, 'item': esc_entry, 'log': esc_succ_l}, ensure_ascii=False)}\n\n"
                                                    except Exception as esc_call_e:
                                                        log.warning("Escalated model call failed: %s", esc_call_e)
                                    except Exception as chk_e:
                                        chk_warn_l = add_log("WARN", f"Laya 质量判定跳过或异常: {chk_e}", "verify")
                                        yield f"data: {json.dumps({'type': 'verify_done', 'log': chk_warn_l, 'verification_ms': 0}, ensure_ascii=False)}\n\n"

                                # 通知当前模型调用完成
                                yield f"data: {json.dumps({'type': 'model_success', 'attempt': attempt_idx + 1, 'item': dispatch_entry, 'log': succ_l}, ensure_ascii=False)}\n\n"
                                success = True
                                break
                            else:
                                err_bytes = await resp.aread()
                                err_text = err_bytes.decode("utf-8", errors="replace")[:200]
                                err_detail = f"HTTP {resp.status_code}: {err_text}"
                                cur_up_ms = round((time.perf_counter() - t_up_0) * 1000.0, 2)
                                upstream_ms += cur_up_ms
                                warn_l = add_log("WARN", f"模型 [{cur_model_name}] 响应异常: {err_detail} (耗时 {cur_up_ms}ms)", "upstream")
                                err_item = {
                                    "attempt": attempt_idx + 1,
                                    "model_name": cur_model_name,
                                    "provider": cur_provider_name,
                                    "upstream_id": cur_upstream_id,
                                    "status": "error",
                                    "http_status": resp.status_code,
                                    "latency_ms": cur_up_ms,
                                    "timeout_seconds": effective_timeout,
                                    "output": f"上游接口异常: {err_detail}",
                                    "error": err_detail,
                                    "is_primary": is_first_attempt,
                                }
                                dispatched_models.append(err_item)
                                next_name = candidate_models_to_try[attempt_idx + 1].name if attempt_idx + 1 < max_attempts else None
                                if next_name:
                                    add_log("INFO", f"准备触发自动降级容灾，转向调度下一个候选模型 [{next_name}]...", "upstream")
                                yield f"data: {json.dumps({'type': 'attempt_error', 'item': err_item, 'next_model': next_name, 'log': warn_l}, ensure_ascii=False)}\n\n"
                except httpx.TimeoutException:
                    cur_up_ms = round((time.perf_counter() - t_up_0) * 1000.0, 2)
                    upstream_ms += cur_up_ms
                    err_detail = f"上游端点 {base_url} 连接超时 ({effective_timeout:.0f}s)"
                    warn_l = add_log("WARN", f"模型 [{cur_model_name}] 请求超时 (耗时 {cur_up_ms}ms, 限制 {effective_timeout:.0f}s)", "upstream")
                    err_item = {
                        "attempt": attempt_idx + 1,
                        "model_name": cur_model_name,
                        "provider": cur_provider_name,
                        "upstream_id": cur_upstream_id,
                        "status": "error",
                        "latency_ms": cur_up_ms,
                        "timeout_seconds": effective_timeout,
                        "output": f"连接超时: {err_detail}",
                        "error": err_detail,
                        "is_primary": is_first_attempt,
                    }
                    dispatched_models.append(err_item)
                    next_name = candidate_models_to_try[attempt_idx + 1].name if attempt_idx + 1 < max_attempts else None
                    if next_name:
                        add_log("INFO", f"准备触发自动降级容灾，转向调度下一个候选模型 [{next_name}]...", "upstream")
                    yield f"data: {json.dumps({'type': 'attempt_error', 'item': err_item, 'next_model': next_name, 'log': warn_l}, ensure_ascii=False)}\n\n"
                except Exception as e:
                    cur_up_ms = round((time.perf_counter() - t_up_0) * 1000.0, 2)
                    upstream_ms += cur_up_ms
                    err_detail = str(e)
                    warn_l = add_log("WARN", f"模型 [{cur_model_name}] 连接异常: {err_detail} (耗时 {cur_up_ms}ms)", "upstream")
                    err_item = {
                        "attempt": attempt_idx + 1,
                        "model_name": cur_model_name,
                        "provider": cur_provider_name,
                        "upstream_id": cur_upstream_id,
                        "status": "error",
                        "latency_ms": cur_up_ms,
                        "output": f"网络异常: {err_detail}",
                        "error": err_detail,
                        "is_primary": is_first_attempt,
                    }
                    dispatched_models.append(err_item)
                    next_name = candidate_models_to_try[attempt_idx + 1].name if attempt_idx + 1 < max_attempts else None
                    if next_name:
                        add_log("INFO", f"准备触发自动降级容灾，转向调度下一个候选模型 [{next_name}]...", "upstream")
                    yield f"data: {json.dumps({'type': 'attempt_error', 'item': err_item, 'next_model': next_name, 'log': warn_l}, ensure_ascii=False)}\n\n"

            # 全程结束
            total_latency_ms = round((time.perf_counter() - t_start) * 1000.0, 2)
            done_l = add_log("INFO", f"全路径测试流程执行完毕, 全程总耗时: {total_latency_ms}ms", "done")

            final_data = {
                "type": "done",
                "status": "ok" if success else "error",
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
                    "is_jev": is_jev_clf,
                    "engine_type": "jev" if is_jev_clf else "laya",
                    "engine_name": clf_engine_name,
                    "classifier_model": getattr(router.classifier, "model", "convaiinnovations/laya") if hasattr(router, "classifier") else "laya",
                },
                "decision": {
                    "chosen_model": chosen_name,
                    "provider": provider_name,
                    "upstream_id": upstream_id,
                    "policy": route_res.explanation.selection.policy if (route_res.explanation and hasattr(route_res.explanation, 'selection')) else "F_expected",
                    "reason": decision_reason,
                    "timeout_seconds": chosen_timeout,
                },
                "dispatched_models": dispatched_models,
                "logs": logs,
            }
            yield f"data: {json.dumps(final_data, ensure_ascii=False)}\n\n"

        return StreamingResponse(
            stream_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # --- ⚡ 纯路由预测或非流式模式（原代码保持不变）---
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
    is_jev_clf = isinstance(getattr(router, "classifier", None), JevClassifier) or hasattr(getattr(router, "classifier", None), "systemone_url")
    clf_engine_name = f"Jev AI 云端模型 ({getattr(router.classifier, 'model', 'jev-latest')})" if is_jev_clf else f"Laya 本地模型 ({clf_dev.upper()} 加速)"
    add_log("INFO", f"{clf_engine_name} 特征分类完成: 类别={clf.category if clf else 'general'}, 难度={clf.difficulty if clf else 0.5:.3f}, 错误代价={clf.stakes if clf else 0.2:.3f}, 耗时={laya_ms}ms", "classifier")
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

    raw_cfg = get_current_raw_config()
    global_timeout = float(raw_cfg.get("policy", {}).get("request_timeout_seconds") or 60.0)
    chosen_timeout_val = getattr(chosen_model, "timeout_s", None)
    if chosen_timeout_val is None:
        for rm in raw_cfg.get("models", []):
            if (rm.get("name") == chosen_name or rm.get("upstream_id") == upstream_id) and rm.get("timeout_seconds") is not None:
                try:
                    chosen_timeout_val = float(rm["timeout_seconds"])
                except (ValueError, TypeError):
                    pass
                break
    chosen_timeout = float(chosen_timeout_val or global_timeout)

    # 默认触发端到端真实生成，除非显式指定 predict_only
    if mode != "predict_only":
        mode = "end_to_end"

    dispatched_models: list[dict] = []
    upstream_data = None
    upstream_ms = 0.0
    verify_ms = 0.0

    if mode == "end_to_end":
        # 构建候选调度序列：首选模型优先，随后从 candidates 及 catalog 中挑选备选模型
        candidate_models_to_try = [chosen_model]
        if route_res.explanation and getattr(route_res.explanation, "candidates", None):
            for c in route_res.explanation.candidates:
                c_model = ar_server.config.catalog.get(c.model)
                if c_model and c_model.name not in [m.name for m in candidate_models_to_try]:
                    candidate_models_to_try.append(c_model)

        for m in ar_server.config.catalog.all():
            if m.name not in [x.name for x in candidate_models_to_try]:
                candidate_models_to_try.append(m)

        add_log("INFO", f"开始执行真实模型调用 (首选目标: [{chosen_name}], 候选备选池: {len(candidate_models_to_try)} 个模型)", "upstream")

        # 最多尝试 3 个不同模型进行容灾调度
        max_attempts = min(3, len(candidate_models_to_try))
        for attempt_idx in range(max_attempts):
            cur_target = candidate_models_to_try[attempt_idx]
            cur_provider_name = cur_target.provider
            cur_model_name = cur_target.name
            cur_upstream_id = getattr(cur_target, "upstream_id", cur_model_name)
            is_first_attempt = (attempt_idx == 0)

            cur_provider = ar_server.config.providers.get(cur_provider_name)
            if not cur_provider:
                err_msg = f"未找到提供商 [{cur_provider_name}] 配置"
                add_log("ERROR", f"调度尝试 #{attempt_idx + 1}: {err_msg}", "upstream")
                dispatched_models.append({
                    "attempt": attempt_idx + 1,
                    "model_name": cur_model_name,
                    "provider": cur_provider_name,
                    "upstream_id": cur_upstream_id,
                    "status": "error",
                    "latency_ms": 0.0,
                    "output": err_msg,
                    "error": err_msg,
                    "is_primary": is_first_attempt,
                })
                continue

            base_url = normalize_provider_url(cur_provider.resolved_base_url or cur_provider.base_url, api=getattr(cur_provider, "api", "openai"))
            headers = {"Content-Type": "application/json"}
            api_key = cur_provider.api_key
            if api_key and api_key != "none":
                headers["Authorization"] = f"Bearer {api_key}"
            for k, v in cur_provider.extra_headers.items():
                headers[k] = v

            payload = {
                "model": cur_upstream_id,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
            }

            # 读取当前模型专属超时时间，若未配置则继承全局策略默认超时
            raw_cfg = get_current_raw_config()
            global_timeout = float(raw_cfg.get("policy", {}).get("request_timeout_seconds") or 60.0)
            model_timeout_s = getattr(cur_target, "timeout_s", None)
            if model_timeout_s is None:
                for rm in raw_cfg.get("models", []):
                    if (rm.get("name") == cur_model_name or rm.get("upstream_id") == cur_upstream_id) and rm.get("timeout_seconds") is not None:
                        try:
                            model_timeout_s = float(rm["timeout_seconds"])
                        except (ValueError, TypeError):
                            pass
                        break
            effective_timeout = float(model_timeout_s or global_timeout)

            dispatch_label = f"首选模型" if is_first_attempt else f"自动降级备选模型 #{attempt_idx}"
            add_log("INFO", f"[{dispatch_label}] 开始向上游端点发起调用: {base_url}/chat/completions (模型: {cur_upstream_id}, 超时设定: {effective_timeout:.0f}s)", "upstream")
            t_up_0 = time.perf_counter()
            cur_up_ms = 0.0

            try:
                async with httpx.AsyncClient(timeout=effective_timeout) as client:
                    resp = await client.post(f"{base_url}/chat/completions", headers=headers, json=payload)
                    cur_up_ms = round((time.perf_counter() - t_up_0) * 1000.0, 2)
                    upstream_ms += cur_up_ms

                    if resp.status_code == 200:
                        data = resp.json()
                        choices = data.get("choices") or []
                        msg_obj = choices[0].get("message", {}) if choices else {}
                        content = (msg_obj.get("content") or "").strip()
                        reasoning = (msg_obj.get("reasoning_content") or "").strip()
                        
                        full_display_output = ""
                        if content and reasoning:
                            full_display_output = f"【深度思考推导】\n{reasoning}\n\n【生成回答】\n{content}"
                        elif content:
                            full_display_output = content
                        elif reasoning:
                            full_display_output = reasoning
                        else:
                            full_display_output = "（生成完成，上游返回空白文本）"

                        usage = data.get("usage") or {}
                        add_log("INFO", f"上游响应成功: 状态码 200 OK, 耗时 {cur_up_ms}ms, 消耗 Token: prompt={usage.get('prompt_tokens', 0)}, completion={usage.get('completion_tokens', 0)}", "upstream")

                        dispatch_entry = {
                            "attempt": attempt_idx + 1,
                            "model_name": cur_model_name,
                            "provider": cur_provider_name,
                            "upstream_id": cur_upstream_id,
                            "status": "success",
                            "http_status": 200,
                            "latency_ms": cur_up_ms,
                            "timeout_seconds": effective_timeout,
                            "reply_snippet": full_display_output[:300] + ("..." if len(full_display_output) > 300 else ""),
                            "output": full_display_output,
                            "full_reply": full_display_output,
                            "usage": usage,
                            "is_primary": is_first_attempt,
                        }
                        dispatched_models.append(dispatch_entry)
                        upstream_data = dispatch_entry

                        # 触发 Laya 质量验收判定 (如果有)
                        ans_to_check = content or full_display_output
                        if ans_to_check and hasattr(router, "check"):
                            t_chk_0 = time.perf_counter()
                            add_log("INFO", "触发 Laya 质量验收判定 (Adequacy Verification)...", "verify")
                            try:
                                verdict = await asyncio.to_thread(router.check, route_res, prompt, ans_to_check)
                                verify_ms = round((time.perf_counter() - t_chk_0) * 1000.0, 2)
                                add_log("INFO", f"Laya 质量判定完成: 满意度预估={getattr(verdict, 'p_adequate', 'N/A')}, 是否建议升级={getattr(verdict, 'escalate', False)}, 耗时={verify_ms}ms", "verify")
                                dispatch_entry["verification"] = {
                                    "verified": True,
                                    "score": getattr(verdict, "p_adequate", 0.5),
                                    "escalate": getattr(verdict, "escalate", False),
                                    "failure": getattr(verdict, "failure", "fine"),
                                    "reason": getattr(verdict, "reason", ""),
                                    "latency_ms": verify_ms,
                                }

                                # 若判定需要升级 (escalate == True)，触发第二跳高阶模型升级调用
                                if getattr(verdict, "escalate", False):
                                    try:
                                        test_msgs = [{"role": "user", "content": prompt}]
                                        retry_res, _, _ = await asyncio.to_thread(
                                            router.escalate_after_verdict, route_res, verdict, test_msgs, ans_to_check
                                        )
                                    except Exception as esc_err:
                                        log.warning("escalate_after_verdict error: %s", esc_err)
                                        retry_res = None
                                    
                                    esc_model = retry_res.model if (retry_res and retry_res.model) else None
                                    if not esc_model or esc_model.name == cur_model_name:
                                        pro_model = None
                                        for m in getattr(router.config.catalog, "models", []):
                                            if m.name != cur_model_name and (not pro_model or m.cap("general") > pro_model.cap("general")):
                                                pro_model = m
                                        esc_model = pro_model

                                    if esc_model and esc_model.name != cur_model_name:
                                        esc_prov = router.config.providers.get(esc_model.provider)
                                        if esc_prov:
                                            add_log("WARN", f"首选模型 [{cur_model_name}] 质量未达标 (满意度评分={getattr(verdict, 'p_adequate', 'N/A')})，触发两跳升级，调度高阶模型 [{esc_model.name}]...", "verify")
                                            esc_t0 = time.perf_counter()
                                            esc_payload = {**payload, "model": esc_model.upstream_id or esc_model.name, "stream": False}
                                            esc_headers = {"Content-Type": "application/json"}
                                            if esc_prov.api_key:
                                                esc_headers["Authorization"] = f"Bearer {expand_env_vars(esc_prov.api_key)}"
                                            esc_url = f"{normalize_provider_url(expand_env_vars(esc_prov.base_url))}/chat/completions"
                                            try:
                                                async with httpx.AsyncClient(timeout=effective_timeout) as esc_client:
                                                    esc_resp = await esc_client.post(esc_url, headers=esc_headers, json=esc_payload)
                                                    if esc_resp.status_code == 200:
                                                        esc_data = esc_resp.json()
                                                        esc_up_ms = round((time.perf_counter() - esc_t0) * 1000.0, 2)
                                                        upstream_ms += esc_up_ms
                                                        esc_content = esc_data.get("choices", [{}])[0].get("message", {}).get("content", "")
                                                        content = esc_content
                                                        dispatch_entry = {
                                                            "attempt": attempt_idx + 2,
                                                            "model_name": esc_model.name,
                                                            "provider": esc_prov.name,
                                                            "upstream_id": esc_model.upstream_id,
                                                            "status": "success",
                                                            "http_status": 200,
                                                            "latency_ms": esc_up_ms,
                                                            "timeout_seconds": effective_timeout,
                                                            "output": esc_content,
                                                            "full_reply": esc_content,
                                                            "is_primary": False,
                                                            "is_escalation": True,
                                                        }
                                                        dispatched_models.append(dispatch_entry)
                                                        upstream_data = dispatch_entry
                                                        add_log("INFO", f"升级模型 [{esc_model.name}] 成功生成优质答案 (耗时 {esc_up_ms}ms)", "upstream")
                                            except Exception as esc_call_e:
                                                log.warning("Escalated model call failed in non-streaming: %s", esc_call_e)
                            except Exception as chk_e:
                                add_log("WARN", f"Laya 质量判定跳过或异常: {chk_e}", "verify")

                        # 成功获取回答，结束调度
                        break
                    else:
                        err_text = resp.text[:200]
                        err_detail = f"HTTP {resp.status_code}: {err_text}"
                        add_log("WARN", f"模型 [{cur_model_name}] 响应异常: {err_detail} (耗时 {cur_up_ms}ms)", "upstream")
                        dispatched_models.append({
                            "attempt": attempt_idx + 1,
                            "model_name": cur_model_name,
                            "provider": cur_provider_name,
                            "upstream_id": cur_upstream_id,
                            "status": "error",
                            "http_status": resp.status_code,
                            "latency_ms": cur_up_ms,
                            "timeout_seconds": effective_timeout,
                            "output": f"上游接口异常: {err_detail}",
                            "error": err_detail,
                            "is_primary": is_first_attempt,
                        })
                        if attempt_idx + 1 < max_attempts:
                            next_name = candidate_models_to_try[attempt_idx + 1].name
                            add_log("INFO", f"准备触发自动降级容灾，转向调度下一个候选模型 [{next_name}]...", "upstream")
            except httpx.TimeoutException:
                cur_up_ms = round((time.perf_counter() - t_up_0) * 1000.0, 2)
                upstream_ms += cur_up_ms
                err_detail = f"上游端点 {base_url} 连接超时 ({effective_timeout:.0f}s)"
                add_log("WARN", f"模型 [{cur_model_name}] 请求超时 (耗时 {cur_up_ms}ms, 限制 {effective_timeout:.0f}s)", "upstream")
                dispatched_models.append({
                    "attempt": attempt_idx + 1,
                    "model_name": cur_model_name,
                    "provider": cur_provider_name,
                    "upstream_id": cur_upstream_id,
                    "status": "error",
                    "latency_ms": cur_up_ms,
                    "timeout_seconds": effective_timeout,
                    "output": f"连接超时: {err_detail}",
                    "error": err_detail,
                    "is_primary": is_first_attempt,
                })
                if attempt_idx + 1 < max_attempts:
                    next_name = candidate_models_to_try[attempt_idx + 1].name
                    add_log("INFO", f"准备触发自动降级容灾，转向调度下一个候选模型 [{next_name}]...", "upstream")
            except Exception as e:
                cur_up_ms = round((time.perf_counter() - t_up_0) * 1000.0, 2)
                upstream_ms += cur_up_ms
                err_detail = str(e)
                add_log("WARN", f"模型 [{cur_model_name}] 连接异常: {err_detail} (耗时 {cur_up_ms}ms)", "upstream")
                dispatched_models.append({
                    "attempt": attempt_idx + 1,
                    "model_name": cur_model_name,
                    "provider": cur_provider_name,
                    "upstream_id": cur_upstream_id,
                    "status": "error",
                    "latency_ms": cur_up_ms,
                    "output": f"网络异常: {err_detail}",
                    "error": err_detail,
                    "is_primary": is_first_attempt,
                })
                if attempt_idx + 1 < max_attempts:
                    next_name = candidate_models_to_try[attempt_idx + 1].name
                    add_log("INFO", f"准备触发自动降级容灾，转向调度下一个候选模型 [{next_name}]...", "upstream")

        if not upstream_data and dispatched_models:
            upstream_data = dispatched_models[-1]

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
            "is_jev": is_jev_clf,
            "engine_type": "jev" if is_jev_clf else "laya",
            "engine_name": clf_engine_name,
            "classifier_model": getattr(router.classifier, "model", "convaiinnovations/laya") if hasattr(router, "classifier") else "laya",
        },
        "decision": {
            "chosen_model": chosen_name,
            "provider": provider_name,
            "upstream_id": upstream_id,
            "policy": route_res.explanation.selection.policy if (route_res.explanation and hasattr(route_res.explanation, 'selection')) else "F_expected",
            "reason": decision_reason,
            "candidates": candidates_data,
            "timeout_seconds": chosen_timeout,
        },
        "dispatched_models": dispatched_models,
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
    sort_col: str | None = None,
    sort_order: str = "desc",
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

    # Sort by column or category rating if requested
    reverse = (sort_order.lower() != "asc")
    if sort_col:
        # Check if sort_col is in subsets or top-level
        def _get_sort_val(item: dict) -> float:
            subsets = item.get("subsets", {})
            if sort_col in subsets:
                sub_info = subsets[sort_col]
                return float(sub_info.get("elo") or 0.0)
            if sort_col in item:
                try:
                    return float(item[sort_col])
                except (ValueError, TypeError):
                    return 0.0
            return 0.0
        entries = sorted(entries, key=_get_sort_val, reverse=reverse)
    elif category in ("coding", "math", "hard"):
        cat_to_sub = {"coding": "webdev", "math": "math", "hard": "hard"}
        sub_key = cat_to_sub.get(category, "text")
        def _get_cat_val(item: dict) -> float:
            sub = item.get("subsets", {}).get(sub_key)
            if sub and "elo" in sub:
                return float(sub["elo"])
            return float(item.get(f"rating_{category}") or 0.0)
        entries = sorted(entries, key=_get_cat_val, reverse=True)
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
        "subsets_meta": all_data.get("subsets_meta", {}),
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


class ApplyLeaderboardRequest(BaseModel):
    model_name: str
    target_leaderboard_model: str | None = None


@app.post("/api/leaderboard/apply_to_model")
async def apply_leaderboard_to_model(req: ApplyLeaderboardRequest):
    """将指定权威模型的评测能力分数一键应用并持久化到路由模型配置中."""
    cfg = get_current_raw_config()
    models = cfg.get("models", [])
    target = None
    for m in models:
        if m.get("name") == req.model_name:
            target = m
            break

    if not target:
        raise HTTPException(status_code=404, detail=f"未找到路由候选模型: {req.model_name}")

    # 匹配目标权威模型
    lb_match = None
    if req.target_leaderboard_model:
        all_lb = leaderboard_mgr.get_all().get("models", [])
        for item in all_lb:
            if item.get("model_name") == req.target_leaderboard_model or item.get("display_name") == req.target_leaderboard_model:
                lb_match = item
                break

    if not lb_match:
        matches = leaderboard_mgr.match_model(target.get("upstream_id") or target.get("name"))
        if matches:
            lb_match = matches[0]

    if not lb_match:
        raise HTTPException(status_code=400, detail="未找到对应的权威榜单评测数据")

    prof = lb_match.get("normalized_profile", {})
    c_coding = int(round(prof.get("coding", 0.85) * 100))
    c_math = int(round(prof.get("math", 0.85) * 100))
    c_reasoning = int(round(prof.get("reasoning", 0.85) * 100))
    c_general = int(round(prof.get("general", 0.85) * 100))
    c_agentic = int(round(prof.get("agentic", prof.get("reasoning", 0.85)) * 100))

    if "capability" not in target:
        target["capability"] = {}

    target["capability"]["coding"] = c_coding
    target["capability"]["math"] = c_math
    target["capability"]["reasoning"] = c_reasoning
    target["capability"]["general"] = c_general
    target["capability"]["knowledge"] = c_general
    target["capability"]["summarisation"] = c_general
    target["capability"]["agentic"] = c_agentic
    target["capability"]["tool_use"] = max(c_agentic, c_coding)

    if prof.get("vision", 0.0) >= 0.70:
        target["vision"] = True

    save_config(cfg)
    reload_router_system()

    return {
        "status": "ok",
        "message": f"已将权威模型 [{lb_match.get('display_name')}] (Elo {lb_match.get('rating_overall')}) 的能力评分同步至路由模型 [{req.model_name}]",
        "applied_model": req.model_name,
        "matched_leaderboard": lb_match.get("display_name"),
        "elo": lb_match.get("rating_overall"),
        "capability": target["capability"],
    }


@app.post("/api/leaderboard/apply_all_candidates")
async def apply_all_candidates_leaderboard():
    """批量将所有当前配置的候选模型与权威评测榜单进行匹配并一键校准能力参数."""
    cfg = get_current_raw_config()
    models = cfg.get("models", [])
    updated = []

    for target in models:
        matches = leaderboard_mgr.match_model(target.get("upstream_id") or target.get("name"))
        if not matches:
            continue
        lb_match = matches[0]
        prof = lb_match.get("normalized_profile", {})
        c_coding = int(round(prof.get("coding", 0.85) * 100))
        c_math = int(round(prof.get("math", 0.85) * 100))
        c_reasoning = int(round(prof.get("reasoning", 0.85) * 100))
        c_general = int(round(prof.get("general", 0.85) * 100))
        c_agentic = int(round(prof.get("agentic", prof.get("reasoning", 0.85)) * 100))

        if "capability" not in target:
            target["capability"] = {}

        target["capability"]["coding"] = c_coding
        target["capability"]["math"] = c_math
        target["capability"]["reasoning"] = c_reasoning
        target["capability"]["general"] = c_general
        target["capability"]["knowledge"] = c_general
        target["capability"]["summarisation"] = c_general
        target["capability"]["agentic"] = c_agentic
        target["capability"]["tool_use"] = max(c_agentic, c_coding)

        if prof.get("vision", 0.0) >= 0.70:
            target["vision"] = True

        updated.append({
            "model": target.get("name"),
            "matched_leaderboard": lb_match.get("display_name"),
            "elo": lb_match.get("rating_overall"),
            "capability": target["capability"],
        })

    if updated:
        save_config(cfg)
        reload_router_system()

    return {
        "status": "ok",
        "message": f"成功为 {len(updated)} 个路由候选模型一键同步最新权威评测能力基准",
        "updated_models": updated,
    }


# ---------------------------------------------------------------------------
# WebUI Static Frontend
# ---------------------------------------------------------------------------
@app.api_route("/", methods=["GET", "HEAD"], response_class=HTMLResponse)
async def serve_index():
    index_path = WEB_DIR / "index.html"
    if index_path.exists():
        content = index_path.read_text(encoding="utf-8")
        headers = {
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
        }
        return HTMLResponse(content, headers=headers)
    return HTMLResponse("<h1>Auto-LLM-Router is running. WebUI index.html not found.</h1>")


class NoCacheStaticFiles(StaticFiles):
    async def get_response(self, path: str, scope):
        resp = await super().get_response(path, scope)
        resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        resp.headers["Pragma"] = "no-cache"
        resp.headers["Expires"] = "0"
        return resp


# Mount static assets if web dir has assets
if WEB_DIR.exists():
    app.mount("/web", NoCacheStaticFiles(directory=str(WEB_DIR)), name="web")


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"])
async def catch_all(path: str, request: Request) -> Response:
    """Fallback passthrough for Claude Code telemetry / count_tokens."""
    if not path or path == "favicon.ico" or path.startswith(("api/", "web/", "reports/")):
        raise HTTPException(status_code=404, detail="Not found")
    from auto_router import shim
    body = await request.body()
    return await shim.passthrough(request, body, f"/{path}")

