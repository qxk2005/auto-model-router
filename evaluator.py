"""Benchmark evaluation engine for auto-llm-router with Laya.

Supports:
1. Fast Simulation (极速模拟预测): Zero cost, runs local Laya forward pass on Apple M4 Max,
   simulates policy decisions and calculates projected savings vs Always-Expensive and Always-Cheap baselines.
2. Real Endpoint Benchmark (真实端点压测): Dispatches real requests to configured endpoints (LM Studio,
   DeepSeek, OpenAI), measuring real latency, tokens, actual dollar costs and response adequacy.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import httpx

from auto_router.catalog import Catalog, ModelInfo
from auto_router.jev import Classification
from auto_router.economics import turn_cost
from auto_router.policies import Conversation, Context, TurnRequest, Choice

log = logging.getLogger("auto_router.evaluator")


@dataclass
class CaseResult:
    case_id: str
    category: str
    difficulty_tag: str
    prompt: str
    expected_tier: str
    # Laya classifier outputs
    detected_category: str
    detected_difficulty: float
    detected_stakes: float
    classifier_latency_ms: float
    # Router decisions
    chosen_model: str
    chosen_provider: str
    decision_reason: str
    # Cost metrics
    prompt_tokens: int
    output_tokens: int
    total_tokens: int
    cost_router: float
    cost_expensive: float
    cost_cheap: float
    savings_usd: float
    savings_pct: float
    is_aligned: bool
    # Real test fields
    real_execution: bool = False
    real_latency_s: float = 0.0
    real_response: str = ""
    error: str = ""
    # Execution Trace Chain fields
    trace_id: str = ""
    trace_chain: list[dict] = field(default_factory=list)
    hop_count: int = 1
    escalated: bool = False


@dataclass
class BenchmarkSummary:
    timestamp: str
    mode: str
    total_cases: int
    successful_cases: int
    total_tokens: int
    # Costs
    cost_router_total: float
    cost_expensive_total: float
    cost_cheap_total: float
    total_savings_usd: float
    total_savings_pct: float
    # Latencies
    avg_classifier_latency_ms: float
    avg_total_latency_s: float
    # Quality & distribution
    alignment_rate: float
    model_distribution: dict[str, int]
    category_breakdown: dict[str, dict[str, Any]]
    results: list[CaseResult]
    currency_symbol: str = "¥"
    usd_cny_rate: float = 7.20
    config_snapshot: dict[str, Any] = field(default_factory=dict)
    trace_topology_stats: dict[str, Any] = field(default_factory=dict)


class BenchmarkEvaluator:
    def __init__(self, router, catalog: Catalog, policy_engine, config_raw: dict):
        self.router = router
        self.catalog = catalog
        self.policy = policy_engine
        self.config_raw = config_raw

        curr_cfg = self.config_raw.get("policy", {}).get("currency", {}) if isinstance(self.config_raw, dict) else {}
        self.currency_symbol = "¥" if curr_cfg.get("base", "CNY") == "CNY" else "$"
        self.usd_cny_rate = float(curr_cfg.get("usd_cny_rate", 7.20))

    def _build_config_snapshot(self) -> dict[str, Any]:
        """Snapshot current router policy, Laya settings and active models catalog."""
        policy_cfg = self.config_raw.get("policy", {}) if isinstance(self.config_raw, dict) else {}
        laya_cfg = policy_cfg.get("laya", {})
        curr_cfg = policy_cfg.get("currency", {})
        policy_name = policy_cfg.get("name", "F_expected")
        
        policy_labels = {
            "F_expected": "F_expected (期望成本最小化：综合失误惩罚与验证决策)",
            "B_naive": "B_naive (基础朴素：满足预测成功门槛的最廉价模型)",
            "cheapest": "cheapest (纯贪心极简：始终选用目录中最低单价模型)",
            "best": "best (绝对旗舰：始终选用目录中最高能力画像模型)",
        }
        policy_display = policy_labels.get(policy_name, policy_name)

        device = laya_cfg.get("device", "auto")
        if device == "cuda":
            device_display = "NVIDIA CUDA GPU 加速"
        elif device == "mps":
            device_display = "Apple Metal (MPS) GPU 加速"
        elif device == "cpu":
            device_display = "CPU 多线程运算"
        elif device == "auto":
            device_display = "自动侦测最佳硬件 (Auto)"
        else:
            device_display = str(device)

        import torch
        if torch.cuda.is_available():
            try:
                hw_name = torch.cuda.get_device_name(0)
            except Exception:
                hw_name = "CUDA GPU"
            hw_tag = f"NVIDIA {hw_name} (CUDA) [ACTIVE]"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            hw_tag = "Apple Silicon (Metal MPS) [ACTIVE]"
        else:
            import platform
            hw_tag = f"CPU Multi-threading ({platform.machine()})"

        backend = policy_cfg.get("classifier", {}).get("backend", "local")
        backend_display = "local (本地 Laya 引擎)" if backend == "local" else ("typesafe (云端分类器)" if backend == "typesafe" else "mock (静态测试桩)")

        router_params = {
            "policy_name": policy_name,
            "policy_display": policy_display,
            "usd_cny_rate": self.usd_cny_rate,
            "currency_symbol": self.currency_symbol,
            "currency_base": curr_cfg.get("base", "CNY"),
            "stakes_usd": float(policy_cfg.get("stakes_usd", 2.0)),
            "detect_probability": float(policy_cfg.get("detect_probability", 0.6)),
            "failure_cost_multiplier": float(policy_cfg.get("failure_cost_multiplier", 1.0)),
            "remaining_turns_horizon": int(policy_cfg.get("remaining_turns_horizon", 3)),
        }

        laya_params = {
            "classifier_backend": backend,
            "classifier_backend_display": backend_display,
            "device": device,
            "device_display": device_display,
            "checkpoint": laya_cfg.get("checkpoint", "convaiinnovations/laya"),
            "subfolder": laya_cfg.get("subfolder", "multilingual"),
            "request_chars_cap": int(policy_cfg.get("request_chars_cap", 6000)),
            "hardware_tag": hw_tag,
        }

        active_models = []
        for m in self.catalog.models:
            if getattr(m, "launch_only", False):
                continue
            is_free = getattr(m, "free", False) or (m.prices and getattr(m.prices, "is_free", False))
            p_in = getattr(m.prices, "input", 0.0) if m.prices else 0.0
            p_out = getattr(m.prices, "output", 0.0) if m.prices else 0.0
            active_models.append({
                "name": m.name,
                "provider": getattr(m, "provider", "none"),
                "upstream_id": getattr(m, "upstream_id", m.name),
                "context_tokens": getattr(m, "context_tokens", 128000),
                "is_free": is_free,
                "input_cny": p_in,
                "output_cny": p_out,
                "input_usd": round(p_in / self.usd_cny_rate, 4) if self.usd_cny_rate > 0 else 0.0,
                "output_usd": round(p_out / self.usd_cny_rate, 4) if self.usd_cny_rate > 0 else 0.0,
            })

        return {
            "router_params": router_params,
            "laya_params": laya_params,
            "active_models": active_models,
        }

    def _get_baseline_models(self) -> tuple[ModelInfo | None, ModelInfo | None]:
        """Find the most expensive (frontier) and cheapest (local/free) models in catalog."""
        models = [m for m in self.catalog.models if not m.launch_only]
        if not models:
            return None, None
        expensive = max(models, key=lambda m: (m.prices.input + m.prices.output) if m.prices else 0.0)
        cheap = min(models, key=lambda m: (m.prices.input + m.prices.output) if m.prices else 0.0)
        return expensive, cheap

    def _calc_model_cost(self, model: ModelInfo, prompt_tokens: int, output_tokens: int) -> float:
        if not model.prices or getattr(model, "free", False) or getattr(model.prices, "is_free", False):
            return 0.0
        return (prompt_tokens * model.prices.input + output_tokens * model.prices.output) / 1_000_000.0

    def _build_jaeger_trace(
        self,
        cid: str,
        prompt: str,
        cat_expected: str,
        diff_tag: str,
        policy_name: str,
        raw_stages: list[dict],
        total_tokens: int,
        total_cost_usd: float,
        savings_pct: float,
        has_error: bool = False,
        is_escalated: bool = False,
    ) -> tuple[str, list[dict]]:
        """Build standards-compliant Jaeger / OpenTelemetry Spans hierarchy (Root + Children)."""
        import hashlib
        raw_hash = hashlib.sha256(f"{cid}_{prompt}".encode("utf-8")).hexdigest()
        trace_id = raw_hash[:16]
        root_span_id = trace_id[:8]

        tot_dur_ms = round(max((s.get("start_ms", 0.0) + s.get("duration_ms", 0.0)) for s in raw_stages), 2) if raw_stages else 0.0
        root_status = "error" if has_error else ("escalated" if is_escalated else "ok")

        root_span = {
            "span_id": root_span_id,
            "parent_span_id": None,
            "depth": 1,
            "service": "amra",
            "operation": "amra: /v1/chat/completions",
            "stage_type": "root",
            "name": "amra: /v1/chat/completions",
            "start_ms": 0.0,
            "duration_ms": tot_dur_ms,
            "status": root_status,
            "tags": {
                "component": "amra_router",
                "case.id": cid,
                "case.category": cat_expected,
                "case.difficulty": diff_tag,
                "router.policy": policy_name,
                "tokens.total": total_tokens,
                "cost.usd": f"${total_cost_usd:.6f}",
                "savings.pct": f"{savings_pct}%",
                "has_error": has_error,
                "is_escalated": is_escalated,
            },
            "process": {
                "env": "benchmark",
                "runtime": "python3.11",
                "service.version": "v1.2.0",
            },
            "logs": [
                {"time_ms": 0.0, "event": "request_received", "payload": prompt[:120] + ("..." if len(prompt) > 120 else "")},
                {"time_ms": tot_dur_ms, "event": "request_completed", "payload": f"Completed with status {root_status}"},
            ],
            "snippet": prompt[:140] + ("..." if len(prompt) > 140 else ""),
            "detail": f"全链路总时延: {tot_dur_ms}ms, 状态: {root_status}",
        }

        spans = [root_span]

        for idx, st in enumerate(raw_stages, start=1):
            child_span_id = f"{trace_id[:4]}c{idx:02d}"
            st_type = st.get("stage_type", "stage")
            dur_ms = round(st.get("duration_ms", 0.0), 2)
            start_ms = round(st.get("start_ms", 0.0), 2)
            st_status = st.get("status", "ok")

            if st_type == "classifier":
                svc = "laya"
                op = "laya: /classify & /route"
            elif st_type == "verifier":
                svc = "laya"
                op = "laya: /quality_check"
            elif st_type == "primary_model":
                svc = f"provider[{st.get('provider', 'primary')}]"
                op = f"{st.get('name', 'model')} /generate"
            elif st_type in ("escalate_model", "fallback_model", "retry_model"):
                svc = f"provider[{st.get('provider', 'escalate')}]"
                op = f"{st.get('name', 'model')} /escalated_generation"
            else:
                svc = st.get("provider", "service")
                op = st.get("name", "operation")

            tokens_info = st.get("tokens") or {}
            p_tok = tokens_info.get("prompt", 0)
            c_tok = tokens_info.get("completion", 0)
            t_tok = tokens_info.get("total", p_tok + c_tok)
            speed = round(c_tok / max(0.001, dur_ms / 1000.0), 1) if c_tok > 0 else 0.0

            tags = {
                "stage.type": st_type,
                "status": st_status,
            }
            if "provider" in st:
                tags["peer.service"] = str(st["provider"])
            if "name" in st:
                tags["model.id"] = str(st["name"])
            if t_tok > 0:
                tags["tokens.prompt"] = p_tok
                tags["tokens.completion"] = c_tok
                tags["tokens.total"] = t_tok
            if speed > 0:
                tags["tokens.per_second"] = speed
            if "cost_usd" in st and st["cost_usd"] > 0:
                tags["cost.usd"] = f"${st['cost_usd']:.6f}"
            if "detail" in st and st["detail"]:
                tags["stage.detail"] = str(st["detail"])

            logs = []
            if "snippet" in st and st["snippet"]:
                logs.append({"time_ms": dur_ms, "event": "output_snippet", "payload": str(st["snippet"])})

            child_span = {
                "span_id": child_span_id,
                "parent_span_id": root_span_id,
                "depth": 2,
                "service": svc,
                "operation": op,
                "stage_type": st_type,
                "name": st.get("name", op),
                "provider": st.get("provider", svc),
                "start_ms": start_ms,
                "duration_ms": dur_ms,
                "status": st_status,
                "tokens": tokens_info,
                "cost_usd": st.get("cost_usd", 0.0),
                "tags": tags,
                "process": {
                    "env": "benchmark",
                    "service": svc,
                },
                "logs": logs,
                "snippet": st.get("snippet", ""),
                "detail": st.get("detail", ""),
            }
            spans.append(child_span)

        return trace_id, spans


    def run_simulation(self, cases: list[dict], on_progress: Callable[[int, int], None] | None = None) -> BenchmarkSummary:
        """Run fast simulation across all cases using Laya forward pass on local M4 Max."""
        expensive_model, cheap_model = self._get_baseline_models()
        results: list[CaseResult] = []
        model_dist: dict[str, int] = {}
        cat_stats: dict[str, dict[str, Any]] = {}
        total_classifier_latency = 0.0

        for idx, item in enumerate(cases):
            cid = item.get("id", f"case_{idx+1}")
            prompt = item.get("prompt", "")
            cat_expected = item.get("category", "general")
            diff_tag = item.get("difficulty_tag", "easy")
            expected_tier = item.get("expected_tier", "cheap")
            prompt_tokens = int(item.get("estimated_prompt_tokens", len(prompt) // 2 or 30))
            output_tokens = int(item.get("estimated_output_tokens", 100))

            # 1. Run full router pipeline
            try:
                route_res = self.router.route(
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=output_tokens
                )
                chosen_model = route_res.model
                chosen_name = chosen_model.name
                reason = route_res.reason
                clf = route_res.classification
                lat_ms = route_res.classification_ms or (clf.latency_s * 1000.0 if clf else 30.0)
            except Exception as e:
                chosen_model = cheap_model
                chosen_name = cheap_model.name if cheap_model else "fallback"
                reason = f"error: {e}"
                clf = None
                lat_ms = 30.0

            total_classifier_latency += lat_ms

            # 3. Compute cost comparison
            cost_router = self._calc_model_cost(chosen_model, prompt_tokens, output_tokens) if chosen_model else 0.0
            cost_exp = self._calc_model_cost(expensive_model, prompt_tokens, output_tokens) if expensive_model else cost_router
            cost_chp = self._calc_model_cost(cheap_model, prompt_tokens, output_tokens) if cheap_model else 0.0

            savings_usd = max(0.0, cost_exp - cost_router)
            savings_pct = round((savings_usd / cost_exp * 100.0), 2) if cost_exp > 0 else 0.0

            # Alignment check:
            # If expected_tier is cheap/mid and router chose cheap/mid -> aligned.
            # If expected_tier is expensive and router chose expensive -> aligned.
            tier_chosen = "cheap" if (chosen_model and (getattr(chosen_model, "free", False) or getattr(chosen_model.prices, "is_free", False))) else ("expensive" if (chosen_model == expensive_model) else "mid")
            is_aligned = (expected_tier == "expensive" and tier_chosen == "expensive") or (expected_tier in ("cheap", "mid") and tier_chosen in ("cheap", "mid"))

            model_dist[chosen_name] = model_dist.get(chosen_name, 0) + 1

            cat_info = cat_stats.setdefault(cat_expected, {"total": 0, "savings_usd": 0.0, "cheap_count": 0, "exp_count": 0})
            cat_info["total"] += 1
            cat_info["savings_usd"] += savings_usd
            if tier_chosen == "cheap":
                cat_info["cheap_count"] += 1
            elif tier_chosen == "expensive":
                cat_info["exp_count"] += 1

            # 构建仿真模拟的完整执行链路 (Trace Chain)
            dev_str = getattr(self.router.classifier, "actual_device", "mps").upper() if hasattr(self.router, "classifier") else "MPS"
            raw_stages: list[dict] = [{
                "stage_index": 1,
                "stage_type": "classifier",
                "name": f"Laya 决策引擎 ({dev_str})",
                "provider": "local_engine",
                "status": "success",
                "start_ms": 0.0,
                "duration_ms": round(lat_ms, 2),
                "tokens": {"prompt": prompt_tokens, "completion": 0, "total": prompt_tokens},
                "cost_usd": 0.0,
                "detail": f"类别: {clf.category if clf else cat_expected} | 难度: {clf.difficulty if clf else 0.5:.2f} | 风险代价: {clf.stakes if clf else 0.2:.2f}",
                "snippet": f"策略仲裁选定首选目标: [{chosen_name}] ({reason})",
            }]

            is_chosen_local = getattr(chosen_model, "free", False) or "local" in getattr(chosen_model, "provider", "").lower()
            sim_primary_ms = 140.0 if is_chosen_local else 420.0

            # 模拟推演：若为高难度/高风险任务但初选分配了较弱模型，按概率推演触发质量验收不合格并升级
            p_failure_sim = max(0.0, min(0.85, ((clf.difficulty if clf else 0.5) - 0.45) * 1.5)) if (clf and tier_chosen == "cheap" and diff_tag == "hard") else 0.0
            will_escalate = (p_failure_sim > 0.40)

            if not will_escalate:
                raw_stages.append({
                    "stage_index": 2,
                    "stage_type": "primary_model",
                    "name": chosen_name,
                    "provider": getattr(chosen_model, "provider", "none"),
                    "status": "success",
                    "start_ms": round(lat_ms, 2),
                    "duration_ms": round(sim_primary_ms, 2),
                    "tokens": {"prompt": prompt_tokens, "completion": output_tokens, "total": prompt_tokens + output_tokens},
                    "cost_usd": round(cost_router, 6),
                    "detail": "首选模型推演完成 (HTTP 200 模拟响应)",
                    "snippet": f"首选模型生成完毕，Token 消耗: {prompt_tokens + output_tokens}",
                })
                raw_stages.append({
                    "stage_index": 3,
                    "stage_type": "verifier",
                    "name": "Laya 质量验收裁决 (Adequacy Check)",
                    "provider": "local_engine",
                    "status": "adequate",
                    "start_ms": round(lat_ms + sim_primary_ms, 2),
                    "duration_ms": 12.0,
                    "tokens": {"prompt": 0, "completion": 0, "total": 0},
                    "cost_usd": 0.0,
                    "detail": "质量判定满意度高 (p_adequate ≈ 0.95)，通过验收无需二次升级",
                    "snippet": "裁决通过，直接采纳首选模型输出作为最终交付",
                })
                hop_count = 1
                is_case_escalated = False
            else:
                raw_stages.append({
                    "stage_index": 2,
                    "stage_type": "primary_model",
                    "name": chosen_name,
                    "provider": getattr(chosen_model, "provider", "none"),
                    "status": "unmet",
                    "start_ms": round(lat_ms, 2),
                    "duration_ms": round(sim_primary_ms, 2),
                    "tokens": {"prompt": prompt_tokens, "completion": output_tokens, "total": prompt_tokens + output_tokens},
                    "cost_usd": round(cost_router, 6),
                    "detail": "初选轻量模型生成完成，但推演质量存疑",
                    "snippet": "生成结果较为简略，未充分覆盖复杂难点",
                })
                ver_ms = 16.0
                raw_stages.append({
                    "stage_index": 3,
                    "stage_type": "verifier",
                    "name": "Laya 质量验收裁决 (Adequacy Check)",
                    "provider": "local_engine",
                    "status": "escalate_recommended",
                    "start_ms": round(lat_ms + sim_primary_ms, 2),
                    "duration_ms": ver_ms,
                    "tokens": {"prompt": 0, "completion": 0, "total": 0},
                    "cost_usd": 0.0,
                    "detail": f"质量判定预估满意度不足 (p_adequate ≈ {1.0 - p_failure_sim:.2f} < 0.65)，建议重新转发至高阶模型",
                    "snippet": "触发 Scott-Shapiro 升级规约，转向高阶基准模型",
                })
                esc_model = expensive_model or chosen_model
                esc_cost = self._calc_model_cost(esc_model, prompt_tokens, output_tokens)
                esc_dur_ms = 460.0
                raw_stages.append({
                    "stage_index": 4,
                    "stage_type": "escalate_model",
                    "name": getattr(esc_model, "name", "escalated-model"),
                    "provider": getattr(esc_model, "provider", "none"),
                    "status": "success",
                    "start_ms": round(lat_ms + sim_primary_ms + ver_ms, 2),
                    "duration_ms": esc_dur_ms,
                    "tokens": {"prompt": prompt_tokens, "completion": output_tokens, "total": prompt_tokens + output_tokens},
                    "cost_usd": round(esc_cost, 6),
                    "detail": "二次升级高阶模型执行成功，深度解答难例",
                    "snippet": f"高阶模型重新推演生成完毕，单次消耗费用: ${esc_cost:.6f}",
                })
                hop_count = 2
                is_case_escalated = True

            policy_name = (self.config_raw.get("policy", {}) or {}).get("name", "F_expected")
            trace_id, full_spans = self._build_jaeger_trace(
                cid=cid,
                prompt=prompt,
                cat_expected=cat_expected,
                diff_tag=diff_tag,
                policy_name=policy_name,
                raw_stages=raw_stages,
                total_tokens=prompt_tokens + output_tokens,
                total_cost_usd=cost_router,
                savings_pct=savings_pct,
                has_error=False,
                is_escalated=is_case_escalated,
            )

            results.append(
                CaseResult(
                    case_id=cid,
                    category=cat_expected,
                    difficulty_tag=diff_tag,
                    prompt=prompt,
                    expected_tier=expected_tier,
                    detected_category=clf.category if clf else cat_expected,
                    detected_difficulty=round(clf.difficulty, 3) if clf else 0.5,
                    detected_stakes=round(clf.stakes, 3) if clf else 0.5,
                    classifier_latency_ms=round(lat_ms, 2),
                    chosen_model=chosen_name,
                    chosen_provider=chosen_model.provider if chosen_model else "none",
                    decision_reason=reason,
                    prompt_tokens=prompt_tokens,
                    output_tokens=output_tokens,
                    total_tokens=prompt_tokens + output_tokens,
                    cost_router=round(cost_router, 6),
                    cost_expensive=round(cost_exp, 6),
                    cost_cheap=round(cost_chp, 6),
                    savings_usd=round(savings_usd, 6),
                    savings_pct=savings_pct,
                    is_aligned=is_aligned,
                    trace_id=trace_id,
                    trace_chain=full_spans,
                    hop_count=hop_count,
                    escalated=is_case_escalated,
                )
            )
            if on_progress:
                on_progress(idx + 1, len(cases))

        # Overall summary
        tot_router = sum(r.cost_router for r in results)
        tot_exp = sum(r.cost_expensive for r in results)
        tot_chp = sum(r.cost_cheap for r in results)
        tot_savings = max(0.0, tot_exp - tot_router)
        tot_savings_pct = round((tot_savings / tot_exp * 100.0), 2) if tot_exp > 0 else 0.0
        aligned_cnt = sum(1 for r in results if r.is_aligned)

        # 构造宏观拓扑流向统计
        trace_topology_stats = {
            "total_cases": len(results),
            "direct_success_count": sum(1 for r in results if not r.escalated and not r.error),
            "escalated_count": sum(1 for r in results if r.escalated),
            "error_count": sum(1 for r in results if r.error),
            "hop_distribution": {
                "1_hop": sum(1 for r in results if r.hop_count == 1),
                "2_hops": sum(1 for r in results if r.hop_count >= 2),
            },
            "avg_classifier_ms": round(total_classifier_latency / max(1, len(results)), 2),
            "avg_primary_ms": round(sum(s["duration_ms"] for r in results for s in r.trace_chain if s.get("stage_type") == "primary_model") / max(1, len(results)), 2),
            "avg_escalate_ms": round(sum(s["duration_ms"] for r in results for s in r.trace_chain if s.get("stage_type") in ("escalate_model", "fallback_model", "retry_model")) / max(1, sum(1 for r in results if r.escalated)), 2) if any(r.escalated for r in results) else 0.0,
        }

        return BenchmarkSummary(
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            mode="simulation",
            total_cases=len(results),
            successful_cases=len(results),
            total_tokens=sum(r.total_tokens for r in results),
            cost_router_total=round(tot_router, 6),
            cost_expensive_total=round(tot_exp, 6),
            cost_cheap_total=round(tot_chp, 6),
            total_savings_usd=round(tot_savings, 6),
            total_savings_pct=tot_savings_pct,
            avg_classifier_latency_ms=round(total_classifier_latency / max(1, len(results)), 2),
            avg_total_latency_s=round((total_classifier_latency / 1000.0) / max(1, len(results)), 3),
            alignment_rate=round(aligned_cnt / max(1, len(results)) * 100.0, 1),
            model_distribution=model_dist,
            category_breakdown=cat_stats,
            currency_symbol=self.currency_symbol,
            usd_cny_rate=self.usd_cny_rate,
            config_snapshot=self._build_config_snapshot(),
            trace_topology_stats=trace_topology_stats,
            results=results,
        )

    async def run_real_benchmark(self, cases: list[dict], on_progress: Callable[[int, int], None] | None = None) -> BenchmarkSummary:
        """Run real benchmark with controlled concurrency (Semaphore=2) to avoid timeouts."""
        expensive_model, cheap_model = self._get_baseline_models()
        model_dist: dict[str, int] = {}
        cat_stats: dict[str, dict[str, Any]] = {}
        total_classifier_latency = 0.0
        total_request_latency = 0.0

        sem = asyncio.Semaphore(2)
        loop = asyncio.get_event_loop()

        async with httpx.AsyncClient(timeout=25.0) as client:
            async def run_single_case(idx: int, item: dict) -> CaseResult:
                nonlocal total_classifier_latency, total_request_latency
                cid = item.get("id", f"case_{idx+1}")
                prompt = item.get("prompt", "")
                cat_expected = item.get("category", "general")
                diff_tag = item.get("difficulty_tag", "easy")
                expected_tier = item.get("expected_tier", "cheap")
                prompt_tokens = int(item.get("estimated_prompt_tokens", len(prompt) // 2 or 30))
                output_tokens = int(item.get("estimated_output_tokens", 100))

                # 1. 运行标准路由决策流程（在线程池中运行 Laya 推断）
                try:
                    route_res = await loop.run_in_executor(
                        None,
                        lambda: self.router.route(
                            messages=[{"role": "user", "content": prompt}],
                            max_tokens=output_tokens
                        )
                    )
                    chosen_model = route_res.model
                    chosen_name = chosen_model.name
                    reason = route_res.reason
                    clf = route_res.classification
                    lat_ms = route_res.classification_ms or (clf.latency_s * 1000.0 if clf else 30.0)
                except Exception as e:
                    chosen_model = cheap_model
                    chosen_name = cheap_model.name if cheap_model else "fallback"
                    reason = f"route error: {e}"
                    clf = None
                    lat_ms = 30.0

                total_classifier_latency += lat_ms

                raw_stages: list[dict] = []
                dev_str = getattr(self.router.classifier, "actual_device", "mps").upper() if hasattr(self.router, "classifier") else "MPS"
                raw_stages.append({
                    "stage_index": 1,
                    "stage_type": "classifier",
                    "name": f"Laya 决策引擎 ({dev_str})",
                    "provider": "local_engine",
                    "status": "success",
                    "start_ms": 0.0,
                    "duration_ms": round(lat_ms, 2),
                    "tokens": {"prompt": prompt_tokens, "completion": 0, "total": prompt_tokens},
                    "cost_usd": 0.0,
                    "detail": f"类别: {clf.category if clf else cat_expected} | 难度: {clf.difficulty if clf else 0.5:.2f} | 风险代价: {clf.stakes if clf else 0.2:.2f}",
                    "snippet": f"策略仲裁首选: [{chosen_name}] ({reason})",
                })

                # 2. 向选中模型发起首次真实调用
                provider_id = getattr(chosen_model, "provider", None) if chosen_model else None
                provider = None
                if provider_id and hasattr(self.router, "config") and hasattr(self.router.config, "providers"):
                    provider = self.router.config.providers.get(provider_id)

                real_response_text = ""
                real_p_tokens = prompt_tokens
                real_o_tokens = output_tokens
                real_dur = 0.0
                err_msg = ""
                hop_count = 1
                is_case_escalated = False
                cur_start_ms = lat_ms

                async def call_model_api(target_m, target_p, max_tok):
                    if not target_p or not target_p.base_url:
                        return False, 0.0, "", 0, 0, f"未配置提供商或端点 (provider: {getattr(target_m, 'provider', 'none')})"
                    base_u = (getattr(target_p, "resolved_base_url", None) or target_p.base_url or "").strip().rstrip("/")
                    endpoint = base_u if base_u.endswith("/chat/completions") else f"{base_u}/chat/completions"
                    hdrs = {"Content-Type": "application/json"}
                    if target_p.api_key:
                        hdrs["Authorization"] = f"Bearer {target_p.api_key}"
                    if hasattr(target_p, "extra_headers") and target_p.extra_headers:
                        hdrs.update(target_p.extra_headers)
                    up_id = getattr(target_m, "upstream_id", None) or getattr(target_m, "name", "default")
                    pld = {
                        "model": up_id,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": min(max_tok, 150),
                        "temperature": 0.7,
                    }
                    to_s = float(
                        getattr(target_m, "timeout_s", None)
                        or (self.config_raw.get("policy", {}) or {}).get("request_timeout_seconds")
                        or 60.0
                    )
                    t0 = time.perf_counter()
                    try:
                        resp = await client.post(endpoint, json=pld, headers=hdrs, timeout=to_s)
                        dur = time.perf_counter() - t0
                        if resp.status_code == 200:
                            d = resp.json()
                            chs = d.get("choices", [])
                            txt = chs[0].get("message", {}).get("content", "") if chs else ""
                            usg = d.get("usage", {})
                            pt = usg.get("prompt_tokens", prompt_tokens)
                            ct = usg.get("completion_tokens", len(txt) // 2 or max_tok)
                            return True, dur, txt, pt, ct, ""
                        else:
                            return False, dur, "", 0, 0, f"HTTP {resp.status_code}: {resp.text[:160]}"
                    except httpx.TimeoutException:
                        dur = time.perf_counter() - t0
                        return False, dur, "", 0, 0, f"调用超时 ({to_s:.0f}s)"
                    except Exception as e:
                        dur = time.perf_counter() - t0
                        return False, dur, "", 0, 0, f"网络异常: {e}"

                async with sem:
                    # 尝试首选模型
                    succ, dur_s, ans_txt, pt, ct, err = await call_model_api(chosen_model, provider, output_tokens)
                    real_dur += dur_s
                    stage_dur_ms = round(dur_s * 1000.0, 2)

                    if succ:
                        real_response_text = ans_txt
                        real_p_tokens = pt
                        real_o_tokens = ct
                        raw_stages.append({
                            "stage_index": 2,
                            "stage_type": "primary_model",
                            "name": chosen_name,
                            "provider": getattr(chosen_model, "provider", "none"),
                            "status": "success",
                            "start_ms": round(cur_start_ms, 2),
                            "duration_ms": stage_dur_ms,
                            "tokens": {"prompt": pt, "completion": ct, "total": pt + ct},
                            "cost_usd": round(self._calc_model_cost(chosen_model, pt, ct), 6),
                            "detail": f"首选模型响应成功 (200 OK)",
                            "snippet": (ans_txt[:140] + "...") if len(ans_txt) > 140 else ans_txt,
                        })
                        cur_start_ms += stage_dur_ms

                        # 进行 Laya 质量验收判定
                        if hasattr(self.router, "check") and ans_txt:
                            t_chk_0 = time.perf_counter()
                            try:
                                verdict = await loop.run_in_executor(None, lambda: self.router.check(route_res, prompt, ans_txt))
                                chk_dur_ms = round((time.perf_counter() - t_chk_0) * 1000.0, 2)
                                should_escalate = getattr(verdict, "escalate", False)

                                raw_stages.append({
                                    "stage_index": 3,
                                    "stage_type": "verifier",
                                    "name": "Laya 质量验收裁决 (Adequacy Check)",
                                    "provider": "local_engine",
                                    "status": "escalate_recommended" if should_escalate else "adequate",
                                    "start_ms": round(cur_start_ms, 2),
                                    "duration_ms": chk_dur_ms,
                                    "tokens": {"prompt": 0, "completion": 0, "total": 0},
                                    "cost_usd": 0.0,
                                    "detail": f"满意度评分: {getattr(verdict, 'p_adequate', 'N/A')}, 是否升级: {should_escalate}",
                                    "snippet": getattr(verdict, "reason", "质量验收裁决完成"),
                                })
                                cur_start_ms += chk_dur_ms

                                # 若建议升级，且有更强的高阶模型，触发升级重新转发
                                if should_escalate and expensive_model and expensive_model.name != chosen_name:
                                    esc_prov = self.router.config.providers.get(expensive_model.provider) if hasattr(self.router.config, "providers") else None
                                    e_succ, e_dur, e_txt, e_pt, e_ct, e_err = await call_model_api(expensive_model, esc_prov, output_tokens)
                                    real_dur += e_dur
                                    esc_dur_ms = round(e_dur * 1000.0, 2)
                                    if e_succ:
                                        real_response_text = e_txt
                                        real_p_tokens = e_pt
                                        real_o_tokens = e_ct
                                        chosen_model = expensive_model
                                        chosen_name = expensive_model.name
                                        is_case_escalated = True
                                        hop_count = 2
                                        raw_stages.append({
                                            "stage_index": 4,
                                            "stage_type": "escalate_model",
                                            "name": expensive_model.name,
                                            "provider": getattr(expensive_model, "provider", "none"),
                                            "status": "success",
                                            "start_ms": round(cur_start_ms, 2),
                                            "duration_ms": esc_dur_ms,
                                            "tokens": {"prompt": e_pt, "completion": e_ct, "total": e_pt + e_ct},
                                            "cost_usd": round(self._calc_model_cost(expensive_model, e_pt, e_ct), 6),
                                            "detail": f"升级至高阶模型调用成功 (200 OK)",
                                            "snippet": (e_txt[:140] + "...") if len(e_txt) > 140 else e_txt,
                                        })
                            except Exception:
                                pass
                    else:
                        # 首选模型调用失败或超时，记录并尝试容灾降级调度
                        err_msg = err
                        raw_stages.append({
                            "stage_index": 2,
                            "stage_type": "primary_model",
                            "name": chosen_name,
                            "provider": getattr(chosen_model, "provider", "none"),
                            "status": "timeout" if "超时" in err else "error",
                            "start_ms": round(cur_start_ms, 2),
                            "duration_ms": stage_dur_ms,
                            "tokens": {"prompt": prompt_tokens, "completion": 0, "total": prompt_tokens},
                            "cost_usd": 0.0,
                            "detail": f"调用失败: {err}",
                            "snippet": "首次选择模型未解决问题，准备自动容灾重试",
                        })
                        cur_start_ms += stage_dur_ms

                        # 挑选备选模型容灾
                        fallback_cand = expensive_model if (expensive_model and expensive_model.name != chosen_name) else cheap_model
                        if fallback_cand and fallback_cand.name != chosen_name:
                            fb_prov = self.router.config.providers.get(fallback_cand.provider) if hasattr(self.router.config, "providers") else None
                            fb_succ, fb_dur, fb_txt, fb_pt, fb_ct, fb_err = await call_model_api(fallback_cand, fb_prov, output_tokens)
                            real_dur += fb_dur
                            fb_dur_ms = round(fb_dur * 1000.0, 2)
                            if fb_succ:
                                real_response_text = fb_txt
                                real_p_tokens = fb_pt
                                real_o_tokens = fb_ct
                                chosen_model = fallback_cand
                                chosen_name = fallback_cand.name
                                is_case_escalated = True
                                hop_count = 2
                                err_msg = ""
                                raw_stages.append({
                                    "stage_index": 3,
                                    "stage_type": "fallback_model",
                                    "name": fallback_cand.name,
                                    "provider": getattr(fallback_cand, "provider", "none"),
                                    "status": "success",
                                    "start_ms": round(cur_start_ms, 2),
                                    "duration_ms": fb_dur_ms,
                                    "tokens": {"prompt": fb_pt, "completion": fb_ct, "total": fb_pt + fb_ct},
                                    "cost_usd": round(self._calc_model_cost(fallback_cand, fb_pt, fb_ct), 6),
                                    "detail": f"容灾转发备选模型执行成功",
                                    "snippet": (fb_txt[:140] + "...") if len(fb_txt) > 140 else fb_txt,
                                })

                total_request_latency += real_dur

                cost_router = self._calc_model_cost(chosen_model, real_p_tokens, real_o_tokens) if chosen_model else 0.0
                cost_exp = self._calc_model_cost(expensive_model, real_p_tokens, real_o_tokens) if expensive_model else cost_router
                cost_chp = self._calc_model_cost(cheap_model, real_p_tokens, real_o_tokens) if cheap_model else 0.0
                savings_usd = max(0.0, cost_exp - cost_router)
                savings_pct = round((savings_usd / cost_exp * 100.0), 2) if cost_exp > 0 else 0.0

                tier_chosen = "cheap" if (chosen_model and (getattr(chosen_model, "free", False) or getattr(chosen_model.prices, "is_free", False))) else ("expensive" if (chosen_model == expensive_model) else "mid")
                is_aligned = (expected_tier == "expensive" and tier_chosen == "expensive") or (expected_tier in ("cheap", "mid") and tier_chosen in ("cheap", "mid"))

                if on_progress:
                    on_progress(idx + 1, len(cases))

                policy_name = (self.config_raw.get("policy", {}) or {}).get("name", "F_expected")
                trace_id, full_spans = self._build_jaeger_trace(
                    cid=cid,
                    prompt=prompt,
                    cat_expected=cat_expected,
                    diff_tag=diff_tag,
                    policy_name=policy_name,
                    raw_stages=raw_stages,
                    total_tokens=real_p_tokens + real_o_tokens,
                    total_cost_usd=cost_router,
                    savings_pct=savings_pct,
                    has_error=bool(err_msg),
                    is_escalated=is_case_escalated,
                )

                return CaseResult(
                    case_id=cid,
                    category=cat_expected,
                    difficulty_tag=diff_tag,
                    prompt=prompt,
                    expected_tier=expected_tier,
                    detected_category=clf.category if clf else cat_expected,
                    detected_difficulty=round(clf.difficulty, 3) if clf else 0.5,
                    detected_stakes=round(clf.stakes, 3) if clf else 0.5,
                    classifier_latency_ms=round(lat_ms, 2),
                    chosen_model=chosen_name,
                    chosen_provider=getattr(chosen_model, "provider", "none") if chosen_model else "none",
                    decision_reason=reason,
                    prompt_tokens=real_p_tokens,
                    output_tokens=real_o_tokens,
                    total_tokens=real_p_tokens + real_o_tokens,
                    cost_router=round(cost_router, 6),
                    cost_expensive=round(cost_exp, 6),
                    cost_cheap=round(cost_chp, 6),
                    savings_usd=round(savings_usd, 6),
                    savings_pct=savings_pct,
                    is_aligned=is_aligned,
                    real_execution=True,
                    real_latency_s=round(real_dur, 3),
                    real_response=real_response_text,
                    error=err_msg,
                    trace_id=trace_id,
                    trace_chain=full_spans,
                    hop_count=hop_count,
                    escalated=is_case_escalated,
                )

            tasks = [run_single_case(i, case) for i, case in enumerate(cases)]
            results = await asyncio.gather(*tasks)

        # 汇总统计
        for r in results:
            model_dist[r.chosen_model] = model_dist.get(r.chosen_model, 0) + 1
            cat_info = cat_stats.setdefault(r.category, {"total": 0, "savings_usd": 0.0, "cheap_count": 0, "exp_count": 0})
            cat_info["total"] += 1
            cat_info["savings_usd"] += r.savings_usd
            if r.cost_router == 0.0 or r.savings_pct > 80.0:
                cat_info["cheap_count"] += 1
            elif r.cost_router == r.cost_expensive:
                cat_info["exp_count"] += 1

        tot_router = sum(r.cost_router for r in results)
        tot_exp = sum(r.cost_expensive for r in results)
        tot_chp = sum(r.cost_cheap for r in results)
        tot_savings = max(0.0, tot_exp - tot_router)
        tot_savings_pct = round((tot_savings / tot_exp * 100.0), 2) if tot_exp > 0 else 0.0
        aligned_cnt = sum(1 for r in results if r.is_aligned)

        trace_topology_stats = {
            "total_cases": len(results),
            "direct_success_count": sum(1 for r in results if not r.escalated and not r.error),
            "escalated_count": sum(1 for r in results if r.escalated),
            "error_count": sum(1 for r in results if r.error),
            "hop_distribution": {
                "1_hop": sum(1 for r in results if r.hop_count == 1),
                "2_hops": sum(1 for r in results if r.hop_count >= 2),
            },
            "avg_classifier_ms": round(total_classifier_latency / max(1, len(results)), 2),
            "avg_primary_ms": round(sum(s["duration_ms"] for r in results for s in r.trace_chain if s.get("stage_type") == "primary_model") / max(1, len(results)), 2),
            "avg_escalate_ms": round(sum(s["duration_ms"] for r in results for s in r.trace_chain if s.get("stage_type") in ("escalate_model", "fallback_model", "retry_model")) / max(1, sum(1 for r in results if r.escalated)), 2) if any(r.escalated for r in results) else 0.0,
        }

        return BenchmarkSummary(
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            mode="real_benchmark",
            total_cases=len(results),
            successful_cases=sum(1 for r in results if not r.error),
            total_tokens=sum(r.total_tokens for r in results),
            cost_router_total=round(tot_router, 6),
            cost_expensive_total=round(tot_exp, 6),
            cost_cheap_total=round(tot_chp, 6),
            total_savings_usd=round(tot_savings, 6),
            total_savings_pct=tot_savings_pct,
            avg_classifier_latency_ms=round(total_classifier_latency / max(1, len(results)), 2),
            avg_total_latency_s=round(total_request_latency / max(1, len(results)), 3),
            alignment_rate=round(aligned_cnt / max(1, len(results)) * 100.0, 1),
            model_distribution=model_dist,
            category_breakdown=cat_stats,
            currency_symbol=self.currency_symbol,
            usd_cny_rate=self.usd_cny_rate,
            config_snapshot=self._build_config_snapshot(),
            trace_topology_stats=trace_topology_stats,
            results=list(results),
        )

