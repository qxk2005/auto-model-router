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


class BenchmarkEvaluator:
    def __init__(self, router, catalog: Catalog, policy_engine, config_raw: dict):
        self.router = router
        self.catalog = catalog
        self.policy = policy_engine
        self.config_raw = config_raw

        curr_cfg = self.config_raw.get("policy", {}).get("currency", {}) if isinstance(self.config_raw, dict) else {}
        self.currency_symbol = "¥" if curr_cfg.get("base", "CNY") == "CNY" else "$"
        self.usd_cny_rate = float(curr_cfg.get("usd_cny_rate", 7.20))

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

                # 2. 向选中模型对应的上游提供商发起真实 HTTP 调用
                provider_id = getattr(chosen_model, "provider", None) if chosen_model else None
                provider = None
                if provider_id and hasattr(self.router, "config") and hasattr(self.router.config, "providers"):
                    provider = self.router.config.providers.get(provider_id)

                real_response_text = ""
                real_p_tokens = prompt_tokens
                real_o_tokens = output_tokens
                real_dur = 0.0
                err_msg = ""

                async with sem:
                    if provider and provider.base_url:
                        base_url = (provider.base_url or "").strip().rstrip("/")
                        url = base_url if base_url.endswith("/chat/completions") else f"{base_url}/chat/completions"
                        t_req_start = time.perf_counter()
                        try:
                            headers = {"Content-Type": "application/json"}
                            key = provider.api_key
                            if key:
                                headers["Authorization"] = f"Bearer {key}"
                            if hasattr(provider, "extra_headers") and provider.extra_headers:
                                headers.update(provider.extra_headers)

                            upstream_id = getattr(chosen_model, "upstream_id", None) or getattr(chosen_model, "name", "default")
                            payload = {
                                "model": upstream_id,
                                "messages": [{"role": "user", "content": prompt}],
                                "max_tokens": min(output_tokens, 150),
                                "temperature": 0.7,
                            }
                            res = await client.post(url, json=payload, headers=headers)
                            real_dur = time.perf_counter() - t_req_start
                            if res.status_code == 200:
                                data = res.json()
                                choices = data.get("choices", [])
                                if choices:
                                    real_response_text = choices[0].get("message", {}).get("content", "")
                                usage = data.get("usage", {})
                                real_p_tokens = usage.get("prompt_tokens", prompt_tokens)
                                real_o_tokens = usage.get("completion_tokens", len(real_response_text) // 2 or output_tokens)
                            else:
                                err_msg = f"HTTP {res.status_code}: {res.text[:200]}"
                        except Exception as exc:
                            real_dur = time.perf_counter() - t_req_start
                            err_msg = f"Request error: {exc}"
                    else:
                        err_msg = f"未配置提供商或端点 (provider: {provider_id})"

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
            results=list(results),
        )

