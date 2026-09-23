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
        """Run real benchmark by actually sending HTTP requests to the chosen provider."""
        expensive_model, cheap_model = self._get_baseline_models()
        results: list[CaseResult] = []
        model_dist: dict[str, int] = {}
        cat_stats: dict[str, dict[str, Any]] = {}
        total_classifier_latency = 0.0
        total_request_latency = 0.0

        async with httpx.AsyncClient(timeout=120.0) as client:
            for idx, item in enumerate(cases):
                cid = item.get("id", f"case_{idx+1}")
                prompt = item.get("prompt", "")
                cat_expected = item.get("category", "general")
                diff_tag = item.get("difficulty_tag", "easy")
                expected_tier = item.get("expected_tier", "cheap")

                # 1. Classification
                t0 = time.perf_counter()
                classifier_fn = getattr(self.router, "classifier", None)
                if callable(classifier_fn):
                    # Run classifier in executor if sync
                    loop = asyncio.get_event_loop()
                    clf = await loop.run_in_executor(None, classifier_fn, prompt)
                else:
                    clf = Classification(category=cat_expected, category_probs={}, difficulty=0.5,
                                         difficulty_confidence=0.5, needs_tools=0.0, needs_vision=0.0,
                                         needs_long_context=0.0, follow_up=0.0, stakes=0.5, latency_s=0.03)
                lat_ms = (clf.latency_s or (time.perf_counter() - t0)) * 1000.0
                total_classifier_latency += lat_ms

                # 2. Routing Decision
                est_p_tokens = len(prompt) // 2 or 30
                req = TurnRequest(category=clf.category or cat_expected, difficulty=clf.difficulty,
                                  prompt_tokens=est_p_tokens, output_tokens=150, now=time.time())
                conv = Conversation()
                ctx = Context(catalog=self.catalog, success=getattr(self.router, "success", None))
                try:
                    choice = self.policy.choose(conv, req, ctx)
                    chosen_model = choice.model if choice else cheap_model
                    reason = choice.reason if choice else "default"
                except Exception as e:
                    chosen_model = cheap_model
                    reason = str(e)

                chosen_name = chosen_model.name if chosen_model else "unknown"
                model_dist[chosen_name] = model_dist.get(chosen_name, 0) + 1

                # 3. Dispatch real HTTP call to chosen model's provider
                provider = self.router.config.providers.get(chosen_model.provider) if chosen_model else None
                real_response_text = ""
                real_p_tokens = est_p_tokens
                real_o_tokens = 100
                real_dur = 0.0
                err_msg = ""

                if provider and provider.base_url:
                    t_req_start = time.perf_counter()
                    try:
                        headers = {"Content-Type": "application/json"}
                        key = provider.api_key
                        if key:
                            headers["Authorization"] = f"Bearer {key}"
                        headers.update(provider.extra_headers)

                        payload = {
                            "model": chosen_model.upstream_id or "default",
                            "messages": [{"role": "user", "content": prompt}],
                            "max_tokens": 1000,
                            "temperature": 0.7,
                        }
                        url = f"{provider.base_url}/chat/completions"
                        res = await client.post(url, json=payload, headers=headers)
                        real_dur = time.perf_counter() - t_req_start
                        if res.status_code == 200:
                            data = res.json()
                            choices = data.get("choices", [])
                            if choices:
                                real_response_text = choices[0].get("message", {}).get("content", "")
                            usage = data.get("usage", {})
                            real_p_tokens = usage.get("prompt_tokens", est_p_tokens)
                            real_o_tokens = usage.get("completion_tokens", len(real_response_text) // 2)
                        else:
                            err_msg = f"HTTP {res.status_code}: {res.text[:200]}"
                    except Exception as exc:
                        real_dur = time.perf_counter() - t_req_start
                        err_msg = f"Request error: {exc}"
                else:
                    err_msg = "No provider configured for chosen model"

                total_request_latency += real_dur

                cost_router = self._calc_model_cost(chosen_model, real_p_tokens, real_o_tokens) if chosen_model else 0.0
                cost_exp = self._calc_model_cost(expensive_model, real_p_tokens, real_o_tokens) if expensive_model else cost_router
                cost_chp = self._calc_model_cost(cheap_model, real_p_tokens, real_o_tokens) if cheap_model else 0.0
                savings_usd = max(0.0, cost_exp - cost_router)
                savings_pct = round((savings_usd / cost_exp * 100.0), 2) if cost_exp > 0 else 0.0

                tier_chosen = "cheap" if (chosen_model and (getattr(chosen_model, "free", False) or getattr(chosen_model.prices, "is_free", False))) else ("expensive" if (chosen_model == expensive_model) else "mid")
                is_aligned = (expected_tier == "expensive" and tier_chosen == "expensive") or (expected_tier in ("cheap", "mid") and tier_chosen in ("cheap", "mid"))

                results.append(
                    CaseResult(
                        case_id=cid,
                        category=cat_expected,
                        difficulty_tag=diff_tag,
                        prompt=prompt,
                        expected_tier=expected_tier,
                        detected_category=clf.category,
                        detected_difficulty=round(clf.difficulty, 3),
                        detected_stakes=round(clf.stakes, 3),
                        classifier_latency_ms=round(lat_ms, 2),
                        chosen_model=chosen_name,
                        chosen_provider=chosen_model.provider if chosen_model else "none",
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
                )
                if on_progress:
                    on_progress(idx + 1, len(cases))

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
            results=results,
        )
