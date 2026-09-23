"""Open LLM Leaderboard Data Manager & Evaluator Bridge.

Provides:
1. Local caching and persistence (data/open_llm_leaderboard.json)
2. Live dataset synchronization from Hugging Face Open LLM Leaderboard (contents)
3. Fuzzy matching between configured model ID/name and benchmark models
4. Capability conversion (Weyaxi scrape-open-llm-leaderboard metrics -> Laya Router scores)
"""

from __future__ import annotations

import asyncio
import difflib
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any

import httpx

log = logging.getLogger("auto_router.leaderboard")

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
CACHE_FILE = DATA_DIR / "open_llm_leaderboard.json"

# Default seed fallback models covering common architectures and flagships
DEFAULT_SEED_MODELS = [
    {
        "fullname": "deepseek-ai/DeepSeek-V2.5",
        "name": "DeepSeek-V2.5",
        "average": 68.35,
        "ifeval": 78.42,
        "bbh": 76.50,
        "math": 56.40,
        "gpqa": 43.30,
        "musr": 48.60,
        "mmlu_pro": 60.10,
        "params": 236.0,
        "architecture": "DeepseekV2ForCausalLM",
        "precision": "bfloat16",
        "type": "💬 chat models",
        "license": "deepseek-license",
        "likes": 1280,
        "hf_url": "https://huggingface.co/deepseek-ai/DeepSeek-V2.5",
    },
    {
        "fullname": "Qwen/Qwen2.5-72B-Instruct",
        "name": "Qwen2.5-72B-Instruct",
        "average": 74.20,
        "ifeval": 84.10,
        "bbh": 82.50,
        "math": 68.90,
        "gpqa": 49.80,
        "musr": 54.20,
        "mmlu_pro": 66.80,
        "params": 72.7,
        "architecture": "Qwen2ForCausalLM",
        "precision": "bfloat16",
        "type": "💬 chat models",
        "license": "apache-2.0",
        "likes": 2150,
        "hf_url": "https://huggingface.co/Qwen/Qwen2.5-72B-Instruct",
    },
    {
        "fullname": "Qwen/Qwen2.5-Coder-32B-Instruct",
        "name": "Qwen2.5-Coder-32B-Instruct",
        "average": 70.80,
        "ifeval": 81.20,
        "bbh": 78.60,
        "math": 64.50,
        "gpqa": 46.20,
        "musr": 51.00,
        "mmlu_pro": 68.40,
        "params": 32.5,
        "architecture": "Qwen2ForCausalLM",
        "precision": "bfloat16",
        "type": "💬 chat models",
        "license": "apache-2.0",
        "likes": 1420,
        "hf_url": "https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct",
    },
    {
        "fullname": "Qwen/Qwen2.5-7B-Instruct",
        "name": "Qwen2.5-7B-Instruct",
        "average": 62.40,
        "ifeval": 74.80,
        "bbh": 68.20,
        "math": 48.60,
        "gpqa": 36.40,
        "musr": 42.10,
        "mmlu_pro": 51.20,
        "params": 7.6,
        "architecture": "Qwen2ForCausalLM",
        "precision": "bfloat16",
        "type": "💬 chat models",
        "license": "apache-2.0",
        "likes": 890,
        "hf_url": "https://huggingface.co/Qwen/Qwen2.5-7B-Instruct",
    },
    {
        "fullname": "meta-llama/Llama-3.1-70B-Instruct",
        "name": "Llama-3.1-70B-Instruct",
        "average": 71.90,
        "ifeval": 82.60,
        "bbh": 79.40,
        "math": 61.20,
        "gpqa": 47.90,
        "musr": 52.80,
        "mmlu_pro": 63.50,
        "params": 70.6,
        "architecture": "LlamaForCausalLM",
        "precision": "bfloat16",
        "type": "💬 chat models",
        "license": "llama3.1",
        "likes": 3200,
        "hf_url": "https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct",
    },
    {
        "fullname": "meta-llama/Llama-3.1-8B-Instruct",
        "name": "Llama-3.1-8B-Instruct",
        "average": 59.80,
        "ifeval": 72.30,
        "bbh": 64.90,
        "math": 44.10,
        "gpqa": 34.20,
        "musr": 40.50,
        "mmlu_pro": 47.80,
        "params": 8.0,
        "architecture": "LlamaForCausalLM",
        "precision": "bfloat16",
        "type": "💬 chat models",
        "license": "llama3.1",
        "likes": 1890,
        "hf_url": "https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct",
    },
    {
        "fullname": "google/gemma-2-27b-it",
        "name": "gemma-2-27b-it",
        "average": 68.70,
        "ifeval": 79.50,
        "bbh": 75.30,
        "math": 55.80,
        "gpqa": 44.50,
        "musr": 49.20,
        "mmlu_pro": 59.40,
        "params": 27.2,
        "architecture": "Gemma2ForCausalLM",
        "precision": "bfloat16",
        "type": "💬 chat models",
        "license": "gemma",
        "likes": 980,
        "hf_url": "https://huggingface.co/google/gemma-2-27b-it",
    },
    {
        "fullname": "google/gemma-2-9b-it",
        "name": "gemma-2-9b-it",
        "average": 61.20,
        "ifeval": 73.10,
        "bbh": 66.40,
        "math": 46.20,
        "gpqa": 35.80,
        "musr": 41.70,
        "mmlu_pro": 49.30,
        "params": 9.2,
        "architecture": "Gemma2ForCausalLM",
        "precision": "bfloat16",
        "type": "💬 chat models",
        "license": "gemma",
        "likes": 740,
        "hf_url": "https://huggingface.co/google/gemma-2-9b-it",
    },
    {
        "fullname": "mistralai/Mistral-Large-Instruct-2407",
        "name": "Mistral-Large-Instruct-2407",
        "average": 72.10,
        "ifeval": 83.00,
        "bbh": 79.80,
        "math": 62.40,
        "gpqa": 48.10,
        "musr": 53.00,
        "mmlu_pro": 64.20,
        "params": 123.0,
        "architecture": "MistralForCausalLM",
        "precision": "bfloat16",
        "type": "💬 chat models",
        "license": "mistral-commercial",
        "likes": 1150,
        "hf_url": "https://huggingface.co/mistralai/Mistral-Large-Instruct-2407",
    },
    {
        "fullname": "THUDM/glm-4-9b-chat",
        "name": "glm-4-9b-chat",
        "average": 60.50,
        "ifeval": 71.90,
        "bbh": 65.80,
        "math": 45.30,
        "gpqa": 35.10,
        "musr": 41.20,
        "mmlu_pro": 48.70,
        "params": 9.4,
        "architecture": "ChatGLMForConditionalGeneration",
        "precision": "bfloat16",
        "type": "💬 chat models",
        "license": "other",
        "likes": 620,
        "hf_url": "https://huggingface.co/THUDM/glm-4-9b-chat",
    },
]


def clean_score(val: Any) -> float:
    """Safely parse float score."""
    if val is None:
        return 0.0
    try:
        f = float(val)
        return round(f, 2)
    except (ValueError, TypeError):
        return 0.0


def normalize_row(row: dict) -> dict[str, Any]:
    """Normalize a raw row from open-llm-leaderboard/contents."""
    fullname = row.get("fullname") or ""
    if not fullname and row.get("Model"):
        # Extract from link if necessary
        match = re.search(r">([^<]+)</a>", row["Model"])
        fullname = match.group(1) if match else str(row["Model"])

    short_name = fullname.split("/")[-1] if "/" in fullname else fullname

    return {
        "fullname": fullname,
        "name": short_name,
        "average": clean_score(row.get("Average ⬆️")),
        "ifeval": clean_score(row.get("IFEval")),
        "bbh": clean_score(row.get("BBH")),
        "math": clean_score(row.get("MATH Lvl 5")),
        "gpqa": clean_score(row.get("GPQA")),
        "musr": clean_score(row.get("MUSR")),
        "mmlu_pro": clean_score(row.get("MMLU-PRO")),
        "params": clean_score(row.get("#Params (B)")),
        "architecture": str(row.get("Architecture") or "Unknown"),
        "precision": str(row.get("Precision") or "bfloat16"),
        "type": str(row.get("Type") or "💬 chat"),
        "license": str(row.get("Hub License") or "unknown"),
        "likes": int(row.get("Hub ❤️") or 0),
        "hf_url": f"https://huggingface.co/{fullname}" if fullname else "",
        "raw_data": {
            "eval_name": row.get("eval_name"),
            "MoE": row.get("MoE"),
            "Flagged": row.get("Flagged"),
            "Chat Template": row.get("Chat Template"),
            "Upload To Hub Date": row.get("Upload To Hub Date"),
            "Submission Date": row.get("Submission Date"),
            "IFEval Raw": clean_score(row.get("IFEval Raw")),
            "BBH Raw": clean_score(row.get("BBH Raw")),
            "MATH Lvl 5 Raw": clean_score(row.get("MATH Lvl 5 Raw")),
            "GPQA Raw": clean_score(row.get("GPQA Raw")),
            "MUSR Raw": clean_score(row.get("MUSR Raw")),
            "MMLU-PRO Raw": clean_score(row.get("MMLU-PRO Raw")),
        },
    }


def compute_laya_capabilities(item: dict) -> dict[str, float]:
    """Convert Weyaxi / Open LLM Leaderboard benchmark metrics to Laya capability scores (0.00 ~ 1.00).

    Mapping rules:
    - Coding: MMLU-PRO (broad coding, CS & engineering tests)
    - Math: MATH Lvl 5 (high-school & competition math)
    - Reasoning: Weighted ensemble of BBH (50%), GPQA (30%), MUSR (20%)
    - General: Weighted ensemble of IFEval (60%) and Average (40%)
    """
    avg = item.get("average") or 50.0

    # Math
    math_raw = item.get("math") or 0.0
    math_score = math_raw if math_raw > 0 else (avg * 0.9)

    # Coding
    mmlu_pro = item.get("mmlu_pro") or 0.0
    coding_score = mmlu_pro if mmlu_pro > 0 else (avg * 0.95)

    # Reasoning
    bbh = item.get("bbh") or avg
    gpqa = item.get("gpqa") or avg
    musr = item.get("musr") or avg
    reasoning_score = bbh * 0.50 + gpqa * 0.30 + musr * 0.20

    # General
    ifeval = item.get("ifeval") or avg
    general_score = ifeval * 0.60 + avg * 0.40

    def clamp(v: float) -> float:
        return max(0.10, min(1.00, round(v / 100.0, 2)))

    return {
        "coding": clamp(coding_score),
        "math": clamp(math_score),
        "reasoning": clamp(reasoning_score),
        "general": clamp(general_score),
    }


class LeaderboardManager:
    """Manages leaderboard persistence, querying, fuzzy matching, and live sync."""

    def __init__(self):
        self._data: list[dict[str, Any]] = []
        self._last_updated: float = 0.0
        self._is_syncing: bool = False
        self.load_cache()

    def load_cache(self) -> None:
        """Load from local JSON cache or initialize with default seed."""
        if CACHE_FILE.exists():
            try:
                with open(CACHE_FILE, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                    self._data = payload.get("models", [])
                    self._last_updated = payload.get("updated_at", time.time())
                log.info("Loaded %d leaderboard models from %s", len(self._data), CACHE_FILE)
                return
            except Exception as e:
                log.warning("Failed to read leaderboard cache %s: %s", CACHE_FILE, e)

        # Fallback to seed
        self._data = list(DEFAULT_SEED_MODELS)
        self._last_updated = time.time()
        self.save_cache()

    def save_cache(self) -> None:
        """Save models to local cache file."""
        try:
            with open(CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "source": "https://huggingface.co/datasets/open-llm-leaderboard/contents",
                        "total": len(self._data),
                        "updated_at": self._last_updated,
                        "updated_at_str": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self._last_updated)),
                        "models": self._data,
                    },
                    f,
                    indent=2,
                    ensure_ascii=False,
                )
        except Exception as e:
            log.error("Failed to save leaderboard cache: %s", e)

    def query(
        self,
        q: str = "",
        sort_by: str = "average",
        order: str = "desc",
        limit: int = 100,
        offset: int = 0,
        architecture: str = "",
    ) -> dict[str, Any]:
        """Search, filter, and paginate models."""
        filtered = self._data

        if q:
            q_lower = q.strip().lower()
            filtered = [
                m
                for m in filtered
                if q_lower in m["fullname"].lower()
                or q_lower in m["architecture"].lower()
                or q_lower in m.get("license", "").lower()
            ]

        if architecture:
            arch_lower = architecture.strip().lower()
            filtered = [m for m in filtered if arch_lower in m["architecture"].lower()]

        # Sorting
        valid_sort_keys = {
            "average": "average",
            "ifeval": "ifeval",
            "bbh": "bbh",
            "math": "math",
            "gpqa": "gpqa",
            "musr": "musr",
            "mmlu_pro": "mmlu_pro",
            "params": "params",
            "likes": "likes",
        }
        key = valid_sort_keys.get(sort_by, "average")
        reverse = order.lower() == "desc"

        sorted_list = sorted(filtered, key=lambda x: (x.get(key) is not None, x.get(key) or 0.0), reverse=reverse)

        total = len(sorted_list)
        paged = sorted_list[offset : offset + limit] if limit > 0 else sorted_list

        return {
            "total": total,
            "count": len(paged),
            "offset": offset,
            "limit": limit,
            "updated_at": self._last_updated,
            "updated_at_str": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self._last_updated)),
            "models": paged,
        }

    def match_model(self, model_id_or_name: str, top_k: int = 5) -> dict[str, Any]:
        """Fuzzy match a model ID / name to the best leaderboard model record."""
        if not model_id_or_name:
            # Fallback to top model in list
            best = self._data[0] if self._data else DEFAULT_SEED_MODELS[0]
            return {
                "query": "",
                "best_match": best,
                "confidence": 0.5,
                "laya_capabilities": compute_laya_capabilities(best),
                "candidates": self._data[:top_k],
            }

        q_clean = model_id_or_name.lower().replace(":", "-").replace("/", "-")
        # Extract meaningful tokens
        tokens = [t for t in re.split(r"[-_.\s]+", q_clean) if t and t not in ("model", "chat", "v1", "v2", "v3", "v4")]

        scored_candidates = []
        for m in self._data:
            m_full = m["fullname"].lower().replace("/", "-")
            m_short = m["name"].lower()

            score = 0.0

            # Substring exact check
            if q_clean in m_full or m_short in q_clean:
                score += 80.0

            # Token overlap
            m_tokens = re.split(r"[-_.\s]+", m_full)
            overlap = set(tokens).intersection(set(m_tokens))
            score += len(overlap) * 20.0

            # Check core vendor keywords
            for kw in ("deepseek", "qwen", "llama", "mistral", "gemma", "glm", "yi", "claude", "gpt"):
                if kw in q_clean and kw in m_full:
                    score += 40.0

            # Check parameter size match e.g. 7b, 8b, 70b, 72b
            size_match = re.search(r"(\d+b)", q_clean)
            if size_match and size_match.group(1) in m_full:
                score += 35.0

            # String similarity ratio
            ratio = difflib.SequenceMatcher(None, q_clean, m_short).ratio()
            score += ratio * 30.0

            scored_candidates.append((score, m))

        scored_candidates.sort(key=lambda x: x[0], reverse=True)

        if not scored_candidates or scored_candidates[0][0] <= 0:
            # Fallback to first available or seed
            best = self._data[0] if self._data else DEFAULT_SEED_MODELS[0]
            confidence = 0.4
            candidates = [m for _, m in scored_candidates[:top_k]]
        else:
            best = scored_candidates[0][1]
            raw_top_score = scored_candidates[0][0]
            confidence = min(0.99, round(raw_top_score / 150.0, 2))
            candidates = [m for _, m in scored_candidates[:top_k]]

        return {
            "query": model_id_or_name,
            "best_match": best,
            "confidence": confidence,
            "laya_capabilities": compute_laya_capabilities(best),
            "candidates": candidates,
        }

    async def sync_from_huggingface(self, max_pages: int = 15) -> dict[str, Any]:
        """Fetch latest benchmark records concurrently from Hugging Face dataset server."""
        if self._is_syncing:
            return {"status": "in_progress", "message": "正在同步中，请稍候..."}

        self._is_syncing = True
        t0 = time.time()
        log.info("Starting live sync from Hugging Face Open LLM Leaderboard...")

        try:
            sem = asyncio.Semaphore(8)
            all_rows = []

            async with httpx.AsyncClient(
                headers={"User-Agent": "Mozilla/5.0 (AutoLLMRouter/1.0)"},
                timeout=25.0,
            ) as client:

                async def fetch_page(offset: int):
                    async with sem:
                        url = (
                            f"https://datasets-server.huggingface.co/rows"
                            f"?dataset=open-llm-leaderboard%2Fcontents"
                            f"&config=default&split=train&offset={offset}&limit=100"
                        )
                        try:
                            resp = await client.get(url)
                            if resp.status_code == 200:
                                d = resp.json()
                                return [r["row"] for r in d.get("rows", [])]
                            else:
                                log.warning("Offset %d returned HTTP %d", offset, resp.status_code)
                                return []
                        except Exception as e:
                            log.warning("Offset %d fetch error: %s", offset, e)
                            return []

                offsets = [page * 100 for page in range(max_pages)]
                tasks = [fetch_page(off) for off in offsets]
                results = await asyncio.gather(*tasks)

                for sub in results:
                    all_rows.extend(sub)

            if all_rows:
                # Merge and deduplicate by fullname
                model_map = {m["fullname"]: m for m in self._data}
                # Also include default seeds
                for s in DEFAULT_SEED_MODELS:
                    model_map.setdefault(s["fullname"], s)

                for r in all_rows:
                    norm = normalize_row(r)
                    if norm["fullname"]:
                        model_map[norm["fullname"]] = norm

                # Sort by average desc
                self._data = sorted(
                    model_map.values(),
                    key=lambda x: x.get("average", 0.0),
                    reverse=True,
                )
                self._last_updated = time.time()
                self.save_cache()

                duration = round(time.time() - t0, 2)
                log.info("Sync finished in %ss: total %d models stored.", duration, len(self._data))
                return {
                    "status": "success",
                    "total": len(self._data),
                    "newly_fetched": len(all_rows),
                    "duration_s": duration,
                    "updated_at": self._last_updated,
                    "updated_at_str": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self._last_updated)),
                }
            else:
                return {
                    "status": "warning",
                    "message": "未能从远程接口拉取到新数据，已保留现有本地缓存。",
                    "total": len(self._data),
                }

        except Exception as exc:
            log.error("Live sync failed: %s", exc)
            return {"status": "error", "message": f"同步失败: {exc}", "total": len(self._data)}
        finally:
            self._is_syncing = False


# Global singleton instance
leaderboard_mgr = LeaderboardManager()
