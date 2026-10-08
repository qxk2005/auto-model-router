"""Cloudflare Clef (and Clef-flash) decision model integration.

Cloudflare Clef is an open-weight family of decision models hosted on Cloudflare Workers AI:
- ``@cf/cloudflare/clef-flash`` (9B): Optimized for low latency (~39ms) and high-throughput routing.
- ``@cf/cloudflare/clef`` (27B): Higher precision for complex task classification and judging.

This client interfaces with the Workers AI REST API, requiring:
- Cloudflare Account ID (CLOUDFLARE_ACCOUNT_ID)
- Cloudflare API Token (CLOUDFLARE_API_TOKEN)
- Optional custom Base URL (for AI Gateway or custom reverse proxy)

Free tier:
Cloudflare accounts receive 10,000 free Neurons daily. Since decision models have no output
token fees, Clef-flash can process over 1.2 million tokens per day for free.
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.request
from typing import Any

from .jev import (
    DEFAULT_REQUEST_HEADERS,
    FALLBACK,
    JUDGE_QUESTION,
    MAX_ATTEMPTS,
    QUESTIONS,
    REQUEST_CHARS,
    RESPONSE_CHARS,
    RETRY_STATUS,
    Classification,
    Judgement,
    _confidence,
    _retry_after_seconds,
    _score01,
    scrub,
    verify_questions,
)

logger = logging.getLogger(__name__)

DEFAULT_CLEF_MODEL = "@cf/cloudflare/clef-flash"
CLOUDFLARE_API_BASE = "https://api.cloudflare.com/client/v4"


def build_clef_endpoint(
    account_id: str | None = None,
    model: str = DEFAULT_CLEF_MODEL,
    base_url: str | None = None,
) -> str:
    """Construct the endpoint URL for Cloudflare Clef inference."""
    raw_base = (base_url or "").strip().rstrip("/")
    if raw_base:
        if raw_base.endswith("/" + model):
            return raw_base
        if "/systemone" in raw_base:
            return raw_base
        if raw_base.endswith("/run") or "/ai/run" in raw_base or "/workers-ai" in raw_base:
            return f"{raw_base}/{model}"
        if account_id and "/accounts/" not in raw_base:
            return f"{raw_base}/accounts/{account_id}/ai/run/{model}"
        return f"{raw_base}/{model}"

    acc = (account_id or os.environ.get("CLOUDFLARE_ACCOUNT_ID") or "").strip()
    return f"{CLOUDFLARE_API_BASE}/accounts/{acc}/ai/run/{model}"


class ClefClassifier:
    """Classifier and Judge powered by Cloudflare Clef / Clef-flash on Workers AI.

    Zero-GPU, pure cloud inference utilizing Cloudflare's daily free Neurons allowance.
    On network error or rate limit exhaustion (429/403), gracefully falls back to
    the safe neutral ``FALLBACK`` without invoking any heavy local models.
    """

    def __init__(
        self,
        account_id: str | None = None,
        api_token: str | None = None,
        model: str = DEFAULT_CLEF_MODEL,
        base_url: str | None = None,
        timeout: float = 15.0,
        attempts: int = MAX_ATTEMPTS,
    ):
        self.account_id = (
            account_id
            or os.environ.get("CLOUDFLARE_ACCOUNT_ID")
            or ""
        ).strip()
        self.api_token = (
            api_token
            or os.environ.get("CLOUDFLARE_API_TOKEN")
            or os.environ.get("CLOUDFLARE_TOKEN")
            or ""
        ).strip()
        self.model = (model or DEFAULT_CLEF_MODEL).strip()
        self.base_url = (base_url or "").strip()
        self.timeout = float(timeout)
        self.attempts = max(1, int(attempts))
        self.endpoint = build_clef_endpoint(self.account_id, self.model, self.base_url)

    @classmethod
    def from_config(cls, cfg: dict | None) -> "ClefClassifier":
        cfg = cfg or {}
        # Support nested clef / cloudflare block or top-level keys
        c_cfg = cfg.get("clef") or cfg.get("cloudflare") if isinstance(cfg.get("clef") or cfg.get("cloudflare"), dict) else cfg
        account_id = (
            c_cfg.get("account_id")
            or cfg.get("account_id")
            or os.environ.get("CLOUDFLARE_ACCOUNT_ID")
        )
        api_token = (
            c_cfg.get("api_token")
            or c_cfg.get("api_key")
            or cfg.get("api_token")
            or cfg.get("api_key")
            or os.environ.get("CLOUDFLARE_API_TOKEN")
            or os.environ.get("CLOUDFLARE_TOKEN")
        )
        model = (
            c_cfg.get("model")
            or cfg.get("model")
            or DEFAULT_CLEF_MODEL
        )
        base_url = (
            c_cfg.get("base_url")
            or cfg.get("base_url")
            or ""
        )
        timeout = float(c_cfg.get("timeout") or cfg.get("timeout") or 15.0)
        return cls(
            account_id=account_id,
            api_token=api_token,
            model=model,
            base_url=base_url,
            timeout=timeout,
        )

    def test_connection(self) -> dict:
        """Verify API token credentials and measure roundtrip network latency."""
        token = self.api_token
        if not token:
            return {
                "status": "error",
                "error": "未配置 Cloudflare API Token (可在环境变量 CLOUDFLARE_API_TOKEN 或配置中设置)",
            }

        headers = {
            **DEFAULT_REQUEST_HEADERS,
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

        # If custom base_url is specified and doesn't point to api.cloudflare.com, probe base_url
        if self.base_url and "api.cloudflare.com" not in self.base_url:
            probe_url = self.base_url
        else:
            # Cloudflare official token verification endpoint
            probe_url = f"{CLOUDFLARE_API_BASE}/user/tokens/verify"

        last_error = None
        for attempt in range(2):
            req = urllib.request.Request(probe_url, headers=headers)
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({})) if attempt > 0 else urllib.request.build_opener()
            started = time.perf_counter()
            try:
                with opener.open(req, timeout=min(10.0, self.timeout)) as resp:
                    elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
                    raw_data = resp.read().decode("utf-8")
                    data = json.loads(raw_data) if raw_data else {}
                    success = data.get("success", True) if isinstance(data, dict) else True
                    
                    if success:
                        return {
                            "status": "ok",
                            "latency_ms": elapsed_ms,
                            "endpoint": self.endpoint,
                            "target_model": self.model,
                            "account_id": (self.account_id[:6] + "..." + self.account_id[-4:]) if len(self.account_id) > 10 else (self.account_id or "未指定"),
                            "details": data.get("messages") or ["Token 有效且鉴权成功"],
                        }
                    else:
                        errors = data.get("errors") or []
                        err_text = "; ".join(e.get("message", str(e)) for e in errors) if errors else "Cloudflare API 验证失败"
                        return {
                            "status": "error",
                            "error": err_text,
                            "http_status": resp.status,
                        }
            except urllib.error.HTTPError as exc:
                err_body = ""
                msg = exc.reason
                try:
                    raw_bytes = exc.read()
                    err_body = raw_bytes.decode("utf-8", errors="ignore")
                    parsed = json.loads(err_body)
                    if isinstance(parsed, dict):
                        errors = parsed.get("errors") or []
                        if errors and isinstance(errors, list):
                            msg = "; ".join(e.get("message", str(e)) for e in errors)
                        else:
                            msg = parsed.get("message") or msg
                except Exception:
                    pass

                if exc.code == 401:
                    friendly_err = f"HTTP 401 令牌鉴权失败: Cloudflare API Token 无效或无权限 ({msg})"
                elif exc.code == 403:
                    friendly_err = f"HTTP 403 访问受限: 账号权限受限或需要开通 Workers AI 权限 ({msg})"
                elif exc.code == 404:
                    friendly_err = f"HTTP 404 端点未找到: 请检查 Account ID 是否填写正确 ({self.endpoint})"
                elif exc.code == 429:
                    friendly_err = f"HTTP 429 访问受限: 今日 Workers AI 免费额度耗尽或遇到频率限制 ({msg})"
                else:
                    friendly_err = f"HTTP {exc.code} {exc.reason}: {msg}"

                return {
                    "status": "error",
                    "error": friendly_err,
                    "endpoint": self.endpoint,
                    "http_status": exc.code,
                    "raw_error": err_body[:300],
                }
            except Exception as exc:
                last_error = exc
                time.sleep(0.3 * (attempt + 1))

        err_msg = str(last_error) if last_error else "网络异常"
        if isinstance(last_error, urllib.error.URLError):
            err_msg = f"网络连接失败 ({last_error.reason})"
        return {
            "status": "error",
            "error": f"无法连接至 Cloudflare 服务器: {err_msg}",
            "endpoint": self.endpoint,
        }

    def _post(self, state: dict, questions: dict, timeout: float | None = None) -> tuple[dict, float]:
        token = self.api_token
        if not token:
            raise RuntimeError("CLOUDFLARE_API_TOKEN is not set")
        if not self.endpoint or "accounts//ai" in self.endpoint:
            raise RuntimeError("CLOUDFLARE_ACCOUNT_ID is not set or endpoint invalid")

        timeout = timeout or self.timeout
        # Workers AI Clef payload format (System One API compatible)
        body = json.dumps({"state": state, "questions": questions}).encode()
        started = time.perf_counter()
        last: Exception = RuntimeError("no attempt made")

        for attempt in range(self.attempts):
            headers = {
                **DEFAULT_REQUEST_HEADERS,
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            }
            req = urllib.request.Request(self.endpoint, data=body, headers=headers)
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({})) if attempt > 0 else urllib.request.build_opener()
            try:
                with opener.open(req, timeout=timeout) as resp:
                    elapsed = time.perf_counter() - started
                    res_body = resp.read().decode("utf-8")
                    return json.loads(res_body), elapsed
            except urllib.error.HTTPError as exc:
                last = exc
                if exc.code not in RETRY_STATUS or attempt == self.attempts - 1:
                    raise
                time.sleep(_retry_after_seconds(exc, attempt))
            except Exception as exc:
                last = exc
                if attempt == self.attempts - 1:
                    raise
                time.sleep(0.5 * (attempt + 1))
        raise last

    def __call__(self, request: str, context: str = "") -> Classification:
        """Classify user request to assess category, difficulty, tool/vision/context needs."""
        started = time.perf_counter()
        try:
            state = {
                "request": scrub(request, REQUEST_CHARS),
                "context": scrub(context, REQUEST_CHARS) or "(new conversation)",
            }
            payload, latency = self._post(state, QUESTIONS, self.timeout)
            # Unpack from Cloudflare Workers AI envelope (result.answers or answers)
            result = payload.get("result") if isinstance(payload.get("result"), dict) else payload
            a = result.get("answers") or payload.get("answers") or {}
            usage = payload.get("usage") or result.get("usage") or {}

            cat_dict = a.get("category") or {}
            diff_dict = a.get("difficulty") or {}
            tools_dict = a.get("needs_tools") or {}
            vis_dict = a.get("needs_vision") or {}
            ctx_dict = a.get("needs_long_context") or {}
            fup_dict = a.get("follow_up") or {}
            stk_dict = a.get("stakes") or {}

            return Classification(
                category=str(cat_dict.get("choice") or "general"),
                category_probs=cat_dict.get("probabilities") or {},
                category_confidence=_confidence(cat_dict),
                difficulty=_score01(diff_dict, len(QUESTIONS["difficulty"]["criteria"])),
                difficulty_confidence=_confidence(diff_dict),
                needs_tools=float(tools_dict.get("noul", 0.0)),
                needs_vision=float(vis_dict.get("noul", 0.0)),
                needs_long_context=float(ctx_dict.get("noul", 0.0)),
                follow_up=float(fup_dict.get("noul", 0.5)),
                stakes=_score01(stk_dict, len(QUESTIONS["stakes"]["criteria"])),
                latency_s=latency or (time.perf_counter() - started),
                model=str(payload.get("model") or self.model),
                input_tokens=int(usage.get("input_tokens") or 0),
                output_tokens=int(usage.get("output_tokens") or 0),
                raw=a,
                source_name=f"clef[{self.model.replace('@cf/cloudflare/', '')}]",
            )
        except Exception as exc:
            logger.warning("Cloudflare Clef classification failed: %s; returning neutral FALLBACK", exc)
            return FALLBACK

    def judge(self, request: str, response: str, category: str = "") -> Judgement:
        """Evaluate response adequacy and classify failure type if inadequate."""
        started = time.perf_counter()
        try:
            state = {
                "request": scrub(request, REQUEST_CHARS),
                "response": scrub(response, RESPONSE_CHARS),
            }
            questions = verify_questions(category) if category else JUDGE_QUESTION
            payload, latency = self._post(state, questions, self.timeout)
            result = payload.get("result") if isinstance(payload.get("result"), dict) else payload
            answers = result.get("answers") or payload.get("answers") or {}
            usage = payload.get("usage") or result.get("usage") or {}

            adq_dict = answers.get("adequate") or {}
            failure_answer = answers.get("failure") or {}
            choice = failure_answer.get("choice")

            return Judgement(
                p_adequate=float(adq_dict.get("noul", 0.5)),
                latency_s=latency or (time.perf_counter() - started),
                failure=str(choice) if choice else "unknown",
                failure_probs={k: float(v) for k, v in (failure_answer.get("probabilities") or {}).items()},
                model=str(payload.get("model") or self.model),
                input_tokens=int(usage.get("input_tokens") or 0),
                output_tokens=int(usage.get("output_tokens") or 0),
            )
        except Exception as exc:
            logger.warning("Cloudflare Clef judge failed: %s", exc)
            return Judgement(0.5, 0.0, failed=True)
