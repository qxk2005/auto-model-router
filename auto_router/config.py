"""Provider configuration and catalog assembly.

The router ships with no providers. You describe your own OpenAI-compatible
endpoints (and, optionally, a flat-rate subscription tier) in a YAML or JSON
file and point ``AUTO_ROUTER_CONFIG`` at it. See ``examples/config.example.yaml``.

Schema (YAML shown)::

    providers:
      my-host:
        base_url: https://api.example.com/v1
        api_key_env: EXAMPLE_API_KEY          # name of the env var, never the key
        cache: openai                          # key into DEFAULT_CACHE_RULES
        extra_headers: {User-Agent: "..."}
    subscriptions:
      claude:
        usage_command: [my-usage-reader, --json]   # your own reader, percentages only
        budget_file: ~/.agent-budget.json      # optional cached usage source
        weekly_reserve: 0.65                   # see quota.py
    models:
      - name: cheap-coder
        provider: my-host
        upstream_id: vendor/model-x
        bench_id: model-x::default             # capability + list price lookup
        success_key: model-x                   # name in the measured success table
        bench_offer: {platform: OpenRouter, provider: SomeHost}
        prices: {input: 0.1, output: 0.4, cache_read: 0.01}   # overrides list price
        free: true                              # shorthand for all-zero prices
        cache: {ttl_seconds: 300, hit_rate: 0.93}             # measured values
        capability: {coding: 55}                # overrides benchmark data
        vision: false
        tools: true
        context_tokens: 128000
        launch_only: true                       # only reachable by launching its own client
        runner:                                 # optional: how route-run starts this route
          cmd: [the-official-cli, --model, vendor/model-x]
          stdin: true                           # pass the task on stdin
          env: {SOME_API_KEY: ""}               # cleared, never a literal secret
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

try:
    import dotenv
    dotenv.load_dotenv()
except ImportError:
    pass

from .bench import BenchmarkClient, capability_evidence, context_length, pick_offer
from .catalog import CONFIG_OVERRIDE, DEFAULT_CACHE_RULES, CacheRules, Catalog, ModelInfo, Prices


def expand_env_vars(text: str | None) -> str | None:
    """Expand ${VAR} and $VAR environment variable placeholders."""
    if not text or not isinstance(text, str):
        return text
    try:
        import dotenv
        dotenv.load_dotenv()
    except ImportError:
        pass
    def _sub(match):
        var = match.group(1)
        default = match.group(3) if match.group(2) else ""
        return os.environ.get(var, default)
    expanded = re.sub(r"\$\{([A-Za-z0-9_]+)(:-(.*?))?\}", _sub, text)
    if expanded.startswith("$") and len(expanded) > 1 and re.match(r"^\$[A-Za-z0-9_]+$", expanded):
        expanded = os.environ.get(expanded[1:], "")
    return expanded


def normalize_provider_url(url: str | None, api: str = "openai") -> str:
    """Normalize provider base URL, expanding env vars and auto-completing /v1 for OpenAI/Anthropic endpoints."""
    if not url or not isinstance(url, str):
        return ""
    expanded = expand_env_vars(url) or url
    u = expanded.strip().rstrip("/")
    if not u:
        return ""

    # Strip trailing endpoint path if user pasted a full endpoint
    for suffix in ("/chat/completions", "/models", "/messages", "/embeddings"):
        if u.endswith(suffix):
            u = u[:-len(suffix)].rstrip("/")
            break

    # If it is OpenAI-compatible (default) or Anthropic, auto-complete /v1 if missing version specifier
    # e.g., http://8.148.249.98:88 -> http://8.148.249.98:88/v1
    # http://localhost:1234 -> http://localhost:1234/v1
    # https://api.openai.com -> https://api.openai.com/v1
    # But preserve if already has /v1, /v2, etc.
    if api in ("openai", "anthropic"):
        if not re.search(r"/v\d+$", u, re.IGNORECASE):
            u = f"{u}/v1"

    return u


@dataclass
class Provider:
    name: str
    base_url: str
    api_key_env: str | None = None
    api_key_literal: str | None = None
    cache: str = "generic"
    extra_headers: dict[str, str] = field(default_factory=dict)
    #: "openai" (chat completions) or "anthropic" (messages passthrough).
    api: str = "openai"

    @property
    def resolved_base_url(self) -> str:
        url = expand_env_vars(self.base_url) or self.base_url
        return normalize_provider_url(url, api=self.api)

    @property
    def api_key(self) -> str | None:
        if self.api_key_literal:
            expanded = expand_env_vars(self.api_key_literal)
            return expanded if expanded else self.api_key_literal
        if self.api_key_env:
            return os.environ.get(self.api_key_env)
        return None


@dataclass
class RouterConfig:
    providers: dict[str, Provider]
    catalog: Catalog
    subscriptions: dict[str, dict[str, Any]] = field(default_factory=dict)
    policy: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)


def _load_file(path: str | Path) -> dict:
    p = Path(path).expanduser()
    try:
        text = p.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        try:
            text = p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = p.read_text(encoding="gbk", errors="replace")
    if str(path).endswith((".yaml", ".yml")):
        import yaml  # optional dependency, only needed for YAML configs
        return yaml.safe_load(text) or {}
    return json.loads(text)


def build_model(entry: dict, providers: dict[str, Provider],
                bench: BenchmarkClient | None) -> ModelInfo:
    provider = providers.get(entry.get("provider", ""))
    cache_family = (entry.get("cache_family") or (provider.cache if provider else "generic"))
    rules = DEFAULT_CACHE_RULES.get(cache_family, DEFAULT_CACHE_RULES["generic"])
    if isinstance(entry.get("cache"), dict):
        rules = replace(rules, **entry["cache"])

    prices: Prices | None = None
    capability: dict[str, float] = {}
    basis: dict[str, str] = {}
    strength: dict[str, str] = {}
    ctx: int | None = None
    benchmaxxing = 0.0
    cap_source = "none"
    evidence: dict = {"source": "none"}
    stale = True

    bench_id = entry.get("bench_id")
    if bench and bench_id:
        doc, prov = bench.model_with_provenance(bench_id)
        evidence = {"bench_id": bench_id, **prov.to_dict()}
        stale = prov.stale
        if doc:
            ev = capability_evidence(doc)
            capability = {k: round(e.value, 2) for k, e in ev.items()}
            basis = {k: e.basis for k, e in ev.items()}
            strength = {k: e.strength for k, e in ev.items()}
            cap_source = "bench" if capability else "none"
            evidence["release_date"] = doc.get("release_date")
            ctx = context_length(doc)
            offer_sel = entry.get("bench_offer") or {}
            offer = pick_offer(doc, offer_sel.get("platform"), offer_sel.get("provider"))
            if offer:
                prices = Prices(
                    input=float(offer["input_per_1m"]),
                    output=float(offer.get("output_per_1m") or 0.0),
                    cache_read=offer.get("cache_read_per_1m"),
                    cache_write=offer.get("cache_write_per_1m"),
                )
                evidence["offer"] = {"platform": offer.get("platform"),
                                     "provider": offer.get("provider")}
        # Fetched separately from the model document and able to be stale on
        # its own, so it carries its own provenance and a stale score is not
        # allowed to silently move a capability that is otherwise current.
        score, bm_prov = bench.benchmaxxing_with_provenance(bench_id)
        evidence["benchmaxxing"] = bm_prov.to_dict()
        if score is not None and not bm_prov.stale:
            benchmaxxing = score
        elif score is not None:
            evidence["benchmaxxing_ignored"] = "stale benchmaxxing report; penalty not applied"

    if isinstance(entry.get("prices"), dict):
        p = entry["prices"]
        prices = Prices(float(p["input"]), float(p["output"]), p.get("cache_read"), p.get("cache_write"))
    if entry.get("free") or entry.get("subscription"):
        prices = Prices.free()
    if prices is None:
        raise ValueError(f"model {entry.get('name')!r}: no prices (set prices, free, or a bench_id with offers)")

    if isinstance(entry.get("capability"), dict):
        # An explicit config number is a deliberate statement by the operator,
        # so it counts as direct evidence and is never discounted as stale.
        overrides = {k: float(v) for k, v in entry["capability"].items()}
        capability = {**capability, **overrides}
        basis = {**basis, **{k: CONFIG_OVERRIDE for k in overrides}}
        strength = {**strength, **{k: "direct" for k in overrides}}
        cap_source = "config" if cap_source == "none" else cap_source + "+config"
        if cap_source == "config":
            stale = False
            evidence = {"source": "config", "stale": False}
    if "benchmaxxing" in entry:
        benchmaxxing = float(entry["benchmaxxing"])

    return ModelInfo(
        name=entry["name"],
        provider=entry.get("provider", ""),
        upstream_id=entry.get("upstream_id", entry["name"]),
        prices=prices,
        cache=rules,
        context_tokens=int(entry.get("context_tokens") or ctx or 128_000),
        max_output_tokens=int(entry.get("max_output_tokens") or 32_000),
        vision=bool(entry.get("vision", False)),
        tools=bool(entry.get("tools", True)),
        capability=capability,
        benchmaxxing=benchmaxxing,
        subscription=entry.get("subscription"),
        capability_source=cap_source,
        capability_basis=basis,
        capability_strength=strength,
        evidence_stale=bool(stale and cap_source != "config"),
        evidence=evidence,
        bench_id=bench_id,
        latency_s=float(entry.get("latency_s", 5.0)),
        success_key=entry.get("success_key"),
        runner=entry.get("runner"),
        launch_only=bool(entry.get("launch_only", False)),
        timeout_s=(
            float(entry["timeout_seconds"])
            if entry.get("timeout_seconds") is not None
            else (float(entry["timeout_s"]) if entry.get("timeout_s") is not None else None)
        ),
        enabled=bool(entry.get("enabled", True)),
    )


def for_http(config: RouterConfig) -> RouterConfig:
    """The same configuration, minus routes that only a launched client can reach.

    A subscription that is served exclusively through its own CLI is not an
    endpoint: no HTTP request from another client can be answered from it. It
    stays in the catalog for the launcher and disappears here.
    """
    return replace(config, catalog=Catalog(config.catalog.http_routable()))


def load_config(path: str | Path | None = None, *, bench: BenchmarkClient | None = None,
                use_bench: bool = True) -> RouterConfig:
    try:
        import dotenv
        dotenv.load_dotenv()
    except ImportError:
        pass

    path = path or os.environ.get("AUTO_ROUTER_CONFIG")
    if not path:
        local_p = Path("config/router_config.local.json")
        default_p = Path("config/router_config.json")
        if local_p.exists():
            path = local_p
        elif default_p.exists():
            path = default_p
    if not path or not Path(path).exists():
        return RouterConfig(providers={}, catalog=Catalog([]))
    raw = _load_file(path)
    providers = {
        name: Provider(
            name=name,
            base_url=p["base_url"].rstrip("/"),
            api_key_env=p.get("api_key_env"),
            api_key_literal=p.get("api_key") or p.get("api_key_literal"),
            cache=p.get("cache", "generic"),
            extra_headers=p.get("extra_headers") or {},
            api=p.get("api", "openai"),
        )
        for name, p in (raw.get("providers") or {}).items()
    }
    if use_bench and bench is None:
        bench = BenchmarkClient(offline=os.environ.get("AUTO_ROUTER_BENCH_OFFLINE") == "1")
    models = [build_model(m, providers, bench if use_bench else None) for m in raw.get("models") or []]
    return RouterConfig(providers=providers, catalog=Catalog([m for m in models if m.enabled]),
                        subscriptions=raw.get("subscriptions") or {},
                        policy=raw.get("policy") or {}, raw=raw)


# 常见中文提供商到环境变量前缀映射
KNOWN_PROVIDER_ENV_PREFIXES = {
    "具生涌动": "JUSHENG",
    "智谱": "ZHIPU",
    "智谱AI": "ZHIPU",
    "通义千问": "QWEN",
    "通义": "QWEN",
    "阿里": "ALIYUN",
    "阿里云": "ALIYUN",
    "月之暗面": "MOONSHOT",
    "Kimi": "MOONSHOT",
    "kimi": "MOONSHOT",
    "零一万物": "YI",
    "百川智能": "BAICHUAN",
    "百川": "BAICHUAN",
    "商汤": "SENSETIME",
    "阶跃星辰": "STEPFUN",
    "MiniMax": "MINIMAX",
    "minimax": "MINIMAX",
    "深度求索": "DEEPSEEK",
    "幻方": "DEEPSEEK",
}


def generate_env_var_name(provider_name: str) -> str:
    """Generate a clean, standardized uppercase environment variable name for a provider.
    e.g. 'deepseek' -> 'DEEPSEEK_API_KEY', '具生涌动' -> 'JUSHENG_API_KEY'
    """
    clean_p = str(provider_name or "").strip()
    if clean_p in KNOWN_PROVIDER_ENV_PREFIXES:
        return f"{KNOWN_PROVIDER_ENV_PREFIXES[clean_p]}_API_KEY"

    # 若已经是纯英文字符/下划线/数字
    ascii_key = re.sub(r"[^A-Za-z0-9_]+", "_", clean_p).strip("_").upper()
    if not ascii_key:
        import hashlib
        h = hashlib.md5(clean_p.encode("utf-8")).hexdigest()[:6].upper()
        ascii_key = f"PROV_{h}"
    return f"{ascii_key}_API_KEY"


def update_local_env_file(var_name: str, var_val: str, env_path: str | Path | None = None) -> bool:
    """Safely append or update an environment variable in the local private .env file."""
    if not var_name or not var_val:
        return False
    path = Path(env_path) if env_path else Path(".env")

    # 同步注入到当前进程环境变量，供即时读取
    os.environ[var_name] = var_val

    lines = []
    found = False
    if path.exists():
        try:
            content = path.read_text(encoding="utf-8")
            lines = content.splitlines()
        except Exception:
            lines = []

    pattern = re.compile(rf"^\s*{re.escape(var_name)}\s*=")
    new_lines = []
    for line in lines:
        if pattern.match(line):
            new_lines.append(f"{var_name}={var_val}")
            found = True
        else:
            new_lines.append(line)

    if not found:
        if new_lines and new_lines[-1].strip() != "":
            new_lines.append("")
        new_lines.append(f"{var_name}={var_val}")

    path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    return True


def sanitize_config_for_public_repo(raw: dict) -> tuple[dict, dict[str, str]]:
    """Inspect raw config dict, extract any plain secrets into a dict of {ENV_VAR: plain_secret},
    and return a sanitized copy of raw config where all plain secrets are replaced by ${ENV_VAR}.
    """
    import copy
    sanitized = copy.deepcopy(raw)
    extracted_secrets = {}

    for prov_name, prov in (sanitized.get("providers") or {}).items():
        if isinstance(prov, dict):
            key = str(prov.get("api_key") or prov.get("api_key_literal") or "").strip()
            # 排除环境变量占位符、lm-studio 及 dummy 测试值
            if key and not key.startswith("$") and key != "lm-studio" and not key.startswith("sk-your-"):
                if len(key) >= 12 or key.startswith("sk-") or "key" in key.lower() or "token" in key.lower():
                    env_var = generate_env_var_name(prov_name)
                    extracted_secrets[env_var] = key
                    prov["api_key"] = f"${{{env_var}}}"

    return sanitized, extracted_secrets


def has_plain_secrets(raw: dict) -> bool:
    """Check if config contains literal/plain secrets rather than ${ENV} placeholders."""
    if not isinstance(raw, dict):
        return False
    # Check providers
    for p in (raw.get("providers") or {}).values():
        if isinstance(p, dict):
            key = str(p.get("api_key") or p.get("api_key_literal") or "").strip()
            if key and not key.startswith("$") and key != "lm-studio" and not key.startswith("sk-your-"):
                if len(key) >= 12 or key.startswith("sk-"):
                    return True
    # Check classifier policy
    clf = (raw.get("policy") or {}).get("classifier") or {}
    for field in ("api_key", "api_token"):
        k = str(clf.get(field) or "").strip()
        if k and not k.startswith("$") and not k.startswith("sk-your-"):
            if len(k) >= 12 or k.startswith("sk-"):
                return True
    for sub in ("jev", "clef", "cloudflare"):
        if isinstance(clf.get(sub), dict):
            for field in ("api_key", "api_token"):
                k = str(clf[sub].get(field) or "").strip()
                if k and not k.startswith("$") and not k.startswith("sk-your-"):
                    if len(k) >= 12 or k.startswith("sk-"):
                        return True
    return False


def save_config(raw: dict, path: str | Path | None = None) -> str:
    """Save raw configuration dict to JSON or YAML.

    If configuration contains literal plain secrets, automatically routes
    save destination to config/router_config.local.json (which is git-ignored)
    to prevent accidental credential leakage to public version control.
    """
    if not path:
        env_p = os.environ.get("AUTO_ROUTER_CONFIG")
        if env_p:
            path = env_p
        else:
            local_p = Path("config/router_config.local.json")
            if local_p.exists() or has_plain_secrets(raw):
                path = local_p
            else:
                path = Path("config/router_config.json")
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if str(p).endswith((".yaml", ".yml")):
        import yaml
        p.write_text(yaml.safe_dump(raw, sort_keys=False, allow_unicode=True), encoding="utf-8")
    else:
        p.write_text(json.dumps(raw, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(p.resolve())

