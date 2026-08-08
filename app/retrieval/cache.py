import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any

CACHE_ROOT = Path("data/cache")
CACHE_ROOT.mkdir(parents=True, exist_ok=True)


def _cache_file(cache_name: str) -> Path:
    return CACHE_ROOT / f"{cache_name}.json"


def compute_cache_key(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_cache(cache_name: str) -> dict[str, Any]:
    path = _cache_file(cache_name)
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as fp:
        try:
            return json.load(fp)
        except json.JSONDecodeError:
            return {}


def save_cache(cache_name: str, data: dict[str, Any]) -> None:
    path = _cache_file(cache_name)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(data, fp)


def get_cached(cache_name: str, key: str) -> Any:
    return load_cache(cache_name).get(key)


def set_cached(cache_name: str, key: str, value: Any) -> None:
    cache = load_cache(cache_name)
    cache[key] = value
    save_cache(cache_name, cache)


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    tokens = re.findall(r"\w+|[^\s\w]", text)
    return max(1, len(tokens))


def get_cost_per_token(model: str) -> float:
    rates = {
        "openai/gpt-oss-120b": 0.0001,
    }
    return rates.get(model, 0.0001)


def record_llm_call(
    model: str,
    prompt: str,
    response: str,
    latency_ms: float,
    call_type: str,
    cache_hit: bool = False,
) -> dict[str, Any]:
    prompt_tokens = estimate_tokens(prompt)
    completion_tokens = estimate_tokens(response)
    total_tokens = prompt_tokens + completion_tokens
    cost = 0.0 if cache_hit else total_tokens * get_cost_per_token(model)

    metrics = load_cache("llm_metrics")
    summary = metrics.get(
        "summary",
        {
            "calls": 0,
            "api_calls": 0,
            "cache_hits": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "total_cost": 0.0,
            "total_latency_ms": 0.0,
        },
    )

    summary["calls"] += 1
    summary["cache_hits"] += 1 if cache_hit else 0
    summary["api_calls"] += 0 if cache_hit else 1
    summary["prompt_tokens"] += prompt_tokens
    summary["completion_tokens"] += completion_tokens
    summary["total_tokens"] += total_tokens
    summary["total_cost"] += cost
    summary["total_latency_ms"] += latency_ms

    entry = {
        "timestamp": int(time.time()),
        "call_type": call_type,
        "model": model,
        "cache_hit": cache_hit,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "cost": cost,
        "latency_ms": latency_ms,
    }

    history = metrics.get("history", [])
    history.insert(0, entry)
    history = history[:100]

    metrics["summary"] = summary
    metrics["history"] = history
    save_cache("llm_metrics", metrics)

    return entry


def get_llm_metrics() -> dict[str, Any]:
    return load_cache("llm_metrics")
