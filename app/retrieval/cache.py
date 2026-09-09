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


# ---------------------------------------------------------------------------
# Semantic cache
#
# Unlike get_cached/set_cached above (exact string match via SHA-256), this
# variant embeds the cache key text and matches on cosine similarity, so a
# paraphrased question ("What's the refund window?" vs "How long do I have
# to request a refund?") can still hit a previously cached entry. Entries
# are stored as a list per cache_name so we can scan for the closest match;
# fine for the scale of a JSON-file cache (hundreds/low thousands of
# entries), not meant for a large shared cache.
# ---------------------------------------------------------------------------

# Calibrated empirically against all-MiniLM-L6-v2 (the model used for
# retrieval): true paraphrases of the same question score ~0.65-0.85,
# related-but-different questions ("refund policy" vs "shipping policy",
# "attention mechanism" vs "multi-head attention") score ~0.25-0.55, and
# unrelated questions score ~0.05-0.15. 0.92 (the original guess) is above
# the paraphrase cluster entirely, so it never matched anything but
# near-identical text. 0.75 sits just below the paraphrase cluster.
DEFAULT_SEMANTIC_THRESHOLD = 0.75


def _embed_text(text: str) -> list[float]:
    from app.ingestion.embedder import get_embedding_model

    model = get_embedding_model()
    return model.embed_query(text)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    import numpy as np

    va, vb = np.array(a, dtype=float), np.array(b, dtype=float)
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if denom == 0:
        return 0.0
    return float(np.dot(va, vb) / denom)


def get_semantic_cached(
    cache_name: str,
    text: str,
    threshold: float = DEFAULT_SEMANTIC_THRESHOLD,
) -> Any:
    """Return a cached value for `text`, matching by exact hash first and
    falling back to the closest embedding match above `threshold`.

    Returns None on a cache miss, or if embeddings are unavailable.
    """
    cache = load_cache(cache_name)
    entries: list[dict] = cache.get("entries", [])
    if not entries:
        return None

    exact_hash = compute_cache_key(text)
    for entry in entries:
        if entry.get("key_hash") == exact_hash:
            return entry.get("value")

    try:
        query_embedding = _embed_text(text)
    except Exception:
        # Embeddings unavailable (e.g. heavy ML deps missing) — no semantic
        # match possible, behave as a cache miss.
        return None

    best_score = 0.0
    best_value = None
    for entry in entries:
        embedding = entry.get("embedding")
        if not embedding:
            continue
        score = _cosine_similarity(query_embedding, embedding)
        if score > best_score:
            best_score = score
            best_value = entry.get("value")

    return best_value if best_score >= threshold else None


def set_semantic_cached(
    cache_name: str, text: str, value: Any, max_entries: int = 500
) -> None:
    """Cache `value` under `text`, storing its embedding for future
    semantic lookups via get_semantic_cached. Keeps at most `max_entries`,
    evicting the oldest first.
    """
    cache = load_cache(cache_name)
    entries: list[dict] = cache.get("entries", [])

    try:
        embedding = _embed_text(text)
    except Exception:
        embedding = None

    key_hash = compute_cache_key(text)
    entries = [e for e in entries if e.get("key_hash") != key_hash]
    entries.insert(
        0,
        {
            "key_hash": key_hash,
            "text": text,
            "embedding": embedding,
            "value": value,
            "timestamp": int(time.time()),
        },
    )

    cache["entries"] = entries[:max_entries]
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
