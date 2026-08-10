import json
import os
import time
from typing import Iterator

import requests
from groq import Groq

from app.retrieval.cache import (
    compute_cache_key,
    get_cached,
    record_llm_call,
    set_cached,
)

GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
FALLBACK_MODEL = os.getenv("FALLBACK_LLM_MODEL", "llama3.2:3b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")

client = Groq(api_key=os.getenv("GROQ_API_KEY")) if os.getenv("GROQ_API_KEY") else None


def _call_groq(prompt: str) -> str:
    if client is None:
        raise RuntimeError("GROQ_API_KEY is not configured")

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return response.choices[0].message.content or ""


def _call_groq_stream(prompt: str) -> Iterator[str]:
    if client is None:
        raise RuntimeError("GROQ_API_KEY is not configured")

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        stream=True,
    )

    for chunk in response:
        if not getattr(chunk, "choices", None):
            continue
        delta = chunk.choices[0].delta
        if delta is None:
            continue
        text = getattr(delta, "content", None)
        if text:
            yield text


def _call_ollama(prompt: str) -> str:
    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={"model": FALLBACK_MODEL, "prompt": prompt, "stream": False},
        timeout=120,
    )
    response.raise_for_status()
    payload = response.json()
    return payload.get("response", "")


def _call_ollama_stream(prompt: str) -> Iterator[str]:
    with requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={"model": FALLBACK_MODEL, "prompt": prompt, "stream": True},
        timeout=120,
        stream=True,
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line:
                continue
            payload = json.loads(line)
            text = payload.get("response", "")
            if text:
                yield text


def _generate_with_fallback(prompt: str) -> str:
    last_error = None

    if client is not None:
        try:
            return _call_groq(prompt)
        except Exception as exc:  # pragma: no cover - exercised in runtime
            last_error = exc

    try:
        return _call_ollama(prompt)
    except Exception as exc:  # pragma: no cover - exercised in runtime
        last_error = exc

    raise RuntimeError(
        f"LLM generation failed via Groq and Ollama: {last_error}"
    ) from last_error


def _generate_with_fallback_stream(prompt: str) -> Iterator[str]:
    last_error = None

    if client is not None:
        try:
            yield from _call_groq_stream(prompt)
            return
        except Exception as exc:  # pragma: no cover - exercised in runtime
            last_error = exc

    try:
        yield from _call_ollama_stream(prompt)
        return
    except Exception as exc:  # pragma: no cover - exercised in runtime
        last_error = exc

    raise RuntimeError(
        f"LLM streaming failed via Groq and Ollama: {last_error}"
    ) from last_error


def build_generate_answer_prompt(question: str, context: str) -> str:
    return f"""
You are a helpful RAG assistant.

Answer ONLY from the provided context.

If the answer is not present,
say:

"I could not find the answer in the documents."

Context:
{context}

Question:
{question}
"""


def generate_answer(question: str, context: str):
    prompt = build_generate_answer_prompt(question, context)
    cache_key = compute_cache_key(f"generate_answer:{prompt}")
    cached = get_cached("llm_responses", cache_key)
    start = time.time()
    if cached is not None:
        latency_ms = (time.time() - start) * 1000
        record_llm_call(
            model=GROQ_MODEL,
            prompt=prompt,
            response=cached,
            latency_ms=latency_ms,
            call_type="generate_answer",
            cache_hit=True,
        )
        return cached

    output = _generate_with_fallback(prompt)
    latency_ms = (time.time() - start) * 1000
    set_cached("llm_responses", cache_key, output)
    record_llm_call(
        model=GROQ_MODEL,
        prompt=prompt,
        response=output,
        latency_ms=latency_ms,
        call_type="generate_answer",
    )
    return output


def generate_llm_response_stream(prompt: str) -> Iterator[str]:
    cache_key = compute_cache_key(f"generate_llm_response:{prompt}")
    cached = get_cached("llm_responses", cache_key)
    start = time.time()
    if cached is not None:
        latency_ms = (time.time() - start) * 1000
        record_llm_call(
            model="openai/gpt-oss-120b",
            prompt=prompt,
            response=cached,
            latency_ms=latency_ms,
            call_type="generate_llm_response",
            cache_hit=True,
        )
        yield cached
        return

    partial_output = []
    for chunk in _generate_with_fallback_stream(prompt):
        partial_output.append(chunk)
        yield chunk

    output = "".join(partial_output)
    latency_ms = (time.time() - start) * 1000
    set_cached("llm_responses", cache_key, output)
    record_llm_call(
        model="openai/gpt-oss-120b",
        prompt=prompt,
        response=output,
        latency_ms=latency_ms,
        call_type="generate_llm_response",
    )


def evaluate_context(question: str, context: str):
    prompt = f"""Evaluate whether the following context is relevant to the question.
Context:
{context}

Question:
{question}

Return ONLY a single decimal number between 0 and 1, where 0 means not
relevant and 1 means fully relevant.
"""

    cache_key = compute_cache_key(f"evaluate_context:{prompt}")
    cached = get_cached("llm_responses", cache_key)
    start = time.time()
    if cached is not None:
        latency_ms = (time.time() - start) * 1000
        record_llm_call(
            model=GROQ_MODEL,
            prompt=prompt,
            response=cached,
            latency_ms=latency_ms,
            call_type="evaluate_context",
            cache_hit=True,
        )
        return cached

    output = _generate_with_fallback(prompt)
    latency_ms = (time.time() - start) * 1000
    set_cached("llm_responses", cache_key, output)
    record_llm_call(
        model=GROQ_MODEL,
        prompt=prompt,
        response=output,
        latency_ms=latency_ms,
        call_type="evaluate_context",
    )
    return output


def verify_answer(question: str, context: str, answer: str):
    prompt = f"""Assess whether the answer below is fully supported by the provided
context.

Context:
{context}

Question:
{question}

Answer:
{answer}

Return a JSON object ONLY with these fields:
{{
  "supported": true|false,
  "confidence": <number between 0 and 1>,
  "reasoning": "one sentence explaining whether the answer is grounded"
}}
"""

    cache_key = compute_cache_key(f"verify_answer:{prompt}")
    cached = get_cached("llm_responses", cache_key)
    start = time.time()
    if cached is not None:
        latency_ms = (time.time() - start) * 1000
        record_llm_call(
            model=GROQ_MODEL,
            prompt=prompt,
            response=cached,
            latency_ms=latency_ms,
            call_type="verify_answer",
            cache_hit=True,
        )
        return cached

    output = _generate_with_fallback(prompt)
    latency_ms = (time.time() - start) * 1000
    set_cached("llm_responses", cache_key, output)
    record_llm_call(
        model=GROQ_MODEL,
        prompt=prompt,
        response=output,
        latency_ms=latency_ms,
        call_type="verify_answer",
    )
    return output


def generate_llm_response(prompt: str):
    cache_key = compute_cache_key(f"generate_llm_response:{prompt}")
    cached = get_cached("llm_responses", cache_key)
    start = time.time()
    if cached is not None:
        latency_ms = (time.time() - start) * 1000
        record_llm_call(
            model=GROQ_MODEL,
            prompt=prompt,
            response=cached,
            latency_ms=latency_ms,
            call_type="generate_llm_response",
            cache_hit=True,
        )
        return cached

    output = _generate_with_fallback(prompt)
    latency_ms = (time.time() - start) * 1000
    set_cached("llm_responses", cache_key, output)
    record_llm_call(
        model=GROQ_MODEL,
        prompt=prompt,
        response=output,
        latency_ms=latency_ms,
        call_type="generate_llm_response",
    )
    return output
