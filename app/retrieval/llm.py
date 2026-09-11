import json
import os
import time
from typing import Iterator

from groq import Groq

from app.retrieval.cache import (
    compute_cache_key,
    get_cached,
    record_llm_call,
    set_cached,
)

from app.core.config import get_settings


client = Groq(
    api_key=get_settings().groq_api_key,
)


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
            model="openai/gpt-oss-120b",
            prompt=prompt,
            response=cached,
            latency_ms=latency_ms,
            call_type="generate_answer",
            cache_hit=True,
        )
        return cached

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    output = response.choices[0].message.content
    latency_ms = (time.time() - start) * 1000
    set_cached("llm_responses", cache_key, output)
    record_llm_call(
        model="openai/gpt-oss-120b",
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

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        stream=True,
    )

    partial_output = []
    for chunk in response:
        if not getattr(chunk, "choices", None):
            continue
        delta = chunk.choices[0].delta
        if delta is None:
            continue
        text = getattr(delta, "content", None)
        if text:
            partial_output.append(text)
            yield text

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
            model="openai/gpt-oss-120b",
            prompt=prompt,
            response=cached,
            latency_ms=latency_ms,
            call_type="evaluate_context",
            cache_hit=True,
        )
        return cached

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    output = response.choices[0].message.content
    latency_ms = (time.time() - start) * 1000
    set_cached("llm_responses", cache_key, output)
    record_llm_call(
        model="openai/gpt-oss-120b",
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
            model="openai/gpt-oss-120b",
            prompt=prompt,
            response=cached,
            latency_ms=latency_ms,
            call_type="verify_answer",
            cache_hit=True,
        )
        return cached

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    output = response.choices[0].message.content
    latency_ms = (time.time() - start) * 1000
    set_cached("llm_responses", cache_key, output)
    record_llm_call(
        model="openai/gpt-oss-120b",
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
            model="openai/gpt-oss-120b",
            prompt=prompt,
            response=cached,
            latency_ms=latency_ms,
            call_type="generate_llm_response",
            cache_hit=True,
        )
        return cached

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    output = response.choices[0].message.content
    latency_ms = (time.time() - start) * 1000
    set_cached("llm_responses", cache_key, output)
    record_llm_call(
        model="openai/gpt-oss-120b",
        prompt=prompt,
        response=output,
        latency_ms=latency_ms,
        call_type="generate_llm_response",
    )
    return output
