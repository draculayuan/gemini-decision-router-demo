# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""Async client for Gemini Decision 1 on the Gemini Decisions API (v1beta).

Uses `genai.Client.aio.decisions.create` natively when `google-genai >= 3.2` is
installed, and falls back seamlessly to the official REST endpoint
(`POST https://generativelanguage.googleapis.com/v1beta/decisions`) via HTTP/2
`httpx.AsyncClient` when `google-adk` pins `google-genai < 3.0`.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any

import httpx
from google import genai

from app.pricing import (
    ARGON_MODEL_ID,
    DECISION_MODEL_ID,
    FLASH_LITE_MODEL_ID,
    HYDROGEN_MODEL_ID,
)

DECISIONS_REST_URL = "https://generativelanguage.googleapis.com/v1beta/decisions"

ROUTING_QUESTIONS: dict[str, dict[str, Any]] = {
    "model_tier": {
        "type": "choice",
        "instructions": (
            "Which Gemini generative model tier best fits the reasoning depth, "
            "domain complexity, and accuracy requirements of the user's latest request?"
        ),
        "criteria": {
            FLASH_LITE_MODEL_ID: (
                "Greetings, small talk, simple factual Q&A, translation, short "
                "summarization, formatting, or straightforward field extraction "
                "from text, receipts, or images."
            ),
            HYDROGEN_MODEL_ID: (
                "Standard software engineering, interactive web/UI apps, multi-step "
                "synthesis, multimodal document analysis, or everyday agentic workflows."
            ),
            ARGON_MODEL_ID: (
                "Frontier reasoning: formal math or proofs, subtle distributed-systems "
                "race-condition or deadlock debugging, deep system architecture "
                "trade-offs, or dense legal/financial/scientific analysis."
            ),
        },
    },
    "thinking_level": {
        "type": "choice",
        "instructions": (
            "If this request runs on a thinking-capable model, what thinking "
            "effort level is appropriate?"
        ),
        "criteria": {
            "LOW": "Direct response requiring minimal internal deliberation.",
            "MEDIUM": "Balanced multi-step reasoning for moderately complex tasks.",
            "HIGH": "Deep, exhaustive verification for subtle or high-stakes problems.",
        },
    },
    "needs_search": {
        "type": "noul",
        "instructions": (
            "Does answering the user's latest request accurately require live web "
            "search, current events, recent news, real-time prices, or fresh facts?"
        ),
    },
    "complexity": {
        "type": "score",
        "instructions": "Rate the cognitive complexity of the user's latest request.",
        "criteria": [
            "Trivial lookup, greeting, or mechanical formatting",
            "Routine single-step task or straightforward explanation",
            "Multi-constraint engineering or analytical synthesis",
            "Frontier-grade multi-hop reasoning, proof, or deep debugging",
        ],
    },
}

_shared_http_client: httpx.AsyncClient | None = None


def _get_http_client() -> httpx.AsyncClient:
    global _shared_http_client
    if _shared_http_client is None or _shared_http_client.is_closed:
        _shared_http_client = httpx.AsyncClient(
            http2=True,
            timeout=httpx.Timeout(20.0, connect=5.0),
        )
    return _shared_http_client


@dataclass(frozen=True)
class RoutingDecisionResult:
    """Structured telemetry returned by a single-prefill Gemini Decision 1 call."""

    decision_model: str
    raw_choice: str
    tier_probabilities: dict[str, float]
    tier_confidence: float
    thinking_level: str
    thinking_probabilities: dict[str, float]
    thinking_confidence: float
    needs_search_prob: float
    needs_search_confidence: float
    complexity_score: float
    complexity_max: int
    complexity_probabilities: dict[str, float]
    complexity_legend: dict[str, str]
    decision_input_tokens: int
    decision_output_tokens: int
    decision_ms: float
    transport: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _to_plain_dict(obj: Any) -> dict[str, Any]:
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    raise TypeError(f"Unexpected Decisions API response type: {type(obj)!r}")


async def evaluate_routing_decision(
    *,
    api_key: str,
    decision_input: str | list[dict[str, Any]],
    questions: dict[str, dict[str, Any]] | None = None,
) -> RoutingDecisionResult:
    """Evaluate all routing questions in one single-prefill Gemini Decision 1 pass."""
    active_questions = questions or ROUTING_QUESTIONS
    sdk_client = genai.Client(
        api_key=api_key,
        vertexai=False,
        http_options={"api_version": "v1beta"},
    )

    t0 = time.perf_counter()
    if hasattr(sdk_client.aio, "decisions"):
        sdk_resp = await sdk_client.aio.decisions.create(
            model=DECISION_MODEL_ID,
            input=decision_input,
            questions=active_questions,
        )
        decision_ms = (time.perf_counter() - t0) * 1000.0
        payload = _to_plain_dict(sdk_resp)
        transport = "google-genai-sdk"
    else:
        http_client = _get_http_client()
        resp = await http_client.post(
            DECISIONS_REST_URL,
            headers={
                "x-goog-api-key": api_key,
                "Content-Type": "application/json",
            },
            json={
                "model": DECISION_MODEL_ID,
                "input": decision_input,
                "questions": active_questions,
            },
        )
        decision_ms = (time.perf_counter() - t0) * 1000.0
        if resp.status_code >= 400:
            raise RuntimeError(
                f"Gemini Decisions API HTTP {resp.status_code}: {resp.text}"
            )
        payload = resp.json()
        transport = "rest-v1beta"

    answers = payload.get("answers") or {}
    usage = payload.get("usage") or {}

    tier_ans = answers["model_tier"]
    thinking_ans = answers["thinking_level"]
    search_ans = answers["needs_search"]
    complexity_ans = answers["complexity"]

    return RoutingDecisionResult(
        decision_model=DECISION_MODEL_ID,
        raw_choice=str(tier_ans["choice"]),
        tier_probabilities={
            str(k): float(v) for k, v in (tier_ans.get("probabilities") or {}).items()
        },
        tier_confidence=float(tier_ans.get("confidence", 0.0)),
        thinking_level=str(thinking_ans["choice"]),
        thinking_probabilities={
            str(k): float(v)
            for k, v in (thinking_ans.get("probabilities") or {}).items()
        },
        thinking_confidence=float(thinking_ans.get("confidence", 0.0)),
        needs_search_prob=float(search_ans["noul"]),
        needs_search_confidence=float(search_ans.get("confidence", 0.0)),
        complexity_score=float(complexity_ans["score"]),
        complexity_max=max(len(active_questions["complexity"]["criteria"]) - 1, 1),
        complexity_probabilities={
            str(k): float(v)
            for k, v in (complexity_ans.get("probabilities") or {}).items()
        },
        complexity_legend={
            str(k): str(v) for k, v in (complexity_ans.get("legend") or {}).items()
        },
        decision_input_tokens=int(usage.get("total_input_tokens", 0)),
        decision_output_tokens=int(usage.get("total_output_tokens", 0)),
        decision_ms=round(decision_ms, 2),
        transport=transport,
    )
