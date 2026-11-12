# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""ADK `before_model_callback` dynamic model router powered by Gemini Decision 1."""

from __future__ import annotations

import asyncio
import base64
import contextvars
import time
from dataclasses import dataclass, field
from typing import Any

from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types

from app.decisions_client import RoutingDecisionResult, evaluate_routing_decision
from app.pricing import (
    ARGON_MODEL_ID,
    FLASH_LITE_MODEL_ID,
    FLASH_MODEL_ID,
    TIER_ORDER,
)
from app.secret_loader import get_gemini_api_key

SEARCH_GROUNDING_THRESHOLD = 0.70
VALID_THINKING_LEVELS = frozenset({"LOW", "MEDIUM", "HIGH"})


@dataclass
class TurnRoutingContext:
    """Per-request context shared between FastAPI SSE handler and ADK callback."""

    confidence_threshold: float = 0.65
    forced_model: str | None = None
    enable_search_grounding: bool = True
    event_queue: asyncio.Queue[dict[str, Any]] | None = None
    decision_result: RoutingDecisionResult | None = None
    selected_model: str | None = None
    escalated_by_confidence: bool = False
    applied_thinking_level: str | None = None
    applied_search_grounding: bool = False
    llm_dispatch_perf_ts: float | None = None
    extra_notes: list[str] = field(default_factory=list)


routing_context_var: contextvars.ContextVar[TurnRoutingContext | None] = (
    contextvars.ContextVar("routing_context_var", default=None)
)


def serialize_contents_for_decisions(
    contents: list[types.Content],
    *,
    max_history_turns: int = 6,
) -> list[dict[str, Any]]:
    """Convert ADK `LlmRequest.contents` into Decisions API input blocks.

    Per Decisions API rules:
    - Conversation history, thought blocks, and tool-call blocks are serialized
      as plain text (not structured message blocks).
    - Attached images, PDFs, audio, and video from the latest user message are
      forwarded as native multimodal Decisions API blocks (`resolution` only set
      on `image` and `video`, never on `document` or `audio`).
    """
    recent = contents[-max_history_turns:] if max_history_turns > 0 else contents
    transcript_lines: list[str] = []
    media_blocks: list[dict[str, Any]] = []

    last_user_idx = -1
    for idx in range(len(recent) - 1, -1, -1):
        if (recent[idx].role or "user") == "user":
            last_user_idx = idx
            break

    for idx, content in enumerate(recent):
        role_label = "User" if (content.role or "user") == "user" else "Assistant"
        text_fragments: list[str] = []
        for part in content.parts or []:
            if getattr(part, "thought", False):
                continue
            if part.text:
                text_fragments.append(part.text.strip())
            elif part.inline_data and idx == last_user_idx:
                mime = (part.inline_data.mime_type or "").lower()
                raw_bytes = part.inline_data.data or b""
                if not raw_bytes or not mime:
                    continue
                b64_str = (
                    base64.b64encode(raw_bytes).decode("ascii")
                    if isinstance(raw_bytes, (bytes, bytearray))
                    else str(raw_bytes)
                )
                if mime.startswith("image/"):
                    media_blocks.append(
                        {
                            "type": "image",
                            "data": b64_str,
                            "mime_type": mime,
                            "resolution": "low",
                        }
                    )
                elif mime == "application/pdf":
                    media_blocks.append(
                        {
                            "type": "document",
                            "data": b64_str,
                            "mime_type": mime,
                        }
                    )
                elif mime.startswith("audio/"):
                    media_blocks.append(
                        {
                            "type": "audio",
                            "data": b64_str,
                            "mime_type": mime,
                        }
                    )
                elif mime.startswith("video/"):
                    media_blocks.append(
                        {
                            "type": "video",
                            "data": b64_str,
                            "mime_type": mime,
                            "resolution": "low",
                        }
                    )

        if text_fragments:
            transcript_lines.append(f"{role_label}: {' '.join(text_fragments)}")

    blocks: list[dict[str, Any]] = []
    transcript_text = "\n".join(transcript_lines).strip()
    if transcript_text:
        blocks.append({"type": "text", "text": transcript_text + "\n"})
    elif media_blocks:
        blocks.append({"type": "text", "text": "User uploaded media attachment.\n"})
    else:
        blocks.append({"type": "text", "text": "Empty user turn."})

    blocks.extend(media_blocks)
    return blocks


def resolve_tier_with_confidence_escalation(
    *,
    raw_choice: str,
    tier_confidence: float,
    confidence_threshold: float,
) -> tuple[str, bool]:
    """Escalate one model tier up when Decision 1 confidence is below threshold."""
    chosen = raw_choice if raw_choice in TIER_ORDER else FLASH_MODEL_ID
    idx = TIER_ORDER.index(chosen)
    if tier_confidence < confidence_threshold and idx < len(TIER_ORDER) - 1:
        return TIER_ORDER[idx + 1], True
    return chosen, False


def apply_model_specific_config(
    llm_request: LlmRequest,
    *,
    selected_model: str,
    thinking_level: str,
    attach_google_search: bool,
) -> str | None:
    """Mutate `llm_request.model` and `llm_request.config` safely per Gemini tier.

    Enforces documented Gemini 4 Argon & Gemini 3.8 Flash API rules:
    - Thinking levels must be one of `LOW`, `MEDIUM`, `HIGH` (`MINIMAL` or
      `thinking_budget` raises HTTP 400 on 3.8 Flash and 4 Argon).
    - Custom sampling parameters (`temperature`, `top_p`, `top_k`) are stripped
      on Gemini 4 Argon.
    """
    llm_request.model = selected_model
    llm_request.config = llm_request.config or types.GenerateContentConfig()

    safe_level = (
        thinking_level.upper()
        if thinking_level.upper() in VALID_THINKING_LEVELS
        else "MEDIUM"
    )

    applied_level: str | None = None
    if selected_model in (FLASH_MODEL_ID, ARGON_MODEL_ID):
        llm_request.config.thinking_config = types.ThinkingConfig(
            thinking_level=safe_level,  # type: ignore[arg-type]
        )
        applied_level = safe_level
    else:
        llm_request.config.thinking_config = None

    if selected_model == ARGON_MODEL_ID:
        llm_request.config.temperature = None
        llm_request.config.top_p = None
        llm_request.config.top_k = None

    has_function_tools = bool(llm_request.tools_dict)
    if attach_google_search and not has_function_tools:
        llm_request.config.tools = [types.Tool(google_search=types.GoogleSearch())]
    elif not has_function_tools:
        llm_request.config.tools = None

    return applied_level


async def route_with_gemini_decision_1(
    callback_context: CallbackContext,
    llm_request: LlmRequest,
) -> LlmResponse | None:
    """ADK `before_model_callback` executing single-prefill Gemini Decision 1 routing."""
    turn_ctx = routing_context_var.get()
    confidence_threshold = (
        turn_ctx.confidence_threshold if turn_ctx is not None else 0.65
    )
    forced_model = turn_ctx.forced_model if turn_ctx is not None else None
    allow_search = (
        turn_ctx.enable_search_grounding if turn_ctx is not None else True
    )

    api_key = get_gemini_api_key()
    decision_blocks = serialize_contents_for_decisions(llm_request.contents)

    decision = await evaluate_routing_decision(
        api_key=api_key,
        decision_input=decision_blocks,
    )

    if forced_model and forced_model in TIER_ORDER:
        selected_model = forced_model
        escalated = False
    else:
        selected_model, escalated = resolve_tier_with_confidence_escalation(
            raw_choice=decision.raw_choice,
            tier_confidence=decision.tier_confidence,
            confidence_threshold=confidence_threshold,
        )

    attach_search = (
        allow_search
        and decision.needs_search_prob >= SEARCH_GROUNDING_THRESHOLD
        and not bool(llm_request.tools_dict)
    )

    applied_thinking = apply_model_specific_config(
        llm_request,
        selected_model=selected_model,
        thinking_level=decision.thinking_level,
        attach_google_search=attach_search,
    )

    routing_payload: dict[str, Any] = {
        **decision.to_dict(),
        "selected_model": selected_model,
        "forced_model": forced_model,
        "confidence_threshold": confidence_threshold,
        "escalated_by_confidence": escalated,
        "applied_thinking_level": applied_thinking,
        "applied_search_grounding": attach_search,
    }

    callback_context.state["last_routing_decision"] = routing_payload

    if turn_ctx is not None:
        turn_ctx.decision_result = decision
        turn_ctx.selected_model = selected_model
        turn_ctx.escalated_by_confidence = escalated
        turn_ctx.applied_thinking_level = applied_thinking
        turn_ctx.applied_search_grounding = attach_search
        turn_ctx.llm_dispatch_perf_ts = time.perf_counter()
        if turn_ctx.event_queue is not None:
            await turn_ctx.event_queue.put(
                {
                    "type": "routing_decision",
                    "data": routing_payload,
                }
            )

    return None
