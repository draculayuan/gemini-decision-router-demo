# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""FastAPI SSE server for the Gemini Decision 1 Dynamic Model Routing Demo.

Streams two-stage telemetry per turn:
1. Instant `routing_decision` SSE event as soon as `gemini-decision-1` completes
   its single-prefill pass (~150-250 ms, 0 output tokens).
2. Live `ttft`, `text_delta`, and `turn_complete` SSE events with exact
   `perf_counter()` Latency Waterfall breakdowns and Always-Argon cost savings.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from google.adk.agents.run_config import RunConfig, StreamingMode
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from pydantic import BaseModel, Field

from app.agent import app as adk_app
from app.pricing import (
    ARGON_MODEL_ID,
    MODEL_RATES_PER_1M_USD,
    calculate_turn_cost,
)
from app.router_callback import TurnRoutingContext, routing_context_var
from app.sample_media import build_sample_receipt_png_b64
from app.secret_loader import (
    SecretConfigurationError,
    clear_secret_cache,
    inspect_secret_status,
)

load_dotenv()
logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "static"

demo_app = FastAPI(
    title="Gemini Decision 1 — Dynamic Model Routing Demo",
    version="0.1.0",
)

_session_service = InMemorySessionService()
_runner = Runner(
    app=adk_app,
    session_service=_session_service,
    auto_create_session=True,
)


class MediaAttachment(BaseModel):
    """Optional inline media attachment (image, PDF, audio, or video)."""

    mime_type: str
    data_b64: str
    filename: str | None = None


class ChatTurnRequest(BaseModel):
    """Incoming chat request payload from the Split-Screen Demo UI."""

    user_id: str = Field(default_factory=lambda: f"user-{uuid.uuid4().hex}")
    session_id: str = Field(default_factory=lambda: f"sess-{uuid.uuid4().hex}")
    message: str = ""
    confidence_threshold: float = Field(default=0.65, ge=0.0, le=1.0)
    forced_model: str | None = None
    enable_search_grounding: bool = True
    attachment: MediaAttachment | None = None


def _sse_frame(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


@demo_app.get("/")
async def index() -> FileResponse:
    return FileResponse(
        STATIC_DIR / "index.html",
        headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"},
    )


@demo_app.get("/api/status")
async def status() -> dict[str, Any]:
    secret_status = inspect_secret_status(force_refresh=False)
    if not secret_status.configured and secret_status.error:
        logger.warning("Secret Manager check failed: %s", secret_status.error)
    return {
        "ready": secret_status.configured,
        "rates_per_1m_usd": MODEL_RATES_PER_1M_USD,
    }


@demo_app.post("/api/status/refresh")
async def refresh_status() -> dict[str, Any]:
    clear_secret_cache()
    secret_status = inspect_secret_status(force_refresh=True)
    if not secret_status.configured and secret_status.error:
        logger.warning("Secret Manager refresh failed: %s", secret_status.error)
    return {
        "ready": secret_status.configured,
        "rates_per_1m_usd": MODEL_RATES_PER_1M_USD,
    }


@demo_app.get("/api/sample-receipt")
async def sample_receipt() -> dict[str, str]:
    return {
        "filename": "bay_view_roasters_receipt.png",
        "mime_type": "image/png",
        "data_b64": build_sample_receipt_png_b64(),
    }


@demo_app.post("/api/chat")
async def chat_stream(req: ChatTurnRequest) -> StreamingResponse:
    event_queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()

    parts: list[types.Part] = []
    if req.message.strip():
        parts.append(types.Part.from_text(text=req.message.strip()))
    if req.attachment is not None:
        raw_bytes = base64.b64decode(req.attachment.data_b64)
        parts.append(
            types.Part.from_bytes(
                data=raw_bytes,
                mime_type=req.attachment.mime_type,
            )
        )
    if not parts:
        parts.append(types.Part.from_text(text="Hello"))

    user_content = types.Content(role="user", parts=parts)

    async def _drive_turn() -> None:
        turn_start_ts = time.perf_counter()
        turn_ctx = TurnRoutingContext(
            confidence_threshold=req.confidence_threshold,
            forced_model=req.forced_model or None,
            enable_search_grounding=req.enable_search_grounding,
            event_queue=event_queue,  # type: ignore[arg-type]
        )
        token = routing_context_var.set(turn_ctx)
        first_token_ts: float | None = None
        emitted_partial = False
        prompt_tokens = 0
        output_tokens = 0
        thought_tokens = 0
        grounding_sources: list[dict[str, str]] = []

        try:
            async for event in _runner.run_async(
                user_id=req.user_id,
                session_id=req.session_id,
                new_message=user_content,
                run_config=RunConfig(streaming_mode=StreamingMode.SSE),
            ):
                if event.error_message:
                    raise RuntimeError(event.error_message)

                is_partial = bool(getattr(event, "partial", False))
                text_chunks: list[str] = []
                if event.content and event.content.parts:
                    for part in event.content.parts:
                        if getattr(part, "thought", False):
                            continue
                        if part.text:
                            text_chunks.append(part.text)

                combined_chunk = "".join(text_chunks)

                if is_partial and combined_chunk:
                    now = time.perf_counter()
                    if first_token_ts is None:
                        first_token_ts = now
                        dispatch_ts = turn_ctx.llm_dispatch_perf_ts or turn_start_ts
                        ttft_ms = max((first_token_ts - dispatch_ts) * 1000.0, 0.0)
                        await event_queue.put(
                            {"type": "ttft", "ttft_ms": round(ttft_ms, 2)}
                        )
                    emitted_partial = True
                    await event_queue.put(
                        {"type": "text_delta", "delta": combined_chunk}
                    )
                elif not is_partial:
                    if not emitted_partial and combined_chunk:
                        now = time.perf_counter()
                        first_token_ts = now
                        dispatch_ts = turn_ctx.llm_dispatch_perf_ts or turn_start_ts
                        ttft_ms = max((first_token_ts - dispatch_ts) * 1000.0, 0.0)
                        await event_queue.put(
                            {"type": "ttft", "ttft_ms": round(ttft_ms, 2)}
                        )
                        await event_queue.put(
                            {"type": "text_delta", "delta": combined_chunk}
                        )

                    if event.usage_metadata is not None:
                        prompt_tokens = int(
                            event.usage_metadata.prompt_token_count or prompt_tokens
                        )
                        output_tokens = int(
                            event.usage_metadata.candidates_token_count or output_tokens
                        )
                        thought_tokens = int(
                            getattr(event.usage_metadata, "thoughts_token_count", 0)
                            or thought_tokens
                        )

                    gm = getattr(event, "grounding_metadata", None)
                    if gm and getattr(gm, "grounding_chunks", None):
                        for gc in gm.grounding_chunks:
                            web = getattr(gc, "web", None)
                            if web and getattr(web, "uri", None):
                                grounding_sources.append(
                                    {
                                        "title": getattr(web, "title", "") or web.uri,
                                        "uri": web.uri,
                                    }
                                )

            turn_end_ts = time.perf_counter()
            decision_ms = (
                turn_ctx.decision_result.decision_ms
                if turn_ctx.decision_result is not None
                else 0.0
            )
            dispatch_ts = turn_ctx.llm_dispatch_perf_ts or turn_start_ts
            if first_token_ts is not None:
                ttft_ms = max((first_token_ts - dispatch_ts) * 1000.0, 0.0)
                stream_ms = max((turn_end_ts - first_token_ts) * 1000.0, 0.0)
            else:
                ttft_ms = max((turn_end_ts - dispatch_ts) * 1000.0, 0.0)
                stream_ms = 0.0

            total_ms = max((turn_end_ts - turn_start_ts) * 1000.0, 0.01)
            decision_pct = round((decision_ms / total_ms) * 100.0, 1)

            routed_model = turn_ctx.selected_model or ARGON_MODEL_ID
            decision_in_tokens = (
                turn_ctx.decision_result.decision_input_tokens
                if turn_ctx.decision_result is not None
                else 0
            )

            cost_breakdown = calculate_turn_cost(
                routed_model=routed_model,
                decision_input_tokens=decision_in_tokens,
                generation_input_tokens=prompt_tokens,
                generation_output_tokens=output_tokens,
                generation_thought_tokens=thought_tokens,
            )

            await event_queue.put(
                {
                    "type": "turn_complete",
                    "data": {
                        "session_id": req.session_id,
                        "routed_model": routed_model,
                        "latency": {
                            "decision_ms": round(decision_ms, 2),
                            "ttft_ms": round(ttft_ms, 2),
                            "stream_ms": round(stream_ms, 2),
                            "total_ms": round(total_ms, 2),
                            "decision_share_pct": decision_pct,
                        },
                        "cost": cost_breakdown.to_dict(),
                        "grounding_sources": grounding_sources[:6],
                    },
                }
            )
        except SecretConfigurationError as exc:
            logger.error("Secret Manager configuration error: %s", exc)
            await event_queue.put(
                {
                    "type": "error",
                    "code": "backend_auth_unavailable",
                    "message": "Backend API credentials are not configured. Check server logs.",
                }
            )
        except Exception as exc:
            await event_queue.put(
                {
                    "type": "error",
                    "code": "turn_execution_error",
                    "message": str(exc),
                }
            )
        finally:
            routing_context_var.reset(token)
            await event_queue.put(None)

    async def _sse_generator():
        worker = asyncio.create_task(_drive_turn())
        try:
            while True:
                item = await event_queue.get()
                if item is None:
                    break
                yield _sse_frame(item)
        finally:
            if not worker.done():
                worker.cancel()

    return StreamingResponse(
        _sse_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "8080"))
    uvicorn.run(demo_app, host="0.0.0.0", port=port)
