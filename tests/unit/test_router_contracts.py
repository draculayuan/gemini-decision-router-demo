# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""Unit, API contract, and static anti-mocking tests for the Gemini Decision 1 Router."""

from __future__ import annotations

import ast
import base64
import struct
from pathlib import Path

from fastapi.testclient import TestClient
from google.adk.models.llm_request import LlmRequest
from google.genai import types

from app.agent import root_agent
from app.demo_server import demo_app
from app.pricing import (
    ARGON_MODEL_ID,
    FLASH_LITE_MODEL_ID,
    FLASH_MODEL_ID,
    calculate_turn_cost,
)
from app.router_callback import (
    apply_model_specific_config,
    resolve_tier_with_confidence_escalation,
    serialize_contents_for_decisions,
)
from app.sample_media import build_sample_receipt_png_b64

PROJECT_ROOT = Path(__file__).resolve().parents[2]
APP_DIR = PROJECT_ROOT / "app"


def test_static_anti_mocking_and_zero_hardcoded_secrets() -> None:
    """Enforce zero runtime mock fallbacks and zero hardcoded API keys in app/ and .env."""
    forbidden_tokens = (
        "mock_success",
        "mock_fallback",
        "fake_response",
        "dummy_decision",
        "BEGIN PRIVATE KEY",
    )
    for py_file in APP_DIR.rglob("*.py"):
        source = py_file.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(py_file))
        assert tree is not None
        for token in forbidden_tokens:
            assert token not in source, f"Forbidden token {token!r} found in {py_file}"

    env_file = PROJECT_ROOT / ".env"
    if env_file.exists():
        env_text = env_file.read_text(encoding="utf-8")
        assert "GEMINI_API_KEY=" not in env_text


def test_root_agent_wires_before_model_callback() -> None:
    """Verify ADK root_agent uses SecretManagerGemini and before_model_callback."""
    assert root_agent.name == "dynamic_router_assistant"
    assert root_agent.before_model_callback is not None


def test_serialize_contents_for_decisions_multimodal_wire_contract() -> None:
    """Verify thought stripping, image resolution='low', and PDF without resolution."""
    png_bytes = base64.b64decode(build_sample_receipt_png_b64())
    pdf_bytes = b"%PDF-1.4 sample"

    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text="Hi there")],
        ),
        types.Content(
            role="model",
            parts=[
                types.Part(text="internal chain", thought=True),
                types.Part.from_text(text="Hello! How can I help?"),
            ],
        ),
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text="Parse this receipt and invoice PDF."),
                types.Part.from_bytes(data=png_bytes, mime_type="image/png"),
                types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf"),
            ],
        ),
    ]

    blocks = serialize_contents_for_decisions(contents)
    assert blocks[0]["type"] == "text"
    assert "internal chain" not in blocks[0]["text"]
    assert "User: Parse this receipt and invoice PDF." in blocks[0]["text"]

    img_block = next(b for b in blocks if b["type"] == "image")
    assert img_block["mime_type"] == "image/png"
    assert img_block["resolution"] == "low"

    doc_block = next(b for b in blocks if b["type"] == "document")
    assert doc_block["mime_type"] == "application/pdf"
    assert "resolution" not in doc_block


def test_confidence_escalation_ladder() -> None:
    """Verify low-confidence decisions escalate one tier and cap at Gemini 4 Argon."""
    model, esc = resolve_tier_with_confidence_escalation(
        raw_choice=FLASH_LITE_MODEL_ID,
        tier_confidence=0.52,
        confidence_threshold=0.65,
    )
    assert model == FLASH_MODEL_ID
    assert esc is True

    model2, esc2 = resolve_tier_with_confidence_escalation(
        raw_choice=FLASH_MODEL_ID,
        tier_confidence=0.40,
        confidence_threshold=0.65,
    )
    assert model2 == ARGON_MODEL_ID
    assert esc2 is True

    model3, esc3 = resolve_tier_with_confidence_escalation(
        raw_choice=ARGON_MODEL_ID,
        tier_confidence=0.35,
        confidence_threshold=0.65,
    )
    assert model3 == ARGON_MODEL_ID
    assert esc3 is False


def test_apply_model_specific_config_argon_and_flash_guardrails() -> None:
    """Verify Argon strips sampling params and clamps invalid MINIMAL thinking."""
    req = LlmRequest(
        model=FLASH_LITE_MODEL_ID,
        config=types.GenerateContentConfig(temperature=0.7, top_p=0.9, top_k=40),
    )

    level = apply_model_specific_config(
        req,
        selected_model=ARGON_MODEL_ID,
        thinking_level="MINIMAL",
        attach_google_search=True,
    )
    assert req.model == ARGON_MODEL_ID
    assert level == "MEDIUM"
    assert req.config.thinking_config is not None
    assert str(req.config.thinking_config.thinking_level).endswith("MEDIUM")
    assert req.config.temperature is None
    assert req.config.top_p is None
    assert req.config.top_k is None
    assert req.config.tools is not None
    assert req.config.tools[0].google_search is not None

    # Switching back to 3.5 Flash-Lite disables thinking_config
    level_lite = apply_model_specific_config(
        req,
        selected_model=FLASH_LITE_MODEL_ID,
        thinking_level="HIGH",
        attach_google_search=False,
    )
    assert req.model == FLASH_LITE_MODEL_ID
    assert level_lite is None
    assert req.config.thinking_config is None
    assert req.config.tools is None


def test_pricing_counterfactual_savings() -> None:
    """Verify Decision 1 0-output-token billing and savings vs. Always-Argon."""
    breakdown = calculate_turn_cost(
        routed_model=FLASH_LITE_MODEL_ID,
        decision_input_tokens=250,
        generation_input_tokens=500,
        generation_output_tokens=400,
        generation_thought_tokens=0,
    )
    assert breakdown.decision_output_tokens == 0
    assert breakdown.actual_total_cost_usd < breakdown.always_argon_cost_usd
    assert breakdown.savings_pct > 70.0


def test_sample_receipt_png_and_http_routes() -> None:
    """Verify valid PNG binary generation and FastAPI UI routes."""
    raw = base64.b64decode(build_sample_receipt_png_b64())
    assert raw[:8] == b"\x89PNG\r\n\x1a\n"
    width, height = struct.unpack(">II", raw[16:24])
    assert (width, height) == (420, 320)

    client = TestClient(demo_app)
    res_html = client.get("/")
    assert res_html.status_code == 200
    assert "Gemini Decision 1" in res_html.text
    assert "secretPill" not in res_html.text

    res_status = client.get("/api/status")
    assert res_status.status_code == 200
    status_body = res_status.json()
    assert "ready" in status_body
    for sensitive_key in ("secret", "project_id", "secret_id", "resource_name", "key_fingerprint"):
        assert sensitive_key not in status_body

    res_receipt = client.get("/api/sample-receipt")
    assert res_receipt.status_code == 200
    assert res_receipt.json()["mime_type"] == "image/png"
