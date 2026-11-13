# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""Token pricing & counterfactual cost calculator (November 2026 Gemini API rates)."""

from __future__ import annotations

from dataclasses import asdict, dataclass

DECISION_MODEL_ID = "gemini-decision-1"
FLASH_LITE_MODEL_ID = "gemini-3.5-flash-lite"
HYDROGEN_MODEL_ID = "gemini-4-hydrogen"
ARGON_MODEL_ID = "gemini-4-argon"

# Paid-tier standard rates in USD per 1,000,000 tokens (November 2026 Gemini API docs).
# Note: output_per_1m includes thinking tokens for thinking-capable models.
MODEL_RATES_PER_1M_USD: dict[str, dict[str, float | str]] = {
    DECISION_MODEL_ID: {
        "label": "Gemini Decision 1",
        "input_per_1m": 0.10,
        "output_per_1m": 0.00,  # Always 0 output tokens
    },
    FLASH_LITE_MODEL_ID: {
        "label": "Gemini 3.5 Flash-Lite",
        "input_per_1m": 0.30,
        "output_per_1m": 2.50,
    },
    HYDROGEN_MODEL_ID: {
        "label": "Gemini 4 Hydrogen",
        "input_per_1m": 1.50,
        "output_per_1m": 7.50,
    },
    ARGON_MODEL_ID: {
        "label": "Gemini 4 Argon",
        "input_per_1m": 2.00,
        "output_per_1m": 10.00,
    },
}

TIER_ORDER: tuple[str, ...] = (
    FLASH_LITE_MODEL_ID,
    HYDROGEN_MODEL_ID,
    ARGON_MODEL_ID,
)


@dataclass(frozen=True)
class TurnCostBreakdown:
    """Empirical cost breakdown for one routed turn vs. an Always-Argon baseline."""

    routed_model: str
    decision_input_tokens: int
    decision_output_tokens: int
    generation_input_tokens: int
    generation_output_tokens: int
    generation_thought_tokens: int
    decision_cost_usd: float
    generation_cost_usd: float
    actual_total_cost_usd: float
    always_argon_cost_usd: float
    savings_usd: float
    savings_pct: float

    def to_dict(self) -> dict[str, float | int | str]:
        return asdict(self)


def calculate_turn_cost(
    *,
    routed_model: str,
    decision_input_tokens: int,
    generation_input_tokens: int,
    generation_output_tokens: int,
    generation_thought_tokens: int = 0,
) -> TurnCostBreakdown:
    """Compute actual turn cost (Decision 1 + routed model) vs. Always-Argon."""
    if routed_model not in MODEL_RATES_PER_1M_USD:
        raise ValueError(f"Unsupported routed_model: {routed_model}")

    decision_rate = float(MODEL_RATES_PER_1M_USD[DECISION_MODEL_ID]["input_per_1m"])
    routed_in_rate = float(MODEL_RATES_PER_1M_USD[routed_model]["input_per_1m"])
    routed_out_rate = float(MODEL_RATES_PER_1M_USD[routed_model]["output_per_1m"])
    argon_in_rate = float(MODEL_RATES_PER_1M_USD[ARGON_MODEL_ID]["input_per_1m"])
    argon_out_rate = float(MODEL_RATES_PER_1M_USD[ARGON_MODEL_ID]["output_per_1m"])

    billable_output_tokens = generation_output_tokens + generation_thought_tokens

    decision_cost = (decision_input_tokens / 1_000_000.0) * decision_rate
    generation_cost = (
        (generation_input_tokens / 1_000_000.0) * routed_in_rate
        + (billable_output_tokens / 1_000_000.0) * routed_out_rate
    )
    actual_total = decision_cost + generation_cost

    # Counterfactual: sending the same generation tokens directly to Gemini 4 Argon
    # without dynamic model routing.
    always_argon_cost = (
        (generation_input_tokens / 1_000_000.0) * argon_in_rate
        + (billable_output_tokens / 1_000_000.0) * argon_out_rate
    )

    savings_usd = always_argon_cost - actual_total
    savings_pct = (
        (savings_usd / always_argon_cost) * 100.0 if always_argon_cost > 0 else 0.0
    )

    return TurnCostBreakdown(
        routed_model=routed_model,
        decision_input_tokens=decision_input_tokens,
        decision_output_tokens=0,
        generation_input_tokens=generation_input_tokens,
        generation_output_tokens=generation_output_tokens,
        generation_thought_tokens=generation_thought_tokens,
        decision_cost_usd=round(decision_cost, 8),
        generation_cost_usd=round(generation_cost, 8),
        actual_total_cost_usd=round(actual_total, 8),
        always_argon_cost_usd=round(always_argon_cost, 8),
        savings_usd=round(savings_usd, 8),
        savings_pct=round(savings_pct, 2),
    )
