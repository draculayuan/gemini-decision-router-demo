# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""ADK Root Agent wired with Gemini Decision 1 `before_model_callback` routing."""

from __future__ import annotations

from collections.abc import AsyncGenerator

from google.adk.agents import Agent
from google.adk.apps import App
from google.adk.models import Gemini
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types

from app.pricing import FLASH_MODEL_ID
from app.router_callback import route_with_gemini_decision_1
from app.secret_loader import get_gemini_api_key


class SecretManagerGemini(Gemini):
    """Gemini BaseLlm loading its API key lazily in-memory from Secret Manager.

    Explicitly sets `vertexai=False` and `api_version='v1beta'` via public
    `client_kwargs` right before model invocation so `GOOGLE_CLOUD_PROJECT` in
    `.env` (used for Cloud Secret Manager) never forces Vertex AI routing.
    """

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        api_key = get_gemini_api_key()
        merged_kwargs = dict(self.client_kwargs or {})
        if merged_kwargs.get("api_key") != api_key or merged_kwargs.get("vertexai") is not False:
            merged_kwargs["api_key"] = api_key
            merged_kwargs["vertexai"] = False
            self.client_kwargs = merged_kwargs
            # Clear cached per-loop client if secret rotated
            self.__dict__.pop("api_client", None)

        async for response in super().generate_content_async(
            llm_request, stream=stream
        ):
            yield response


SYSTEM_INSTRUCTION = """You are a crisp, insightful AI assistant in a live demonstration of Dynamic Model Routing powered by Gemini Decision 1.

Guidelines:
- Give clear, well-structured Markdown answers tailored to the complexity of the user's prompt.
- For quick lookups, translations, or receipt/image field extractions, keep your response tight and immediate.
- For deep engineering, mathematical, architectural, or debugging questions, show clear, rigorous reasoning and concrete solutions.
"""

root_agent = Agent(
    name="dynamic_router_assistant",
    model=SecretManagerGemini(
        model=FLASH_MODEL_ID,
        api_version="v1beta",
        retry_options=types.HttpRetryOptions(attempts=3),
    ),
    instruction=SYSTEM_INSTRUCTION,
    description=(
        "General-purpose multimodal assistant dynamically routed across "
        "Gemini 3.5 Flash-Lite, Gemini 3.8 Flash, and Gemini 4 Argon by "
        "Gemini Decision 1."
    ),
    before_model_callback=route_with_gemini_decision_1,
)

app = App(
    root_agent=root_agent,
    name="app",
)
