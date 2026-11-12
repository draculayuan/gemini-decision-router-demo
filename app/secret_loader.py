# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""In-memory credential loader for the Gemini Developer API key.

Resolves the API key strictly in memory (never written to disk or logs):
1. Environment variable `GEMINI_API_KEY` if explicitly set (local quickstart).
2. Google Cloud Secret Manager via Application Default Credentials (ADC), using
   `GOOGLE_CLOUD_PROJECT` (or the ambient ADC default project) and
   `GEMINI_API_KEY_SECRET_ID` (default: `gemini_api_key`).
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass

import google.auth
from dotenv import load_dotenv
from google.api_core import exceptions as gcp_exceptions
from google.auth import exceptions as auth_exceptions
from google.cloud import secretmanager

load_dotenv()

DEFAULT_SECRET_ID = "gemini_api_key"
DEFAULT_SECRET_VERSION = "latest"

_lock = threading.Lock()
_cached_api_key: str | None = None


class SecretConfigurationError(RuntimeError):
    """Raised when the Gemini Developer API key cannot be resolved."""


@dataclass(frozen=True)
class SecretStatus:
    """Redacted readiness status for health checks (contains no project or key data)."""

    configured: bool
    error: str | None = None


def _resolve_secret_resource_name() -> str:
    """Build the Secret Manager version resource name from env or ambient ADC."""
    secret_id = os.environ.get(
        "GEMINI_API_KEY_SECRET_ID", DEFAULT_SECRET_ID
    ).strip()
    version = (
        os.environ.get("GEMINI_API_KEY_SECRET_VERSION", DEFAULT_SECRET_VERSION).strip()
        or DEFAULT_SECRET_VERSION
    )

    if not secret_id:
        raise SecretConfigurationError("GEMINI_API_KEY_SECRET_ID is empty.")

    if secret_id.startswith("projects/"):
        if "/versions/" in secret_id:
            return secret_id
        return f"{secret_id.rstrip('/')}/versions/{version}"

    project_id = os.environ.get("GOOGLE_CLOUD_PROJECT", "").strip()
    if not project_id:
        try:
            _, default_project = google.auth.default()
            project_id = (default_project or "").strip()
        except auth_exceptions.DefaultCredentialsError as exc:
            raise SecretConfigurationError(
                "No GOOGLE_CLOUD_PROJECT set and Application Default Credentials "
                "are not configured. Run `gcloud auth application-default login` "
                "or set GEMINI_API_KEY."
            ) from exc

    if not project_id:
        raise SecretConfigurationError(
            "Could not determine GCP project ID. Set GOOGLE_CLOUD_PROJECT or "
            "configure a quota project via `gcloud auth application-default set-quota-project`."
        )

    return f"projects/{project_id}/secrets/{secret_id}/versions/{version}"


def clear_secret_cache() -> None:
    """Clear the in-memory cached API key so the next call re-reads credentials."""
    global _cached_api_key
    with _lock:
        _cached_api_key = None


def get_gemini_api_key(*, force_refresh: bool = False) -> str:
    """Return the Gemini Developer API key in memory."""
    global _cached_api_key
    with _lock:
        if _cached_api_key and not force_refresh:
            return _cached_api_key

        direct_env_key = os.environ.get("GEMINI_API_KEY", "").strip()
        if direct_env_key:
            _cached_api_key = direct_env_key
            return _cached_api_key

        resource_name = _resolve_secret_resource_name()
        try:
            client = secretmanager.SecretManagerServiceClient()
            response = client.access_secret_version(
                request={"name": resource_name},
                retry=None,
                timeout=4.0,
            )
            payload = response.payload.data.decode("utf-8").strip()
        except auth_exceptions.DefaultCredentialsError as exc:
            raise SecretConfigurationError(
                "Application Default Credentials (ADC) are not configured or have "
                "expired. Run `gcloud auth application-default login`."
            ) from exc
        except gcp_exceptions.NotFound as exc:
            raise SecretConfigurationError(
                "Configured Secret Manager secret was not found."
            ) from exc
        except gcp_exceptions.PermissionDenied as exc:
            raise SecretConfigurationError(
                "Permission denied accessing Secret Manager. Grant "
                "`roles/secretmanager.secretAccessor` to the runtime identity."
            ) from exc
        except Exception as exc:
            raise SecretConfigurationError(
                f"Failed to access Secret Manager: {exc}"
            ) from exc

        if not payload:
            raise SecretConfigurationError(
                "Secret Manager secret version payload is empty."
            )

        _cached_api_key = payload
        return _cached_api_key


def inspect_secret_status(*, force_refresh: bool = False) -> SecretStatus:
    """Check whether credentials can be resolved without exposing identifiers."""
    try:
        get_gemini_api_key(force_refresh=force_refresh)
        return SecretStatus(configured=True)
    except SecretConfigurationError as exc:
        return SecretStatus(configured=False, error=str(exc))
