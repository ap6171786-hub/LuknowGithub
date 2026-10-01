"""Shared-secret configuration used by the API and Streamlit frontend."""

from __future__ import annotations

import os

MINIMUM_API_KEY_LENGTH = 32


def configured_api_key() -> str | None:
    """Return a sufficiently strong configured key, or ``None`` if missing/weak."""
    key = os.getenv("INSIGHT_API_KEY", "").strip()
    if len(key) < MINIMUM_API_KEY_LENGTH:
        return None
    return key
