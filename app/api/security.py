"""API-key authentication for protected HTTP routes."""

from __future__ import annotations

import secrets
from typing import Annotated

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

from app.utils.auth import configured_api_key

API_KEY_HEADER = APIKeyHeader(
    name="X-API-Key",
    scheme_name="ApiKeyAuth",
    auto_error=False,
)


async def require_api_key(
    api_key: Annotated[str | None, Security(API_KEY_HEADER)],
) -> None:
    """Require the configured shared API key on protected endpoints."""
    expected = configured_api_key()
    if expected is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="API authentication is not configured. Set a strong INSIGHT_API_KEY.",
        )
    if api_key is None or not secrets.compare_digest(api_key, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="A valid X-API-Key header is required.",
            headers={"WWW-Authenticate": "ApiKey"},
        )
