"""The one retrying JSON POST and the one API-key lookup of the hosted retrieval clients (rerank and embedding APIs,
a vLLM pooling server)."""

from __future__ import annotations

import os
import time
from collections.abc import Sequence
from typing import Any

from rcp_ndcg.errors import CredentialsError, ProviderError
from rcp_ndcg.support.logging import get_logger

logger = get_logger(__name__)

RETRY_STATUS = frozenset({429, 500, 502, 503, 504})
"""Status codes worth another attempt: a rate limit, a busy queue or a restarting server resolve on their own."""


def api_key(explicit: str | None, env_names: Sequence[str], *, what: str) -> str:
    """``explicit``, else the first of ``env_names`` that is set.

    Raises:
        CredentialsError: Neither gives a key.
    """
    key = explicit or next((os.environ[name] for name in env_names if os.environ.get(name)), None)
    if not key:
        raise CredentialsError(f"{what} needs an API key", hint=f"export {' or '.join(env_names)}=...")
    return key


def bearer(key: str | None) -> dict[str, str]:
    """The ``Authorization`` header of ``key``; none without a key (a local server)."""
    return {"Authorization": f"Bearer {key}"} if key else {}


def post_json(
    url: str,
    payload: dict[str, Any],
    *,
    headers: dict[str, str],
    what: str,
    timeout_s: float = 120.0,
    connect_timeout_s: float = 5.0,
    max_retries: int = 8,
) -> Any:
    """POST ``payload`` as JSON and return the decoded response body.

    Retries transport errors and :data:`RETRY_STATUS` responses with exponential backoff (1 s doubling, capped at
    60 s, or the server's ``Retry-After``), up to ``max_retries`` times; each attempt waits ``timeout_s`` seconds
    for the response and ``connect_timeout_s`` for the connection.

    Raises:
        ProviderError: The request failed after the retries, or with a status that is not worth retrying.
    """
    import httpx

    headers = {**headers, "Content-Type": "application/json"}
    timeout = httpx.Timeout(timeout_s, connect=connect_timeout_s)
    for attempt in range(max_retries + 1):
        delay = min(2.0**attempt, 60.0)
        try:
            response = httpx.post(url, json=payload, headers=headers, timeout=timeout)
        except httpx.HTTPError as exc:
            if attempt == max_retries:
                raise ProviderError(f"{what}: {url} failed after {attempt + 1} attempts: {exc}") from exc
            logger.warning(f"{what}: {exc}; retrying in {delay:.0f}s")
            time.sleep(delay)
            continue
        if response.status_code == 200:
            return response.json()
        if response.status_code in RETRY_STATUS and attempt < max_retries:
            delay = _retry_after(response.headers) or delay
            logger.warning(f"{what}: HTTP {response.status_code}; retrying in {delay:.0f}s")
            time.sleep(delay)
            continue
        raise ProviderError(
            f"{what}: {url} failed with HTTP {response.status_code}: {response.text[:500]}",
            retryable=response.status_code in RETRY_STATUS,
        )
    raise AssertionError("unreachable")


def _retry_after(headers: Any) -> float | None:
    raw = headers.get("retry-after") if headers else None
    try:
        return float(raw) if raw else None
    except (TypeError, ValueError):
        return None


__all__ = ["RETRY_STATUS", "api_key", "post_json"]
