"""The provenance probe: what one replica says about itself, read best effort from ``GET {url}/models``.

The transport's :meth:`~rcp_ndcg.inference.transport.Transport.probe` runs :func:`read_replica` once per
replica; the role clients record the result in their stores and run manifests. Nothing engine-specific is
asked, and an endpoint that says nothing readable is recorded, never raised: what an engine reports about
itself is runtime information, never required.
"""

from __future__ import annotations

from collections.abc import Mapping

import httpx

from rcp_ndcg.inference.types import EngineInfo
from rcp_ndcg.support.logging import get_logger

logger = get_logger(__name__)


async def read_replica(
    client: httpx.AsyncClient,
    url: str,
    *,
    model: str,
    headers: Mapping[str, str] | None = None,
    timeout: httpx.Timeout,
    system_fingerprint: str | None = None,
) -> EngineInfo:
    """Read one replica's ``GET {url}/models`` into an :class:`~rcp_ndcg.inference.types.EngineInfo`, best effort.

    Args:
        client: The client to read with (the transport's own pool, bound to its event loop).
        url: The replica's base URL.
        model: The endpoint's served model name; the entry listing it is read, else the first listed one.
        headers: The request headers (the credentials and gateway headers); their values are never logged.
        timeout: The read timeout, the endpoint's.
        system_fingerprint: The ``system_fingerprint`` the replica's first answer reported, preserved onto the
            new record (the role clients report it per reply; see
            :meth:`~rcp_ndcg.inference.transport.Transport.note_system_fingerprint`).

    Returns:
        What the replica reported: the served model id, ``owned_by``, ``max_model_len`` when reported, the
        ``server`` and any version header. A replica that answers nothing readable is recorded with its
        ``error`` -- the probe never raises.
    """
    try:
        response = await client.get(f"{url}/models", headers=headers, timeout=timeout)
        response.raise_for_status()
        entries = [entry for entry in response.json().get("data") or [] if isinstance(entry, dict)]
    except Exception as exc:  # best effort: what the endpoint says is recorded, never required
        return EngineInfo(url=url, system_fingerprint=system_fingerprint, error=f"{type(exc).__name__}: {exc}")
    entry = next((candidate for candidate in entries if candidate.get("id") == model), None)
    if entry is None and entries:
        logger.warning(
            "%s serves %s, not the endpoint's model %r: start the server with --served-model-name %s, or set the "
            "config's model to the served name",
            url,
            [candidate.get("id") for candidate in entries],
            model,
            model,
        )
    entry = entry or (entries[0] if entries else {})
    length = entry.get("max_model_len")
    return EngineInfo(
        url=url,
        model=entry.get("id"),
        owned_by=entry.get("owned_by"),
        max_model_len=length if isinstance(length, int) else None,
        headers={
            name: value
            for name, value in response.headers.items()
            if name.lower() == "server" or "version" in name.lower()
        },
        system_fingerprint=system_fingerprint,
    )


__all__ = ["read_replica"]
