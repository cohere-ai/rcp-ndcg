"""An HTTP embedding endpoint as an encoder (``provider: openai_compatible | cohere | voyage | gemini``). No GPU.

The only path that runs on a laptop, and for Cohere Embed v4 a text-to-image retrieval path that needs no local
weights at all. Which vendors can see pixels is declared in :data:`~rcp_ndcg.retrieval.api_dense.VENDORS`, not
discovered: sending a page corpus to a text-only endpoint would either error 40k times or, worse, embed 40k empty
strings.
"""

from __future__ import annotations

from collections.abc import Sequence

from rcp_ndcg_core.content import Content

from rcp_ndcg.retrieval.encoder import Embeddings, Encoder, EncodeRole


class HostedApiEncoder(Encoder):
    """L2-normalised embeddings from a vendor API.

    Args:
        vendor: One of :data:`~rcp_ndcg.retrieval.api_dense.VENDORS`; ``openai`` is any OpenAI-compatible
            ``/embeddings`` route, which needs a key only when one is given.
        model: The vendor's model id.
        base_url: Endpoint override, for a proxy, a gateway or a served model.
        api_key: The key; the vendor's environment variables when ``None``.
        batch_size: Texts per request, capped at the vendor's limit.
        timeout_s, connect_timeout_s, max_retries: The endpoint's request policy.
        query_prefix / document_prefix: Text prepended to queries / documents (a served model's recipe).
    """

    def __init__(
        self,
        *,
        vendor: str,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        batch_size: int | None = None,
        timeout_s: float = 60.0,
        connect_timeout_s: float = 5.0,
        max_retries: int = 5,
        query_prefix: str = "",
        document_prefix: str = "",
    ) -> None:
        from rcp_ndcg.retrieval.api_dense import EmbeddingAPIClient

        self.client = EmbeddingAPIClient(
            vendor,
            model=model,
            base_url=base_url,
            api_key=api_key,
            batch_size=batch_size,
            key_required=vendor != "openai",
            timeout_s=timeout_s,
            connect_timeout_s=connect_timeout_s,
            max_retries=max_retries,
        )
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix
        # Per-vendor rather than per-class: one client, two capabilities.
        self.supports_media = self.client.spec.supports_media

    def encode(
        self,
        contents: Sequence[Content],
        *,
        role: EncodeRole,
        batch_size: int | None = None,
    ) -> Embeddings:
        self.check_media(contents)
        if not contents:
            return Embeddings.empty(0)
        input_type = "query" if role is EncodeRole.QUERY else "document"
        prefix = self.query_prefix if role is EncodeRole.QUERY else self.document_prefix
        if prefix:
            contents = [content.with_text_prefix(prefix) for content in contents]
        matrix = self.client.embed_contents(contents, input_type, progress=len(contents) > self.client.batch_size)
        return Embeddings.single(matrix)


__all__ = ["HostedApiEncoder"]
