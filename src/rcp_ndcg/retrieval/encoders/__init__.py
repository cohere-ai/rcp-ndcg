"""Encoder backends: where an embedding model runs, one class per kind of encoder config.

* :class:`TorchDenseEncoder` -- ``provider: local`` with ``engine: hf``: in-process HuggingFace weights (the
  ``[local]`` extra).
* :class:`VllmEncoder` -- ``provider: local`` with ``engine: vllm`` (an in-process engine), or
  ``provider: openai_compatible`` with ``pooling: token`` (a running server's ``/pooling``).
* :class:`HostedApiEncoder` -- ``provider: openai_compatible``, ``cohere``, ``voyage`` or ``gemini``: an HTTP
  embedding endpoint; no GPU.

Importing this package is cheap: each module defers ``torch`` / ``vllm`` to construction time.
"""

from rcp_ndcg.retrieval.encoders.hosted_api import HostedApiEncoder
from rcp_ndcg.retrieval.encoders.torch_dense import TorchDenseEncoder
from rcp_ndcg.retrieval.encoders.vllm_encoder import VllmEncoder

__all__ = ["HostedApiEncoder", "TorchDenseEncoder", "VllmEncoder"]
