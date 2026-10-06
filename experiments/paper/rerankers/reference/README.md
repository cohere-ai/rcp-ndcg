# Paper-exact reference implementations (not part of the package)

This directory holds the in-process rerankers and the HF dense encoder that the package shipped
until the unified-inference change removed every in-process model from it. They are kept **unchanged in behaviour** so
the recipe lanes can derive each model's `reference.py` from them and the equivalence check
(the unified-inference design's equivalence check) has the paper's exact code to compare a served recipe against.

**Not part of the package. Nothing in `src/rcp_ndcg/` imports this directory, and nothing here
imports `rcp_ndcg`** (at most `rcp_ndcg_core` content types, and today not even those). The modules
import torch and transformers; install their own environment with the pinned `requirements.txt`
(torch 2.9.1, transformers 4.57.6, accelerate, flash-attn — the versions the paper's `[local]`
extra pinned), outside the package's lock.

| Module | Model family | Was |
| --- | --- | --- |
| `qwen3.py` | Qwen3-Reranker 0.6B / 4B / 8B (yes/no token log-softmax) | `src/rcp_ndcg/retrieval/external_rerankers.py` (`QwenOGRerank`) |
| `zerank.py` | ZeroEntropy ZeRank 1 / 1-small / 2 (chat template + "Yes"-logit/5) | `external_rerankers.py` (`ZerankRerank`) |
| `contextual.py` | ContextualAI ctxl-rerank 1B / 2B / 6B (vocab-position-0 logit) | `external_rerankers.py` (`ContextualRerank`) |
| `jina.py` | Jina reranker v3 (the checkpoint's own `model.rerank()`) | `external_rerankers.py` (`JinaRerank`) |
| `octen.py` | Octen-Embedding-8B (HF `AutoModel`, last-token pooling, L2, `"- "` document prefix) | `src/rcp_ndcg/retrieval/hf_dense.py` + `encoders/torch_dense.py` |

Behaviour is the paper's: the same prompts, padding sides, truncation rules, dtypes and OOM
backoff as the release code (see each module's docstring for its provenance). The only changes from
the deleted modules are the removals that made them importable on their own: the `rcp_ndcg` logging
helper, the `AccelState` multi-GPU sharding (the paper's runs were single-GPU per model; a rerun
that needs several GPUs shards the queries itself) and the `DependencyError` plumbing are gone, and
each module is loaded with the constructor arguments the old
:func:`load_external_model` factory passed (the budgets the paper code set).

The hosted rerank APIs (Cohere, Voyage) are *not* here: they move onto the package's shared
transport as wire profiles (`api: cohere | voyage` on a `RerankEndpoint`), and their wire behaviour
is covered by the package's own adapter tests.
