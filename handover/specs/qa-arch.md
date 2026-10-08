<!-- Handover copy of the operator's working note `drafts/qa-arch.md` (sanitized: private names, the private repository, the job platform and storage locations removed; `<operator-notes>/...` paths point to files that are NOT available — the content you need is in handover/). Treat it as the specification it was for its lane. -->

# QA review `qa-arch`: architecture, layering, seams and public surface

**Crux.** After a large refactor done by many parallel lanes, does the package read as one design? Find every place
where the structure contradicts RFC-0001 or AGENTS.md, or where two lanes built the same thing differently.

## Watch for (each item: check it, report what you found, with evidence)
1. **Layering**: run `tests/test_layering.py`'s scanner logic yourself over the tree and also look for lazy imports that
   hide a cycle (imports inside functions that point outward); list every one with its reason, and judge whether it is
   a sanctioned lazy import (schemas, mcp, the CLI table) or a disguised violation.
2. **One home per concept** (AGENTS.md table + the new `rcp_ndcg.inference` row): for each concept list where it lives;
   flag any concept with two homes (e.g. tokenizer loading, token counting, retries/backoff, HTTP status mapping, auth
   header construction, error classification, media preparation, text cutting, chunking, score pooling, identity
   payloads, engine probing, fake servers).
3. **The transport seam**: do all four roles (judge, embed, rerank, multi-vector) go through `Transport` the same way
   (construction, send, interpret, usage, close, sync bridge)? Any role client that builds its own `httpx` client,
   retry loop, auth header or status map is a finding. Is the adapter `Protocol` actually what the adapters implement
   (signatures, ClassVars), and is the entry-point group `rcp_ndcg.adapters` exercised by a test?
4. **Config shapes across roles**: compare `JudgeConfig`, `EmbeddingEndpoint`, `PoolingEndpoint`, `RerankEndpoint`
   and the retrieval configs field by field: same concept, same name, same type, same default, same identity role?
   Produce the table.
5. **Removal completeness**: no in-process model code left under `rcp-ndcg/src/rcp_ndcg/` (torch, transformers, accelerate,
   vllm, flash-attn imports; `[local]`/`[vllm]` extras; `LocalEncoder`; `accel.py`; hidden constants like
   `MAX_SEQ_LENGTH`); every reference to a removed name in code, configs, docs, examples, skills and experiments is
   gone.
6. **Serving by role**: the phase plan, the runners and the CLI agree on one shape (`ServeByRole`, `JobPhase`,
   `RCP_NDCG_ENGINES`, `--engine`); no leftover of `RCP_NDCG_JUDGE_URLS`, `--judge-urls` or `JobSpec.serve`.
7. **Public surface**: every new public module has `__all__`; every public name is documented on a docs page and in
   the CHANGELOG; nothing public that should be private (and vice versa); the contract snapshots match.
8. **Recipe package boundary**: `rcp-ndcg-vllm` imports nothing from `rcp_ndcg` at runtime and vice versa;
   the recipe `client` fields map one to one onto the endpoint configs (table), with no field that one side has and
   the other ignores.
Deliver also: a module map (one line per module: purpose, layer, public or internal), and the field table of item 4.
