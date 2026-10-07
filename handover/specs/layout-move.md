<!-- Handover copy of the operator's working note `drafts/layout-move.md` (sanitized: private names, the private repository, the job platform and storage locations removed; `<operator-notes>/...` paths point to files that are NOT available — the content you need is in handover/). Treat it as the specification it was for its lane. -->

# Lane `layout-move` (owner decision 14:25; run in the quiet window after the P2.5 lanes merge, before RC0)

Target (conceptual tree agreed with the owner, 14:15):
```
rcp-ndcg-core/   pyproject.toml, src/rcp_ndcg_core/
rcp-ndcg/        pyproject.toml, src/rcp_ndcg/
rcp-ndcg-vllm/   pyproject.toml, src/rcp_ndcg_vllm/{recipe, serve, recipes/ (package data), models/<name>/}
rcp-ndcg-test/   pyproject.toml, src/rcp_ndcg_test/{cases, conformance, fakes, equivalence, record, jobs}, cases/
pyproject.toml   the uv workspace only (members, tooling config); docs/, experiments/, examples/, skills/, schemas/, .github/
```
1. Mechanical moves with `git mv` (history kept): the root package to `rcp-ndcg/`, `packages/rcp-ndcg-core` to
   `rcp-ndcg-core/`, `packages/rcp-ndcg-vllm` to `rcp-ndcg-vllm/`, `packages/rcp-ndcg-test` to `rcp-ndcg-test/`; every
   path in CI, release.yml, AGENTS.md, docs, mkdocs, tests (tests/docs path helpers), snapshots, constraints, MANIFEST.
2. **rcp-ndcg-vllm becomes the lean serving package**: dependencies only what the stock vLLM image ships (pydantic,
   pyyaml); `pip install --no-deps` must work and change `pip freeze` by exactly that wheel. Recipes are package data
   read through `importlib.resources` (the wheel ships them; a test installs the wheel in a fresh venv and lists them).
   New `rcp-ndcg-vllm serve <recipe-id> [--port ...]` builds the `vllm serve` argv (template path from package data,
   media flags, pooler config) and execs it; it refuses a missing plugin with the exact install line. The client block
   stays plain data here (validated by rcp-ndcg when it reads it).
3. **The model plugins fold into rcp-ndcg-vllm** (`rcp_ndcg_vllm/models/topk/`, `models/pplx/`) under one
   `vllm.general_plugins` entry point that registers each architecture lazily (`"module:Class"` strings; importing
   the package never imports torch or vllm); one version guard. The separate plugin distributions are deleted.
4. **The validation tooling moves to rcp-ndcg-test** (unpublished): the equivalence harness, the recorder, the RC build,
   the node bootstrap, the wave runner, wave 0, submit; rcp-ndcg-test depends on rcp-ndcg and rcp-ndcg-vllm. The harness
   still sends through the product's role clients (R30).
5. **rcp-ndcg reads recipes optionally** (form fixed in research/docs-firstcontact/work/OPERATOR-ANSWERS.md Q1: a role
   config naming `recipe: <id>` takes its client block from the recipe; RUNTIME fields stay; an explicit CONTENT field
   must equal the recipe's or is refused; CLI shorthand `--reranker recipe:<id>`; `serve: command: [rcp-ndcg-vllm, serve,
   <id>]` allowed, Q2): `recipe:<id>` as a retrieval/rerank config value (and the paper configs point
   at recipes instead of duplicating client fields) resolves through `rcp_ndcg_vllm` when installed (lazy import,
   typed error with the install hint otherwise); the layering charter and AGENTS.md updated (rcp-ndcg may import
   rcp_ndcg_vllm's recipe data lazily; rcp-ndcg-vllm never imports rcp-ndcg).
6. Release: three published dists stay (core, rcp-ndcg, rcp-ndcg-vllm) with the publish order core -> rcp-ndcg ->
   vllm; no plugin wheels to publish.
Gate: everything green, `run_all` unchanged, a fresh-venv install of each wheel, the `--no-deps` freeze check in a venv
that has only `vllm`'s declared deps of pydantic/pyyaml, `rcp-ndcg-vllm serve <id> --dry-run` for every recipe.

## Executor and verifiers for this lane (owner decision 14:30)
