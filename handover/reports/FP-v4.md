# Report FP-v4: the behaviour fingerprint keys everything that shapes the engine's output (freeze-risks R1, R6)

**Status:** DONE. The lane merged `rfc-0001` twice: `446765e5` (l10d + export-seam) and `01f4b9be`
(qa-prep + sz-pplx + media-rules + l10c); the gated head is `8a7adbf4` plus the report commit.

## Commits

- `2e3efbba` The recipe declares the plugin code its engine runs: `serve.plugin_architectures` and
  `serve.patches`, one home for the architecture->module mapping, and `serve` renders the declared patches
  into the engine's `RCP_NDCG_VLLM_PATCHES` (R1)
- `ec518fba` Request packing and the media caps are CONTENT in the endpoint identities: the identity, the
  fingerprint and the cached index check agree (R6)
- `a30ecac7` `rcp-fp/4`: the behaviour fingerprint keys the engine image and version and the plugin code the
  recipe's engine runs, per declared architecture and opted-in patch (R1, R3)
- `cdad8e08` The corpora and goldens follow `rcp-fp/4`: five metadata-only re-keys (each manifest naming the
  move), the seven stale declarations name the new inputs, and the per-variant goldens are regenerated
- `11d186d6` Docs and CHANGELOG: `rcp-fp/4`, the plugin code declaration, the patch opt-in, and the identity
  roles of request packing and the media caps
- `bdd3a54f` Merge branch `rfc-0001` (`446765e5`) into `lane/fp-v4`
- `b3e1d874` Round-1 verifier findings: the wave and e2e engine starts render the recipe's patches (one
  `patches_env_value` helper), the corpus provenance records the opt-in, the plugin package init is keyed,
  the MRL range/projection and the media caps gain fingerprint tests, and `plugin_distribution_name` is one
  home
- `b054224d` Round-2 verifier minors: the config registrations every plugin engine imports are keyed, the
  patch module stays opt-in-keyed, the e2e call sites pin the `patches` argument, and the stale runtime/env
  wording is fixed
- `b471eb41` Merge `rfc-0001` (`01f4b9be`: qa-prep, the pplx-embed-v1 family, pplx-embed-v2-late-9b,
  media-rules) into `lane/fp-v4`: the new plugin recipes declare their registrations (`PplxV1Config`), the
  config registrations every plugin engine imports are keyed, and the per-variant goldens are regenerated on
  the merged tree
- `8a7adbf4` The network-gated family contract tests pin the two new serve fields
  (`plugin_architectures` per recipe, `patches` empty), the way the gate's recipes step runs them

## What changed

Per brief item:

1. **R1 — the plugin code is in the key.** `fingerprint_inputs` now emits
   `plugin_sha256.<module> = sha256:<hex>` for exactly the engine-side modules the recipe's declared plugin
   registrations and patches run: the shared modules every plugin engine imports at registration
   (`rcp_ndcg_vllm.models.PLUGIN_ENGINE_MODULES`) plus each declared registration's modules
   (`rcp_ndcg_vllm.models.ARCHITECTURE_MODULES`) plus each opted-in patch's module
   (`rcp_ndcg_vllm.patches.PATCH_MODULES`). The recipe declares what it runs: `serve.plugin_architectures`
   (required exactly when `serve.plugin` is set; validated against the plugin's own mapping) and
   `serve.patches` (validated against `PATCH_NAMES`). A foreign plugin spec is refused by name at
   fingerprint time. The plugin package owns the one mapping; a module edit moves exactly the recipes whose
   declared registration implements it (a shared module moves every plugin recipe, a patch module only the
   opting-in recipe) — pinned by mutation tests. `stale.json` names the moved module: a plugin-code move
   surfaces as `plugin_sha256.<module>` like any other input.
2. **R6 — one role per field.** `batch_size`, `max_images` and `max_videos` are CONTENT in
   `EmbeddingEndpoint`/`PoolingEndpoint`/`RerankEndpoint` (they change the request bytes, and a bf16 batch's
   composition can move the numbers) and stay `request` in `CLIENT_FIELDS`; the identity, the fingerprint
   and the cached index check agree. A cached index built with one `max_images` is rebuilt when the next run
   declares another (test); the rerank step re-keys on `batch_size` (test).
3. **The engine image and its processing.** `engine.image` and `engine.min_version` are fingerprint inputs
   of their own (`rcp-fp/4`), so a vLLM/transformers change that moves the engine's processing moves the
   key; tests name each input.
4. **Post-processing fields.** `normalize`, `aggregation`, `dim`, the MRL fields (`mrl_kind`, `mrl_dims`,
   `mrl_range`, `mrl_projection`, `mrl_dim`), `document_skip_token_ids` and `outputs` stay out of the replay
   fingerprint (they act on the reply) and are CONTENT in the endpoint identity — the step/stored-reference
   key the stage-2 comparison uses. The cross-check test now asserts the identity payload moves while the
   fingerprint does not, for every one of those fields (the MRL range and projection were added); a
   `CLIENT_FIELDS` ↔ `IDENTITY_ROLES` cross-check pins request→CONTENT, post-processing→CONTENT,
   transport→RUNTIME and the tokenizer's content hash.
5. **The schema bump, the corpora and the goldens.** `FINGERPRINT_SCHEMA` is `rcp-fp/4` (an input of its
   own, so old and new corpora never collide). The five corpora whose only moved inputs were the new
   metadata (engine image/version, `serve.plugin_architectures`, `serve.patches`, the schema) were re-keyed
   with the documented procedure (each manifest carries a `rekeyed` entry with the old fingerprint, both old
   hashes and the exact changed inputs; the records are byte-identical); the seven already-stale corpora
   keep their recordings and their `stale.json` declarations now name the new inputs too. The per-variant
   goldens were regenerated with `--update-goldens` (the documented way; the pre-family deltas were baked in
   as the writer's documented act, `DELTAS.json` is empty and the shrink-only guard still holds).

The verifier rounds then fixed: the wave runner and the e2e driver now render `serve.patches` into the
engine's environment (one `patches_env_value` helper), the corpus provenance records
`RCP_NDCG_VLLM_PATCHES`, the config registrations every plugin engine imports are keyed, the patch module
stays opt-in-keyed, the e2e call sites pin the `patches` argument, and the pip-spec → distribution-name rule
has one home (`plugin_distribution_name`).

## Verification

**Round 1 (two fresh verifiers, DeepSeek-V4.1-flash `xhigh`, lens A correctness / lens B regressions and
hygiene).** Both ran the suites and their own reproductions; both returned **VERDICT: FAIL**.

- Lens A findings: (1) major — the recorder paths (`run_wave._start`, `e2e._serve_config`) never rendered
  `serve.patches`, so the first opting-in recipe would be recorded on an unpatched engine under a key naming
  the patch module; (2) minor — the provenance docstring claimed the opt-in was recorded but `engine_facts`
  filtered env to `VLLM_*`; (3) minor — the post-processing exclusion test never exercised `mrl_range` /
  `mrl_projection`; (4) minor — no test asserted `max_images`/`max_videos` move the fingerprint. Fixed in
  `b3e1d874`, each with a red-first mutation reproduction.
- Lens B findings: (1) the same major (fixed as above); (2) minor — `rcp_ndcg_vllm.models.topk` ran in every
  plugin engine but was not keyed; (3) minor — the provenance claim; (4) minor — stale "runtime" comments
  in `recipe.py` and `experiments/paper/retrieval/octen.yaml`; (5) minor — the pip-spec parsing rule had
  three homes; (6) note — the judge's `max_images`/`max_videos` stay RUNTIME (out of the brief's endpoint
  scope; Open questions). All fixed in `b3e1d874` / `b054224d`.

**Round 2 (one fresh confirmation verifier, both lenses).** Re-verified every round-1 fix by mutation in a
separate scratch worktree; the core claims (the `rcp-fp/4` inputs, the five metadata-only re-keys with
byte-identical records, the exact stale declarations, the CONTENT roles) held. **VERDICT: FAIL** on four
minors, all fixed in `b054224d`: the CHANGELOG's Unreleased section still called the media caps RUNTIME;
the provenance module docstring and the corpus schema still described the recorded env as `VLLM_*` only;
the config registrations every plugin engine imports were still unkeyed; the e2e call sites' `patches`
pass-through was unpinned (now a required keyword). No blocker or major remained.

**Integration after the `rfc-0001` merges.** The `01f4b9be` merge brought new plugin recipes
(`pplx-embed-v1-0.6b`/`-4b`, `pplx-embed-v2-late-9b`) whose engine uses the plugin's config-only
registration; they declare `plugin_architectures: [PplxV1Config]` (one home, added to
`ARCHITECTURE_MODULES`), the goldens were regenerated on the merged tree, and the network-gated family
contract tests now pin the two new serve fields (`8a7adbf4`). The gate's first `recipes` run exposed those
70 unpinned contract tests; the re-run's `recipes` step passed with no failure outside the (empty) baseline.

## Checks

Last runs on the gated head `8a7adbf4` (`bin/gate lane/fp-v4`, retry after one
transient slot crash; `GATE: PASS`):

- `ruff-check` exit=0 `All checks passed!`; `ruff-format` exit=0 `579 files already formatted`;
  `basedpyright` exit=0 `0 errors, 0 warnings, 0 notes`.
- `pytest` (full root suite, `-n 4`) exit=0 `3628 passed, 102 skipped`.
- `contract-docs` exit=0 `304 passed, 55 skipped`; `mkdocs` exit=0.
- `test-pkg` exit=0 `647 passed, 396 skipped`; `recipes` (network-gated, `RCP_NDCG_NETWORK_TESTS=1`) exit=0
  `no failure outside the baseline`.
- `vllm-pkg` exit=0 `49 passed`; `vllm-models` exit=0 `72 passed, 7 skipped`.
- `run_all` exit=0 `1022 checks, 987 match, 35 known deviations, 0 failed`; `human study` `67/67`;
  `external LLM judges` `82/82`; `public-names: clean`; `clean`.
- One earlier gate attempt on the same head failed the `test-pkg` step with exit 139 (a segfault in
  `test_wave_corpus.py::test_changed_since_skips_the_unchanged_and_records_the_changed`); the test passes
  locally in isolation and the retry passed the whole step, so the failure was slot-transient.

## Open questions

- **The judge's media caps.** `JudgeConfig.max_images`/`max_videos` stay RUNTIME
  (`rcp_ndcg/judging/client.py`); the brief's R6 names the three retrieval role configs, and the judge's
  judgement family is keyed by the prompt/decoding/preprocessing payload. If the owner wants one role per
  field across every role that sends media to a model, the judge is the remaining decision.
- **Foreign plugins.** A `serve.plugin` that is not the shipped wheel is refused by the fingerprint (its
  modules cannot be resolved), and the recipe schema cannot declare its architectures. A third-party plugin
  needs its own module declaration/hash seam before it can be recorded.
- **The local-source/wheel gap.** The fingerprint hashes the harness's local plugin source while the engine
  runs the staged wheel; the corpus manifest records the wheel's SHA-256 (`model.plugin.wheel_sha256`) but
  nothing cross-checks the two. The wave's staged wheel is built from the same checkout, so the gap is a
  workflow trust assumption, not a demonstrated mismatch.
- **The seven stale corpora** await re-recording at E2; the three plugin families (and the new variants)
  have no committed corpora yet and must be recorded under `rcp-fp/4`.
- **The versioning page.** No `docs/reference/versioning.md` exists yet; the `rcp-fp/4` scheme and its bump
  rule are documented in `docs/how-to/use-verified-fake-engines.md` §"Behaviour fingerprint versions". The
  docs-final lane's compatibility/versioning page must state `rcp-fp/4` (and `handover/06-docs-final.md`'s
  `rcp-fp/3` reference is now stale).

## CHANGELOG entry

Added under `## Unreleased`.

`### Public surface`:

```markdown
- **The recipe schema declares the plugin code and the engine patches** (freeze-risk R1): `serve` gains
  `plugin_architectures` (the plugin's architectures this recipe's engine registers; required exactly when
  `serve.plugin` is set) and `patches` (the engine patch names this recipe opts into, validated against
  `rcp_ndcg_vllm.patches.PATCH_NAMES`); every engine-start path renders the declared patches into the
  engine's `RCP_NDCG_VLLM_PATCHES` (the `rcp-ndcg-vllm serve` console, the wave runner and the e2e driver,
  overriding an inherited value; the console logs both values), and the corpus provenance records the value
  the engine ran with, so the process runs exactly what the recipe declares. The plugin package declares its
  code one home per concept: `rcp_ndcg_vllm.models.ARCHITECTURE_MODULES` (architecture -> modules),
  `rcp_ndcg_vllm.models.PLUGIN_ENGINE_MODULES`, `rcp_ndcg_vllm.patches.PATCH_MODULES` (name -> module) and
  `rcp_ndcg_vllm.patches.patches_env_value`; `rcp_ndcg_vllm.recipe` gains `plugin_distribution_name` (the
  pip-spec -> distribution-name rule the loader, the console and the fingerprint share).
  `schema/recipe.schema.json` and `schema/family.schema.json` are regenerated.
- **The behaviour fingerprint is `rcp-fp/4`** (freeze-risks R1/R3): `fingerprint_inputs` now keys
  `engine.image` and `engine.min_version` (the engine's processing is versioned by them) and
  `plugin_sha256.<module>` for exactly the engine-side modules the recipe's declared plugin architectures
  and patches run (the shared entry modules, each architecture's modules, each opted-in patch's module), so a
  plugin fix moves the key instead of passing a stale corpus. Every other input is unchanged; old corpora
  never collide with the new rule.
```

`### Fixed`:

```markdown
- **A plugin-code fix moves the behaviour fingerprint** (freeze-risk R1): the plugin was keyed by the bare
  spec `rcp-ndcg-vllm`, so editing a head, quantiser or weight mapping kept every recorded corpus "current".
  The fingerprint now hashes the source of exactly the modules a recipe's engine runs (its declared
  architectures' modules plus its opted-in patches', the shared entry modules included), named
  `plugin_sha256.<module>`, so a staleness failure names the module that moved and `stale.json` declares it
  like any other input. A foreign plugin spec is refused by name (its code cannot be resolved here).
- **The engine paths that record a corpus render the recipe's patches**: the wave runner and the e2e driver
  started `vllm serve` with an inherited environment, so the first recipe opting into `pooling-full-context`
  would have been recorded on an unpatched engine while its fingerprint named the patch module. Both now set
  `RCP_NDCG_VLLM_PATCHES` from `serve.patches` through the one `patches_env_value` helper (empty when the
  recipe opts into none), the same rendering the serve console uses, and the corpus provenance records the
  value the engine ran with.
- **The fingerprint and the run identity agree about request-shaping fields** (freeze-risk R6):
  `batch_size`, `max_images` and `max_videos` were `RUNTIME` in the endpoint identities but request inputs in
  the fingerprint, so a cached index or rerank step could be reused across settings that move the vectors.
  All three are CONTENT now: the identity, the fingerprint and the cached index check agree, and a cached
  index built with one media cap is rebuilt when the next run declares another.
```

`### Changed`:

```markdown
- **The committed corpora are re-keyed to `rcp-fp/4`** (metadata-only): the five corpora whose only moved
  inputs are the new rule's engine image/version and serve plugin/patches metadata
  (`octen-embedding-8b`, `qwen3-embedding-0.6b`, `qwen3-reranker-8b`, `qwen3-vl-reranker-2b`,
  `zembed-1-embedding`) carry the current fingerprint, each manifest naming the move in `recipe.rekeyed`
  (the recording ran on exactly the engine image the key now names; no recorded exchange moved). The seven
  corpora already declared stale keep their recordings and their declarations gain the new metadata inputs;
  the per-variant goldens are regenerated at the new rule.
```

The Unreleased section's earlier `max_images`/`max_videos` sentence was corrected from RUNTIME to CONTENT,
and its fingerprint bullet from `rcp-fp/3` to `rcp-fp/4`.

## Public surface changes

- **Recipe schema** (`rcp-ndcg-vllm/schema/recipe.schema.json`, `family.schema.json`, regenerated):
  `ServeConfig` gains `plugin_architectures` (tuple of strings, default `()`) and `patches` (tuple of
  strings, default `()`), with the load-time refusals (a plugin without its architectures, an architecture
  the shipped mapping does not register, patches without the plugin that applies them, an unknown patch
  name). The three shipped plugin recipes and the merged pplx-embed-v1 family declare them.
- **`rcp_ndcg_vllm.recipe`** gains `plugin_distribution_name` (also re-exported from the package root);
  `tests/contract/snapshots/python_api.json` regenerated.
- **`rcp_ndcg_test.fingerprint`**: `FINGERPRINT_SCHEMA = "rcp-fp/4"`; new inputs `engine.image`,
  `engine.min_version` and `plugin_sha256.<module>`.
- **Endpoint identities** (`rcp_ndcg.inference.config`): `batch_size`, `max_images`, `max_videos` are
  CONTENT; `schemas/index.v1.json` and `schemas/run-config.v1.json` regenerated (descriptions only).
- No CLI command, flag, exit code or MCP tool changed. The corpus/golden artifacts changed as described
  above; no artifact schema version changed.

## Files outside scope

None: every changed file is the fingerprint, the endpoint config, the recipe/plugin schema and its mapping,
the harness's engine-start/provenance paths, their tests, the corpora/goldens/schemas/snapshots, the docs
and the CHANGELOG the brief names. The `experiments/paper/retrieval/octen.yaml` comment and the
`rcp-ndcg-vllm/README.md` sentence were corrected because the lane's role decision made them false.

## Docs updated

- `docs/how-to/use-verified-fake-engines.md` — the `rcp-fp/4` input list, a "Behaviour fingerprint versions"
  section (when the rule bumps), the plugin-code move in the staleness procedure, and the provenance's
  patch opt-in.
- `docs/how-to/add-a-model.md` — the family example and prose gain `plugin_architectures`/`patches` and the
  fingerprint's plugin-code key.
- `docs/reference/recipes.md` — the public-name list gains `plugin_distribution_name`; the plugin paragraph
  and the engine-side patches section say every engine-start path renders the declared patches.
- `docs/api/inference.md`, `docs/concepts/embeddings.md`, `docs/concepts/retrieval.md` — the identity
  paragraphs now list `batch_size`/`max_images`/`max_videos` as CONTENT.
- `rcp-ndcg-vllm/README.md` — the patch opt-in is per recipe (`serve.patches`), and the plugin-code key.
- `rcp-ndcg-test/schema/observation-corpus.md` — the recorded engine environment is `VLLM_*` plus
  `RCP_NDCG_VLLM_PATCHES`.
- `CHANGELOG.md` — the entries above.

Greps run: `git grep -n -i "plugin_architectures"`, `"plugin_sha256"`, `"rcp-fp"`, `"RCP_NDCG_VLLM_PATCHES"`,
`"batch_size.*runtime"`, `"runtime.*batch_size"` over `docs/`, `README.md`, `rcp-ndcg/README.md`,
`rcp-ndcg-vllm/README.md`, `REPRODUCIBILITY.md`, `skills/`, `examples/`, `experiments/`, `CHANGELOG.md`,
`mkdocs.yml`; every remaining `rcp-fp/3` hit is a historical `handover/` document or a re-keyed manifest's
recorded old value.

## For the next lanes

- **E2 (any recording):** record the plugin families and the new variants under `rcp-fp/4`; the seven
  declared-stale corpora are listed in `rcp-ndcg-test/tests/conformance/stale.json` with their exact moved
  inputs. The five re-keyed corpora replay under their new keys.
- **recipe-fix / rf-engine:** a recipe that opts into `pooling-full-context` declares
  `serve.patches: [pooling-full-context]`; every engine path now renders it, and the corpus provenance
  records the value. The patch retires when `engine.image` moves to the first vLLM release carrying
  `e6fc81bc78`.
- **docs-final (06):** create the compatibility/versioning page and state `rcp-fp/4` and its bump rule; the
  `rcp-fp/3` reference in the workstream prompt is stale.
- **A third-party plugin:** needs a module-declaration/hash seam (the shipped wheel's mapping is
  `rcp_ndcg_vllm.models`); today such a recipe is refused by name at fingerprint time.
- **The judge's media caps:** the remaining role decision if "one role per field" is to hold for the judge
  too.
