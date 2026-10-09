# Lane SZ-sz-misc: size variants (Octen 0.6B/4B, jina nano, topk xsmall)

**Status:** DONE.

**Base:** the lane started at `lane/rfam` tip `da344613` (the family layout, not merged into `rfc-0001`). It merged
`lane/rfam` twice: `3ce7357a` (the golden guard's shrink-only/key-unique check, the operator-path redaction) as
`eee157fc`, and `345beab0` (which merges `rfc-0001` workstream 08 A vLLM-only and the MRL cards, and whose `2c45386a`
fixes the same pairs-validator reload through the family directory and makes every family reference read the resolved
recipe it is given) as `3d55ffc5`. The final recipe tree is `3d55ffc5`; the report (and its wording) are the only
commits after it. `rfc-0001` was not merged directly: `lane/rfam` is
still the integration point (it carries `rfc-0001`'s workstreams 08, 09, 10 and the MRL cards).

## 1. Commits

- `2b78b4f1` The octen, jina and topk family references run every variant: model, revision and dim read from the
  resolved recipe the harness passes.
- `875867e9` The request generator validates a multi-size family's own variant: the stage-1 probe never reloads the
  family directory (superseded by `2c45386a`'s family-directory re-read, kept in the history).
- `274b63f1` Four new sizes for the octen, jina and topk families: variant rows, per-variant tests, goldens,
  vendored tokenizers and pairs files (the catalog, release checklist, T3 matrix and recipe counts move to 23).
- `eee157fc` Merge `lane/rfam` (`3ce7357a`).
- `2718ecdb` The verifiers' documentation findings: the jina `-nano` note states the stored dtype and the
  chat-template facts exactly, the octen 4B note scopes its byte-identity claim, the topk `-xsmall` tokenizer_config
  wording is exact, and the three family references name the variant's own vector width.
- `3d55ffc5` Merge `lane/rfam` (`345beab0`).

## 2. What changed (per brief item)

### 2.1 The four variants (Hub API, re-checked 2026-10-09; all ungated)

| variant | model | revision | licence | served width | context | recipe budget | weights (bf16) | KV at budget | GPUs |
|---|---|---|---|---|---|---|---|---|---|
| `octen-embedding-0.6b` | Octen/Octen-Embedding-0.6B | `d715b32ee68f057b54dff09fc93c23485bc403d3` | apache-2.0 | hidden 1024 | 32768 | 8192 / 8192 | 595,776,512 params; 1,191,586,416 B | 0.94 GB | 1 |
| `octen-embedding-4b` | Octen/Octen-Embedding-4B | `fea468fae3f0caffbae8a12ba792d1c394b6277d` | apache-2.0 | hidden 2560 | 40960 | 8192 / 8192 | 4,021,774,336 params; 8,043,592,088 B | 1.21 GB | 1 |
| `jina-embeddings-v5-text-nano` | jinaai/jina-embeddings-v5-text-nano | `8a7f00aac812071b69403df470f1038ec85f8925` | cc-by-nc-4.0 | hidden 768 | 8192 | 8192 / 8192 | 211,766,016 params; 423,543,680 B + 54,348,064 B adapters | 0.30 GB | 1 |
| `topk-embed-v1-xsmall` | topk-io/topk-embed-v1-xsmall | `210ebf2a25fb7128480f9b9c8f228e8d65c433a7` | apache-2.0 | dim/output_dim 1024 | 262144 | 8448 / 8192 document, 1024 query | 854,034,496 params; 1,708,127,768 B | 0.10 GB | 1 |

Every per-size fact is cited file:key in the variant's own `notes`/`sources` in `family.yaml`: `config.json`
(`hidden_size`/`dim`, `max_position_embeddings`, `dtype`, `tie_word_embeddings`, `vocab_size`, `is_decoder`,
`text_config`/`vision_config`), `1_Pooling/config.json` (`word_embedding_dimension`), `config_sentence_transformers.json`,
`sentence_bert_config.json` (`query_length` 1024 / `document_length` 8192), `modules.json`, `processor_config.json`,
`adapters/*/adapter_config.json`, the tokenizer files, the card, the Hub tree API (safetensors sizes) and the Hub model
API (BF16 parameter censuses). KV bytes recomputed from the configs (heads x head_dim x layers x 2 x 2 B); all four fit
one 80 GB-class GPU with large headroom, so the family's `resources.gpus: 1` is unchanged.

### 2.2 Family membership (the diffs against each family's existing variant)

- **octen-embedding (8B exists).** 0.6B/4B are `Qwen3Model` with the same `config_sentence_transformers.json`
  prompts, the same `modules.json` chain (Transformer + Pooling + Normalize), the same `1_Pooling` last-token flag and
  the same post-processor anchor (the end-of-text added token, id 151643). Differences: hidden size, context limit,
  `tie_word_embeddings` (true for 0.6B/4B, false for 8B -- the converted pooling class replaces the lm_head with a
  `StageMissingLayer` either way, so the shared `hf_overrides: {}` still needs none), vocab, the 0.6B's own tokenizer
  file (4B's tokenizer is byte-identical to the 8B's) and the 0.6B's newer ChatML template (never served:
  `chat_template: null`). No new field, no new family.
- **jina-embeddings-v5-text (small exists).** The `-nano` is a different backbone (EuroBERT-210m, `is_decoder false`)
  under the same `architectures` entry: vLLM v0.31.0's `JinaEmbeddingsV5Model.__new__` dispatches on `is_decoder` to
  `JinaEmbeddingsV5EncoderModel` (whose docstring names this checkpoint), and its config handler sets `is_causal false`
  -- the serving path (`--runner pooling`, the load-time `jina_task` adapter merge, `trust_remote_code`) is the
  family's. Same prompts, same four task adapters (alpha 32, r 32), same mask-based last-real-token pooling, same L2
  normalisation, no head. Differences beyond the per-size fields: the tokenizer (EuroBERT vocab 128256 vs 151936) and
  the fact that `add_special_tokens=true` **appends** the end-of-text token (id 128001) where the `-small`'s appends
  nothing; the card path appends it too, so both sides pool the appended tail, and the family template's
  `add_special_tokens: true` reserves it (the harness's `last_content` audit's tail branch asserts it closes every
  render). `torch_dtype` in `config.json` says float32 but the safetensors are stored BF16 and the card recommends
  bf16 for GPUs, so the shared `serve.dtype: bfloat16` stands. Overrides: `serve.max_model_len: 8192`,
  `serve.hf_overrides` (its own Matryoshka list 32..768), `client.max_tokens: 8192`.
- **topk-embed-v1 (small exists).** Same architecture (`TopkEmbedModel`), same plugin registration and config class,
  byte-identical `modeling_topk_embed.py`/`topk_embed_st.py`/`hf_backbone.py`, byte-identical tokenizer.json
  (sha256 `e56427d6...`), same `chat_template.jinja`, `processor_config.json`, `sentence_bert_config.json`, prompts,
  41 `scoring_skip_ids` and `image_token_budget`. Differences: `dim`/`output_dim` 1024 (override `client.dim: 1024`),
  text hidden 1024 over Qwen3.5-0.8B, vision 12/768/1024, head.weight (1024, 1024) BF16. No new field, no new family.

### 2.3 Rows, tests, pairs, goldens, docs

- Variant rows with overrides/notes/sources in the three `family.yaml` files (status `unverified`; the operator's GPU
  wave sets it from evidence).
- The family test modules are parametrized per variant: every resolved serve/client/reference field is pinned per
  variant, the stage-1 network tests run the real tokenizers (one parameter per variant), the mutants stay red per
  family, and the new variants' notes tests pin their per-size facts. The topk module also carries the reference's
  `--recipe` mismatch refusal (adapted per variant).
- Goldens: the four new variants' goldens (the guard's writer) plus the tokenizer store now covers every new spec
  (the 0.6B/4B index entries point at the existing byte-identical files; the `-nano`'s and `-xsmall`'s files are
  vendored). The notes deltas for the existing variants the lane's prose changes moved (`jina-...-small`,
  `octen-embedding-8b`, `topk-embed-v1-small`) are declared in `golden/DELTAS.json`; the merged guard's
  shrink-only/key-unique check passes.
- Pairs: the four files were generated with `python -m rcp_ndcg_test.observe.requests --reference-python ...`
  (octen 0.6B/4B 23 rows, jina nano 23, topk xsmall 31; render checks passed, xsmall's media check passed, two rows
  pruned) and the manifest merged; a verifier regenerated all four byte-identically.
- Docs: the `rcp-ndcg-vllm` README catalog (23 rows), `docs/reference/recipes.md` (13 families / 23 variants),
  `docs/index.md`, `docs/quickstart.md`, `CHANGELOG.md`, `handover/RELEASE-CHECKLIST.md` (23 retrieval recipes).
  Count-coverage updates outside the brief: `rcp_ndcg_test.quality.TASK_MATRIX`, `tests/retrieval/test_paper_configs.py`
  and `rcp-ndcg-vllm/tests/models/test_wheel_contract.py` (`N_RECIPES = 23`).

### 2.4 The request generator

`_validate_and_prune` re-loaded the recipe with `load_recipe(recipe._dir)`; `_dir` is the family directory, which the
standalone path refuses for a multi-variant family, so no multi-size family could regenerate its pairs files. The
lane's `875867e9` fixed it by probing the resolved recipe; the merged `2c45386a` (lane rfam, same finding) re-reads the
variant through its family directory (`load_family` + `load_recipes_of`), which the merge keeps. The generator tests
carry the two-variant regression.

## 3. Verification

**Round 1** (two fresh-context verifiers on `eee157fc`, DeepSeek-V4.1-flash, `:xhigh`; lens A correctness, lens B
regressions/hygiene; COMMON's owner decision 2026-10-09 replaces the GLM verifier). Both ran the suites, the network
recipe tests, their own reproductions and mutation tests; each verdict:

- **Lens A: VERDICT PASS.** Independently re-derived the Hub shas/licences/gated flags (25 checks), every per-size fact
  from the checkpoints' files, the KV/weight arithmetic, the family-membership diffs, the resolved overrides, the
  vendored tokenizer hashes, the pairs provenance (regenerated all four pairs byte-identically) and the goldens/DELTAS
  (31 entries, key-unique, none stale). Five minor documentation findings:
  - F1: the `-nano` note said the `-small` ships a `chat_template.jinja`; it does not (its ChatML template is embedded
    in `tokenizer_config.json`). Fixed in `2718ecdb`.
  - F2: the `-nano` note called the config's float32 "the stored dtype"; the safetensors are stored BF16. Fixed.
  - F3: the octen 4B note claimed the ST configs are byte-identical to the 8B's; `1_Pooling`'s
    `word_embedding_dimension` necessarily differs. Fixed.
  - F4: three reference docstrings still stated the original variant's vector width. Fixed.
  - F5: the topk `-xsmall` note's `fix_mistral_regex` wording. Fixed.
- **Lens B: VERDICT PASS.** Full root suite (3203 passed), `rcp-ndcg-test` (601), contract/docs (270), the vllm suite
  (62), lint/types clean, the network recipe tests (124) and `public-names`/`mkdocs --strict` clean; mutation (a) the
  generator fix reverted → the new regression test red; mutation (b) the topk `-xsmall` `client.dim` and the jina
  `-nano` `max_tokens` mutated → the family contract tests red. One minor: dropping a vendored tokenizer spec from the
  store does not red the offline golden guard because `_tokenizer_sha`/`_write_golden` fall back to the golden's own
  hash (documented fallback, pre-existing lane-rfam design). Not changed: the lane's specs are all vendored and
  hash-verified, and the network tests re-derive the Hub sha; the finding is recorded in Open questions as an optional
  hardening.

Round 2 was not run: round 1 found no blocker and no major (COMMON's owner decision: round 2 only for those). After
the merge of `lane/rfam` (`345beab0`) the affected tests were re-run and the full gate re-ran at `3d55ffc5` (below).

## 4. Checks (the gate ran on `3d55ffc5` and again on the final tip; both PASS)

- `bin/gate lane/sz-misc` → **GATE: PASS**:
  - `ruff-check` 0; `ruff-format` 527 files already formatted; `basedpyright` 0 errors/0 warnings/0 notes.
  - `pytest` (root, `-n 8`): **3269 passed, 93 skipped**.
  - `contract-docs`: **287 passed, 52 skipped**; `mkdocs build --strict` clean.
  - `test-pkg` (`rcp-ndcg-test/tests`): **605 passed, 283 skipped**.
  - `recipes` (network-gated, real tokenizers): **no failure outside the baseline** (0 baseline failures remain,
    34 fixed; pytest exit 0).
  - `vllm-pkg` 1 passed; `vllm-models` **70 passed, 7 skipped**.
  - `run_all`: leaderboards 1022 checks / 987 match / 35 known deviations / 0 failed; human study 67/67;
    external judges 82/82.
  - `public-names` clean (0 baselined hits remain); `clean` exit 0.
- Narrow runs before the gate: the four new variants' network tests (**99 passed** in the family modules) with
  `RCP_NDCG_NETWORK_TESTS=1` and `timeout 900`; the golden guard **27 passed**; `test_observe_requests` +
  `test_quality` **68 passed**.

## 5. Open questions

- **The jina `-nano`'s declared anchor.** Its tokenizer appends the end-of-text token (id 128001) under
  `add_special_tokens`, so the model pools that appended tail, while the family template declares `anchor: last_content`
  (the last kept content token). Both the served route and the reference append and pool the same tail, the budget
  reserves it and the audit's tail branch asserts it, so the numbers are right -- but the declaration's semantics
  ("pools the last real token of raw text") is inexact for this size. A per-variant template/anchor field or a separate
  family would be the exact fix; the family schema shares content shapes by design (decision 34), so it was not
  changed. The owner may want to decide before a second EuroBERT-backed size arrives.
- **Golden-store fallback.** The offline guard accepts the committed golden's own `tokenizer_sha256` when the store
  does not cover a spec, so removing a vendored spec is silent (verifier lens B, minor). All four new specs are
  vendored and hash-verified; an optional hardening is to assert every shipped recipe's spec resolves from a store.
- **GPU waves (operator).** `octen-embedding-0.6b`, `octen-embedding-4b` and `jina-embeddings-v5-text-nano` have
  `client.max_tokens == serve.max_model_len` (8192 = 8192), so an exactly-at-budget pooling prompt is exposed to the
  known vLLM v0.31.0 hang (GPU-E1: any chunked pooling prompt of exactly `max_model_len` never completes) and would
  need the recipe-fix lane's opt-in patch, exactly like the existing octen-8B and jina-small. `topk-embed-v1-xsmall`
  keeps 256 tokens of headroom (8448 > 8192), so a client request never reaches `max_model_len`. At these budgets the
  32-bit warmup limits from GPU-E1 (q bytes and MLP activation elements below 2^31) are not approached (q bytes at
  8192/8448 are 13-68 MB; MLP activation elements are below 8e7). The references must run on GPU (E1's CPU-reference
  precision class), and the embedders have no score head, so the octen bf16 engine-vs-reference precision class from
  E1 (min cosine 0.9936 on the 8B) applies to the 0.6B/4B and the nano waves.
- **Status.** All four variants are `status: unverified`; only the operator's GPU evidence flips them.

## Docs updated

- `rcp-ndcg-vllm/README.md`: four catalog rows (nano/0.6b/4b/xsmall), the plugin sentence and the
  "four released checkpoints" count.
- `docs/reference/recipes.md`: 13 families / 23 variants, the three family rows and the plugin sentence.
- `docs/index.md` and `docs/quickstart.md`: "23 recipes".
- `handover/RELEASE-CHECKLIST.md`: retrieval recipes (23) with the four ids.
- `CHANGELOG.md`: the two entries below.
- The three family `reference.py` files and the variant `notes`/`sources` (the per-size facts and the corrected
  dtype/chat-template/byte-identity wording).
- Greps run over `docs/`, `README.md`, `REPRODUCIBILITY.md`, `skills/`, `examples/`, `experiments/**/*.md`,
  `mkdocs.yml` and the docstrings: `git grep -n "octen-embedding-0.6b"`, `... "octen-embedding-4b"`,
  `... "jina-embeddings-v5-text-nano"`, `... "topk-embed-v1-xsmall"`, and
  `git grep -n "19 variants\|19 recipes\|three released checkpoints\|their 19"` (no stale hits remain).

## 6. CHANGELOG entry

Added under `## Unreleased` (exact text):

Public surface:
```
- **Four new sizes for three shipped families** (decision 34): `octen-embedding-0.6b` and
  `octen-embedding-4b` (the Octen family's 0.6B and 4B checkpoints, last-token pooling and the paper's
  `"- "` document frame), `jina-embeddings-v5-text-nano` (the EuroBERT-210m encoder under the same vLLM
  `JinaEmbeddingsV5Model` dispatch as the family's Qwen3-based `-small`; its own 8192-token budget and
  Matryoshka list) and `topk-embed-v1-xsmall` (the 1024-dim sibling of the plugin-served topk retriever).
  Each is a full recipe id with its own pinned revision, per-size overrides, contract pins, stage-1 test,
  golden and pairs file; the catalog, the release checklist and the request generator's four new pairs
  files gain the rows.
```

Fixed:
```
- **The request generator validates a multi-size family's variant**: `_validate_and_prune` re-loaded
  the recipe with `load_recipe(recipe._dir)`, and `_dir` is the family directory, which the standalone
  path refuses for a family with more than one variant, so no multi-size family could regenerate its
  pairs files. It now re-reads the variant through its family directory (`load_family` +
  `load_recipes_of`).
```

## 7. Public surface changes

- New recipe ids (catalog entries, `rcp-ndcg-vllm serve <id>`, `recipe: <id>`): `octen-embedding-0.6b`,
  `octen-embedding-4b`, `jina-embeddings-v5-text-nano`, `topk-embed-v1-xsmall`. The resolved recipe schema, the family
  schema, the CLI tree, the exit codes and the contract snapshots are unchanged (the variant rows resolve to the
  unchanged `Recipe` schema; the goldens pin each resolved contract).
- No new CLI flag, exit code or schema; the generator function is internal.

## 8. Files outside scope

- `rcp-ndcg-test/src/rcp_ndcg_test/quality.py` and `rcp-ndcg-test/tests/test_quality.py`: the T3 task matrix must name
  every shipped variant (the completeness test), so the four new ids were added to their family rows.
- `rcp-ndcg-vllm/tests/models/test_wheel_contract.py` (`N_RECIPES = 23`), `tests/retrieval/test_paper_configs.py`
  (`checked == 23`), `docs/index.md`, `docs/quickstart.md`: recipe-count completeness/truth.
- `rcp-ndcg-test/src/rcp_ndcg_test/observe/requests.py` + `rcp-ndcg-test/tests/test_observe_requests.py`: the
  multi-size family fix (the lane's commit and, after the merge, lane rfam's own fix of the same bug).
- The two zerank goldens/DELTAS: the lane redacted their pre-family operator paths first; the merged `lane/rfam`
  carries its own redaction (`REDACTIONS.json`), which supersedes it.

## 9. For the next lanes

- The merged `lane/rfam` (`345beab0`) is the integration point; it carries `rfc-0001` workstreams 08 A, 09, 10 and the
  MRL cards. Merge it before touching the recipe families.
- The jina `-nano` anchor question above is the one modelling decision this lane leaves open.
- The operator's GPU waves should run the four variants with the recipes' `status` set from evidence; the octen/nano
  waves need the pooling-hang patch (or a run at `max_model_len - 1`) and GPU references.
