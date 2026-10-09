# Lane `mrl-core`: first-class, efficient MRL in the product (owner decision 39)

## 1. Status

DONE. The MRL head, the config declaration (set or range), the per-row records, the full-width embedding
store and the ex-post sweep are implemented, tested and documented; `bin/gate lane/mrl-core` passes on the
merged tip.

## 2. Commits

| Commit | Subject |
|---|---|
| `87a6199f` | The MRL head has one home: the truncation cut moves to `rcp_ndcg.data.mrl` beside the learned-projection head (a minimal safetensors reader through storage, float32 chains, renormalised), and the `mrl_cut` mechanism joins `ProcessingRecord` with the kind, k and full width it applied |
| `a2cd4606` | MRL is declared before it is selected: `mrl_kind`, `mrl_dims` and `mrl_projection` on both role configs with load-time validation (a k outside the set, `dimensions` beside `mrl_dim`, `dimensions` off the truncation kind, a projection kind without its source), `mrl_dim` on the dense client too, and the clients normalise the full-width reply before the head so a direct k run and the ex-post sweep agree bit for bit |
| `682c3b39` | One forward pass, every declared width: the full-width `EmbeddingStore` (corpus and query vectors, ragged offsets, provenance, the retrieval identity plus a full-width marker) with `build_store`/`load_store`/`sweep` and the retrieval `store` and `sweep` commands (per-k Rankings `<model>@<k>`, then `evaluate`/`compare`) |
| `95cf31be` | The MRL kind, set and projection are post-processing fingerprint fields: the declared head never moves the request bytes |
| `750c0792` | Docs and the public surface for the MRL head and the sweep: a matryoshka concept page, the embeddings/late-interaction/retrieval pages, the CLI reference, the changelog, the regenerated snapshots and schemas |
| `b13ec8af` | Merge branch `rfc-0001` into `lane/mrl-core` (the l10a data-model and reader merge) |
| `7a4c0ab4` | `ruff format` of the MRL files, the sweep handler and the touched tests |
| `c338a852` | Merge branch `rfc-0001` into `lane/mrl-core` (the mrl-cards spec, `e3a356f1`) |
| `cbc51196` | A card's prose range is a first-class MRL declaration: `mrl_range [min, max]` beside `mrl_dims` (one of the two, validated), every k in the closed interval selectable with the floor enforced client-side, a projection kind keeps the discrete set, and the sweep takes explicit dims for a range |
| `68fccdd3` | Docs and snapshots for the MRL range declaration |
| `c924cdc4` | Verifier round 1 fixes: projection chains normalise their keys to strings, a chain that does not end at k is refused, a store rebuilt across layouts ignores stale offsets, an explicitly empty sweep dims is refused, and the reader/cache/BF16 paths are tested |
| `f4542a96` | Verifier round 1 docs fixes: the one-home table points the MRL head at `rcp_ndcg.data.mrl`, the duplicate `mrl_kind` is gone, the `mrl_dims`/`mrl_range` docstrings and the store command's `mrl_range` field are current, schemas regenerated |
| `bba1b9b7` | Round-2 verifier minors: the chain-key check refuses non-ASCII digits before `int()` sees them, and the safetensors reader validates the header object, the shape's integer entries and the span's exact byte count |
| `a507e285` | Review notes resolved: the projection chain has one naming convention (the file's tensor names are their target widths, so the declaration is the source alone), the projection computes and returns float32 over a float16 store (stated and tested), and M7's legacy cut paths are pinned by a refusal test |
| `fe094987` | handover: the mrl-core report (this file; a later commit carries the review-note update) |

## 3. What changed

**1. Config (`rcp_ndcg/inference/config.py`).** `mrl_dim` moved to `EmbeddingEndpoint` (parity with
`PoolingEndpoint`), and the declaration is `mrl_kind` (`truncation`/`projection`/unset) plus exactly one of
`mrl_dims` (the card's discrete table) and `mrl_range` (`[min, max]`, the card's prose range; the client
enforces the floor). `mrl_projection` carries a projection kind's safetensors source (one naming
convention: the file's tensor names are their target widths, and the declared `mrl_dims` are the projected
sizes). Validation at load refuses: a k outside the declaration; `dimensions` + `mrl_dim` together;
`dimensions` on a non-truncation kind or on the pooling wire; a projection kind without its source or with
a range (a range names no chain); `mrl_dims` beside `mrl_range`; a declared kind without a
declaration; a set, range, projection or selection without a kind. Each refusal names the field and the
fix. All five fields are CONTENT for identities. This is the M7 fix: no `mrl_dim`/`dimensions` cut can run
without a declared `mrl_kind`, so a projection-kind checkpoint is never silently sliced as truncation.

**2. One MRL head home (`rcp_ndcg/data/mrl.py`).** The truncation cut moved there from
`data/postprocess.py` (no second home; `postprocess.py` points at it), and the projection head loads the
checkpoint's learned matrices from a declared `repo@revision` file through `rcp_ndcg.storage` (local path,
`gs://`, or `hf://org/model@revision/path.safetensors`), parses the minimal safetensors format in-process
(no new dependency; F32/F16/F64/I8..I64/U8/BOOL/BF16), caches per source, applies the chain (the tensors
named for every declared dimension at or above k, widest first) in float32 and renormalises. The projection
computes and returns float32 even over a float16 store; a truncation cut keeps the input dtype.
`MrlHead` is the one selection check: declared kind, k in the set or the closed
range, k no wider than the vectors, and a projection chain that ends exactly at k. The embed and pooling
clients construct one head each and apply it client-side; both normalise the full-width reply first (when
`normalize`) and then run the head, so a direct k run and the ex-post sweep compute bit-identical vectors.

**3. Records (`rcp_ndcg/data/text_budget.py`).** `mrl_cut` joins `ChangeMechanism`/`CHANGE_MECHANISMS`;
`ProcessingRecord` gained `mrl_kind`, `mrl_dim` and `full_width` (all `None` on other rows and in
`as_row`). Every row the head changed carries one record; a row without a selection carries none.

**4. The full-width store and the sweep (`rcp_ndcg/data/embedding_store.py`,
`rcp_ndcg/retrieval/store.py`).** `EmbeddingStore` (schema `rcp-ndcg.embedding-store.v1`) records the
corpus and query vectors at full width (ragged offsets for late interaction), the provenance (model,
revision, recipe, prompt digest, tokenizer and its digest, budget, full width, dtype, the declared MRL
head) and the doc/query ids; bytes and URIs go through `rcp_ndcg.storage`. Its identity is the retrieval
identity of the full-width encoder plus a full-width marker, so k never enters it and a cut store can
never be mistaken for a full-width one; the saver and reader both refuse vectors narrower than the
record's `full_width`. `build_store` encodes corpus and queries once (the configured selection is stripped
for that pass); `sweep` applies the declared head per k, scores with `score_topk` (dense dot or MaxSim)
and returns one `Rankings` per k with system `<model>@<k>`; the new `retrieval store` and
`retrieval sweep` commands build the store and, with `--dataset`, run `evaluate` and `compare` across k
and write the report and comparison. A store that declares only `mrl_range` needs explicit `--dims`.

**5. Docs.** New `docs/concepts/matryoshka.md` (the two kinds, the order, the set/range, the store, the
sweep), updated `docs/concepts/embeddings.md`, `late-interaction.md`, `retrieval.md`,
`docs/reference/cli.md`, `AGENTS.md`'s one-home table, `CHANGELOG.md`; contract snapshots and `schemas/`
regenerated (new `embedding-store.v1.json`, `store-build.v1.json`, `retrieval-sweep-result.v1.json`).

## 4. Verification

**Round 1 (two independent fresh-context verifiers, lens A correctness and lens B regressions/hygiene,
on tip `68fccdd3`; model DeepSeek-V4.1-flash, xhigh).**

- **Lens A: FAIL.** F1 (major, blocker candidate): `mrl_projection.chains` was keyed by `int`, so
  `identity_payload` -> `hash_payload` raised `UnhashableValueError` on every identity-bearing path
  (`index`, `retrieve`, `build_store`, run steps); `retrieval store` exited `INTERNAL`. F2 (minor): a
  projection chain that does not end at k was not checked. F3 (minor): stale `*_offsets.npy` made a store
  rebuilt across layouts unreadable. F4/F5 (minor docs): duplicate `mrl_kind` in `embeddings.md`; the
  `preprocess.py` docstring still named `postprocess` as the MRL cut's home. F6 (minor): `StoreBuild` did
  not report `mrl_range`. F7 (minor): `sweep(dims=())` returned `[]` silently. All other acceptance
  items passed, including bit-for-bit sweep-vs-direct equality for dense/late interaction and
  set/range/projection.
- **Lens B: FAIL.** B1 (major hygiene): the binding one-home table in `AGENTS.md` still named
  `rcp_ndcg.data.postprocess` as the MRL cut's home. B2-B8 (minor): the `embeddings.md` duplicate; stale
  `dimensions`/`mrl_kind` attribute docstrings exported into the schemas; `StoreBuild` without
  `mrl_range`; the `retrieval store` CLI handler untested; dead `MrlHead.declared` /
  `clear_projection_cache`; untested BF16/reader-error/projection-store branches; stale handover notes.
  Four scratch mutations each turned the corresponding lane tests red.
- **Fixes:** `c924cdc4` (F1 string-keyed chains with a `mode="before"` normaliser, F2 final-width check,
  F3 layout-gated offsets, F7 empty-dims refusal, F6 CLI field, B5 CLI test, B6 removal/wiring, B7
  BF16/error/cache/projection tests) and `f4542a96` (F4/F5 docs, B1 table row, B3 docstrings + schemas).
  Each got a failing test first (the F1 test hashes an identity with int-keyed chains; the store test
  runs a projection store through sweep). The operator review note below then removed the `chains` field
  altogether, so the F1 crash class is gone at the root rather than patched.
- **Note on an earlier pair:** a first pair of verifier runs on the pre-addendum tip died as an
  infrastructure failure (the async runner process disappeared before writing a result; no verdict) and
  was relaunched fresh after the addendum's range work.

**Round 2 (one fresh confirmation verifier, lens A+B, on tip `f4542a96`).**

- **PASS**, with two new minor findings, both fixed in `bba1b9b7` with tests: (1) the chain-key check
  called `int()` on `str.isdigit()` keys that are not ASCII digits (a superscript raised an unhelpful
  pydantic error instead of the intended refusal) — fixed by gating on `isascii()`; (2) the safetensors
  reader raised untyped errors on malformed headers and could silently read across tensors when a shape
  did not match its span — fixed by validating the header object, the shape's integer entries and
  `end - start == count * itemsize`, and wrapping `np.frombuffer`. The verifier confirmed F1 end to end
  (pre-fix crash, fixed pass), mutation-tested four round-1 fixes in scratch copies, and re-ran the full
  suite, contract/docs, fingerprint tests, `basedpyright` and `ruff` (all green). These are robustness
  minors with tests, covered by the final gate; no third round was needed.

**Operator review note (2026-10-09, pre/post-processing review, sections 2 M7 and 5.2), resolved in
`a507e285`.**

- **M7 covered.** The review's finding was that `mrl_dim` could apply a truncation cut with no declared MRL
  kind on projection-kind checkpoints. Every legacy cut path is now refused without a declared
  `mrl_kind`: the dense and pooling `mrl_dim`, the dense engine-side `dimensions`, and an explicit
  `mrl_kind: none`; the pooling `dimensions` stays refused as inert, and the `postprocess.py` cut no
  longer exists (the head has one home). `test_a_truncation_cut_without_a_declared_kind_is_refused` pins
  all five cases.
- **Projection output dtype resolved.** The chain computes and returns float32 even over a float16 store
  (the learned matrices are F32; the truncation cut keeps the input dtype). `MrlHead.apply` and `_project`
  state it, and `test_the_projection_returns_float32_over_a_float16_store` pins it.
- **Chain naming convention resolved.** One convention: the file's tensor names are their target widths,
  the declared `mrl_dims` are the projected sizes (the full width is served without a head), and the
  chain for k is the tensors named for every declared dimension at or above k, widest first. The optional
  per-k `chains` override was removed; the matryoshka page and the schemas state the convention, and the
  reader still refuses a chain that does not line up or does not end at k.

## 5. Checks

Last runs on the final tip (`a507e285`):

```text
uv run --no-sync ruff check .                         -> All checks passed!
uv run --no-sync ruff format --check .                -> 545 files already formatted
uv run --no-sync basedpyright                         -> 0 errors, 0 warnings, 0 notes
PYTHONPATH=<scratch entry-point shim> pytest tests/ -q -n 4
                                                      -> 3370 passed, 94 skipped
pytest tests/contract tests/docs -q                   -> 294 passed, 53 skipped
pytest rcp-ndcg-test/tests/test_fingerprint.py -q     -> 24 passed
pytest rcp-ndcg-test/tests -q                         -> 570 passed, 225 skipped
experiments/run_all.py                                -> leaderboards 1022 checks, 987 match,
                                                         35 known deviations, 0 failed;
                                                         human study 67/67; external judges 82/82
bin/gate lane/mrl-core                                -> GATE: PASS (see below)
```

The lane venv predates the merged `rcp_ndcg.readers`/`rcp_ndcg.writers` entry points, so the suite runs
used a scratch-only `PYTHONPATH` shim (a `*.dist-info` with the merged entry points; it changes neither
the venv nor the lock, and it was never committed). The gate's integration worktree syncs the environment
from the lock and ran the same suites without the shim.

**Gate result** (`bin/gate lane/mrl-core`, revision `a507e285`):

```text
ruff-check exit=0 / ruff-format exit=0 / basedpyright exit=0
pytest exit=0 -> 3370 passed, 94 skipped
contract-docs exit=0 -> 294 passed, 53 skipped
mkdocs exit=0 / test-pkg exit=0 -> 570 passed, 225 skipped
recipes exit=0 (no failure outside the baseline) / vllm-pkg exit=0
run_all exit=0 -> 1022/987/35/0, 67/67, 82/82
public-names exit=0 / clean exit=0
GATE: PASS
```

## 6. Open questions

- **`k == full width`.** A truncation head allows a k equal to the observed width (a renormalising
  no-op), because a card's set may include the full width and a range's ceiling is the width; the pooling
  config still refuses `mrl_dim >= dim`. If the owner wants the head to refuse k at the width too, that is
  a one-line check (and the range cards would need their ceilings handled).
- **`normalize` is not in the store record.** The sweep-vs-direct bit identity relies on the store holding
  the client's full-width form (the head renormalises and the projection is linear, so this is
  mathematically irrelevant), but another consumer of the record would not see whether `normalize` was
  applied. Adding the field is additive if wanted.
- **`dimensions` remains a user option, never a recipe default** (per the addendum); a recipe that wants
  the engine-side cut must declare `is_matryoshka` in its serve block. The recipes lane owns that choice.
- **Handover specs** still say `mrl_cut` lives in `data/postprocess.py` (`handover/specs/mrl-cards.md`);
  `handover/` is temporary and deleted before release, so it was not rewritten.

## CHANGELOG entry

```markdown
- **First-class, efficient Matryoshka support (owner decision 39)**: every embedding and multi-vector
  endpoint declares its MRL head once -- `mrl_kind` (`truncation`, `projection` or unset), the card's
  supported output dimensions as `mrl_dims` (a discrete table) or `mrl_range` (`[min, max]` prose, with the
  floor enforced client-side) and, for a projection kind, `mrl_projection` (the checkpoint's
  learned `*.safetensors` matrices, read through `rcp_ndcg.storage`) -- and a run selects `k` from that
  declaration (`mrl_dim` on both role configs, client-side; the engine-side `dimensions` stays dense-only
  and truncation-kind-only). Every refusal names the field and the fix: a `k` outside the declaration,
  `mrl_dims` beside `mrl_range`, `dimensions` beside `mrl_dim`, `dimensions` on another kind, a declared
  kind without a declaration, and a projection kind without its source (or with a range, which names no
  chain). The one head home is `rcp_ndcg.data.mrl` (`MrlHead`, `mrl_cut`, `MrlProjection`): the truncation
  cut moves there from `rcp_ndcg.data.postprocess`, and the projection head loads the declared chain in
  float32 and renormalises. Every row the head changed carries a `ProcessingRecord` with the new `mrl_cut`
  mechanism and its kind, `k` and full width (`mrl_cut` joins `CHANGE_MECHANISMS`). The full-width
  `EmbeddingStore` (`rcp_ndcg.data.EmbeddingStore`, `StoredVectors`, `load_embedding_store`) holds corpus
  and query vectors, ragged offsets for late interaction, and a `store.json` with the schema and provenance
  (model, revision, recipe, prompt digest, tokenizer, budget, full width, dtype, the declared MRL head),
  content-addressed by the retrieval identity plus a full-width marker; `rcp_ndcg.retrieval.build_store`/
  `load_store`/`sweep` wire it, and the new `rcp-ndcg retrieval store` and `rcp-ndcg retrieval sweep`
  commands build it and evaluate every declared `k` from it (per-k rankings `<model>@<k>`, then
  `evaluate`/`compare`) in one forward pass.
```

and the `### Changed` bullet:

```markdown
- **The Matryoshka selection is declared before it is selected**: a pooling `mrl_dim` now needs its
  `mrl_kind` and `mrl_dims`/`mrl_range` (the card's set) and a dense `mrl_dim` is new; a `k` outside the
  declaration is refused at load. When `mrl_dim` is set, the client normalises the full-width reply first
  (when `normalize`) and then applies the head, so a direct `k` run and the ex-post sweep over a full-width
  store compute bit-identical vectors (the head renormalises the cut, and the learned projection is
  linear).
```

## Public surface changes

- **Python names**: `rcp_ndcg.data.EmbeddingStore`, `rcp_ndcg.data.StoredVectors`,
  `rcp_ndcg.data.load_embedding_store`; `rcp_ndcg.retrieval.build_store`, `rcp_ndcg.retrieval.load_store`,
  `rcp_ndcg.retrieval.sweep`; the internal-but-documented `rcp_ndcg.data.mrl` (`MrlHead`, `MrlKind`,
  `MrlProjection`, `mrl_cut`).
- **Config fields**: `EmbeddingEndpoint.mrl_kind`, `.mrl_dims`, `.mrl_range`, `.mrl_projection`,
  `.mrl_dim` (the last inherited by `PoolingEndpoint`); the `mrl_cut` `ChangeMechanism`; the
  `ProcessingRecord.mrl_kind`/`mrl_dim`/`full_width` fields.
- **CLI**: `rcp-ndcg retrieval store` and `rcp-ndcg retrieval sweep` (no new exit codes).
- **Schemas**: new `schemas/embedding-store.v1.json`, `store-build.v1.json`,
  `retrieval-sweep-result.v1.json`; regenerated `index.v1.json`, `run-config.v1.json`, and the contract
  snapshots (`cli.json`, `python_api.json`).

## Files outside scope

- `AGENTS.md` (the one-home table; verifier B1).
- `rcp-ndcg-test/src/rcp_ndcg_test/fingerprint.py` and `rcp-ndcg-test/tests/test_fingerprint.py` (the
  fingerprint classification is in the brief; the fixture test needed the new field).
- `tests/_safetensors.py` (new test support for the projection file).
- `handover/reports/MRL-core.md` (this report).

## For the next lanes

- **Equivalence harness (the study's items 5-8) is still open**: the product fake/emulators' `/pooling`
  should refuse `dimensions`, `/embeddings` should validate `is_matryoshka`/the set, and stage 2 should
  gate every k ex-post from one full-width reference run. The fingerprint classification
  (`mrl_kind`/`mrl_dims`/`mrl_range`/`mrl_projection` as `post_processing`) is in place, so the declared
  head never re-keys a request.
- **Recipes**: use the client-side `mrl_dim` for every kind. zembed uses `mrl_kind: projection` with
  `mrl_projection.source` alone (the file's tensor names are the target widths, so the chain derives from
  the declared projected sizes); pplx-context and topk use `mrl_kind: truncation` + `mrl_dim` after the
  plugin's projection;
  Qwen3-Embedding and Qwen3-VL-Embedding declare `mrl_range: [32, W]`/`[64, W]` (the floor is enforced
  client-side); pplx-embed-v1, octen, harrier-oss-v1 and pplx-embed-v2-late declare no head. The
  engine-side `dimensions` is a user option only (dense + truncation), never a recipe default.
- **`run_all` did not move**: 1022/987/35/0, 67/67, 82/82 on the merged tree.
