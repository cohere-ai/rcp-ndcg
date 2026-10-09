# Lane `mrl-cards` report: every variant's MRL kind and set, from the model cards (spec only)

## Status

DONE. Deliverable `handover/specs/mrl-cards.md` (spec only; no code, no recipe). Two verifier rounds: round 1 FAIL
(two majors, seven minors, all evidence-attribution defects), round 2 PASS after the fixes. `bin/gate lane/mrl-cards`
on the spec head `868447e3`: **GATE: PASS**; the report-only commit after it was re-checked with the gate's
`public-names` step (clean).

## Commits

- `a41deeef` handover: the MRL model-card spec: every embedding and multi-vector variant's kind, set, head and engine path at its pinned revision
- `927c7ad6` handover: the MRL spec names the field's absence by the product trees it was grepped in
- `e9ead059` handover: the MRL spec fixes the verifiers' evidence attributions (octen/zembed note, qwen3-vl serve block) and the line cites
- `868447e3` handover: the MRL spec polishes the G3 wording, the fake /pooling range and the Octen note line

Base `rfc-0001` @ `681a8cea`; `origin/rfc-0001` was still `681a8cea` when fetched, so no merge was needed. The lane
changed only `handover/specs/mrl-cards.md` (plus this report).

## What changed

Per brief item:

- **All 22 variants** (8 shipped, 4 in flight, 10 new sizes) are in the spec's §2 table and in §3: the MRL kind
  (`truncation` / `projection` / `none`), the supported set and its source (card, `config_sentence_transformers.json`,
  `config.json` `matryoshka_dimensions`, reference code), where the full width comes from (backbone / ST `Dense` /
  projection), the per-request `dimensions` path and the serve-time `pooler_config.dimensions` path at vLLM v0.31.0.
- **Projection kinds carry the file, shapes and order**: `zembed`'s `projections.safetensors` is six chained F32
  matrices (`2560->1280->640->320->160->80->40`, applied in sequence); the projected full widths of `topk` (head
  `[W, W]`), `pplx-context` (`contextual_head.safetensors` `[2048, 4096]`) and `embeddinggemma-2`
  (`embedding_projection` `[768, 512]`) are documented with their orders.
- **Engine facts at vLLM v0.31.0** with `file:line`: the seqwise and tokwise head orders (projector -> slice -> L2),
  the `is_matryoshka` / range / set-membership gates, the `/pooling` request refusal, the registry entries and the
  `Model` suffix rule, the jina encoder dispatch, the transformers pin, and which architectures have no class at all.
- **Product facts at this tree**: the two MRL paths (`dimensions` dense, `mrl_dim` pooling), the identity/fingerprint
  treatment, the missing mechanism record, the fake engine's two wrong routes, and the harness's missing MRL gate.
- **Eleven gaps, G1-G11**: a card that claims MRL without a set (pplx-embed-v1); a set the engine path cannot serve
  (zembed's projection chain); a recipe note the shipped declaration does not back (qwen3-embedding); card ranges the
  engine cannot enforce; card sets not declared to the engine; topk's `output_dim` not being the served knob;
  projection kinds needing a product field; the study's product-side gaps; in-flight variants without a recipe; the
  pplx-context pin drift; tested sets vs code bounds.
- **Five open decisions** for the owner / `mrl-core` in §5.

## Verification

Round 1, two fresh verifiers in parallel (DeepSeek-V4.1-flash `:xhigh`):

- **Lens A, correctness: VERDICT FAIL.** It reproduced all 22 pins, the kinds, sets, full-width sources, projection
  shapes/orders, both engine routes and the product records; it found two majors and seven minors, all
  evidence-attribution defects: (1) the Octen section quoted zembed's note and cited it to the Octen recipe, which
  has no MRL text; (2) the Qwen3-VL section claimed a `dimensions: null` key and a recipe note that the recipe does
  not carry; (3-9) the Octen ">32K" vs ">40K" quote, `config/model.py` 2291 vs 2288, the embeddinggemma-2
  `truncate_dim` line, an unpinned `EmbeddingGemma2TextModel` quote, the `embedding_size` range, the fake `/pooling`
  range, and the observe probe's `mrl_dim` width bound.
- **Lens B, regressions and hygiene: VERDICT FAIL.** Same two majors; plus the table's route column was ambiguous
  for the multi-vector rows, the pplx-v1 reference is 92 content lines, and "decision 39" is not in `00-MASTER.md`.
  Scope clean (only the spec in the diff), `public-names-step` clean, no CHANGELOG owed (handover-only, no public
  surface), `tests/contract tests/docs` 270 passed / 52 skipped.
- **Fixes:** every agreed finding fixed in `e9ead059` (spec text only).

Round 2, one fresh confirmation verifier (same model and thinking), lens A+B: **VERDICT PASS.** It re-derived the two
majors independently, verified every fix-introduced citation against the tree, fresh Hub bytes at the pinned shas and
the vLLM v0.31.0 tag, confirmed 22/22 coverage and that no new wrong citation was introduced. Three trivial polish
items (G3 wording, the fake `/pooling` range one line late, the Octen 0.6B/4B note line) were fixed in `868447e3`.

## Checks

Last commands and their result lines:

- `bin/gate lane/mrl-cards` (spec head `868447e3`, slot 3): `GATE: PASS`; `ruff-check exit=0 All checks passed`;
  `ruff-format exit=0 532 files already formatted`; `basedpyright exit=0 0 errors, 0 warnings, 0 notes`;
  `pytest exit=0 3203 passed, 82 skipped`; `contract-docs exit=0 270 passed, 52 skipped`; `mkdocs exit=0`;
  `test-pkg exit=0 570 passed, 225 skipped`; `recipes exit=0 recipes: no failure outside the baseline`;
  `vllm-pkg exit=0 1 passed`; `vllm-models exit=0 70 passed, 7 skipped`; `run_all exit=0 leaderboards: 1022 checks,
  987 match, 35 known deviations, 0 failed; human study: 67/67; external LLM judges: 82/82`; `public-names exit=0
  public-names: clean (2 baselined hits remain)`; `clean exit=0 clean`.
- Round-1 verifiers: Hub re-fetch of all 22 pinned revisions (all pins match the API `sha` except the declared
  pplx-context drift); targeted vLLM v0.31.0 reads; `tests/contract tests/docs` 270 passed / 52 skipped;
  `public-names-step` clean.
- Round-2 verifier: independent re-fetch and re-read of every fix-introduced citation; worktree clean at `e9ead059`.

## Open questions

The spec's §5 lists five decisions for the owner / `mrl-core`:

1. Range cards (Qwen3-Embedding 32..W, Qwen3-VL-Embedding 64..W): declare `is_matryoshka: true` with no set (the
   engine accepts 1..W) or mirror the card's range as a discrete `matryoshka_dimensions` (loses the floor, pins the
   tested grid)?
2. `zembed`: refuse MRL in the recipe validator (projection kind, chain unloaded) or build a plugin that loads
   `projections.safetensors`?
3. `pplx-embed-v1`: treat the card's `MRL: Yes` as unbacked (kind `none`) until a set and a served path exist, or
   add a plugin and adopt prefix truncation?
4. Does the engine-side path become the one home for MRL cuts where the card's set exists, with `mrl_dim` reserved
   for `/pooling` and the ex-post sweep?
5. `pplx-context` and `topk`: adopt MRL now (`is_matryoshka` + `pooler_config.dimensions`, or `mrl_dim`) or leave it
   to the wave? Both cards give an explicit set.

## CHANGELOG entry

None. The lane is spec-only under `handover/`, which is deleted before the release and excluded from the docs tests;
no public name, CLI, exit code or schema changed, so no `## Unreleased` entry is owed.

## Public surface changes

None.

## Files outside scope

None. Only `handover/specs/mrl-cards.md` and this report.

## For the next lanes

- **`mrl-core`**: the spec's §2/§3 is the per-variant input to the MRL fields and the one MRL head home; the engine
  path for every dense card needs `serve.hf_overrides: {is_matryoshka: true}` (plus `matryoshka_dimensions` where the
  card gives a set); `/pooling` recipes use `pooler_config.dimensions` or the client `mrl_dim`; `zembed` must be
  refused, never cut; the product still has no `matryoshka_dims` field, no `mrl_cut` mechanism record and no dense
  `mrl_dim` (G8).
- **Recipe lanes (`rfam`, `rec-harrier`, `rec-egemma2`, the new sizes)**: the §2 table is the card-side input.
  `jina-v5-text-nano` is natively served through the encoder dispatch (`is_decoder: false`); `embeddinggemma-2` and
  `pplx-embed-v1` need a plugin or a newer engine (no v0.31.0 class); the pplx-context recipe pin drifted from Hub
  `main`, but its Matryoshka section and `embedding_dim` are unchanged there (G10).
