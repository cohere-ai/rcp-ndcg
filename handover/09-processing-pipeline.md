# Workstream 09: one ordered processing pipeline (decision 23)

Read `handover/00-MASTER.md` first. Run right after workstream 05 (before module paths freeze for the release). One
builder, two verifiers. Mostly internal: the public configuration surface changes little; record every public change
in the CHANGELOG.

## Why
The declarations are good (`TextPolicy`, `ChunkPolicy`, `TextBudget`, `ImagePolicy`, `VideoPolicy`,
`ProcessorGeometry`, the recipe client block), and each mechanism has one home (`fit`/`token_prefix` in
`data.preprocess`, media preparation in `data.prepare`, geometry in `data.resolution`, I/O in `data.media`). But the order
in which a role client applies them is implicit: each client's `_prepare` composes normalisation, empty handling, media
preparation, template rendering, the budget fit and the wire lowering itself. Two bugs fixed in lane H came from exactly
that (`empty_doc` applied after the template; the `messages` route framed twice).

## Target
1. **One declared stage order** in `RoleClient`, shared by every role: content normalisation -> empty handling ->
   media preparation -> template render -> budget fit (query share, document cap, pair budget) -> wire lowering. Role
   clients supply only role-specific stages; nothing re-orders them.
2. **The pipeline emits the per-row `ProcessingRecord`** (already introduced in lane H, `client.processing`) as its one
   output describing what it changed; the census, the equivalence harness and the identity read it and never recompute.
3. **A `postprocess` home**: L2 normalisation (today `inference.types.l2_normalize`), chunk-score aggregation (today
   `max_pool_scores_by_document` and `max_pool_rubric_window_by_document` in `data.preprocess`), the client-side MRL cut,
   and late-interaction skip ids (today inside the clients).
4. **Split `data/preprocess.py`** (about 1,700 lines): census file I/O (`drop_torn_last_line`, `census_sink_lock`,
   `append_census_rows`, `read_census_rows`) moves to `storage`; the judge text policy, chunking and the client fit get
   separate modules. Update AGENTS.md's one-home table.

## Gates
Behaviour-preserving except where a test pins a fixed bug: the full suite, every recipe's network-gated test file, the
conformance suite and `run_all` unchanged; a test per stage-order invariant (e.g. `empty_doc` before the frame, one
frame on every route, the record emitted for every change and only for changes). Report in
`handover/reports/09-processing-pipeline.md`.
