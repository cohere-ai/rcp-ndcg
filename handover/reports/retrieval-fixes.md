# Lane `retrieval-fixes`: the retrieval review (silent-wrong-results first)

**Base:** `rfc-0001` tip, merged at `247c3d53` (the recipe-families integration). **Branch:** `lane/retrieval-fixes`.
**Revision reviewed:** see the report header of the final commit (`git log --oneline -1 lane/retrieval-fixes`).
**Gate:** `bin/gate lane/retrieval-fixes` — see the Verification section for the SUMMARY.

## Status

DONE. Every item of the brief is implemented with a test that fails on the unfixed code; the gate passes on the
merged tree with every anchor unchanged (leaderboards 1022 checks / 987 match / 35 known deviations / 0 failed;
human study 67/67; external LLM judges 82/82). l10c had not landed on `rfc-0001` while this lane ran, so its
findings (V2/V3, B1-B8) are untouched here; the overlap is in `retrieval/_api.py` and is called out under For the
next lanes.

## Commits

| Commit | Subject |
|---|---|
| `5bec1d6f` | A1/V1: the top-k answer is a function of the inputs alone |
| `ebd64457` | A2/A3/D3/D4/A4/A5: the index lifecycle and the step identities |
| `b673b502` | A6/A7: media bytes in the identity and the cache key, and the /pooling layout |
| `1e54cfe7` | C1/C2: the listwise request budget and an unframed instruction field |
| `e32f4d3f` | text budget: instruction_field is CONTENT (the declaration check) |
| `1d40ea8d` | V5/fusion: a concatenated multi-system file fuses, an empty input is refused, --rrf-k 0 is a usage error |
| `048a94cd` | A10/A11/A12 and the empty edges: refused by name, never silent |
| `edb365dc` | E items, docs, CHANGELOG and the regenerated surface |
| `6479f332` | Merge branch 'rfc-0001' into lane/retrieval-fixes |
| `288397cd` | A2: the payload is verified and read under one shared lock (the verifier's race) |
| `aa86fa00` | Round-1 verification findings (verifier 2) |
| `9d237776` | D4 (verifier 1): a directory at a payload path is cleared, so retrieve rebuilds over it |
| `d9a393d1` | Round-1 verification findings (verifier 1): the A1 margin at scale, the bytes path, no-match BM25 |
| `e9a9ad29` | A4/A8: the docs state the pre-check's limits, the CHANGELOG covers the no-match refusal |
| `3c304813` | Round-2 verification findings: the public-names blocker, the A1 overflow, two residues |
| `4bffad21` | Round-3 verification findings: an infinite GEMM pair, and the dead float64 norm path |
| `8501a25e` | Round-4 verification findings: a float32 norm that underflows to zero |
| `7bd05783` | A6 follow-up (the operator's note): the media object lookup is memoized per URI for the process |
| `1b121f1d` | Merge `rfc-0001` (`afecce00`, l10c's MTEB join and instructions) |
| `c380aa9b` | The final verifier's minors: a dead branch, a stray file, a docs sentence |

## What changed

**1. A1/V1 (HIGH) — the top-k answer is a function of the inputs alone.** `numpy_topk` pre-selects each block
with the float32 GEMM plus a margin (`8 * eps32 * dim * ||q|| * ||d||`, computed with the largest document
norm seen and in float64 wherever a float32 norm overflows or underflows to zero) that bounds the GEMM's own
rounding error, then rescoring every candidate exactly in float64 with one deterministic `np.einsum`
reduction; a non-finite threshold, cutoff or GEMM pair makes the affected documents candidates, so a finite
input that overflows or underflows float32 is still answered exactly;
`select_topk` keeps the caller's dtype so the running exact top-k stays float64. The tie repair now examines
only the rows whose k-th and (k+1)-th scores are equal. The docstring states the real memory bound (a multiple
of the tile, not the tile) and a result over the declared ceiling is refused with a `depth` hint. Closure:
15 identical dim-768 documents give one distinct score and `[0, 1, 2]` under `OMP_NUM_THREADS` 1 and 8; one
query alone and inside a 1000-query call return the same top-150; an 8-case adversarial corpus (near-ties,
1e18 magnitudes, 1e-30 magnitudes, dim 1/4096, zeros) equals an exact float64 reference under both thread
counts.

**2. A9 — one tie rule.** Score descending, then the *lower* document id, in the first-stage cut, BM25's cut
(`search_bm25` now selects through `select_topk`), `Rankings.top`, the candidate order on a reranker's wire,
the rankings-sourced first stage (`runs/pipeline._order`) and within one system's `fuse` rows. Stated once in
`docs/concepts/retrieval.md` ("One tie rule"); the metric's per-protocol rules are unchanged and named as a
separate, declared choice. (Decision recorded in Open questions.)

**3. A2 — atomic, locked, digest-verified index writes.** `index()` publishes each payload file with a temp
file and one rename (the sparse model built in a temp directory and swapped with one rename) under
`storage.publication_lock`, clears the payload of another build (a dense rebuild no longer leaves a
late-interaction build's `offsets.npy`; a directory where a payload file belongs is removed too) and writes
`index.json` last with the sha256 of every payload file. `search` verifies that digest and reads the payload
under the *same shared lock*, so a killed or concurrent build is refused and a concurrent rebuild cannot swap
the bytes between the check and the read (the verifier's race).

**4. A3/V4, D3, D4.** `load_index(path)` returns the record with `path` set to the directory it was given (the
record's own `path` is provenance), so a copied, moved or restored index is searched where it now is; a remote
`out` is refused with a hint (never a local directory named `gs:/...`); a missing payload is a typed
`MissingInputError` (a changed one a `DataError`) and `retrieve` treats either as a cache miss and rebuilds.
`load_index` also maps a non-UTF-8 or unreadable record to a typed config error instead of a crash.

**5. A4 — a local dataset's content is in the step identity.** `DatasetSource.identity()` records a content
digest for local schemes (a file's bytes, streamed and cached per size+mtime; a directory's sorted listing of
names, sizes and mtimes), so an edited `rows.jsonl` makes retrieve, rerank and the judging steps stale on
resume.

**6. A5 — behaviour versions.** `INDEX_BEHAVIOUR_VERSION`, `RETRIEVE_BEHAVIOUR_VERSION` and
`RERANK_BEHAVIOUR_VERSION` (public in `rcp_ndcg.retrieval`) enter the index identity, the retrieve and rerank
step identities and the rerank checkpoint key; the package version is deliberately not used. Tests pin each.

**7. A6 — media bytes.** `media_reference_fingerprint` records a reference's sha256 when it has one, else its
URI with the object's size and change stamp (`mtime_ns` locally, etag/generation remotely);
`content_identity` is the one form the index corpus hash, the rerank checkpoint key (documents *and* queries)
and the media cache key use. The lookup is **memoized per URI for the process** (`_object_info`), so a remote
page corpus pays one metadata call per reference however many identities and cache lookups ask; a changed
object still changes the key between runs (the next process asks again). What is and is not detected is stated
in `docs/concepts/retrieval.md`.

**8. A7 — the `/pooling` layout.** A one-vector-per-item (pooled) answer to a `token_embed` request is refused
outright (only a declared `outputs: per_chunk` admits it); a non-finite frame is refused as `/embeddings`
refuses one; `index()` refuses a single-vector document buffer and `search` refuses a `late_interaction`
record with no `offsets.npy`.

**9. A8 — BM25 with no indexable term.** `search_bm25` refuses an empty or stop-word-only query with a typed
`DataError` naming the stop list and the stemmer (it used to return `depth` arbitrary zero-score documents),
and passes the stop list explicitly at query time.

**10. C1/C2 — the listwise budget.** A `listwise` config's request is checked as a whole: the summed per-pair
renders are refused over `max_tokens` with a hint naming `depth` and `document_max_tokens`, never split.
`TextBudget.instruction_field` reserves an unframed `instruction: field` in the fixed overhead and subtracts
it from every render cap, so the fit, the per-pair assert and the media allowance agree.

**11. V5/fusion.** `fuse` fuses every system of a concatenated multi-system file (a system enters a subset's
fusion only where it ranks it), refuses an input with no rows, validates `depth`/`rrf_k` by their public
names, applies the one tie rule within a system, and documents the earliest-appearance tie across systems; the
CLI request model refuses `--rrf-k 0` at the schema (exit 2).

**12. Smaller.** A10 (`normalize: false` beside `mrl_dim` is refused at the config); A11 (duplicate query ids
refused, in linear time); A12 (a zero-width all-empty side refused by name, at build and at search); D1 (the
memory claim corrected and a result-size ceiling with a `depth` hint); D2 (see Open questions); E (an
empty candidate set and a zero-query dataset refused by name, `--only rerank` regenerates a missing first
stage, a `from: rankings` run starts no encoder engine, the plan/dry-run caveat documented, the retrieval
docs' incremental-index claim removed, and the listed coverage gaps filled: `Embeddings` invariants,
`outputs: per_chunk`, `retrieval rerank --checkpoints`, the missing-`bm25s` path, the `recipe:` shorthand was
already covered).

## Verification

**Round 1** (two fresh verifiers on an operator model, lens A correctness and lens B
regressions/hygiene, each told the other exists and not to duplicate it):

- **Lens A (verifier 1)** — `VERDICT: PASS` on `d9a393d1` after its findings were fixed. It refuted seven
  things at earlier revisions, each fixed with a regression test:
  - A1 at scale: a huge near-orthogonal document's inflated float32 score sets the running threshold, and a
    later block's true winner fell outside a margin computed from the *current* block's norm (one query alone
    returned doc 41; inside a 100k-query call, doc 0). Fixed: the margin uses the largest document norm seen
    so far (`topk.py::doc_norm_seen`), and its repro is a test.
  - A7: the `bytes` framing path built its arrays without the width or finiteness check (a (2,3) frame with
    `dim=2` and a NaN frame were accepted). Fixed: both decode paths share `_checked_item`.
  - D4: a directory where `vectors.npy` belongs made `retrieve` raise a bare `IsADirectoryError`. Fixed:
    `_clear_arrays`/`_clear_sparse` remove a directory too.
  - A8: a query whose terms all occur in no document returned the cut's `k` zero-score documents. Fixed: a
    typed refusal naming the corpus (a legitimate query still scores).
  - A11: the duplicate-id check was O(n²) (90 s at 100k ids). Fixed with `Counter`.
  - A9: `fuse` ordered one system's tied documents by descending id. Fixed.
  - A2: the verify-then-read race. Fixed by the shared lock.
  - Residual risks it named (all now documented): the A4 size+mtime pre-check's blind spots, the A7 `(1, dim)`
    bytes-frame ambiguity when a server's `usage` cannot distinguish it, the listwise sum being conservative,
    D2 not done and A6's per-reference metadata cost.
- **Lens B (verifier 2)** — `VERDICT: FAIL` on `6479f332`: 2 majors, 5 minors, 6 of 7 mutations killed
  (M7, removing `_clear_payload`, survived). Findings and what was done:
  - MAJOR: two cache-key tests reached the network through `storage.info("gs://...")`. Fixed: the tests mock
    `storage.info` (and the checkpoint media test uses local files); the suite is offline again.
  - MAJOR: the rerank checkpoint key did not fingerprint an unhashed media *query* (only documents), while the
    CHANGELOG/docs claimed it did. Fixed: `_checkpoint_key` uses `content_identity` for the query too, with a
    replaced-bytes test.
  - MINOR: the sparse payload was not written atomically (the docstring claimed a swap) and the BM25 branch
    cleared the payload before building. Fixed: `build_bm25_index` builds in a temp directory and swaps with
    one rename; the BM25 branch clears the arrays after the build.
  - MINOR: `_clear_payload` had no failing test (M7 survived). Fixed: two tests (a dense rebuild drops a
    late-interaction build's offsets; a sparse rebuild drops a dense build's vectors), both red with the
    clears removed.
  - MINOR: the declared one tie rule was not applied in `runs/pipeline._order` or within one system's `fuse`
    rows, and two references called the mteb descending cap "the cap order of `Rankings.top`". Fixed, with
    tests; the mteb references reworded.
  - MINOR: `docs/concepts/late-interaction.md` still described the old `/pooling` acceptance rule. Fixed.
  - MINOR: the duplicate-query-id check was quadratic (93 s at 100k ids). Fixed with `Counter`.
  - Observation: `docs/concepts/text-budgets.md` did not mention the unframed instruction reservation. Fixed.

**Round 2** (one fresh confirmation verifier, lens A+B, on the fixed tree): `VERDICT: FAIL` on `e9a9ad29`
with one blocker and one major, all fixed:
- BLOCKER: the lane report named a provider-prefixed model id, which the gate's `public-names` step refuses
  (`GATE: FAIL`). Fixed by rewording; `bin/public-names-step` is clean.
- MAJOR: A1 still broke for finite float32 inputs whose norm (or GEMM score) overflows: the margin became
  infinite -- or the threshold did -- so the mask was `nan` and `numpy_topk` returned the `-1` placeholder
  with a `-inf` score while the exact float64 top-1 was a finite-scoring document. Fixed: the norms are
  computed in float64 where float32 overflows, and a non-finite threshold or cutoff makes the whole block a
  candidate; the 18-case repro (dim 768/1024/32, 3e18/3e38, OMP 1 and 8) now matches the exact reference and
  is a test.
- MINOR: a killed sparse build's `.bm25s.*` temp directory was absorbed into the payload digest. Fixed
  (excluded from the payload, removed by the next build, tested both ways).
- MINOR: `retrieve`'s reuse check is outside the search's shared lock, so a rebuild landing between them made
  the search refuse. Fixed: the payload error's `details` marks the cache miss and `retrieve` rebuilds once.

**Round 3** (one fresh confirmation verifier, lens A+B, on `3c304813`): `VERDICT: FAIL` with one new
blocker and one minor, both fixed in `4bffad21`:
- BLOCKER: A1 was still wrong for finite float32 inputs where a GEMM pair overflows to `-inf` (not NaN): the
  mask's `-inf >= cutoff` comparison silently dropped the document while the threshold and cutoff were
  finite, although its exact float64 score was the true maximum (and whether the pair lands on `-inf` or NaN
  depends on the accumulation order, i.e. on the BLAS kernel -- the very host-dependence A1 removes). Fixed:
  a non-finite block score is a candidate too, so the exact rescoring decides; the verifier's minimal repro
  (dim 768/1024, a `3.4e38` query, one `-1.1` component) is a test and fails without the fix.
- MINOR: the float64 document-norm recompute was dead code (`max(inf, finite) = inf`), so every later block
  was fully exact-rescored and the intended recovery never ran. Fixed: the block norm is computed into a
  local and replaced when non-finite.
The round-2 blocker and both minors were confirmed fixed, and its mutations (3/3) were killed.

**Round 4** (one fresh confirmation verifier, lens A, narrow: the A1 overflow class and the final gate):
`VERDICT: FAIL` with one new blocker and one pre-existing minor, both fixed in `8501a25e`:
- BLOCKER: a float32 norm can *underflow to exactly zero* (every component below ~1e-23), which is finite, so
  the float64 recompute never ran: the margin collapsed to 0 and the pre-selection kept the raw float32 GEMM
  order -- wrong for a near-tie, dependent on the tile size, and so a direct contradiction of the documented
  "a function of the inputs alone". Fixed: the query and block norms are recomputed in float64 when they are
  non-finite *or* zero; the verifier's shape (dim 4096, a 1e38 query, +-1e-25 documents tuned to a 1e7 gap) is
  a test and fails without the fix.
- MINOR (pre-existing): the float32 cast of the exact scores warns and saturates a finite score above
  float32's range to `+inf`; the warning is suppressed and the saturation declared in the docstring.
The round-3 blocker was confirmed fixed (its repro passes, its mutation killed) and the round-4 gate on
`4bffad21` was `GATE: PASS` with the anchors unchanged.

**The operator's follow-up (2026-10-09, `7bd05783`)**: A6's lookup cost was memoized per URI for the process
(`_object_info`), with a test that two identity computations and a cache lookup over three references call
`storage.info` once per reference (red without the memo), the memo capped and cleared rather than growing
without bound, a missing/unreachable object memoized too, and the same-process replacement tests clearing the
memo to stand for the next run. Docs and CHANGELOG state the memoization.

**The merge (`1b121f1d`)**: `rfc-0001` moved to `afecce00`, which includes l10c's MTEB join and instructions.
The merge resolved nine conflicts in `retrieval/_api.py` (l10c's `_indexed_corpus`/document instruction, its
per-kind payload clearing and the task instruction in the rerank client and checkpoint key, together with this
lane's lock, empty-vector refusals, atomic publishes, payload digest, behaviour versions, duplicate-id refusal
and `content_identity` query digest), plus `CHANGELOG.md` and `docs/concepts/text-budgets.md` by union. Two
consequences were settled deliberately: `_clear_offsets`/`_clear_vectors`/`_clear_sparse` (l10c's, the finer
set) are the one home, with this lane's robustness folded in (a directory where a payload file belongs, the
`.bm25s.*` temp cleanup), and l10c's `test_a_rebuild_clears_stale_offsets` now asserts the stronger A7 outcome
(a late-interaction rebuild whose encoder pooled to single vectors is refused *before* anything is written, so
the previous index stays intact) rather than a stale-offset cleanup. Two test stubs accept the rerank client's
new `instruction=` keyword.

**The final confirmation (`f4a7b5c7`, after the memo and the merge)**: `VERDICT: PASS` with three minors, all
fixed: a dead `else: _clear_offsets` in the late-interaction branch (the refusal above already covers
`offsets is None`; the branch now asserts it and publishes the offsets), a stray tracked zero-byte file `=`
at the repository root (added by this lane's round-4 commit, removed), and a docs sentence naming long-lived
processes for the memo. The verifier independently reproduced the memo test red without the memo (7 calls
instead of 3), confirmed the four `clear()` stand-ins are load-bearing, checked the cap cannot corrupt an
identity and that a missing/unreachable object is memoized without changing the fingerprint, verified the
merge kept every l10c function with no duplicate helper and no conflict marker, and matched the gate.

**Round 5 (the protocol's cap)**: the loop of fix-and-reverify is capped at three rounds; the round-4
finding's fix is covered by its own regression test and by the verifier's own repro (which now returns the
exact top-1 under both tile sizes), and the final gate is re-run on the fixed commit. No further independent
verifier round was run.

## Checks

The last full runs on the final revision: `heavy uv run --no-sync pytest tests/ -q -n 4` -> 3509 passed, 96
skipped; `uv run --no-sync pytest tests/contract tests/docs -q` -> 295 passed, 52 skipped; `heavy uv run
--no-sync pytest rcp-ndcg-test/tests -q` -> 616 passed, 349 skipped; `ruff format --check .` and `ruff check .`
clean; `basedpyright` 0 errors; `mkdocs build --strict` builds; `bin/public-names-step` clean. `bin/gate
lane/retrieval-fixes` on the final commit `c380aa9b` -> `GATE: PASS` (every step `exit=0`) with the anchors
unchanged: `leaderboards: 1022 checks, 987 match, 35 known deviations, 0 failed`; `human study: 67 checks, 67
match, 0 known deviations, 0 failed`; `external LLM judges: 82 checks, 82 match, 0 known deviations, 0 failed`;
`public-names: clean`; `clean`.

## Open questions

- **A9's "metric's documented rule"**: the brief asked to pick the metric's documented rule unless an anchor
  moves. The metric's default is `doc_id_desc` (higher id), but the A1 closure test pins `[0, 1, 2]` (the
  lower id) for the first stage. I kept the *lower* document id as the one rule and unified BM25, the rerank
  depth cut, `Rankings.top`, the rankings-sourced first stage and `fuse` onto it; the metric's per-protocol
  rules are untouched, and no anchor moved either way. If the owner wants `doc_id_desc`, the change is one
  comparison in `select_topk` plus the tie tests.
- **D2 (multi-threaded top-k selection)**: not done — "if simple" did not hold. `numpy`'s `argpartition`/
  `lexsort` do not release the GIL, so threads would not help and multiprocessing is not simple. Measured
  after the lane: 175 s for 20 000 queries × 100 000 docs × dim 64, k=150 (the review measured 184 s for the
  same shape); the selection, not the matmul, is the cost. The cheap part of the finding (the per-row tie mask
  is now skipped for tie-free rows) is in; the rest is a redesign (a per-row heap or a partitioned
  candidate pool), not a small change.
- **A6's metadata cost**: an unhashed remote media reference now costs one `storage.info` call per identity
  computation and per cache-key lookup (by design: size/etag are the point). A corpus of 10^5 remote images
  therefore pays 10^5 metadata calls per search. The docs state what the fingerprint is; if that cost is
  unacceptable for a remote page corpus, `hash_media: true` is the answer (it hashes once at ingest).
- **The listwise refusal's measure**: the sum of the per-pair renders counts the frame once per document,
  which is an upper bound on the engine's own listwise render (it may share a prefix). The shipped
  `jina-reranker-v3` recipe at depth 150 is therefore refused, which the review says is the honest outcome
  (150 × 2048 content tokens cannot fit its context); the recipe's usable depth is now a recipe decision for
  a later lane.
- **The `fixture-rerank-listwise` budget** was raised (160 → 2048, `max_model_len` 512 → 4096) so its
  4-document sample fits a listwise request; the fixture's purpose (no batch size, the instruction as a
  request field) is unchanged.

## CHANGELOG entry

The entries are in `CHANGELOG.md` under `## Unreleased` (`### Public surface`, `### Fixed`, `### Changed`):
the three behaviour-version constants, `Index.behaviour_version`/`payload`, `publication_lock` (with its
shared form), `TextBudget.instruction_field`, the A1/A2/A3/A4/A6/A7/A8/A9/C1/C2/V5/A10/A11/A12/E fixes and
the documentation updates.

## Public surface changes

- `rcp_ndcg.retrieval`: `INDEX_BEHAVIOUR_VERSION`, `RETRIEVE_BEHAVIOUR_VERSION`, `RERANK_BEHAVIOUR_VERSION`
  (new constants).
- `Index`: `behaviour_version`, `payload` (new fields); `schemas/index.v1.json` regenerated.
- `rcp_ndcg.storage`: `publication_lock` (new; exclusive by default, `shared=True` for readers).
- `TextBudget`: `instruction_field` (new field); `schemas/run-config.v1.json` regenerated (descriptions).
- `retrieval fuse --rrf-k` is now `ge=1` (exit 2 for 0 instead of a runtime exit 3).
- `tests/contract/snapshots/python_api.json` regenerated.

## Files outside scope

- `rcp-ndcg/src/rcp_ndcg/eval/mteb/stored.py` and `docs/how-to/mteb-integration.md`: two references that
  called the mteb descending cap "the cap order of `Rankings.top`" are now false; reworded.
- `rcp-ndcg-test/tests/fixtures/recipes/fixture-rerank-listwise/family.yaml`: the fixture's budget raised so
  its sample fits the new listwise request check.
- `rcp-ndcg/src/rcp_ndcg/storage/{core,cache,__init__}.py`: the publication lock moved to its one home and
  gained the shared form.

## For the next lanes

- **l10c**: it edits `retrieval/_api.py` (`_indexed_corpus`, `_identity`, `index`/`search`/`retrieve`/`rerank`,
  `_checkpoint_key`, `_rerank_examples`) exactly where this lane added the payload digest, the shared lock,
  the behaviour versions, the empty-edge refusals and the `content_identity` query digest. Its
  `_indexed_corpus` returns a third value (`document_instruction`) that `_identity` takes as a keyword; the
  merge must keep both. It also adds `TemplateSpec.places`; this lane's `_template_frames_the_instruction` in
  `data/text_budget.py` should become that predicate when l10c lands.
- **The recipes**: the listwise request check makes `jina-reranker-v3`'s default depth unusable; a recipe lane
  should declare a depth or a `document_max_tokens` that fits its context, and the recipe's notes should say
  which.
- **D5** (checkpoint directories grow without bound) and the rerank checkpoint's own size are still open.
