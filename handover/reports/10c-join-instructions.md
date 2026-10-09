# Lane l10c: workstream 10 C2 and C3 — one MTEB join for title and body; the two instructions

## Status

DONE. The gate passes on the merged tree (`GATE: PASS`, revision `61b3c1d1`, the final tip; the same PASS was
obtained on the merged head `5b4851cd` before the report commit), `run_all` is unchanged
(1022/987/35/0, 67/67, 82/82), and two independent adversarial verifiers (correctness; regressions/hygiene)
both returned **PASS** after the operator's review findings M8–M11 and the verifiers' minor findings were fixed.
(One intermediate gate run on `61b3c1d1` failed the `pytest` step on a single unrelated flake —
`tests/judging/test_store_multiprocess.py::test_the_append_holds_the_store_writer_lock`, a
`TemporaryDirectory` cleanup race (`OSError: [Errno 39] Directory not empty`) under xdist; the test passes 5/5
standalone and the re-run is green. Pre-existing, not this lane's.)

Base: `rfc-0001` tip `89e7a3b6` (w09 and l10a merged). Merged `rfc-0001` once before the report:
`5b4851cd` (the merge brought lane rf-research, judge-gemma, l08-sglang, mrl-cards and the master's decisions
34–41; the merge was conflict-free and the whole bar was re-run on it).

## Commits (lane branch `lane/l10c`)

| Commit | Subject |
|---|---|
| `71cbcb76` | The one join and the two instructions in the data layer: `Document.model_content` (MTEB's `(title + " " + body).strip()`, the body alone without a title; `DocumentTitle`; a declared `title: separate` takes the title as its own part), `Query.format_query`/`format_content` (the task instruction's generic prefix; the per-query instruction appended exactly as mteb's dataloader appends it), `Dataset.task_instruction_for(side)`, and the Hub reader's lift of a uniform per-query instruction (refusing a subset that instructs only some of its queries) |
| `a4e94860` | The formatting stages: `title` on every role config, `instruction` on the embed/pool configs, the `normalise` stage's placement (the generic fold, or the template's own span; never both), the rerank client's two instructions (task and per-query), and the refusal of a document-side instruction with no span |
| `6070feea` | The retrieval and judging steps consume it: index/search/retrieve/rerank and the judge read each document as MTEB's dataloader does, the task instruction reaches the encoder, the reranker and the judge's query slot, and the checkpoint key and the judging identity record it |
| `42459746` | The harness consumes the two instructions (the conformance rerank send passes the case's instruction as the task instruction; the fold helper renders the generic prefix) and the fingerprint classifies the new `title` client field |
| `9d1d5034` | Docs and the public surface: the one-home row, the CHANGELOG, `REPRODUCIBILITY.md`, the role-config pages, the regenerated schemas and contract snapshot, the `datasets` dependency-gate row |
| `6c1d1cbd` | The review's M8–M11 (see below) |
| `0d22f71e` | The verifiers' minor findings (see Verification) |
| `5b4851cd` | Merge `rfc-0001` into `lane/l10c` |

## What changed (per brief item)

**1. One join, MTEB's.** The rule lives once, in `rcp_ndcg_core._records`: `mteb_document_text(title, body)`
(`(title + " " + body).strip()` when the title is non-empty, else `body.strip()`) and
`Document.model_content(title=...)`. A `title: separate` (declared on the role config) sends the title as its
own leading text part, the body untouched. Every place a document is materialised for a model reads it that
way: `retrieval.index`/`search`/`retrieve`/`rerank`, the judge's windows (`_dataset_rows`), and the derived
ranking shape (`SourceReader.examples`). Nothing joins at read time (l10a's fields stay the data's). The
sparse (BM25) path is the declared exception: mteb's own BM25 is not a served model and reads no dataloader,
so it follows mteb's BM25 byte for byte (`title + "\n" + body`, both as given) — stated in
`REPRODUCIBILITY.md` and `docs/concepts/retrieval.md`.

**2. Two instructions, two fields.** The **task instruction** (`Dataset.task_instruction`, plus
`task_instruction_for(side)`) is placed by the role config's `instruction` mode: `fold` is the generic
default (`Task: <instruction>\nQuery: <text>`, the core record's one formatter), a template's `instruction`
span carries it instead (the template's own placement wins; a rerank span is rendered by the engine from the
request's `instruction` field, so a wire without that field refuses the combination at construction), and
`instruction: none` sends none. The **per-query instruction** (`Query.instruction`, mteb's
InstructionRetrieval data) is appended exactly as mteb's dataloader appends it
(`query + " " + instruction`) — data, never folded as a task instruction, and never both appended and
slotted. Each reaches the model once. The Hub reader **lifts** a uniform per-query instruction to
`task_instruction` (BRIGHT's per-domain instructions; the card's declared `instruction` feature or the first
queries file's schema says whether the column exists, so a repository without instructions never pays for a
table read) and refuses a subset that instructs only some of the queries mteb keeps; instructions that differ
per query stay per query (Core17/FollowIR). The judge reads the query side; a reranker refuses a
document-side instruction (its slot is the query's); an encoder takes one through the template's document
span, refused where there is none.

**3. The run identity.** The new fields are CONTENT on the role configs (`title` on `Endpoint`,
`instruction` on the embed/pool configs), so every step identity, index identity and checkpoint key covers
them; the index identity also covers the resolved document-side task instruction; the rerank checkpoint key
covers the task instruction; and the judging dataset identity records a declared `task_instruction` (and now
names an in-memory dataset by its content instead of crashing on its absent URI). The **behaviour fingerprint
is unchanged (`rcp-fp/3`)** — see the note under Open questions.

**4. `run_all` unchanged**: 1022/987/35/0, 67/67, 82/82 on every run (the experiments score stored runs, so
the join cannot move them; verified after each commit and on the merged tree).

## The operator's review findings (M8–M11, fixed in `6c1d1cbd`)

- **M8** — a template `instruction` span is rendered by the engine from the request's `instruction` field:
  `_sends_the_instruction_field` is gated on the adapter's `HAS_INSTRUCTION_FIELD`, and a span on a wire
  without the field is refused at construction (the adapter names the field and the ways out).
- **M9** — `_identity` folds the resolved document-side task instruction into the index identity: two builds
  differing only in it never share an identity, and `search` refuses an index built for the other one.
- **M10** — the sparse path follows mteb's own BM25 (`models/model_implementations/bm25.py`: the corpus as
  `"\n".join([title, text])`, the query as `_combine_queries_with_instruction_text`, no task instruction);
  `REPRODUCIBILITY.md` states the divergence from the paper's blank-line join. No anchor test pins a BM25
  number and none moved (checked: no BM25 test or golden replay uses a title-bearing corpus).
- **M11** — `instruction: None` on an embed/pool endpoint means UNDECLARED: a request carrying a task
  instruction is refused, naming `fold`/`none`, so a recipe that declares nothing is never silently
  re-formatted. **The shipped embed and multi-vector recipes need a declaration** (they declare none today;
  all of them bake the card's own instruction as fixed template segments, so `instruction: none` is the
  honest value for each, `fold` only for a recipe that wants the dataset's task instruction in the generic
  frame): `jina-embeddings-v5-text-small`, `octen-embedding-8b`, `pplx-embed-v2-context-9b-preview`,
  `pplx-embed-v2-late-0.6b`, `qwen3-embedding-0.6b`, `qwen3-vl-embedding-2b`, `topk-embed-v1-small`,
  `zembed-1-embedding`. (Every rerank recipe already declares `instruction: none`.) Recipes are rfam's
  files: none was edited here.

## Verification

**Round 1** — two fresh verifiers on the owner's verifier model (DeepSeek-V4.1-flash, xhigh), lenses A
(correctness) and B (regressions/hygiene), in parallel; each was told the other exists.

- **Lens A: PASS.** Independently reproduced every brief item and M8–M11: a 104-document/8-query sweep
  against mteb's own `_corpus_to_dict`/`_combine_queries_with_instruction_text` (byte-identical; the only
  divergence is the documented safe one: a `None` title is no title for us, mteb raises); mteb's own
  `BM25Search.index` corpus texts against `_sparse_corpus` (identical on five edge cases); each instruction
  exactly once on the wire; the reader lift (uniform/differing/mixed, Core17 unbroken); the checkpoint key
  and index identity; the full bar (tests 3318 passed/94 skipped at the verified tip, contract+docs 289,
  rcp-ndcg-test 570, run_all unchanged, the join gate 17 passed). Six **minor** findings, all fixed in
  `0d22f71e`: `instruction: none` beside a span still sent the field (now refused at the config); the Hub
  reader's column completeness counted queries the qrels cut drops (now decided over the kept queries); the
  rerank path silently dropped a document-side instruction (now refused); the empty-query refusal was
  bypassed by a task frame around an empty query (now decided on the data's query); the checkpoint key's
  `task_instruction` input had no test (now covered); a plain-string `TaskMetadata.prompt` reaches only the
  query side (now stated in `task_instruction_for`'s docstring as the mteb divergence).
- **Lens B: PASS.** Full bar green on the tip (tests 3318/94, contract+docs 289/52, rcp-ndcg-test 570/225,
  run_all unchanged, mkdocs strict, `git status` clean after every run), the join gate re-run with real mteb
  2.21.6 (17 passed), and **nine mutation tests** (in throwaway worktrees, never the lane's): dropping the
  title from the join, defaulting `None` to `fold`, dropping the document instruction from the identity, the
  dataloader join on the sparse path, removing the adapter's wire-fact refusal, neutering the lift, never
  refusing a mixed column, the `Task:` frame on a BM25 query, and folding beside a template span — each made
  the new tests red (7/1/1/1/1/2/1/1/1 tests respectively). It verified the fingerprint claim independently
  (all 19 shipped recipes' fingerprints byte-identical at base and tip), the schemas/snapshot regeneration,
  the commit identity (no co-author/AI attribution), and that no local/private paths entered the diff. Four
  **minor** findings, all fixed in `0d22f71e`: a wrong `httpx.Any` test annotation (basedpyright error outside
  the bar's `include`), two homes for the instruction-span predicate (now `TemplateSpec.places`, used by the
  clients and the adapters), the judging identity keying `task_instruction: None` (now only a declared one),
  and the report itself.

No round 2 was needed (no blocker or major in round 1); the fixes were re-run through the full bar and the
gate on the merged tree.

**Failing test first (red runs behind the fixes).** The core formatting tests failed with
`AttributeError: 'Document' object has no attribute 'model_content'` (10 failed); the join gate failed on
`DocumentRow.model_content`/`QueryRow.format_query(task_instruction=...)` (1 of 17) before the data layer
landed; the instruction-stage tests failed with `TypeError: encode() got an unexpected keyword argument
'instruction'`; the rerank tests with the old per-query fold expectations; the lift tests with the reader's
`task_instruction` returning `None`; the M8 test with `DID NOT RAISE`; the M11 test with the client sending
the request instead of refusing; the M9 test with equal identities; the M10 tests with the dataloader join and
the `Task:` frame. The verifiers' own reproductions (their scratch scripts) are the red evidence for their
findings, and lens B's mutation table is the red evidence that the new tests can fail.

## Checks (last runs, on the merged tree; the final gate on `61b3c1d1`, the tip)

```
bin/gate lane/l10c
  -> GATE: PASS (slot 4, rev 61b3c1d1): ruff-check/format 0; basedpyright 0 errors; pytest 3311 passed, 94
     skipped; contract-docs 287 passed, 52 skipped; mkdocs ok; test-pkg 570 passed, 225 skipped; recipes "no
     failure outside the baseline (34 baseline failures remain, 0 fixed)"; vllm-pkg 1 passed; vllm-models 70
     passed, 7 skipped; run_all 1022/987/35/0, 67/67, 82/82; public-names clean; clean tree
     (the same PASS was obtained on the merged head 5b4851cd; an intermediate run on 61b3c1d1 failed only the
     unrelated multiprocess-store flake noted under Status)
uv run --no-sync pytest tests/ -q -n 8 -p no:cacheprovider      -> 3311 passed, 94 skipped
uv run --no-sync pytest tests/contract tests/docs -q            -> 287 passed, 52 skipped
uv run --no-sync pytest rcp-ndcg-test/tests -q -p no:cacheprovider -> 570 passed, 225 skipped
RCP_EXPERIMENTS_DATA=<data> uv run --no-sync python experiments/run_all.py
  -> leaderboards: 1022 checks, 987 match, 35 known deviations, 0 failed
     human study: 67 checks, 67 match, 0 failed; external LLM judges: 82 checks, 82 match, 0 failed
the join gate (the [mteb] environment, mteb 2.21.6): tests/data/test_mteb_join.py -> 17 passed
uv run --no-sync mkdocs build --strict -d <scratch>/site       -> Documentation built
```

The `[mteb]`-gated join test was run in a scratch venv outside the repository (mteb 2.21.6 + CPU torch, the
lane's sources on `PYTHONPATH`), because the lane's own environment has no `mteb`; CI's `gated` job runs it
with the extra installed (the `datasets` row was added to `DEPENDENCY_GATES`).

## Open questions

- **The behaviour fingerprint stays `rcp-fp/3`.** The new fields are optional and skipped when unset, so the
  input map of every shipped recipe is unchanged (verified independently by lens B: all 19 fingerprints are
  byte-identical at base and tip), and the formatting is upstream of the wire — the recorded corpora replay
  the same requests (the harness passes raw strings to the clients, never a `Document`). A schema bump would
  have invalidated every committed corpus without changing a single recorded exchange. If the owner wants the
  formatting rule itself keyed, it is a `rcp-fp/4` change plus a re-recording wave, not this lane's.
- **The lift's "mixed subset" reading.** Implemented as: a uniform instruction over every kept query lifts; a
  table that instructs only some of the kept queries is refused; differing-but-complete instructions stay
  per query (Core17/FollowIR). The brief's "refuses a mixed subset" could also be read as refusing differing
  instructions, which would break mteb's InstructionRetrieval data; the chosen reading is documented in the
  reader and this report.
- **`instruction: none` on the embed/pool recipes.** Until rfam declares the field, a dataset carrying a
  `task_instruction` (e.g. `mteb:BrightBiologyRetrieval`, or a lifted Hub subset) is refused by those
  recipes instead of silently folded — M11's intended strictness, and the eight recipes are listed above.
- **The harness's pairs do not carry a task instruction yet.** The pairs generator samples `hf://` data
  (whose cards carry no instruction), so the wave pairs are unaffected; a future `mteb:`-sourced pairs file
  should write `dataset.task_instruction_for("query")` into the pairs row's `instruction` (the harness
  already treats that field as the run-level task text).
- **mteb's own BM25 and a lifted instruction.** mteb appends the column per query; our lift clears it and
  sends no task instruction on the sparse path. Byte-identity therefore holds for absent/per-query
  instructions, not for a lifted one (documented in `REPRODUCIBILITY.md`).
- **The gate's first run** failed in its environment step because the invoking shell exported
  `VIRTUAL_ENV` (the gate's `uv sync` targeted the lane's venv); re-run with `env -u VIRTUAL_ENV -u
  UV_NO_SYNC` it passed. Worth pinning in the gate script (`unset VIRTUAL_ENV UV_NO_SYNC`).
- **A flaky multiprocess-store test.** `tests/judging/test_store_multiprocess.py::test_the_append_holds_the_store_writer_lock`
  failed once in a gate run on a `TemporaryDirectory` cleanup race (`OSError: [Errno 39] Directory not empty`)
  while its child writer was still finishing; 5/5 standalone runs and the re-run are green. Worth a
  `tmp_path`-based cleanup wait in that test (workstream 07's QA territory).

## CHANGELOG entry

Under `## Unreleased`, grouped as the repository asks; the exact bullets added (see `CHANGELOG.md`):

- **### Public surface** — *One join and the two instructions (workstream 10 C2/C3, owner decisions 27, 33)*:
  `mteb_document_text`, `Document.model_content(title=...)`, `DocumentRow.model_content`, `DocumentTitle`;
  `Query.format_query(task_instruction=...)`/`format_content(task_instruction=...)`; `Dataset.task_instruction_for`;
  the `title` field on every role config and `instruction` on the embed/pool configs; `EmbeddingClient.encode`/
  `PoolingClient.encode(instruction=)`; `RerankClient.rerank`/`rerank_many`'s task instruction; the Hub reader's
  lift. *The judge's identity records the task instruction* (and names an in-memory dataset by its content).
- **### Fixed** — *The formatting's own edges* (M8–M11 and the verifiers' findings): the adapter-gated
  instruction field; `instruction: none` beside a span refused; the index identity covering the document-side
  instruction; `retrieval.rerank` refusing a document-side instruction; the sparse path following mteb's BM25;
  the empty-query refusal decided on the data's query; the Hub reader's completeness over kept queries; the
  judging identity keying only a declared instruction; the embed/pool undeclared-policy refusal.
- **### Changed** — *A model's text is formatted where the model's text is formatted*: the corpus
  materialisation and the judge read MTEB's join; the query texts apply the two generic instruction defaults
  once each; the paper's blank line is stated in `REPRODUCIBILITY.md`; the fingerprint stays `rcp-fp/3`.

## Public surface changes

- New public names: `rcp_ndcg_core._records.{mteb_document_text, DocumentTitle}` and
  `Document.model_content`; `Query.format_query(task_instruction=)`/`format_content(task_instruction=)`;
  `Dataset.task_instruction_for`; `DocumentRow.model_content`; `QueryRow.format_query/format_content(task_instruction=)`;
  `TemplateSpec.places`; `Endpoint.title`; `EmbeddingEndpoint.instruction`; `EmbeddingClient.encode`/
  `PoolingClient.encode(instruction=)`; `RerankClient.rerank(instruction=, query_instruction=)` and
  `rerank_many(examples, *, instruction=, checkpoint=)`.
- Schemas: `schemas/{index,run-config,judge-config}.v1.json` regenerated (the `title`/`instruction` fields).
- Snapshots: `tests/contract/snapshots/python_api.json` regenerated (the new signatures and `TemplateSpec.places`).
- CLI/exit codes: unchanged.
- Manifest: unchanged (`datasets` already mapped to `[mteb]`; only the gate table gained its row).

## Files outside scope

- `rcp-ndcg-test/src/rcp_ndcg_test/conformance.py` and `cases.py` — the harness must consume the product's
  two instructions (the conformance rerank send passes the case's instruction as the task instruction; the
  fold helper renders the generic prefix). Two lines each; the alternative was a red harness.
- `rcp-ndcg-test/src/rcp_ndcg_test/fingerprint.py` — `CLIENT_FIELDS["title"]` (a new client field must be
  classified or the fingerprint refuses to compute).
- `rcp-ndcg-test/tests/test_conformance.py` — the three tests pinning the old fold; updated with the harness.
- `tests/docs/test_packaging.py` — the `datasets` dependency-gate row for the join gate's mteb 2.21 query path.
- `rcp-ndcg/src/rcp_ndcg/inference/adapters/rerank.py` — the wire-fact refusal (M8) and the one-home span
  predicate; the adapters were otherwise untouched.

## For the next lanes

- **rfam (recipe-fix)**: declare `instruction:` on the eight embed/multi-vector recipes listed under M11
  (`none` for the card-frame ones). A recipe with a template `instruction` span and `instruction: none` is now
  refused at the config, and a span on a hosted wire needs `instruction: field`.
- **l10b (MTEB export/scoring)**: the writer/reader round trip must preserve `title` (the corpus config's
  `title` column) and `task_instruction` (the card's `TaskMetadata.prompt` is not written by
  `push_dataset_to_hub`; our `-excluded`/extras rule applies). `Dataset.task_instruction_for(side)` is the
  side accessor.
- **Harness/waves**: a pairs file sourced through `mteb:` should write the task instruction into the pairs
  row's `instruction` (the field the harness already treats as the run-level task text); the recorded corpora
  are unaffected by this lane.
- **Anyone adding a reader**: a per-query instruction source that is uniform should go through
  `rcp_ndcg.data.io.base.lift_task_instruction` (one home); the Hub reader is the worked example.
