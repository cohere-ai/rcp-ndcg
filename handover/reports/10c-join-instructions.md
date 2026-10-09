# Lane l10c: workstream 10 C2 and C3 — one MTEB join for title and body; the two instructions

## Status

DONE. The gate passes on the merged tree and on the lane's final tip (`GATE: PASS`; the code tip is `301b6856`,
the report commit is the tip the gate ran on), `run_all` is unchanged (1022/987/35/0, 67/67, 82/82), and three
verifier rounds ran: round 1 (two fresh lenses, both PASS after the operator's review findings M8–M11 and their
minors were fixed), round 2 (two fresh lenses on the retrieval review's findings B1–B8: correctness PASS,
hygiene FAIL on a MAJOR — committed conflict markers in `docs/data.md`, inherited from the merge's upstream
side — fixed by merging the current `rfc-0001`), round 3 (one fresh confirmation verifier, both lenses: PASS).

Base: `rfc-0001` tip `89e7a3b6` (w09 and l10a merged). Merged `rfc-0001` three times: `5b4851cd` (the first
review's tip), `cd371001` (B6: the 66-commit catch-up) and `025fc79d` (the recipe-families restructure and the
`docs/data.md` marker fix). Every merge was re-run through the whole bar.

## Commits (lane branch `lane/l10c`)

| Commit | Subject |
|---|---|
| `71cbcb76` | The one join and the two instructions in the data layer: `Document.model_content` (MTEB's `(title + " " + body).strip()`, the body alone without a title; `DocumentTitle`; a declared `title: separate` takes the title as its own part), `Query.format_query`/`format_content` (the task instruction's generic prefix; the per-query instruction appended exactly as mteb's dataloader appends it), `Dataset.task_instruction_for(side)`, and the Hub reader's lift of a uniform per-query instruction (refusing a subset that instructs only some of its queries) |
| `a4e94860` | The formatting stages: `title` on every role config, `instruction` on the embed/pool configs, the `normalise` stage's placement (the generic fold, or the template's own span; never both), the rerank client's two instructions, and the refusal of a document-side instruction with no span |
| `6070feea` | The retrieval and judging steps consume it: index/search/retrieve/rerank and the judge read each document as MTEB's dataloader does, the task instruction reaches the encoder, the reranker and the judge's query slot, and the checkpoint key and the judging identity record it |
| `42459746` | The harness consumes the two instructions (the conformance rerank send passes the case's instruction as the task instruction; the fold helper renders the generic prefix) and the fingerprint classifies the new `title` client field |
| `9d1d5034` | Docs and the public surface: the one-home row, the CHANGELOG, `REPRODUCIBILITY.md`, the role-config pages, the regenerated schemas and contract snapshot, the `datasets` dependency-gate row |
| `6c1d1cbd` | The review's M8–M11 (the adapter-gated instruction field, the index identity covering the document-side instruction, the sparse path following mteb's BM25, `instruction: None` = undeclared) |
| `0d22f71e` | The round-1 verifiers' minor findings (`TemplateSpec.places`, `instruction: none` beside a span refused, the empty-query refusal before the task frame, the rerank document-side refusal, the Hub completeness over kept queries, the judging identity keying only a declared instruction, the checkpoint-key test) |
| `5b4851cd`, `cd371001`, `025fc79d` | Merges of `rfc-0001` (the first review's tip, the B6 catch-up, the recipe-families restructure with the `docs/data.md` marker fix) |
| `64d09574` | The lane report (the first revision) |
| `94e04bac` | The retrieval review's B1–B5, B8 (below) |
| `73ca28ce` | Docs and CHANGELOG for those findings |
| `301b6856` | The round-2 verifiers' findings (the run-identity docs, the BM25 tokenisation scoping, the other-kind index cleanup, the recipe loader's messages refusal, the family-layout tests, the AGENTS row) |

## What changed (per brief item)

**1. One join, MTEB's.** The rule lives once, in `rcp_ndcg_core._records`: `mteb_document_text(title, body)`
(`(title + " " + body).strip()` when the title is non-empty, else `body.strip()`) and
`Document.model_content(title=...)`. A `title: separate` (declared on the role config) sends the title as its
own leading text part, the body untouched. Every place a document is materialised for a model reads it that
way: `retrieval.index`/`search`/`retrieve`/`rerank`, the judge's windows (`_dataset_rows`), and the derived
ranking shape (`SourceReader.examples`). Nothing joins at read time (l10a's fields stay the data's). The
sparse (BM25) path is the declared exception: mteb's own BM25 is not a served model and reads no dataloader,
so it follows the TEXT mteb's BM25 indexes, byte for byte (`title + "\n" + body`, both as given, and the
per-query append alone for the query) — stated in `REPRODUCIBILITY.md` and `docs/concepts/retrieval.md`, with
the scoring scoped (`bm25s` on both sides; for the shipped English config the tokenisation coincides with
mteb's `BM25Tokenizer` for `eng`).

**2. Two instructions, two fields.** The **task instruction** (`Dataset.task_instruction`, plus
`task_instruction_for(side)`) is placed by the role config's `instruction` mode: `fold` is the generic
default (`Task: <instruction>\nQuery: <text>`, the core record's one formatter), a template's `instruction`
span carries it instead (the template's own placement wins; a rerank span is rendered by the engine from the
request's `instruction` field, so a wire without that field refuses the combination at construction, and
`instruction: none` or `request_shape: messages` beside a span is refused too), and `instruction: none` sends
none. The **per-query instruction** (`Query.instruction`, mteb's InstructionRetrieval data) is appended
exactly as mteb's dataloader appends it (`query + " " + instruction`) — data, never folded as a task
instruction, and never both appended and slotted. Each reaches the model once. The Hub reader **lifts** a
uniform per-query instruction to `task_instruction` (BRIGHT's per-domain instructions; the card's declared
`instruction` feature or the first queries file's schema says whether the column exists) and refuses a subset
that instructs only some of the queries mteb keeps; instructions that differ per query stay per query
(Core17/FollowIR). The judge reads the query side; a reranker refuses a document-side instruction (its slot is
the query's); an encoder takes one through the template's document span, refused where there is none.

**3. The run identity.** The new fields are CONTENT on the role configs (`title` on `Endpoint`, `instruction`
on the embed/pool configs), so every step identity, index identity and checkpoint key covers them; the index
identity also covers the resolved document-side task instruction; the rerank checkpoint key covers the task
instruction; the judging dataset identity records a declared `task_instruction` (and names an in-memory
dataset by its content instead of crashing on its absent URI); the judging identity and every run-step
identity carry the text-formatting rule's version (`TEXT_FORMATTING_VERSION`). The **behaviour fingerprint is
unchanged (`rcp-fp/3`)** — see Open questions.

**4. `run_all` unchanged**: 1022/987/35/0, 67/67, 82/82 on every run (the experiments score stored runs, so
the join cannot move them; verified after each commit and on every merged tree).

## The operator's review findings (M8–M11, fixed in `6c1d1cbd`)

- **M8** — a template `instruction` span is rendered by the engine from the request's `instruction` field:
  `_sends_the_instruction_field` is gated on the adapter's `HAS_INSTRUCTION_FIELD`, and a span on a wire
  without the field is refused at construction (the adapter names the field and the ways out).
- **M9** — `_identity` folds the resolved document-side task instruction into the index identity.
- **M10** — the sparse path follows mteb's own BM25's text; `REPRODUCIBILITY.md` states the divergence from
  the paper's blank-line join. No anchor test pins a BM25 number and none moved.
- **M11** — `instruction: None` on an embed/pool endpoint means UNDECLARED: a request carrying a task
  instruction is refused, naming `fold`/`none`.

## The retrieval review's findings (B1–B8, fixed in `94e04bac`, `73ca28ce`, `301b6856`)

- **B1/V3** — the refusal stays, and every paper config that encodes or scores a query now declares the
  policy with the value the paper's code used. Evidence: the pre-unified dense path passed the bare query
  (`encoders/hosted_api.py` prepends only the profile's `query_prefix`; the old `_api.py` folded a PER-QUERY
  instruction, and the paper's datasets carry no instruction column), and the hosted rerank path sent
  `{"model", "query", "documents"}` (`external_rerankers.py`'s `_HostedRerank._payload`). So
  `experiments/paper/retrieval/{octen,cohere_embed_v4}.yaml` and the four hosted rerank configs declare
  `instruction: none` (the ctxl/qwen3/jina/zerank rerank configs already did; `bm25s.yaml` takes none by
  construction and says so). Test: the paper's dense configs `index`+`search` a BRIGHT-like instructed dataset.
  **The shipped embed and multi-vector recipes still declare none** (rfam's files): they need `instruction:`
  — `none` for every one of them, since each bakes the card's own instruction as fixed template segments
  (`jina-embeddings-v5-text-small`, `octen-embedding-8b`, `pplx-embed-v2-context-9b-preview`,
  `pplx-embed-v2-late-0.6b`, `qwen3-embedding-0.6b`, `qwen3-vl-embedding-2b`, `topk-embed-v1-small`,
  `zembed-1-embedding`); `fold` only for a recipe that wants the dataset's task instruction in the generic
  frame. Every rerank recipe already declares `none`.
- **B2/V2** — the sparse corpus builder reads a `content`-carrying row's body (`as_content`), never the raw
  `text` field a media row leaves empty.
- **B3** — a `request_shape: messages` recipe with a template `instruction` span is refused (the engine's
  chat template frames the content and cannot render the span), at the product config and at the recipe
  loader.
- **B4** — the run-step identities carry the text-formatting rule's version and the title/instruction policy
  (CONTENT fields); the judging identity carries the version beside the dataset's instruction. A resume check
  stays metadata-only (it never loads the dataset), so the resolved instruction is not read there — see Open
  questions.
- **B5** — the recipe loader and the harness case guard state the new capability (an embed/multi-vector span
  needs `instruction: fold`; the conformance embed/pool send passes the case's instruction).
- **B6** — the branch merged the current `rfc-0001` (66 commits) and then the next tip; no later decision was
  reverted (verified by a verifier's remerge-diff audit: the file set is a pure union, only the CHANGELOG
  hand-resolved).
- **B8** — the BM25 claim is scoped to the text (the scoring is `bm25s` on both sides); a card config that
  only `dataset_info` lists no longer shadows the conventional `{subset}/{part}.parquet` path; an index
  rebuild clears the other kind's artifacts (`vectors.npy`/`offsets.npy` on a sparse rebuild, `bm25s/` on a
  vector rebuild) and a stale `offsets.npy` when the new vectors are single.

## Verification

**Round 1** — two fresh verifiers (DeepSeek-V4.1-flash, xhigh), lenses A (correctness) and B
(regressions/hygiene), in parallel.

- **Lens A: PASS.** Independently reproduced every brief item and M8–M11: a 104-document/8-query sweep
  against mteb's own `_corpus_to_dict`/`_combine_queries_with_instruction_text` (byte-identical; the only
  divergence is the documented safe one: a `None` title is no title for us, mteb raises); mteb's own
  `BM25Search.index` corpus texts against `_sparse_corpus`; each instruction exactly once on the wire; the
  reader lift (uniform/differing/mixed, Core17 unbroken); the checkpoint key and index identity; the full bar.
  Six **minor** findings, all fixed in `0d22f71e` (listed in the commit).
- **Lens B: PASS.** The full bar green, the join gate re-run with real mteb, and **nine mutation tests** in
  throwaway worktrees (dropping the title from the join, defaulting `None` to `fold`, dropping the document
  instruction from the identity, the dataloader join on the sparse path, removing the adapter's refusal,
  neutering the lift, never refusing a mixed column, the `Task:` frame on a BM25 query, folding beside a span)
  — each made the new tests red. It verified the fingerprint claim independently (all shipped recipes'
  fingerprints byte-identical at base and tip), the schemas/snapshot regeneration, the commit identity and
  the absence of local/private paths. Four **minor** findings, all fixed in `0d22f71e`.

**Round 2** — two fresh verifiers on the retrieval review's fixes (`73ca28ce`).

- **Lens A (correctness): PASS.** Own reproductions of B1–B5/B8 (the paper configs' values against the
  pre-unified code; a media/OCR row reaching the sparse index by its part text; the messages refusal; the
  identity version with a real bump; the loader/guard refusals; the card fallback and the offsets cleanup);
  run_all exact; the join gate 17 passed; the bar green. Four **minor** findings, fixed in `301b6856`: the
  runs.md resume sentence overpromised; the BM25 tokenisation contrast was inaccurate for the shipped English
  config (mteb's `BM25Tokenizer` for `eng` resolves the same bm25s stop list and English Snowball stemmer);
  a sparse rebuild left a stale `vectors.npy` (and a vector rebuild `bm25s/`); the recipe loader restated only
  the `fold` rule.
- **Lens B (regressions/hygiene): FAIL** on one **MAJOR**: `docs/data.md` carried committed conflict markers
  (`<<<<<<< HEAD`/`=======`/`>>>>>>>`), inherited from the merge's upstream side (`c5ccc851`), which
  `rfc-0001` had already fixed in `d54b672c`. Fixed by merging the current `rfc-0001` (`025fc79d`), whose
  resolution the verifier confirmed (no markers anywhere, both sentences of the file kept, the lane's own
  paragraph byte-identical). Three minors fixed in `301b6856` (the AGENTS.md MRL-cut phrase, the runs.md
  grammar, this report).

**Round 3** — one fresh confirmation verifier (both lenses) on `301b6856`: **PASS**. It confirmed the major's
fix with its own greps and a remerge-diff audit (a pure union, only the CHANGELOG hand-resolved), reproduced
the resume behaviour (an unchanged resume skips; a formatting bump re-runs `evaluate` and a full resume
refuses with the actionable hint), compared mteb's `BM25Tokenizer('eng')` output byte for byte with this
package's, exercised the rebuild cleanup both ways, and confirmed the loader/guard refusals and the
family-layout tests. Two non-blocking findings, both fixed after the round: the runs.md sentence now scopes
the universal claim (the `calibrate` step keys on the stores it reads) and the recipe loader's article
grammar is right for `multi_vector`.

**Failing test first (red runs behind the fixes).** The core formatting tests failed with
`AttributeError: 'Document' object has no attribute 'model_content'` (10 failed); the join gate failed on the
data-layer rows (1 of 17); the instruction-stage tests with `TypeError: encode() got an unexpected keyword
argument 'instruction'`; the rerank tests with the old per-query fold expectations; the lift tests with the
reader's `task_instruction` returning `None`; the M8 test with `DID NOT RAISE`; the M11 test with the client
sending the request instead of refusing; the M9 test with equal identities; the M10 tests with the dataloader
join and the `Task:` frame; the paper-config test with `ConfigError` when the declaration was removed (the
round-2 verifier's mutation); the sparse-body, messages-span, offsets and card-fallback tests with their old
behaviour. The verifiers' scratch reproductions are the red evidence for their findings, and their mutation
tables are the red evidence that the new tests can fail.

## Checks (last runs, on the merged tree; the final gate on the lane's tip)

```
bin/gate lane/l10c
  -> GATE: PASS (rev 301b6856): ruff-check/format 0; basedpyright 0 errors; pytest 3490 passed, 97 skipped;
     contract-docs 295 passed, 52 skipped; mkdocs ok; test-pkg 619 passed, 349 skipped; recipes "no failure
     outside the baseline (0 baseline failures remain, 34 fixed; pytest exit 0)"; vllm-pkg 40 passed;
     vllm-models 71 passed, 7 skipped; run_all 1022/987/35/0, 67/67, 82/82; public-names clean; clean tree
uv run --no-sync pytest tests/ -q -n 8 -p no:cacheprovider      -> 3490 passed, 97 skipped
uv run --no-sync pytest tests/contract tests/docs -q            -> 295 passed, 52 skipped
uv run --no-sync pytest rcp-ndcg-test/tests -q -p no:cacheprovider -> 619 passed, 349 skipped
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
  input map of every shipped recipe is unchanged (verified independently: all fingerprints byte-identical at
  base and tip), and the formatting is upstream of the wire — the recorded corpora replay the same requests.
  A schema bump would invalidate every committed corpus without changing a single recorded exchange. If the
  owner wants the formatting rule itself keyed, it is a `rcp-fp/4` change plus a re-recording wave.
- **B4's second half: the resolved task instruction is not read by a run-step resume check.** The check is
  deliberately metadata-only (an existing test pins "one lookup per repository per process"; reading the
  dataset would load a `mteb:` task's whole corpus). The instruction a Hub source carries is part of the
  repository at the resolved commit, and the policy is a CONTENT config field, so both enter the identity;
  a `mteb:` task's prompt comes from mteb's metadata (the review's A5 code-version gap, another lane's) and a
  local dataset's content is in no identity (A4). The judging store does key the resolved instruction.
- **The lift's "mixed subset" reading.** A uniform instruction over every kept query lifts; a table that
  instructs only some of the kept queries is refused; differing-but-complete instructions stay per query
  (Core17/FollowIR). The brief's "refuses a mixed subset" could also be read as refusing differing
  instructions, which would break mteb's InstructionRetrieval data; the chosen reading is documented in the
  reader.
- **`instruction: none` on the shipped embed/pool recipes.** Until rfam declares the field, a dataset
  carrying a `task_instruction` (e.g. `mteb:BrightBiologyRetrieval`, or a lifted Hub subset) is refused by
  those recipes instead of silently folded — M11's intended strictness. The eight recipes and their
  card-implied values are listed under B1.
- **The harness's pairs do not carry a task instruction yet.** The pairs generator samples `hf://` data
  (whose cards carry no instruction), so the wave pairs are unaffected; a future `mteb:`-sourced pairs file
  should write `dataset.task_instruction_for("query")` into the pairs row's `instruction`.
- **A flaky multiprocess-store test.** `tests/judging/test_store_multiprocess.py::test_the_append_holds_the_store_writer_lock`
  failed once in a gate run and once in a verifier's package run on a `TemporaryDirectory` cleanup race
  (`OSError: [Errno 39] Directory not empty`) under load; it passes standalone (5/5) and in re-runs. Worth a
  `tmp_path`-based cleanup wait (workstream 07's QA territory).
- **The gate's first run** failed in its environment step because the invoking shell exported `VIRTUAL_ENV`
  (the gate's `uv sync` targeted the lane's venv); re-run with `env -u VIRTUAL_ENV -u UV_NO_SYNC` it passed.
  Worth pinning in the gate script (`unset VIRTUAL_ENV UV_NO_SYNC`).
- **Base drift.** The lane merged `rfc-0001` three times; each time the resolution was re-verified (round 3's
  remerge-diff audit). The next merge after this report should be a fresh resolution, not a replay.

## CHANGELOG entry

Under `## Unreleased`, grouped as the repository asks; the exact bullets added (see `CHANGELOG.md`):

- **### Public surface** — *One join and the two instructions (workstream 10 C2/C3, owner decisions 27, 33)*:
  `mteb_document_text`, `Document.model_content(title=...)`, `DocumentRow.model_content`, `DocumentTitle`,
  `TEXT_FORMATTING_VERSION`; `Query.format_query(task_instruction=...)`/`format_content(task_instruction=...)`;
  `Dataset.task_instruction_for`; the `title` field on every role config and `instruction` on the embed/pool
  configs; `EmbeddingClient.encode`/`PoolingClient.encode(instruction=)`; `RerankClient.rerank`/`rerank_many`'s
  task instruction; `TemplateSpec.places`; the Hub reader's lift. *The judge's identity records the task
  instruction.*
- **### Fixed** — *The formatting's own edges* (M8–M11, the round-1 verifiers' findings and the review's
  B1–B5/B8): the adapter-gated instruction field; `instruction: none` and `request_shape: messages` beside a
  span refused; the index identity covering the document-side instruction; `retrieval.rerank` refusing a
  document-side instruction; the sparse path reading mteb's BM25 text and a `content` row's body; the
  empty-query refusal decided on the data's query; the Hub reader's completeness over kept queries and its
  card fallback; the judging identity keying only a declared instruction; the embed/pool undeclared-policy
  refusal; the rebuild cleanup; the paper configs' declared policies.
- **### Changed** — *A model's text is formatted where the model's text is formatted*: the corpus
  materialisation and the judge read MTEB's join; the query texts apply the two generic instruction defaults
  once each; the paper's blank line is stated in `REPRODUCIBILITY.md`; the run-step identities carry the
  text-formatting version; the fingerprint stays `rcp-fp/3`.

## Public surface changes

- New public names: `rcp_ndcg_core._records.{mteb_document_text, DocumentTitle, TEXT_FORMATTING_VERSION}` and
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
  embed/pool send passes it to the client; the guard's premise is the new capability).
- `rcp-ndcg-test/src/rcp_ndcg_test/fingerprint.py` — `CLIENT_FIELDS["title"]` (a new client field must be
  classified or the fingerprint refuses to compute).
- `rcp-ndcg-vllm/src/rcp_ndcg_vllm/recipe.py` — the recipe-level restatement of the product's instruction
  rules (span needs `fold`; `messages` + span refused), so a recipe fails at load.
- `experiments/paper/retrieval/{octen,cohere_embed_v4,bm25s}.yaml` and the four hosted rerank configs — the
  declared policies (B1).
- `tests/docs/test_packaging.py` — the `datasets` dependency-gate row for the join gate's mteb query path.

## For the next lanes

- **rfam (recipe-fix)**: declare `instruction:` on the eight embed/multi-vector recipes listed under B1
  (`none` for the card-frame ones). A recipe with a template `instruction` span and no `fold` policy, or on
  the `messages` route, is now refused at load.
- **l10b (MTEB export/scoring)**: the writer/reader round trip must preserve `title` and `task_instruction`;
  `Dataset.task_instruction_for(side)` is the side accessor.
- **Harness/waves**: a pairs file sourced through `mteb:` should write the task instruction into the pairs
  row's `instruction` (the field the harness already treats as the run-level task text); the recorded corpora
  are unaffected by this lane.
- **Anyone adding a reader**: a per-query instruction source that is uniform should go through
  `rcp_ndcg.data.io.base.lift_task_instruction` (one home); the Hub reader is the worked example.
- **The runs layer**: a resume check is metadata-only; the resolved instruction of a `mteb:` task and a local
  dataset's content are still outside the step identity (the review's A4/A5), a product gap for a later lane.
