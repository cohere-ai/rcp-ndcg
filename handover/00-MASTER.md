# Handover: finish RFC-0001 on `rfc-0001` and make 0.0.1 release-ready (CPU side)

**State: milestone M3 reached on 2026-10-07.** Everything that was in flight on `wip/*` branches is integrated into
`rfc-0001`, verified and gated (section 3). What remains is well specified and starts from a quiet tree: the layout
move, vLLM-only plus judge recipes, the processing-pipeline refactor, data I/O and MTEB interoperability, the final
docs, QA and release preparation
(section 8). Read this file completely, then `AGENTS.md` (binding), then the workstream prompt you are working on.

`handover/` is temporary scaffolding on `rfc-0001`: read it, write your reports into `handover/reports/`, and delete the
whole directory in one commit before the release (it must not ship; no distribution packages it), together with
the exclusion of `handover/` in `tests/docs/_markdown.py` (`markdown_files`), which keeps these working notes out of
the docs tests while they exist. The reference specifications are in `handover/specs/` (sanitized working notes; a
`<operator-notes>/...` or `<repo>/...` path inside them points to a machine you do not have — the content you need is
in `handover/`).

## 1. Mission and scope

RFC-0001 ("the unified inference layer") replaces every in-process model path with role clients that talk to served
engines over one transport, with client-side text budgets, anchor-preserving cuts, a recipe catalog for serving
models on the stock vLLM image, an equivalence harness that proves a served recipe equals its reference, verified
fake engines that replay recorded engine behaviour, and an unpublished test distribution.

In scope for you: everything that runs on CPU. Out of scope (do not attempt, do not stub, do not claim):
- **GPU work**: serving models on GPUs, the GPU waves (T0-T4), recording observation corpora, filling `pending_gpu`
  expected values, marking recipes `verified`. You may write and CPU-test the code those waves run.
- **Any private repository or private recipes/plugins.** Only public names, everywhere (AGENTS.md placeholders such as
  `gs://YOUR-BUCKET/...`, `registry.example.com`).
- **Releasing**: merging to `main`, tagging `v0.0.1`, publishing. The owner gives that go; you prepare it.

## 2. Repository facts you need on day one


- Three published distributions + one unpublished: `rcp-ndcg-core` (`rcp-ndcg-core`, numpy + pydantic),
  `rcp-ndcg` (root, the pipeline and CLI), `rcp-ndcg-vllm` (`rcp-ndcg-vllm`, deliberately OUTSIDE the uv
  workspace today: recipes, the equivalence harness, the recorder, the wave runner, node scripts), and
  `rcp-ndcg-test` (`rcp-ndcg-test`, a workspace member, never published: reference cases, the conformance
  suite, fakes). The layout move (workstream 05) changes this to per-distribution top-level directories.
- Layering inside `rcp-ndcg/src/rcp_ndcg` (imports point inward only), the one-home table and every rule: `AGENTS.md`.
- The quality bar (run all of it before you call anything done; this is what the previous operator called "the gate"):

```bash
.github/scripts/cpu-env.sh dev docs                      # the locked env with CPU torch (Linux); then always --no-sync
uv run --no-sync ruff format --check . && uv run --no-sync ruff check .
uv run --no-sync basedpyright                            # 0 errors
uv run --no-sync pytest tests/ -q -n 4                   # the whole root suite, offline
uv run --no-sync pytest tests/contract tests/docs -q     # public surface, docs snippets
uv run --no-sync pytest rcp-ndcg-test/tests -q  # the unpublished test distribution
uv run --no-sync mkdocs build --strict
# rcp-ndcg-vllm in its own venv, exactly like CI's vllm-recipes job:
uv venv /tmp/vpkg --python 3.12 && uv pip install --python /tmp/vpkg/bin/python \
    -e rcp-ndcg-core -e . -e 'rcp-ndcg-vllm[test]'
/tmp/vpkg/bin/python -m pytest rcp-ndcg-vllm/tests -q
# the paper's numbers must not move (fetch once, ~150 MB of public data at pinned revisions):
python experiments/fetch_data.py && uv run --no-sync python experiments/run_all.py
#   -> leaderboards: 1022 checks, 987 match, 35 known deviations, 0 failed
#      human study: 67 checks, 67 match, 0 failed;  external LLM judges: 82 checks, 82 match, 0 failed
git status --porcelain --untracked-files=all              # must print nothing: tests write only to tmp_path
```

- Network-gated tests (real tokenizers from the Hugging Face Hub, public models only) run only with
  `RCP_NDCG_NETWORK_TESTS=1`. Run them one file at a time under `timeout 900` with
  `-o faulthandler_timeout=300`; several recipe tests used to hang (see section 7).
- GitHub CI runs on pull requests and `workflow_dispatch` (`gh workflow run ci.yml --ref rfc-0001`), not on pushes to
  `rfc-0001`. Every job was green at `22c0cec`; dispatch it after each integration batch.

## 3. State at M3 (what `rfc-0001` holds; do not redo)

Every earlier `wip/*` branch is integrated or superseded; treat `origin/wip/*` as history only.

| Integrated | What it brought | Report |
|---|---|---|
| Workstream 01 (`wip/fix-inference`) | inference-layer fixes; the key-host rule per replica in the transport (a vendor key only ever reaches its profile's home; a named `api_key_env` only the config's own URLs); URL credentials redacted from every error, log line, engine record and chained cause (`rcp_ndcg.support.urls`); one home for usage folding, reply alignment, batch caps and adapter shape checks; linear media preparation | `reports/01-fix-inference.md` |
| Workstream 02 C | harness G1-G3 (token_ids anchor audit and render check, messages extraction, head edge from the assembled render), the offline fake's exact per-vector draw, plugin wheelhouses from `EXTRA_DIRS`, `RCP_GCLOUD_SDK_DIRS` and hermetic node-script tests | `reports/02-recipe-common.md` |
| Workstream 02 A+B | recipe-common (shared `_served.py`/`_contract.py` helpers, one NOTICE), the recipe line brought up to date | `reports/02-recipe-common.md` |
| Workstream 03 (six families, run as three lanes) | all 18 public recipes and the topk and pplx plugins, deep-reviewed against their model cards at the pinned revisions and the vLLM v0.31.0 source; decision 9 enforced (six references were porting the client's cut; they now render the paper's or the card's own cut, with the over-cap deviation declared); contract tests with mutants; stage 1 on real tokenizers; the R20 pixel pin decided (nested `images_kwargs`) | `reports/03-families.md` |
| Lane H (harness and product follow-ups) | the role clients' per-row `ProcessingRecord` (`client.processing`: `budget_cut`, `query_share`, `document_share`, `empty_doc`, `media_resize`, `media_drop`, with uncut and kept totals including the frame) and per-text gating in the harness; the `last_content` anchor audit; `RerankEndpoint.document_max_tokens`; `ImagePolicy.engine_pixel_pinning`; the embed `messages` route framed once by the engine (one conversation per item); `empty_doc` before the frame; token-accurate fake counts | `reports/H-harness.md` |
| Workstream 04 (`wip/gpu-e2e`, `wip/gpu-quality`, `wip/fake-engines`) | T4 scenarios and the in-pod driver; the observation corpus, the request generator (pairs for all 18 recipes), the T3 quality stage and negative controls; verified fake engines (`rcp_ndcg.testing.engines`), conformance, golden replays, fingerprint `rcp-fp/3`, one corpus reader `rcp_ndcg.testing.corpus`; corpus staleness decided per corpus | `reports/04-corpus-fakes-e2e.md` |
| Final M3 lane (media) | the media equivalence stage (part order, prepared geometry, client vs engine media token counts) and the media request set; control (f) on the nested pin; the client keeps the given text/image order; the declared `add_generation_prompt` (messages route); the checkpoint's own chat template; messages-route stage 2 on the stub; recipe follow-ups (jina-reranker-v3 `document_max_tokens`, qwen3-vl pinning and `image_processor`); the release-flag `stale.json` test; duplicate YAML keys refused | `reports/M-media.md` |

Quality bar on the M3 tip: every gate step of section 2 passes, `run_all` gives 1022/987/35/0, 67/67, 82/82, and the tree
is clean. GitHub CI was not run for this milestone (the work was done locally): dispatch it once the branch is pushed
(`gh workflow run ci.yml --ref rfc-0001`) and fix anything runner-specific.

## 4. Branches

`rfc-0001` is the only branch to continue from. The integration history keeps every lane's merge commit; the lane
branches themselves are not needed. `origin/wip/*` are superseded history: do not merge them again.

## 5. Owner decisions (binding; do not reopen)

1. One `0.0.1` with everything: all 18 public recipes ship, each GPU-validated before the tag (you cannot do the GPU
   part; keep every recipe's `status` honest: `unverified` until a wave verifies it).
2. Everything works on the stock image `vllm/vllm-openai:v0.31.0`: no custom images, only lean installs on the fly.
   Three environments on a node, never mixed: the engine (the image's Python, untouched except a recipe's pure-Python
   plugin wheel installed `--no-deps`), the client venv (the rcp-ndcg wheels, no torch), reference venv(s)
   (`--system-site-packages` over the image's torch; install only what is missing).
3. `rcp-ndcg-vllm` becomes the lean serving package (deps pydantic + pyyaml only; recipes as package data;
   `rcp-ndcg-vllm serve <id> [--dry-run]`; the topk and pplx plugins folded in with lazy registration); the harness,
   recorder, wave runner and node scripts move to `rcp-ndcg-test`; `rcp-ndcg` reads `recipe:<id>` optionally.
   Spec: `handover/specs/layout-move.md` + `docs-firstcontact-answers.md` Q1/Q2.
4. The repository moves to per-distribution top-level directories BEFORE the release (workstream 05).
5. `rcp-ndcg-test` is never published; same version number; its own README, LICENSE, NOTICE.
6. One merged NOTICE, byte-identical in every distribution, attributing every third-party file (incl. vendored model
   card scripts and copied templates, with source, revision and licence).
7. Documents default to 32768 tokens for the judge's text policy (done); documented transparently.
8. Explicit budgets: a self-hosted role config declares `tokenizer` + `max_tokens`; a vendor profile without a tokenizer
   records `budget_source: vendor`; over-budget default `cut` (recorded); chunk aggregation `max`; media are counted in
   tokens; anchors are reserved, never cut engine-side.
9. References stay faithful to the paper or the model card. A reference NEVER ports the product client's cut (settle
   rule, query share) to make an over-cap row pass. Over-cap differences are declared:
   `reference.known_deviations: [anchor_drop_over_cap]` when the reference drops an anchor, `[over_cap_cut_differs]`
   when it keeps the anchors but cuts differently; over-cap rows are then reported, not gated; under-cap rows gate
   exactly. (Harness support: `wip/recipe-sweep` commit `9b6bfb5`.)
10. The sweep-recipes decisions in `handover/specs/triage-decisions.md` (section "sweep-recipes" and the
    "OPERATOR DECISION" entry) are binding: ctxl uses `instruction: none` for all three sizes; one R20 pixel-pin shape
    decided from the vLLM v0.31.0 source; renders by offset cuts, never `decode`; the settle-rule wording read from the
    merged rerank client; etc.
11. No GPU tests in pytest: GPU runs produce observation corpora; verified fake engines replay them on CPU
    (`handover/specs/gpu-validation.md` "The GPU run is also the test suite's audit", `observations-spec.md`).
12. Tests are a first-class QA target: fewer, stronger tests; every test must be able to fail.
13. Release order: core -> rcp-ndcg -> rcp-ndcg-vllm, trusted publishing per package (AGENTS.md "Releasing").
14. **vLLM only.** Go all-in on vLLM; remove every explicit SGLang path (workstream 08).
15. **Recipes for every role, judges included**, schema and catalog in 0.0.1 (workstream 08). A judge recipe carries
    `role: judge` and `client.api: chat`, has no `reference`, and its client block is validated by rcp-ndcg's judge
    config (R30). Catalog: `qwen3.5-397b-a17b-nvfp4`, `gpt-oss-120b`, `qwen3.6-27b-fp8` (the paper's TREC-DL judge),
    `qwen3.8-27b-fp8`, `qwen3.8-flash-next-fp8`, `qwen3.8-flash-next-nvfp4`. The self-hosted presets become recipes
    (`qwen35_397b_fp8` is dropped: the paper never ran it); the hosted `gpt5_hosted` stays a vendor profile.
16. **`rcp-ndcg-vllm` installs very leanly on the stock image**: `pip install --no-deps rcp-ndcg-vllm` changes
    `pip freeze` by exactly that wheel, and `rcp-ndcg-vllm serve <id>` serves any role.
17. **Paper configs point at `recipe:<id>`** instead of duplicating client fields; a content override that differs from
    the recipe is refused.
18. **The recipe file format is the versioned contract** between rcp-ndcg and rcp-ndcg-vllm: every recipe carries
    `schema_version`, its JSON Schema is exported, and rcp-ndcg checks compatibility when it reads `recipe:<id>`. There is
    no lockstep version pin between the two packages (core and rcp-ndcg keep their exact pin).
19. **The recipe schema separates the engine-neutral `client` block from the engine-specific `serve`/`engine` blocks.**
20. **Two fakes, two names and homes:** `rcp_ndcg.inference.fake` (the product's offline engine) and the verified
    **emulators**, which move with the observation corpora into `rcp-ndcg-test` (today `rcp_ndcg.testing.engines`,
    `rcp_ndcg.testing.corpus`, and the corpora under `tests/contract/engines/`).
21. **Rename `rcp_ndcg.llm` to `rcp_ndcg.judging`** before 0.0.1.
22. **`rcp-ndcg-test` is never published**; it lives on GitHub and installs from a git subdirectory, and the docs say so.
23. **One ordered processing pipeline** before 0.0.1 (workstream 09).
24. **Fail-closed credentials**: a vendor profile's default key only reaches that profile's home; a named `api_key_env`
    only the config's own URLs; an injected transport aimed elsewhere gets no key.
25. **`empty_query: send` for the qwen3-reranker family** (the paper's predict formats any query; zerank declares it too).
26. **A settled query makes every pair of its row non-gating** (the paper's own cut, e.g. zerank's, may drop the
    document then); a document's own cut affects only that document's pair.

27. **Titles as MTEB does them** (owner, 2026-10-07: "use the title like MTEB to stay consistent"): `Document.title`
    is its own field (an additive core change) and nothing joins title and body at read time; a model reads
    `(title + " " + body).strip()` (the body alone without a title), byte-identical to mteb's dataloader, and a model
    or recipe may take the title separately (mteb keeps it as its own field too). There is no second built-in join.
28. **MTEB ingestion goes two ways**: a card-driven `hf://` reader following MTEB's own layout rules (no `datasets` or
    `mteb` dependency), and `mteb:<Task>` through mteb itself for the custom-loaded tasks. Export to MTEB refuses a
    non-integral `score` unless an integer grade is given; continuous gains travel in extra columns.
29. **A dataset records its `subset`, `split` and optional MTEB `task`** (provenance); exports key on them.
30. **Duplicates**: exact duplicates fold with a note; conflicting ones are refused unless `duplicates: last` (MTEB's
    behaviour), recorded in provenance.
31. **Maximal forward compatibility with MTEB** (no compatibility owed to the paper's data layout): republish every
    rcp-ndcg dataset in exactly the layout `push_dataset_to_hub` writes (eval split `test`; integer `score`;
    `gain`/`theta` and `-excluded` as extras mteb ignores; exclusions also folded into `top_ranked`); paper configs
    pin commits, so they keep working. Align with the owner's local mteb PR.
32. **Retire the column-heuristics `hf` reader**: the Hub contract is MTEB's layout; other data is converted once.
    Its `document_parts` option and image persistence move into the new reader.

33. **Both kinds of instruction, as separate fields** (owner, 2026-10-08: "we should have support for both types").
    The task instruction (`Dataset.task_instruction`, per task, subset or domain, as mteb's `TaskMetadata.prompt`;
    NanoBEIR, BRIGHT) is model-owned and placed by the recipe's template (generic default: today's prefix). The
    per-query instruction (`Query.instruction`, mteb's InstructionRetrieval data) is kept separate, appended as mteb
    does by default, and combined explicitly by recipes with an instruction slot. mteb itself is inconsistent here
    (workstream 10, C.3); ours is specified once and recorded in the run identity. *The two generic defaults are
    confirmed by the owner (2026-10-08): without a model-specific template the task instruction is prefixed
    (`Task: <instruction>\nQuery: <text>`), and a per-query instruction is appended as mteb does
    (`query + " " + instruction`).*
34. **Recipe families: one family, many sizes, every size its own tested recipe id** (owner, 2026-10-08: "Where
    models have shared abstractions we should have a generalized recipe ... avoid code duplication (e.g. the
    various ctxl sizes) ... But it's good to test all individually."). One family directory per model family:
    `family.yaml` (the shared blocks plus a `variants` table carrying only per-size facts), ONE
    `reference.py` parameterised by the variant, the shared template and reference requirements. Every variant
    resolves to a full recipe id (today's `Recipe` schema, byte-identical resolved contract) that is served,
    contract-tested, stage-1-tested and GPU-validated on its own; family ids are never served. Variant overrides
    are restricted to declared per-size fields (resources, engine limits such as `max_model_len`, dim/dims, max
    token lengths, per-size notes, status); anything else that differs is refused with a typed error naming the
    field. Spec: `handover/specs/recipe-families.md`.

38. **Per-recipe engine images: a digest-pinned nightly is allowed when a recipe needs an engine commit the
    released image lacks** (owner, 2026-10-09, on the `embeddinggemma-2` report). The default stays the released
    image (`vllm/vllm-openai:v0.31.0`); a recipe that needs another image pins it by digest
    (`repository:tag@sha256:...`) and carries its switch-to-release note: the engine commit and the transformers
    floor it needs, and "switch `engine.image` to the first release that carries both and re-validate". The
    harness runs one GPU job per engine image.

## 6. Engineering rules (in addition to AGENTS.md)

- **R30 — consume the product, never copy it.** Harnesses, recipes, references' harness glue, plugins, fakes, cases
  and experiments import the product (`rcp_ndcg`, `rcp_ndcg_core`); a local re-implementation of templates, budgets,
  `fit`, tokenizers, role clients, endpoint configs, identities or errors is forbidden. If the product lacks something,
  add a small public accessor to the product (failing test first) — e.g. `RoleClient.text_budget` was added this way.
  The harness's removed helpers (`budget_of`, `fit_rows`, `fold_query`, `chunk_of`, `template_of`) must not come back.
- Failing test first for every fix; show it red, then green.
- One home per concept (AGENTS.md table). Before adding a helper, `git grep` for one.
- Generated files are regenerated, never hand-merged: contract snapshots and `schemas/`
  (`uv run --no-sync pytest tests/contract --update-snapshots`), the recipe schema
  (`python -c "import json,pathlib; from rcp_ndcg_vllm import recipe_json_schema; pathlib.Path('rcp-ndcg-vllm/schema/recipe.schema.json').write_text(json.dumps(recipe_json_schema(), indent=2)+'\n')"`
  in the vllm venv), the lock and the constraints file (`uv lock`, then
  `python .github/scripts/check_constraints.py --write`). Every public-surface change gets a CHANGELOG entry.
- Every `pytest.importorskip("<module>")` must have a row in `DEPENDENCY_GATES` (`tests/docs/test_packaging.py`) and a
  CI job that opens it.
- Wrap every test, generator or reference run in `timeout`; never re-run a timed-out command unchanged — read the
  stack (`-o faulthandler_timeout=...`) and fix the slow path.
- Docs describe current behaviour only (no future-true sentences; the layout move's docs land with the move).

## 7. Known traps (each cost hours)

- **Semantic drift between branches.** Most `wip/*` branches started from older bases. Check every merge for:
  `RankingExample.query` was removed (use `.text`; `model_copy(update={"query": ...})` silently sets an extra field
  because `extra="allow"` — use `"text"`); `JobStatus.SUCCEEDED` is now `COMPLETED`; `storage.atomic_write` became
  `storage.publish`; `judgement_record_id(..., dataset=...)` is required; `calibration insert --plan` is now
  `--dry-run`; `docs/tutorials/` is `docs/how-to/`; `docs/concepts/serving.md` split into `judges.md` + `runs.md`;
  `TemplateSpec` refuses empty fixed segments (`{fixed: ""}`: use a content-final shape with
  `add_special_tokens: true`); `Recipe` refuses image/video input without client `max_images`/`max_videos`, and a
  rerank recipe with `serve.convert` or an `api` other than `rerank`; recipe `known_deviations` accepts
  `anchor_drop_over_cap` and `over_cap_cut_differs`.
- **Branches carrying an old harness.** A merge from a `wip/*` branch can silently bring back pre-R30 harness code in
  non-conflicting hunks. After every merge: `git diff <before>..HEAD -- rcp-ndcg-vllm/src` and read it.
- **Recipe edits outside their owner.** `wip/gpu-e2e`, `wip/fake-engines` and `wip/gpu-quality` edited recipe files to
  make them load; the family branches own those files — take the family's version.
- **Slow paths.** The offline fake engine's multi-vector pooling hashes once per scalar (`rcp_ndcg.inference.fake`,
  `_unit_vector` -> `fake_uniform`): 2048 dims x 16k tokens takes tens of minutes. Recipe tests bound it with an 8-wide
  probe copy of the recipe; the product fix (one seeded draw per vector) is open (workstream 02). The harness sampler's
  over-length padding was quadratic (fixed in `wip/recipe-sweep` `3c9d4e6`).
- **References bent toward the product.** Several family branches "render the wire's spans" in their references; some
  of that is only the required output FORMAT (the harness compares spans), some ports the client's cut — the latter
  violates decision 9 (workstream 03 lists the suspects).
- **Updates at M3.** The offline fake's per-scalar hashing is fixed (one exact per-vector draw, bit-identical across
  BLAS kernels); the 8-wide probe copies in the topk and pplx recipe tests stay only to bound answer size. The decision-9
  suspects are resolved. A strict `xfail` that pins a gap turns red when the gap is fixed: remove it then.
- **Census and record semantics.** The census counts content tokens; the uncut request totals (frame included) live in
  the `ProcessingRecord`. Declared normalisation and the image policy's own resize are never changes; only removals and
  budget-driven media changes are.

## 8. Remaining workstreams, order and parallelism

| # | Prompt | Depends on | Notes |
|---|---|---|---|
| 05 | `05-layout-move.md` and its "Amendments after M3" | M3 (now) | the quiet window; carries decisions 18-22 |
| 08 | `08-vllm-only-and-judge-recipes.md` | 05 | decisions 14-17 |
| 09 | `09-processing-pipeline.md` | 05 (ideally right after it, before module paths freeze) | decision 23 |
| 10 | `10-data-io-and-mteb.md` (evidence: `specs/mteb-data-model.md`) | 09 (both touch `data/`) | decisions 27-33; core record changes, so before 07's surface freeze |
| 06 | `06-docs-final.md` and its amendments | 08, 09, 10 | |
| 07 | `07-qa-and-release-prep.md` and its amendments | 08, 09, 10 | in parallel with 06 |

The completed prompts `01`-`04` stay for reference; their reports are in `reports/`.

How the previous operator ran each workstream (it worked well; reuse it): one builder per lane in its own git
worktree; then two independent adversarial verifiers with different lenses (correctness against the brief and specs;
regressions, hygiene, R30 and the drift catalog) that reproduce every claim and mutation-test the fixes; a fix round;
a confirmation; then a `--no-ff` merge whose tree equals the gated head. Minor findings are folded into the next lane
instead of extra review rounds.

## 9. Open items register (each with its evidence; carry into the matching workstream)

GPU (owner, before the tag; `RELEASE-CHECKLIST.md` lists them per recipe):
- Every recipe needs its GPU waves; every `status` is `unverified`.
- **Corpora to re-record**, declared stale in `tests/conformance/stale.json` (must be empty at release): 
  jina-embeddings-v5-text-small, jina-reranker-v3, qwen3-reranker-0.6b, qwen3-reranker-4b, qwen3-vl-embedding-2b,
  zerank-1-small-reranker, zerank-2-reranker. Re-keyed on metadata only: qwen3-reranker-8b, qwen3-vl-reranker-2b.
  Listwise replay coverage (jina-reranker-v3) is absent until re-recording. All corpora are provisional (shakedown).
- Media: the qwen3-vl-embedding per-clip video pixel budget differs (engine and client 25,165,824 px vs the card's
  7,864,320 px); decide in the media wave. The ViDoRe golden's retrieval view is waived until a corpus observes a page
  image.
- Media gate (M-media): video is not gated (the generator cannot write a video container; a clip's token count is
  reported only); the media stage needs a GPU run per media recipe to compare the engine's media token count.
- **topk-embed-v1-small cannot send images**: its client refuses media whenever `document_skip_token_ids` is declared
  (the recipe's named gap), so its pairs manifest records the media check as failed and its media stage will fail on
  the node. Fixing it is a product change (the skip rule at image positions); do it in 09 or declare the recipe
  text-only for 0.0.1.
- Control (f) is not applicable to qwen3-vl-reranker-2b and topk-embed-v1-small (their pins lie inside the
  checkpoint's own pixel budget, read at the pinned revision); it is served and caught for qwen3-vl-embedding-2b.
  Resolved at M3; on the node, an unreadable checkpoint budget makes (f) a blocker, never a skip.
- Unverified on real infrastructure: SLURM `srun --kill-on-bad-exit/--wait`, Kubernetes, and the stock image's bash,
  python3 and pip for the one-container job.

QA (workstream 07; flagged during M1-M3):
- The 01 follow-up security commits (named-key homes, redactors in `rcp_ndcg.support.urls`, `ValueError` to
  `ConfigError` for URL-list refusals, the deferred pool close, the `--engine` refusal) merged on red-then-green tests
  without a separate verifier round: re-check them in the QA correctness pass.
- Two padding helpers by design: `equivalence/stages.py` `_over_length` (the bounded over-length sampler) and
  `observe/requests.py` `_pad_to_tokens` (an exact target for the generator). Confirm or unify.
- Remaining private-name imports: `rcp_ndcg_core._records` and `irt._*` across rcp-ndcg, and
  `rcp_ndcg_test/conformance.py` importing `_probe_dimensions` (`reports/M-media.md` lists the ones fixed at M3).
- An untested guard: the stage-directory guard in `rcp-ndcg-vllm/jobs/bootstrap.sh` ("not a stage directory").
- The ctxl reference requirements pin `torch==2.9.1` (the paper's pin), against decision 2's reference venv on the
  image's torch. The bootstrap installs reference deps `--no-deps` under the image freeze; confirm or remove the pin.
- The token_ids head-edge audit is a conservative lower bound (it never decodes); an exact check would need the
  client's fitted text per body (a product accessor).
- `rcp_ndcg.testing` `find_corpora` still parses `manifest.json` itself (it only locates corpora); route it through
  the corpus reader for one home.
- qwen3-vl-embedding-2b stage 1 now reads the checkpoint's chat template from the Hub or the cache: offline and
  uncached it fails (by design, never a silent pass); CI must have network for that network-gated test or a cache.
- The pair census's `kept_tokens` counts query and document concatenated without a separator (pre-existing).

## 10. Definition of done (CPU side) and what stays for the owner

Done when workstreams 05, 08, 09, 06 and 07 are complete; the quality bar and GitHub CI are green on `rfc-0001`;
`CHANGELOG.md`'s `## Unreleased` is folded into `## 0.0.1 — <date to be set at tag time>`; and
`handover/RELEASE-CHECKLIST.md` is complete. Do not tag, do not merge to `main`, do not publish. When you finish a
workstream, write `handover/reports/<NN>-<name>.md`: status, commits, decisions and why, what you verified (commands
and results), open questions.

## 11. Working rules for this repository (the owner's)

- Commits use the repository's configured identity (the owner, credited as `fabianschmidt-cohere` on GitHub). Never
  add co-author lines or any AI or assistant attribution to commits, PR texts, code, docs or files.
- Work on branches and merge with `--no-ff` into `rfc-0001` (or rebase/merge commits on GitHub, never squash, so the
  authorship stays). Never rewrite published history. Merging to `main`, tagging and publishing are the owner's.
- Never read, print or commit credentials or tokens; public Hugging Face models and tokenizers are fetched anonymously.

## 12. Deferred beyond 0.0.1 (do not pull into this release)

A dynamic tournament allocation shifting placements toward the top; an explicit re-ask of windows whose answers stayed
unparseable; multi-node replicas (`nodes_per_replica > 1`) and per-dataset fan-out with job dependencies; cross-modal
chunking (video clips as units); the MTEB integration (upstream pull request); the paper's arXiv v2.

## 13. Owner-only items

Move the released Hugging Face datasets from the personal account (`fabianschmidt-cohere`) to an organisation if
wanted (update every pinned reference and `experiments/fetch_data.py`); combine it with the republish in MTEB's
exact layout (decision 31; workstream 10 builds and validates the converter, the owner pushes); confirm the paper's historical engine images
in `experiments/paper/serve/` before 08 replaces them; the go for the release (section 10).
