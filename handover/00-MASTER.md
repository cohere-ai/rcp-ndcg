# Handover: finish RFC-0001 on `rfc-0001` and make 0.0.1 release-ready (CPU side)

You are taking over the integration branch `rfc-0001` of `github.com/cohere-ai/rcp-ndcg` from the previous operator.
Read this file completely, then `AGENTS.md` (binding), then the workstream prompt you are working on
(`handover/01-*.md` ... `handover/07-*.md`). The reference specifications the workstreams cite are in
`handover/specs/` (sanitized copies of the operator's working notes; any `<operator-notes>/...` or `<repo>/...` path
inside them points to the operator's machine and is not available to you — the content you need is in `handover/`).
`handover/` is temporary scaffolding on `rfc-0001`: read it, write your reports into `handover/reports/`, and delete the
whole directory in one commit before the release (it must not ship; no distribution packages it).

## 1. Mission and scope

RFC-0001 ("the unified inference layer") replaces every in-process model path with role clients that talk to served
engines over one transport, with client-side text budgets, anchor-preserving cuts, a recipe catalog for serving
models on the stock vLLM image, an equivalence harness that proves a served recipe equals its reference, and an
unpublished test distribution. The core product work is merged and green on `rfc-0001`. What remains is integration
of in-flight branches, the recipe family work, the repository layout move, the final docs, QA and release prep.

In scope for you: everything that runs on CPU. Out of scope (do not attempt, do not stub, do not claim):
- **GPU work**: serving models on GPUs, the GPU waves (T0-T4), recording observation corpora, filling `pending_gpu`
  expected values, marking recipes `verified`. You may write and CPU-test the code those waves run.
- **Any private repository or private recipes/plugins.** Only public names, everywhere (AGENTS.md: placeholders such as
  `gs://YOUR-BUCKET/...`, `registry.example.com`).
- **Releasing**: merging to `main`, tagging `v0.0.1`, publishing. The owner gives that go; you prepare it (section 9).

## 2. Repository facts you need on day one

- Three published distributions + one unpublished: `rcp-ndcg-core` (`packages/rcp-ndcg-core`, numpy + pydantic),
  `rcp-ndcg` (root, the pipeline and CLI), `rcp-ndcg-vllm` (`packages/rcp-ndcg-vllm`, deliberately OUTSIDE the uv
  workspace today: recipes, the equivalence harness, the recorder, the wave runner, node scripts), and
  `rcp-ndcg-test` (`packages/rcp-ndcg-test`, a workspace member, never published: reference cases, the conformance
  suite, fakes). The layout move (workstream 05) changes this to per-distribution top-level directories.
- Layering inside `src/rcp_ndcg` (imports point inward only), the one-home table and every rule: `AGENTS.md`.
- The quality bar (run all of it before you call anything done; this is what the previous operator called "the gate"):

```bash
.github/scripts/cpu-env.sh dev docs                      # the locked env with CPU torch (Linux); then always --no-sync
uv run --no-sync ruff format --check . && uv run --no-sync ruff check .
uv run --no-sync basedpyright                            # 0 errors
uv run --no-sync pytest tests/ -q -n 4                   # the whole root suite, offline
uv run --no-sync pytest tests/contract tests/docs -q     # public surface, docs snippets
uv run --no-sync pytest packages/rcp-ndcg-test/tests -q  # the unpublished test distribution
uv run --no-sync mkdocs build --strict
# rcp-ndcg-vllm in its own venv, exactly like CI's vllm-recipes job:
uv venv /tmp/vpkg --python 3.12 && uv pip install --python /tmp/vpkg/bin/python \
    -e packages/rcp-ndcg-core -e . -e 'packages/rcp-ndcg-vllm[test]'
/tmp/vpkg/bin/python -m pytest packages/rcp-ndcg-vllm/tests -q
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

## 3. Branch map (all on origin)

| Branch | Base | State |
|---|---|---|
| `rfc-0001` @ `1524a8c` | — | Integration branch. Quality bar green. Contains the core RFC work and every finished fix lane (section 4). |
| `wip/fix-inference` | old `rfc-0001` | Nearly done; in its round-2 review. Workstream 01. |
| `wip/int-recipes` | `rfc-0001`@`9ff594e` | The 18 recipe lanes + the topk and pplx plugin lanes, merged. Red until recipe-common lands. |
| `wip/recipe-sweep` | `wip/int-recipes` | "recipe-common": the shared recipe fixes; functionally done; review findings being triaged. Workstream 02. |
| `wip/fam-qwen3-rerank`, `wip/fam-ctxl`, `wip/fam-zerank`, `wip/fam-vl`, `wip/fam-late`, `wip/fam-dense` | recipe-common @ `d222426` (some merged later recipe-common commits) | The six recipe family lanes, in progress. Workstream 03. |
| `wip/layout-script` | `5359655` (= int-recipes + rfc-0001@`d8a7002`) | The layout move as a script + hand-edit stack; in review. Workstream 05. |
| `wip/gpu-quality` | `5359655` | The observation-request generator, stage-2 pairs, corpus format, T3 quality stage, negative controls (CPU code); in review. Workstream 04. |
| `wip/fake-engines` | `5359655` | Verified fake engines, conformance over a provisional corpus, golden replays, fingerprint; fixing review findings. Workstream 04. |
| `wip/gpu-e2e` | `5359655` | The T4 scenarios and in-pod driver (CPU-tested). DONE and accepted; merge after the recipes. Workstream 04. |

A `wip/*` tip whose subject starts with "WIP snapshot of lane ..." holds that lane's uncommitted work as it stood when
it was stopped: treat it as a draft (it may not pass tests), review it, and finish or redo it.

## 4. What `rfc-0001` already contains (do not redo)

Role clients (embed, pool, rerank, judge) on one transport; client-side text budgets with `fit`, per-shape
`query_max_tokens`, anchors (`last`, `first`, `last_content`, markers), declared `normalize`, `media_sides`,
`empty_query`, late-interaction skip ids, client-side MRL cut, multimodal embeddings incl. video, `token_ids` requests;
the equivalence harness driving the product's role clients (rule R30, section 6); the RC procedure, wave runner,
node bootstrap with three environments (and the shakedown's runtime fixes: one failing recipe never stops a wave,
plugins from the staged wheelhouse, the reference venv kept on the image's torch, short per-slot TMPDIR); the
`rcp-ndcg-test` distribution with the model-card reference cases; the code-quality sweep's fixes (core values,
storage, readers, CLI/MCP (`mcp tools` removed), judging identities and store races, infra/CI/experiments, docs
reorganised into Concepts / How-to / Reference); hosted Cohere and Voyage profiles (verified live by the owner's
operator).

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
  (`python -c "import json,pathlib; from rcp_ndcg_vllm import recipe_json_schema; pathlib.Path('packages/rcp-ndcg-vllm/schema/recipe.schema.json').write_text(json.dumps(recipe_json_schema(), indent=2)+'\n')"`
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
  non-conflicting hunks. After every merge: `git diff <before>..HEAD -- packages/rcp-ndcg-vllm/src` and read it.
- **Recipe edits outside their owner.** `wip/gpu-e2e`, `wip/fake-engines` and `wip/gpu-quality` edited recipe files to
  make them load; the family branches own those files — take the family's version.
- **Slow paths.** The offline fake engine's multi-vector pooling hashes once per scalar (`rcp_ndcg.inference.fake`,
  `_unit_vector` -> `fake_uniform`): 2048 dims x 16k tokens takes tens of minutes. Recipe tests bound it with an 8-wide
  probe copy of the recipe; the product fix (one seeded draw per vector) is open (workstream 02). The harness sampler's
  over-length padding was quadratic (fixed in `wip/recipe-sweep` `3c9d4e6`).
- **References bent toward the product.** Several family branches "render the wire's spans" in their references; some
  of that is only the required output FORMAT (the harness compares spans), some ports the client's cut — the latter
  violates decision 9 (workstream 03 lists the suspects).

## 8. Workstreams, order and parallelism

| # | Prompt | Depends on | Parallel with |
|---|---|---|---|
| 01 | `01-integrate-fix-inference.md` | — | 02, 04 |
| 02 | `02-recipe-common-and-harness.md` | — | 01, 04 |
| 03 | `03-recipe-families.md` | 02 (recipe-common merged into `wip/int-recipes`) | 04 |
| 04 | `04-corpus-fakes-e2e.md` | 03 for the final merge | 01-03 (development) |
| 05 | `05-layout-move.md` | 01-04 integrated into `rfc-0001` (the quiet window) | — |
| 06 | `06-docs-final.md` | 05 | 07 |
| 07 | `07-qa-and-release-prep.md` | 05 | 06 |

Integration target: everything lands on `rfc-0001` (recipes via `wip/int-recipes`, which merges into `rfc-0001` once
the families are in). Keep `rfc-0001` green after every merge; run the full quality bar, push, dispatch CI.

## 9. Definition of done (CPU side) and what stays for the owner

Done when: every `wip/*` branch is integrated or explicitly superseded; the layout move is applied; the final docs and
the QA reports' blockers/majors are fixed; the quality bar and GitHub CI are green on `rfc-0001`; `CHANGELOG.md`'s
`## Unreleased` is folded into `## 0.0.1 — <date to be set at tag time>`; a release checklist exists
(`handover/RELEASE-CHECKLIST.md`, write it) listing: the GPU waves still to run per recipe (T0-T4), the corpora to
re-record and the emulators to re-verify, the `pending_gpu` cases to fill, recipe statuses to flip, Dependabot alerts,
and the owner's go. Do not tag, do not merge to `main`, do not publish.

When you finish a workstream, write `handover/reports/<NN>-<name>.md`: status, commits, decisions taken and why,
what you verified (commands and results), open questions.
