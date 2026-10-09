# Lane report: `l08-sglang` — vLLM only (workstream 08 A, owner decision 14)

Date: 2026-10-09. Branch `lane/l08-sglang`, head `f51bcd84`, based on the `rfc-0001` tip `681a8cea` (re-fetched
before the gate: still the branch's base, so "Already up to date"; the gate ran on `f51bcd84` and passed). The
parallel plan's `l08-sglang` row is the file ownership; the judge-preset YAMLs and the recipes are other lanes'.

## Status

DONE. Every explicit SGLang path is gone from the product, the tests, the schema text, the docs and the paper's
serve scripts; the paper's SGLang judges stay as history. Two verifier rounds: round 1 FAIL (one major, five
minors), all fixed; round 2 PASS with four minor wording findings, all fixed. `bin/gate lane/l08-sglang` is
`GATE: PASS` on `f51bcd84` (`run_all` 1022/987/35/0, 67/67, 82/82).

## Commits

| Hash | Subject |
|---|---|
| `b555e9c1` | vLLM only: the rerank adapter refuses SGLang's bare list; the chat adapter recognises only vLLM's media-limit wording |
| `4e8f617f` | vLLM only: the Qwen2-VL image budget is the checkpoint's own; the SGLang test oracle leaves NOTICE |
| `fd285937` | vLLM only: engine lists, schema text and type-check comments name vLLM alone |
| `01d59c6a` | vLLM only: the paper's SGLang engine commands are history; docs and CHANGELOG say so |
| `02278f04` | vLLM only: the refused bare-list shape names TEI too, and the removed score key is pinned by a test (round 1) |
| `118e31fb` | vLLM only: singular engine wording in the media text and the judges docstring (round 1) |
| `f51bcd84` | vLLM only: round-2 wording fixes and the Voyage-shaped answer test |

This report is committed separately after the gate.

## What changed

- **Rerank adapter (the one named behaviour change).** A bare list of rows — the SGLang shape — is refused by
  name with a hint to serve the model on vLLM (`ProviderError`, non-retryable); the `score` key fallback (the
  same SGLang/TEI rows) is gone, so a row must carry `relevance_score`. The refusal names SGLang and TEI, because
  both answer that shape. `{"results": [...]}` (vLLM, Infinity, Cohere) and `{"data": [...]}` (Voyage) still
  parse and realign by `index`.
- **Chat adapter.** Only vLLM's media-limit wording is mapped to a `CapabilityError`; SGLang's "Image count N
  exceeds limit M per request." is now an ordinary per-request refusal, and the regex has one group again.
- **Qwen2-VL media geometry.** `PROCESSORS["qwen2_vl"].max_pixels` is the checkpoint's own 12,845,056 px, which
  vLLM applies, instead of the 1,003,520 px ceiling SGLang's override imposed. Policies in
  (1,003,520, 12,845,056] are now accepted; no existing policy changes size. The video budget was already the
  checkpoint's.
- **Test oracle and NOTICE.** The SGLang copy in `tests/data/_media_reference.py` (constants, `sglang_smart_resize`,
  the factor helpers, `sglang_frame_indices`, the SGLang `ENGINE_DEFAULTS` rows) is deleted with its NOTICE block;
  the transformers and vLLM oracles stay. All five NOTICE copies are byte-identical and SGLang-free.
- **Paper scripts and history.** `experiments/paper/serve/*.sglang.sh` are deleted; `experiments/paper/README.md`,
  `REPRODUCIBILITY.md` and `docs/how-to/reproduce-the-paper.md` record the paper's judges as SGLang history (the
  paper's submission code is the record; this release serves them on vLLM v0.31.0).
- **Docs, schema text and stale mentions.** The engine lists, serve examples, reasoning-parser column, video
  pinning flags and data-parallelism notes name vLLM alone; the exported schemas are regenerated; the
  basedpyright comment, the doctor extras guard and the heavy-import probe no longer mention SGLang.

## Verification

- **Round 1, lens A (correctness), DeepSeek-V4.1-flash**: **VERDICT FAIL**. Major: the removed `score` fallback
  had no failing-first test (the old source accepted `{"results":[{"index":0,"score":0.5}]}`; the HEAD test file
  passed on it). Minors: the `video_url` message change was not pinned; plural-engine wording remained in
  `resolution.py` and two test modules; a dead SGLang-shaped `list` branch remained in the rerank-client fake;
  the deleted `experiments/paper/serve/` is still cited by the out-of-scope preset YAMLs. Fixed: the score-key
  refusal is now parametrized over the `results` and `data` shapes (round 2 re-confirmed it fails on `681a8cea`
  and passes at HEAD), the `video_url` test asserts `mm-process-config` and `SGLang` are absent, the wording is
  singular, the `list` branch is gone, the refusal and CHANGELOG name TEI, and `judges/__init__.py` says vLLM
  only. The preset YAMLs stay for lane l08-judges, as the brief requires.
- **Round 1, lens B (regressions and hygiene), DeepSeek-V4.1-flash**: **VERDICT PASS**. Full root suite, contract,
  docs, the test distribution, ruff, basedpyright, mkdocs and `run_all` all green; scope matches the brief; three
  mutations (bare-list parse + score fallback, qwen2_vl `max_pixels`, the SGLang media-limit regex) each made the
  new tests go red and were restored byte-identically. Findings, all fixed: TEI was dropped and misattributed
  (refusal and CHANGELOG now name it), the two-engine wording, `judges/__init__.py`, and stale `handover/`
  citations (left as the workstream record; flagged for l08-judges). Nit fixed: the CHANGELOG says "the SGLang
  oracle in `tests/data/_media_reference.py`", not "the test oracle (file)".
- **Round 2 (confirmation, lens A+B, fresh), DeepSeek-V4.1-flash**: **VERDICT PASS**. All six round-1 fixes
  confirmed with evidence; the score-key test was re-mutated red and restored. Four minor wording findings fixed
  in `f51bcd84`: plural engine wording in the two media test modules, the stale short-clip paraphrase in
  `rcp-ndcg-test/tests/test_media.py`, the older CHANGELOG passage now naming SGLang and TEI, and the CHANGELOG
  wording "every SGLang passage describing a live path". The informational finding (the fake's `data` branch was
  never exercised) is closed with a new test that drives the Voyage-shaped answer through the client.

## Checks

The last full run, `bin/gate lane/l08-sglang` on `f51bcd84` (the gate's own worktree, all exit 0):

```
ruff-check    exit=0 All checks passed!
ruff-format   exit=0 532 files already formatted
basedpyright  exit=0 0 errors, 0 warnings, 0 notes
pytest        exit=0 3190 passed, 82 skipped in 82.20s
contract-docs exit=0 268 passed, 52 skipped in 44.38s
mkdocs        exit=0 Documentation built in 1.80 seconds
test-pkg      exit=0 570 passed, 225 skipped in 339.51s
recipes       exit=0 recipes: no failure outside the baseline (34 baseline failures remain, 0 fixed)
vllm-pkg      exit=0 1 passed
vllm-models   exit=0 70 passed, 7 skipped
run_all       exit=0 leaderboards: 1022 checks, 987 match, 35 known deviations, 0 failed
                     human study: 67 checks, 67 match, 0 known deviations, 0 failed
                     external LLM judges: 82 checks, 82 match, 0 known deviations, 0 failed
public-names  exit=0 clean (2 baselined hits remain)
clean         exit=0 clean
GATE: PASS
```

Failing-test-first evidence: against `681a8cea` (a scratch copy outside the repository), the bare-list refusal
test, the chat/judging-client unrecognised-wording tests and the score-key parametrization all failed
(`DID NOT RAISE ProviderError` / `CapabilityError` where `RequestRejectedError` was expected), and
`tests/data/test_prepare.py` failed at collection with `ValidationError: the pixel budget 3136-12845056px lies
outside ... (3136-1003520px)`.

## Open questions

- **TEI rerank.** The pre-change adapter accepted TEI's bare-list/`score` shape; this release refuses it with a
  message naming SGLang and TEI. TEI rerank was never documented for the role (the served wires are vLLM,
  Infinity, Cohere and Voyage), but if the owner wants TEI rerank supported it needs its own decision.
- **The paper's engine scripts.** `experiments/paper/serve/` is deleted entirely; the paper's commands now live
  only in the paper's submission code, with the history in `REPRODUCIBILITY.md` and `experiments/paper/README.md`.
  Keeping a historical (non-shipping) copy is a possible owner preference.
- **The qwen2_vl budget relaxation** (policies up to 12,845,056 px are now accepted) is a deliberate consequence
  of dropping SGLang's lower ceiling; no existing policy or paper number moves. If the owner prefers the old
  conservative ceiling as this project's own policy, it can be re-declared without any engine rationale.
- **Guards.** `sglang` left the heavy-import probe and the doctor extras assertion. If the owner wants the "the
  product must never import sglang" guard kept even without support, it can return as a plain name check.

## CHANGELOG entry

```markdown
- **Every explicit SGLang path** (workstream 08 A, owner decision 14): the release serves every role on vLLM
  v0.31.0. The rerank adapter no longer reads SGLang's (and TEI's) bare list of `{"index", "score"}` rows: that
  shape is refused by name with a hint to serve the model on vLLM, and a row must carry `relevance_score` (the
  `score` key TEI names the relevance by went with it; TEI's rerank shape was never documented for this role,
  whose served wires are vLLM, Infinity, Cohere and Voyage). The chat adapter no longer maps SGLang's
  media-limit wording ("Image count 12 exceeds limit 10 per request.") onto a `CapabilityError`; only vLLM's
  wording is a per-request media limit, and any other refusal is that request's. The Qwen2-VL image budget is
  the checkpoint's own 3,136-12,845,056 px, which vLLM applies, so a policy in the range SGLang's 1,003,520 px
  override used to refuse is accepted. Removed with the paths: the SGLang oracle in
  `tests/data/_media_reference.py`, its NOTICE rows, `experiments/paper/serve/*.sglang.sh`, the engine-script
  test, and every SGLang passage describing a live path; the exported schemas are regenerated. `REPRODUCIBILITY.md`
  records the paper's judges as SGLang history (the paper's submission code is the record; this release serves
  them on vLLM v0.31.0).
```

Four Unreleased entries were corrected in place because they described paths this lane removed (the second
judge's engine script, `VideoPolicy.engine_video_pinning`'s SGLang flag, the rerank answer shapes, the video
counting rationale).

## Public surface changes

No public name, CLI command or flag, exit code, or schema shape changed. The three exported schemas
(`schemas/index.v1.json`, `schemas/judge-config.v1.json`, `schemas/run-config.v1.json`) changed in `description`
text only and were regenerated with `pytest tests/contract --update-snapshots`; `tests/contract/snapshots/` is
unchanged. Behaviour that a user can observe changed in three documented places: the rerank adapter's accepted
answer shapes, the chat adapter's media-limit recognition, and the Qwen2-VL budget range.

## Files outside scope

- `rcp-ndcg/src/rcp_ndcg/judging/judges/__init__.py` (one docstring line: "vLLM or SGLang" → "vLLM"). The
  `judges/*.yaml` presets were left untouched per the brief.
- `pyproject.toml` (the basedpyright comment: `torch or sglang` → `torch or vllm`).
- `tests/contract/surface.py` (the heavy-import probe's `sglang` entry) and `tests/cli/test_doctor.py` (the
  "sglang not in the extras map" assertion).
- `docs/how-to/reproduce-the-paper.md` (its `experiments/paper/serve/` sentence became false with the scripts).
- `rcp-ndcg-test/tests/test_media.py` (one docstring line paraphrasing the changed short-clip message).

## For the next lanes

- **l08-judges**: the two preset YAMLs still say "vLLM or SGLang" and cite the deleted `experiments/paper/serve/`
  (`judging/judges/gpt_oss_120b.yaml:1-2`, `qwen35_397b_nvfp4.yaml:1-2`); decision 15's recipes replace them.
- **rfam / recipe-fix**: the recipes' SGLang mentions are untouched by design (`qwen3-embedding-0.6b/recipe.yaml`,
  `qwen3-reranker-8b/reference.py`).
- **Whoever updates the workstream specs**: `handover/specs/judge-catalog.md` cites the deleted
  `*.sglang.sh` files as provenance; the content is still correct, the paths are not.

## Docs updated

- `docs/api/inference.md` — the rerank answer shapes and the refusal-by-name.
- `docs/concepts/embeddings.md` — the self-hosted engine list drops SGLang.
- `docs/concepts/judges.md` — vLLM-only serve commands and reasoning-parser column, video pinning, data
  parallelism, and the paper's SGLang history.
- `docs/concepts/preprocessing.md` — vLLM-only resize/video rationale, the Qwen2-VL budget row (3,136 to
  12,845,056), the removed SGLang media flags.
- `docs/concepts/runs.md` — the engine image list.
- `docs/concepts/tournament.md` — the `json_schema` enforcement list.
- `docs/how-to/reproduce-the-paper.md` — the paper's SGLang history.
- `rcp-ndcg/README.md` — vLLM commands only.
- `REPRODUCIBILITY.md` and `experiments/paper/README.md` — the history wording.
- `CHANGELOG.md` — the new entry plus four corrected Unreleased entries.

Grep commands run over `docs/`, `README.md`, `REPRODUCIBILITY.md`, `skills/`, `examples/`,
`experiments/**/*.md`, docstrings and `mkdocs.yml`: `git grep -il sglang`, `git grep -n -i sglang`,
`git grep -n "paper/serve\|serve/qwen35\|serve/gpt_oss"`, `git grep -n "1003520"`,
`git grep -n "mm-process-config\|both engines\|engines disagree"`, `git grep -n "score\b"` on the rerank
docs. Every remaining hit is a historical record, the intended refusal-by-name, an out-of-scope preset/recipe
mention, or a planning record under `handover/`.
