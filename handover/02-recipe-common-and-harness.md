# Workstream 02: finish recipe-common, close the harness gaps, bring `wip/int-recipes` up to `rfc-0001`

Read `handover/00-MASTER.md` first. Specs: `handover/specs/recipe-sweep.md`, `triage-decisions.md` (sections
"sweep-recipes" and "OPERATOR DECISION"), `sweep-recipes-report.md`, `gpu-findings.md`.

## A. recipe-common (`wip/recipe-sweep`)

Its job was the part of the recipe sweep every family shares: (1) six recipe test files that imported harness helpers
R30 deleted, rewritten onto the product's role clients through one shared helper
(`packages/rcp-ndcg-vllm/tests/recipes/_served.py`) and a contract helper (`_contract.py`: a recipe's test pins every
`serve`, `client` and `reference` field; mutants must go red); (2) the empty `{fixed: ""}` markers removed; (3) one
merged NOTICE with every third-party file attributed. Commits on top of `wip/int-recipes`:

| Commit | What | Verdict |
|---|---|---|
| `9511c76` | shared served/contract helper; the six test files onto the role clients | keep |
| `f1d1e0a` | qwen3-reranker-8b and zerank-1-small references "render the wire's content spans" | the span OUTPUT format is required (the harness compares spans); the parts that PORT the client's cut (settle rule, query share) violate owner decision 9 — the families qwen3-rerank and zerank restore the paper's cut there (workstream 03); do not "fix" it twice |
| `bc970d1` | empty markers dropped; content-final shapes with `add_special_tokens: true` | keep |
| `d222426` | one merged NOTICE | keep; verify every third-party file in `packages/rcp-ndcg-vllm/recipes/*/` (vendored `qwen3_vl_embedding.py`, copied templates) is attributed with source, revision and licence, each checked at that revision |
| `ea7159b` | topk stage-1 probes at an 8-wide vector (bounds the offline fake's per-scalar hashing); jina counts every document | keep |
| `9b6bfb5`, `090cd3c` | the harness's `over_cap_cut_differs` deviation + CHANGELOG | keep: it is owner decision 9 (a reviewer flagged it as scope creep; it is decided) |
| `3c9d4e6` | the harness sampler's over-length padding bounded (it re-tokenized a growing 65k-token text per step: the network-test hang) | keep; confirm a test bounds its runtime |
| `282f89e` | WIP snapshot: round-1 review minors in progress (3 files) | review, finish or drop |

Round-1 review verdicts were PASS/PASS (one major = the blessed deviation above; 3 + 5 minors). Do your own review
of the minors: read `git diff wip/int-recipes...wip/recipe-sweep`, fix what is true, test-first.

Network check (public tokenizers): all six rewritten files passed on the real tokenizers (jina's file takes ~9 min).
Re-run them one file at a time (MASTER section 2) after your changes.

Then merge `wip/recipe-sweep` into `wip/int-recipes`.

## B. Bring `wip/int-recipes` up to `rfc-0001`
`wip/int-recipes` was cut from `rfc-0001`@`9ff594e`. Merge `origin/rfc-0001` into it (the RC tooling, the shakedown's
runtime fixes, judging/CLI/infra fixes, the docs reorganisation, `rcp-ndcg-test`, `RoleClient.text_budget`). Expect
conflicts in `packages/rcp-ndcg-vllm/` (README, schema — regenerate it, `jobs/`), `CHANGELOG.md`, `pyproject.toml`
(the basedpyright include list must keep `packages/rcp-ndcg-vllm/plugins/*/src` and `packages/rcp-ndcg-test/src`),
`uv.lock` (regenerate; then the constraints file). Run the quality bar on the result. Note: recipes now load through
stricter validators on `rfc-0001` (MASTER section 7); a recipe that fails to load is a family's item — list it for
workstream 03 rather than patching recipe semantics here (patching a load-blocker minimally is fine if a family's branch
already carries the same fix; then take that exact change).

## C. Harness gaps (product code in `packages/rcp-ndcg-vllm/src/rcp_ndcg_vllm/equivalence/`; failing test first each)
Found by validating more recipes than the public ones; all reproduce with public fixtures:
- **G1**: `request_shape: token_ids` request bodies (`{"input": [[ids]]}`) crash the anchor audit
  (`stages._anchor_check` passes a list of ints to `tokenizer.ids`: `TypeError: TextInputSequence must be str`). Read
  the sent ids directly; they already contain the anchor edge.
- **G2**: `messages`-route bodies extract as an empty input list, so the anchor audit checks nothing and passes
  (`checked: 0`). Extract the texts (and media placeholders) from messages; an audit that checked zero inputs fails.
- **G3**: the `anchor: first` head-edge audit compares the fixed head segment's standalone ids verbatim against the
  joined render's head; on BPE tokenizers the head's trailing join-space merges into the first content token, so the
  edge is never found (false failures). Compare on the product's own overhead accounting (it already documents that
  join-merge) instead of standalone ids.
- **Offline fake engine performance** (`src/rcp_ndcg/inference/fake.py`): multi-vector pooling draws one SHA-256 per
  scalar (`_unit_vector` -> `fake_uniform`): 2048 dims x 16k tokens = ~33M hashes per text. Make it one seeded draw per
  vector (e.g. a numpy generator seeded from one hash), deterministic; any test that pins fake values moves
  deliberately (say so in the CHANGELOG). Then the recipe tests' 8-wide probe workaround may stay or go — your call,
  documented.
- **Extra plugin wheels in waves**: the node bootstrap (`packages/rcp-ndcg-vllm/jobs/bootstrap.sh`) installs a recipe's
  named plugin from `<RC>/wheelhouse` only; wheels staged through `EXTRA_DIRS` land under `<RC>/extra/<name>/` and are
  not found. Add `--find-links <RC>/extra/*/wheelhouse` (each existing dir) to the plugin install; CPU test with a fake
  pip like the existing bootstrap tests.

Where: do A and B on `wip/int-recipes`'s line; C may go straight to `rfc-0001` (the harness lives there too) and then
flow into `wip/int-recipes` with the next `rfc-0001` merge. Report in `handover/reports/02-recipe-common.md`.
