# Workstream 01: finish and integrate `wip/fix-inference`

Read `handover/00-MASTER.md` first (rules, quality bar, drift catalog).

## What the branch is
The code-quality sweep's confirmed findings for the inference layer (role clients, transport, adapters, text-budget and
media plumbing), fixed test-first. Base: an old `rfc-0001` (`c23ab01`); `rfc-0001` has moved ~130 commits since.
Commits (oldest first, abridged): retrieval-role usage forwarded; `RoleClient` declares its adapter attribute; the
transport gets a thread-safe sync bridge, a sane `Retry-After`, no userinfo in logs or records; one error shape for the
role-config family and every inert mode refused at the config; the adapter seam's declared contract (`AdapterBase`,
registration checks, a contract kit); data/media: one data-URI home, guarded container probes, a strict budget
identity, typed errors; minors (pooling reply corners refused by name, the fake speaks its wire claims, the hosted
cap, hints); snapshots/schemas/CHANGELOG/docs; two vendor-budget tests own their corpus key; the pooling media
allowance reserves the request shape's own frame; the pair media allowance reserves the share-capped query's render
and counts the frame once; `Transport.close()` atomic against `run()`; docs and typed errors from review.
The tip `a9156c2` is a WIP snapshot: the review's last two "truth" fixes (a docs claim in
`docs/concepts/embeddings.md` about a refusal that cannot fire; a dead batch-cap constant and the test that pinned it).

SECURITY item that must survive the merge intact: a profile's default key variables (`OPENAI_API_KEY`, ...) apply only
when the request goes to that profile's own default host; a custom `base_url` never receives them unless the config
names `api_key_env` explicitly.

## Steps
1. Branch from `wip/fix-inference`; review the WIP snapshot commit: keep it only if each change is true (the refusal
   really cannot fire on the merged code; the constant is really dead), with a test that fails without it.
2. Merge `origin/rfc-0001`. Expected conflict areas: `rcp-ndcg/src/rcp_ndcg/inference/clients/_base.py` (the judge moved onto
   `RoleClient`; `text_budget` was added), `inference/adapters/*`, `inference/config.py`, `inference/transport.py`,
   `data/preprocess.py`, `data/media.py`, docs (the docs were reorganised: `docs/tutorials/` -> `docs/how-to/`,
   `serving.md` -> `judges.md` + `runs.md`), `CHANGELOG.md` (union), snapshots/schemas (regenerate).
   Walk the drift catalog (MASTER section 7). Where this branch and `rfc-0001` both touched one concept (e.g. the media
   data-URI home, typed errors, transport auth), keep exactly one implementation.
3. Read the full `git diff origin/rfc-0001...HEAD -- src` after the merge. For each finding the branch claims fixed,
   confirm a test exists that fails on `origin/rfc-0001` (check it out in a scratch worktree and run that test).
4. Run the whole quality bar (MASTER section 2), plus one shuffled single-process run of the root suite
   (`uv run --no-sync pytest tests/ -p no:xdist -p no:randomly @<shuffled ids file>` — collect ids with
   `--collect-only -q`, shuffle with a fixed seed; the nightly workflow does the same).
5. Merge into `rfc-0001` (`git merge --no-ff`), quality bar again, push, dispatch CI.
6. Report in `handover/reports/01-fix-inference.md`.
