<!-- Handover copy of the operator's working note `drafts/qa-COMMON.md` (sanitized: private names, the private repository, the job platform and storage locations removed; `<operator-notes>/...` paths point to files that are NOT available — the content you need is in handover/). Treat it as the specification it was for its lane. -->

## Context and rules (shared by the three QA reviews)
Read-only review of `<repo>` at the commit named in your brief, the release candidate of rcp-ndcg 0.0.1.
The diff since the last public release is `git diff main...<commit>`; review the **whole package as it now is**, not
only the diff. Also review `packages/rcp-ndcg-vllm/` (the recipe package). Design and rules:
`<repo>/.rfc/RFC-0001-unified-inference.md`, `AGENTS.md` (layering, one home per concept, rules), and
the owner decisions: explicit token budgets, client-side cuts that reserve a model's anchors, never an engine-side cut
of a rendered prompt, over-budget default `cut` (recorded), chunk aggregation `max`, multi-vector transfer float16 by
default, media counted in tokens only (never money), every in-process model removed from the package.

Three reviewers run in parallel with separate mandates: **architecture** (structure, layering, seams, public surface),
**correctness** (bugs, error handling, concurrency, numerics, identities, security), **redundancy** (duplication, dead
code, stale docs and tests). Stay inside yours; when you see something in another's mandate, list it in one line under
"For the other reviewers" without investigating it.

Evidence rules: every finding has `path:line`, a severity (blocker: wrong results or a broken release; major: a real
defect or a rule violation that will bite; minor: worth fixing), the evidence (a command you ran and its output, or a
reproduction script left in your scratch dir), and a proposed fix in one or two lines. No finding without evidence; a
pattern you suspect but did not confirm goes under "Suspected, not confirmed". Count populations before asserting
properties ("all 14 adapters do X" needs the list of 14).

Write deliverables, measurements and logs only under `<operator-notes>/research/<tag>/work/` (persistent: a
node crash wipes `/tmp`); rebuildable things (your detached worktree, your venv) go under `/tmp/<tag>/`. Write nothing else (venv there if you need one; never install into the repository's `.venv`; no git writes;
no network writes). Never type chat-template special tokens literally. Work in small steps, write `report.md`
incrementally, never end a turn with thinking only. Deliver `report.md` (findings table sorted by severity, then the
sections your brief asks for) and `digest.md` (your preamble's format).
