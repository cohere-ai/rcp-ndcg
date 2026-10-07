<!-- Handover copy of the operator's working note `research/docs-site/work/OPERATOR-ANSWERS.md` (sanitized: private names, the private repository, the job platform and storage locations removed; `<operator-notes>/...` paths point to files that are NOT available — the content you need is in handover/). Treat it as the specification it was for its lane. -->

# Operator answers to docs-site's open questions (20:45; binding for docs-early, docs-final and the layout move)
- OQ-1: `rcp_ndcg.testing` (FakeJudge, build_tiny_world) STAYS in rcp-ndcg (it is a public module); the verified
  model fakes built from GPU recordings live in rcp-ndcg-test.
- OQ-2: keep `anchor_report` (no rename); disambiguate in prose from the template anchors.
- OQ-3: settled by fix-cli (merged): `calibration insert --dry-run` is the boolean, `--plan` only ever names a plan
  file. Drop the carve-out; one meaning everywhere.
- OQ-4: publish the tiers T0-T4, the three-environment node runtime and the conformance targets at the level E21
  describes, placeholders only (`gs://YOUR-BUCKET/...`, `registry.example.com`); job-platform specifics stay in the
  contributor page `docs/how-to/release-candidates.md`.
- OQ-5: no `kind` column; the per-recipe status column (docs-release Q5) carries what users need, and the reference
  column says whether the reference is the paper's code or the model card's.
- OQ-6: no extra alias: `recipe:<id>` needs `rcp-ndcg-vllm` installed alongside (pydantic + pyyaml only after the
  layout move); a missing package is a typed error with that install line.
- OQ-7: p1-tail is IN 0.0.1 (merged): document per-shape `query_max_tokens`, late-interaction skip ids, `media_sides`,
  `normalize`, `empty_query`, `anchor: last_content`, `request_shape: token_ids`, the client-side MRL cut.
- OQ-8: three api pages are acceptable for the tag.
- OQ-9: yes: `v0.0.1`, the same version in all three published distributions (and rcp-ndcg-test, unpublished).
- OQ-10: docs spellings only (E24 as written, no code change).
