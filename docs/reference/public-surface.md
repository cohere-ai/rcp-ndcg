# The public surface, frozen

0.0.1 freezes the Python surface deliberately and once. The rules -- what "public" means, how a change is
made, and the deprecation path -- are on
[compatibility and versioning](versioning.md#the-001-freeze); this page is the short form and the index.

## What is public

`rcp_ndcg.support.api_docs.PUBLIC_MODULES` names the public modules: `rcp_ndcg_core` and its documented
submodules, the `rcp_ndcg` facade and the packages the docs present as API (`calibration`, `data`,
`data.preprocess`, `data.revisions`, `errors`, `eval`, `eval.mteb`, `examples`, `inference`, `judging`,
`results`, `retrieval`, `runners`, `runs`, `testing`), and `rcp_ndcg_vllm.recipe` (the serving package's
recipe module). Every other module is internal: it may change without a CHANGELOG entry, and only the names a
public module lists in `__all__` are part of the surface.

The surface is pinned as data in `tests/contract/snapshots/`: `python_api.json` (each public name with its
signature and pydantic fields), `cli.json` (commands, options, defaults), `mcp_tools.json`, `exit_codes.json`
and `packaging.json` (console scripts, entry-point groups, heavy imports); `schemas/` holds the exported JSON
Schemas, compared byte for byte. The snapshot is the definition of the surface; a docs page that misses a
name does not make it private.

## What "frozen" means

- A name in the snapshot is public for the 0.0.1 line. Removing or renaming one is a breaking change: the
  contract test classifies the diff (`tests/contract/diffing.py`) and the CHANGELOG entry goes under
  `### Public surface`.
- **Every public name is documented.** The curated concept and how-to pages carry the narrative, and
  `rcp-ndcg docs api --out docs/reference/api` generates one page per public module -- each name with its
  kind, signature and one-line role -- from the snapshot and the modules' own docstrings. `tests/contract`
  fails when a pinned name is on no page under `docs/`, when a pinned module has no page, or when the
  committed pages differ from a fresh generation, so the docs cannot drift from the surface.
- `rcp_ndcg_test` is never public: it is unpublished, its API moves with the harness, and the product never
  imports it.

## Changing it

```bash
pytest tests/contract                       # a classified BREAKING / ADDITIVE diff
pytest tests/contract --update-snapshots    # rewrite snapshots/ and schemas/
rcp-ndcg docs api --out docs/reference/api  # regenerate the per-module API pages
git diff tests/contract/snapshots schemas docs/reference/api
```

A snapshot change is part of the change's review: read the diff, keep the entry in `CHANGELOG.md` under
`### Public surface`, and say in the pull request which names moved and why.
