# The public surface, frozen

The public surface is the set of names and artifacts that a release keeps stable. 0.0.1 freezes the Python part of
that surface one time, on purpose. This page is the short form and the index. The rules are on
[compatibility and versioning](versioning.md#the-001-freeze). That page gives the meaning of "public" and the steps of
a change. It also gives the deprecation path, which is how a public name goes away.

## What is public

`rcp_ndcg.support.api_docs.PUBLIC_MODULES` names the public modules. The list has three parts. The first part is
`rcp_ndcg_core` and its documented submodules. The second part is the `rcp_ndcg` facade and the packages that the docs
present as API. The facade is the top-level package that a user imports. The API packages are `calibration`, `data`,
`data.preprocess`, `data.revisions`, `errors`, `eval`, `eval.mteb`, `examples`, `inference`, `judging`, `results`,
`retrieval`, `runners`, `runs` and `testing`. The third part is `rcp_ndcg_vllm.recipe`, the recipe module of the
serving package. Every other module is internal. An internal module can change without a CHANGELOG entry. Only the
names that a public module lists in `__all__` are part of the surface.

The surface is pinned as data in `tests/contract/snapshots/`. `python_api.json` holds each public name with its
signature and its pydantic fields. `cli.json` holds the commands, the options and the defaults. `mcp_tools.json` holds
the MCP tools, and `exit_codes.json` holds the exit codes. `packaging.json` holds the packaging: the console scripts,
the entry-point groups and the heavy imports. `schemas/` holds the exported JSON Schemas, and the tests compare them
byte for byte. The snapshot is the definition of the surface. A docs page that misses a name does not make that name
private.

## What "frozen" means

- A name in the snapshot is public for the 0.0.1 line. To remove or rename it is a breaking change. The contract test
  classifies the diff (`tests/contract/diffing.py`), and the CHANGELOG entry goes under `### Public surface`.
- **Every public name is documented.** The curated concept and how-to pages give the explanation.
  `rcp-ndcg docs api --out docs/reference/api` generates one page per public module from the snapshot and the modules'
  own docstrings. Each name on that page has its kind, its signature and a one-line role. `tests/contract` fails when
  a pinned name is on no page under `docs/`, or when a pinned module has no page. It also fails when the committed
  pages differ from a fresh generation. The docs therefore cannot differ from the surface.
- `rcp_ndcg_test` is never public. It is unpublished, its API moves with the harness, and the product never imports
  it.

## Changing it

```bash
pytest tests/contract                       # a classified BREAKING / ADDITIVE diff
pytest tests/contract --update-snapshots    # rewrite snapshots/ and schemas/
rcp-ndcg docs api --out docs/reference/api  # regenerate the per-module API pages
git diff tests/contract/snapshots schemas docs/reference/api
```

A snapshot change is part of the change's review. Read the diff. Keep the entry in `CHANGELOG.md` under
`### Public surface`. Say in the pull request which names moved, and why.
