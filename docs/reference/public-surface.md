# The public surface, frozen

0.0.1 freezes the Python surface deliberately and once. This page says what "public" means here, what is
pinned, and how a change is made.

## What is public

`tests/contract/surface.py` names the public modules: `rcp_ndcg_core` and its documented submodules, the
`rcp_ndcg` facade and the packages the docs present as API (`calibration`, `data`, `data.preprocess`,
`data.revisions`, `errors`, `eval`, `eval.mteb`, `examples`, `inference`, `judging`, `results`, `retrieval`,
`runners`, `runs`, `testing`), and `rcp_ndcg_vllm.recipe` (the serving package's recipe module). Every other
module is internal: it may change without a CHANGELOG entry, and only the names a public module lists in
`__all__` are part of the surface.

The surface is pinned as data in `tests/contract/snapshots/`: `python_api.json` (each public name with its
signature and pydantic fields), `cli.json` (commands, options, defaults), `mcp_tools.json`, `exit_codes.json`
and `packaging.json` (console scripts, entry-point groups, heavy imports); `schemas/` holds the exported JSON
Schemas, compared byte for byte. The snapshot is the definition of the surface; a docs page that misses a
name does not make it private.

## What "frozen" means

- A name in the snapshot is public for the 0.0.1 line. Removing or renaming one is a breaking change: the
  contract test classifies the diff (`tests/contract/diffing.py`) and the CHANGELOG entry goes under
  `### Public surface`.
- Every public name is either mentioned on a page under `docs/` or listed in
  `tests/contract/undocumented_public_names.json` — the **advanced surface** the reference pages do not
  present one by one (a registry constant, an estimator class, a mechanism name). That list is a reviewed
  diff, not a default: a new public name fails the contract suite until it is documented or deliberately
  added, and a name that gains a page fails until it is removed. The list may shrink freely; it grows only
  on purpose.
- `rcp_ndcg_test` is never public: it is unpublished, its API moves with the harness, and the product never
  imports it.

## Changing it

```bash
pytest tests/contract                       # a classified BREAKING / ADDITIVE diff
pytest tests/contract --update-snapshots    # rewrite snapshots/, schemas/ and the freeze list
git diff tests/contract/snapshots schemas tests/contract/undocumented_public_names.json
```

A snapshot change is part of the change's review: read the diff, keep the entry in `CHANGELOG.md` under
`### Public surface`, and say in the pull request which names moved and why.
