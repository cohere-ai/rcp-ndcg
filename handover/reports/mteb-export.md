# MTEB republish export — the local half, validated

**Date:** 2026-10-12 · **Lane:** `lane/mteb` (worktree `wt-mteb`) · **Base:** `rfc-0001` @ `3217b5ed` (merged;
the lane's own tip was an ancestor, so the merge fast-forwarded) · **Gate:** none — the lane changes no code
(one report file); the gate's `public-names` step and a clean tree were re-checked instead.

## Status

DONE. `tools/republish_mteb.py` re-laid all four published repositories in MTEB's exact push layout, and every
written repository was loaded back with mteb's own `RetrievalDatasetLoader`: the qrels (as integers), the
queries, the corpus (media included) and the pool mteb reads all equal the loaded `Dataset`. No mismatch
anywhere, no tool bug, no repo change. Nothing was uploaded and nothing was installed into the repository or
the lane's environment.

## Environment

The lane's worktree environment carries the `dev` extra and CPU torch only: `mteb` and `datasets` are absent
(`ModuleNotFoundError`), so the tool's validation step cannot import mteb's loader there. The validation ran in
a scratch venv outside the repository, built from the current lock's `[mteb]` extra (the dep set CI's `mteb`
job installs):

```bash
uv venv <scratch>/venv-mteb --python 3.12
uv export --frozen --no-hashes --package rcp-ndcg --extra mteb --no-emit-workspace > <scratch>/mteb-reqs.txt
# the lock export minus the torch/torchvision/triton lines (the shape .github/scripts/cpu-env.sh uses)
uv pip install --python <scratch>/venv-mteb/bin/python --no-deps \
    -e ./rcp-ndcg-core -e ./rcp-ndcg -r <scratch>/mteb-reqs-notorch.txt
uv pip install --python <scratch>/venv-mteb/bin/python torch==2.14.0 \
    --index-url https://download.pytorch.org/whl/cpu
```

The verdicts were obtained with the lock's pins: **mteb 2.21.6, datasets 5.0.1, pyarrow 25.0.1,
huggingface-hub 1.32.0, torch 2.14.0+cpu**. A second verdict for NanoBEIR was taken with an independent
interpreter (the earlier MTEB lane's venv, whose mteb 2.21.6 / datasets 5.0.1 are identical to the lock's pins —
no version delta to report): same result, 56,723 documents.

## What was exported

Run from the repository root, into a scratch directory outside the repository:

```bash
python tools/republish_mteb.py --out <scratch>/export --repo rcp-ndcg-nanobeir
python tools/republish_mteb.py --out <scratch>/export --repo rcp-ndcg-bright --repo rcp-ndcg-trecdl
python tools/republish_mteb.py --out <scratch>/export --repo rcp-ndcg-vidore-v3
```

All three exit 0; the union of the `--repo` runs is exactly the tool's default set (`REPOS`), which names the
four paper repositories (`rcp-ndcg-external-validation` holds no retrieval dataset and is not named).

| Repository | Split | Subsets | Configs | Corpus rows | Queries | Qrels rows | Pool rows | Excluded | Loader verdict |
|---|---|---|---|---|---|---|---|---|---|
| `rcp-ndcg-nanobeir` | `train` | 13 | 53 | 56,723 | 649 | 98,003 | 649 | 1 config (21 rows) | PASS |
| `rcp-ndcg-bright` | `standard` | 12 | 51 | 1,333,166 | 1,366 | 190,072 | 1,366 | 3 configs (447 rows) | PASS |
| `rcp-ndcg-trecdl` | `test` | 2 | 8 | 8,841,823 (one shared corpus) | 97 | 27,875 | 97 | — | PASS |
| `rcp-ndcg-vidore-v3` | `test` | 48 | 192 | 19,252 (page images) | 14,514 | 2,177,100 | 14,514 | — | PASS |

Configs are four per subset (corpus, queries, qrels, `top_ranked`) plus one `excluded` per subset that carries
exclusions. The shared corpora (TREC-DL's two years; ViDoRe v3's six languages of a domain) are written once and
the README's `configs:` front matter points every config name at the one file (ViDoRe v3: 48 corpus configs over
8 domain files; TREC-DL: 2 corpus configs over 1 file).

Fields written: corpus `id`/`title`/`text` (plus `image` for ViDoRe v3), queries `id`/`text` (plus `instruction`
where the data carries one — none of the four does), qrels `query-id`/`corpus-id`/`score` (int64) plus the
rcp-ndcg extras `gain`/`theta` (float64), `top_ranked` `query-id`/`corpus-ids`, `excluded`
`query-id`/`excluded-corpus-ids`.

## The loader verdict

Each written subset is validated with mteb's own
`mteb.abstasks.retrieval_dataset_loaders.RetrievalDatasetLoader` (subclassed only to cache one corpus build per
shared group), loading a uniquely named symlink view of the written directory and comparing against the loaded
`Dataset`:

- the qrels as integers (`relevant_docs`),
- the queries mteb keeps — the qrels-bearing ones — their texts and, where present, instructions,
- the corpus texts and every media cell's bytes (`Image`/`Video`, decode off),
- the pool mteb reads (`top_ranked`, exclusions folded out).

Every subset of every repository compared equal; a mismatch raises `DataError` naming the subset, so the four
`[ok]` lines are the verdict. The reads pin the revisions the checks were published against
(`experiments/fetch_data.py`):

- `rcp-ndcg-nanobeir` @ `4517f2cb9e342479725bf0931a329998b4d35038`
- `rcp-ndcg-bright` @ `1bee160c00bf8ea7576a41879a62cb2d4fbd9319`
- `rcp-ndcg-trecdl` @ `67d32cbe133805086ee5ee2600369bf05cb45143`
- `rcp-ndcg-vidore-v3` @ `4e5ca46e2f9c050b18a7f170c23e3ade66f78a4e`

**Against the shipped task definitions.** The published `rcp_ndcg_tasks.py` files (read as data) define, and the
converter checked both directions (a defined subset the repository lacks; a repository subset no definition
reads) and every matching definition's split: NanoBEIR 13 tasks / `train`, BRIGHT 12 / `standard`, TREC-DL 2 /
`test`, ViDoRe v3 16 tasks over 48 language subsets / `test` (every language subset matches a task; the page
image task and its OCR view both have their split checked).

**Against the pinned dataset revisions.** The task files pin their own `dataset.revision`, a later commit than
the checks' revisions. The Hub API's file lists at the two revisions are identical for all four repositories and
every data file's blob id matches; the only differing file is `README.md` (the card). The export therefore holds
exactly the data the published definitions pin, re-laid. The owner bumps `dataset.revision`/`_REVISION` to the
pushed commit after the upload (the new SHA cannot exist before it). One detail: TREC-DL's task file exists only
at the checks' revision, not at the data revision it pins — the pin names the data commit, as expected.

## Owner-side upload

The four written trees are `export/<repo>` in a scratch directory outside the repository, with the run logs
(`republish-*.log`) beside them. Upload each repository root (the card travels with the files):

```bash
hf upload <ORG>/rcp-ndcg-nanobeir  <scratch>/export/rcp-ndcg-nanobeir  . --repo-type dataset
hf upload <ORG>/rcp-ndcg-bright    <scratch>/export/rcp-ndcg-bright    . --repo-type dataset
hf upload <ORG>/rcp-ndcg-trecdl    <scratch>/export/rcp-ndcg-trecdl    . --repo-type dataset
hf upload <ORG>/rcp-ndcg-vidore-v3 <scratch>/export/rcp-ndcg-vidore-v3 . --repo-type dataset
```

(`hf upload REPO_ID [LOCAL_PATH] [PATH_IN_REPO] --repo-type dataset`; `hf` is the `huggingface_hub` CLI. Use the
organisation placeholder, or `fabianschmidt-cohere` if the repositories stay where they are.) Then, per
repository, bump `_REVISION`/`dataset.revision` in `rcp_ndcg_tasks.py` to the pushed commit.

Two owner-side notes. The written card's `dataset.path` (and the reference/source lines) name
`fabianschmidt-cohere/<repo>`; if the repositories move to an organisation, update the card's path after the
move — the tool's `--owner` selects the read source only, and cannot name an owner whose repositories do not
exist yet. And the current hand-written usage cards are not reproduced: the written card is mteb's own template
rendered from the published task metadata, so merge anything the old card says that mteb's template does not.

## Blockers

None for the export or the validation. The only environment gap: the lane's worktree venv has no `[mteb]` extra
(`mteb>=2.10.5` → 2.21.6 and `datasets>=4.1` → 5.0.1 in the lock are both absent), so the release dep set was
installed in a scratch venv outside the repository, as above. Nothing was installed into the repository or the
lane's environment; nothing was uploaded; no dataset was missing (all four read from the Hub, the pinned
revisions were cached locally).
