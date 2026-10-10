# Lane `ref-envs`: per-family reference environments and stored reference outputs (owner decision 35)

**Status:** DONE. Branch `lane/ref-envs`; final head `d2df8066` (the merge of `rfc-0001` at `b18d34c4`).
The gate on the merged tree is **PASS** (ruff/format/basedpyright 0, root suite 3831 passed/102 skipped,
contract+docs 302/55, mkdocs strict, test-pkg 980/223, recipes 0 baseline failures, vllm-pkg 49,
vllm-models 72/7, run_all 1022/987/35/0 + 67/67 + 82/82, public-names clean, checkout clean). An earlier
gate at the same revision failed `test-pkg` with SIGSEGV under a load average of ~35 (a re-run passed, and
the suite also passed locally); the earlier pre-merge head `ef5e1b77` also passed the gate.

## Commits

| Commit | Subject |
|---|---|
| `1db4b1fe` | The per-family reference lock: `reference.in`/`reference.lock` beside every `reference.py` |
| `0e94afcf` | The node builds one reference environment per family, and the wave resolves it per recipe |
| `3c6f0bee` | Stage 2 stores its reference outputs and reuses an unchanged one |
| `a63e9e69` | Stage 2 gates the media rows, and a wave is grouped by engine image |
| `813b178d` | The recipe notes name `reference.in`/`reference.lock` (the migrated reference environment) |
| `27dcc255` | The reference-env docs, the CHANGELOG and the stale file references |
| `1c9ebf13` | Verifier round 1: the fifth media family, the gcloud copy, the migration leftovers |
| `ef5e1b77` | Verifier round 2: the gcloud destination, the nightly's freeze record, the doc stragglers |
| `d362aa0e` | Verifier round 3: the coverage the confirmation verifier asked for |
| `038d32cb` | Merge `rfc-0001` (`b18d34c4`) into `lane/ref-envs` |
| `2f7f6ec6` | The lane report |
| `d2df8066` | The store key carries no MRL selection: the full-width reference is shared by every k |

## What changed (per brief item)

**1. The lock tool and the per-family files.** `rcp_ndcg_test.jobs.reference_lock` (`build`/`check`)
resolves a family's short `reference.in` to exact, hashed pins with `uv pip compile --no-deps`; it
constrains torch and the CUDA stack to the engine image's freeze and nothing else (a family floor on the
stack is checked against the image and recorded as `# image-constraint`, never installed), refuses a stack
pin without the `# own-torch: true` + `# own-torch-evidence` declaration, and hashes the inputs into the
header. The 16 retrieval families migrated their `requirements-reference.txt` into `reference.in` (the
same justified pins) + the generated `reference.lock`; the stock image's torch/CUDA stack is committed as
`rcp-ndcg-vllm/reference-image-v0.31.0.txt`. Own torch: `ctxl-rerank-v2-instruct-multilingual`,
`qwen3-reranker` (flash-attn 2.8.3 needs torch 2.9.1) and `topk-embed-v1` (the checkpoint's own
`requirements.txt`). A family on another image whose stack is not committed records
`image-freeze-sha256: uncommitted` + `image-freeze-source` (embeddinggemma-2's nightly). The 8 judge
families from `rfc-0001` have no reference and no lock (decision 15).

**2. The bootstrap and the runner.** `bootstrap.sh` builds one venv per wave family from its lock
(`--system-site-packages` over the image's torch/CUDA by default, a venv of its own under own-torch),
installs the lock's pins `--no-deps` from the staged wheelhouse(s) (EXTRA_DIRS wheelhouses included for a
pin with no index wheel, e.g. flash-attn), completes the venv's own dependencies with `reference_deps.py`,
writes the family's `freeze.txt`, and runs `reference_env check` (torch imports — the image's build, or the
lock's pin under own-torch — every pin is installed at its version and imports; failures name the family).
`run_wave` gained `--reference-root` (resolving `<root>/<family>/bin/python` per recipe) and
`--reference-store`. `rc_build.sh` builds the unpublished `rcp-ndcg-test` wheel into the wheelhouse (the
client environment runs the wave runner and the checks) and downloads every family lock's wheels — the
own-torch families from PyPI.

**3. Stored reference outputs.** `rcp_ndcg_test.reference_store` keys stage 2's reference vectors and
scores by the family reference hash (every `*.py` in the family directory), the variant revision, the
pairs-file hash, the environment lock hash, the device and the dtype (plus the model, the mode and the
reference block). A wave keeps its store under `<out>/references` (`--reference-store` reuses a previous
wave's), computes only the missing or stale entries, and reports `computed`/`reused` with the fingerprint
and the inputs that moved. Staleness reuses `fingerprint.fingerprint_changes`; an entry is immutable and
its output hash is verified on every load.

**4. CPU tests and the family environment.** The tests run references through the dev interpreter; only
the node's stage 1 render comparison, stage 2 and the recorder use the family venv. The new tests are all
CPU (stub engines, stub references, a local wheelhouse).

**5. `equivalence.json`.** It records `reference_environment` (family, lock SHA-256, the venv's freeze) and
`reference_outputs` (`computed`/`reused`, the fingerprint, the changed inputs).

**Addendum 1 (media stage 2).** Stage 2 now runs over the media rows: the reference receives their `media`
field and the client sends the product's Content path, the same gates apply, and the outputs are stored like
the text rows'. A recipe declaring `reference.known_deviations: [media_approximation]` reports its media
rows non-gating with the reason instead (the media stage still gates placement, geometry and tokens). Five
families declare it — embeddinggemma-2, qwen3-vl-embedding, qwen3-vl-reranker, topk-embed-v1 and
pplx-embed-v2-late — because their references' media score/embed paths land with the E2 wave; this is the
honest half-implementation, recorded under Open questions.

**Addendum 2 (per-image jobs).** `rcp_ndcg_test.jobs.wavegroups` groups a wave list by each variant's
resolved `engine.image`; `submit.sh` downloads the staged wave list and recipes, groups them, and submits
one job per image (`rcp-<wave>-<slug>`, `env.RCP_IMAGE` = the recipe's image, the filtered list mounted and
passed as `--wave-list`); `RCP_GROUP_IMAGES=0` keeps the single-job plan. A stub-gcloud test covers the
gs:// branch.

**Decision 35** is already recorded in `handover/00-MASTER.md` (the owner's list); this lane implemented it
without editing the list.

## Verification

- **Gate** at `ef5e1b77`: PASS. Gate at `038d32cb` (the merged tree): PASS on the re-run. Gate at
  `d2df8066` (the MRL note applied): PASS. (a first run hit a
  SIGSEGV in `test-pkg` under heavy machine load; the identical suite passed locally and on the re-run).
- **Round 1** (two fresh verifiers, `deepseek-v4-1-flash:xhigh`; lens A correctness, lens B
  regressions/hygiene). Both **FAIL**.
  - A1/B1 (blocker): embeddinggemma-2, the fifth media family, did not declare `media_approximation` and
    its reference ignored the `media` field, so its 13 committed media rows would gate against a text-only
    vector. Fixed (declaration + `"media"` in `_MEDIA_COLUMNS`), goldens/contract updated; round 3
    confirmed with a real-call-site reproduction and a mutation.
  - A2 (blocker): `submit.sh`'s gcloud branch could not copy a directory (no `--recursive`, and the
    destination did not exist). Fixed and then re-fixed in round 2 (the destination must be pre-created and
    the source's contents copied — the gcs.sh rule); a stubbed-gcloud test covers it.
  - A3/B2 (major): `requirements-reference.txt` survived in the package manifest, `recipe.py`, the README,
    the pyproject comment and 13 reference docstrings/runtime hints. Fixed; the dead package-level file and
    its MANIFEST line are deleted; `git grep` shows only the migration guard.
  - A4 (minor): the store's `changed_inputs` never named a revision bump. Fixed (the closest entry is the
    same model and mode).
  - A5 (minor): the family reference hash covered only the entry file. Fixed (every `*.py` in the family
    directory; qwen3-vl-embedding's vendored card script included).
  - A6 (minor): `bootstrap.sh` could die on a missing `reference-families.tsv`. Fixed.
  - A7/B4 (minor): `wavegroups`' docstrings described a map no consumer reads. Fixed.
  - A8/B5/B6 (minor): docs corrections (the stage-2 media sentence, the report block's freeze wording, the
    own-torch staging recipe). Fixed.
  - B3 (minor): dead `REFERENCE_REQUIREMENTS` references. Fixed.
  - B7 (minor): the nightly lock's freeze was the stock image's. Fixed with the `uncommitted` +
    `image-freeze-source` record; a lock-image consistency test added.
- **Round 2** (one fresh confirmation verifier, both lenses). **FAIL**, one blocker and four minors:
  - F1 (blocker): the gcloud directory copy still failed (destination must exist; contents copy). Fixed
    (pre-create + wildcard contents, gsutil/local alike) and covered by a strict-stub test.
  - F2 (minor): the nightly freeze inconsistency. Fixed as above.
  - F3/F4/F5 (minor): the CHANGELOG's "four media families" (now five), the store's key docstring, the
    `GROUP_MAP` docstring. Fixed.
- **Round 3** (one fresh confirmation verifier, both lenses). **PASS**; four test-coverage minors, all
  fixed in `d362aa0e`: the gcloud stub now refuses a missing destination and checks the source (so dropping
  the `mkdir` or `--recursive` reds it), an uncommitted freeze requires its source line, the CLI test
  exercises `--image-freeze-source`, and the end-to-end bootstrap test runs once more with no family lock
  (the empty-families path).
- **Merge `rfc-0001` (`b18d34c4`)**: conflicts in `CHANGELOG.md`, `equivalence/stages.py`, `submit.sh`,
  `DELTAS.json` and three family YAMLs, resolved as the merge commit says; the goldens/DELTAS were re-synced
  to the merged notes/sources/`known_deviations`, the schema exports regenerated, the judge families skipped
  by the lock machinery, and `reference_of(recipe)` used where the merged `Recipe.reference` is optional.

## Checks

- `bin/gate lane/ref-envs` at `d2df8066` — **GATE: PASS** (all steps as listed
  in Status).
- `heavy uv run --no-sync pytest rcp-ndcg-test/tests -q -p no:cacheprovider` — 980 passed, 223 skipped.
- `uv run --no-sync pytest rcp-ndcg-test/tests/test_reference_lock.py
  rcp-ndcg-test/tests/test_reference_env.py rcp-ndcg-test/tests/test_reference_store.py
  rcp-ndcg-test/tests/test_wavegroups.py rcp-ndcg-test/tests/test_media.py
  rcp-ndcg-test/tests/test_jobs_scripts.py rcp-ndcg-test/tests/recipes -q -p no:cacheprovider` — 353
  passed, 221 skipped (pre-merge; the merged tree's 980 covers them).
- `uv run --no-sync ruff format . && uv run --no-sync ruff check . && uv run --no-sync basedpyright` —
  clean, 0 errors.
- The committed locks: 16/16 `reference_lock check` exit 0 offline; `test_every_committed_lock_matches_its_inputs`
  and `test_every_lock_names_its_familys_engine_image` green.

## Open questions

- **The MRL seam (operator note, 2026-10-10).** The store key carries no `k` (`client.mrl_dim`): the
  stored reference is the model's full-width output and the next-round harness derives every declared k
  from it (`MRL_GATE_VERSION`), so two k values read one entry (tested at the real `_reference_outputs`
  call site). When the MRL harness merges, it must not re-key the store per k.

- **Media stage 2 is wired but not exercised by a shipped reference.** The five media families declare
  `media_approximation`, so their media rows are reported non-gating; the addendum's "the family reference
  computes their outputs" needs each reference's media score/embed path (qwen3-vl-reranker's score mode
  already handles image columns but reads the old `query_image` fields, not the harness's `media`; the
  others compute text vectors only). The E2 wave should implement those paths and remove the declarations
  so the media gate bites. The reference-side work is GPU-only and was deliberately not faked here.
- **flash-attn has no PyPI wheel.** The two own-torch locks pin `flash-attn==2.8.3`; the operator must stage
  the prebuilt wheel in an `EXTRA_DIRS` wheelhouse (the bootstrap and `rc_build` search those) or the RC
  build fails loudly. The GPU validation of the own-torch venvs (torch 2.9.1/2.11.0 + CUDA) is E2's.
- **The nightly's stack is not committed.** embeddinggemma-2's lock records `uncommitted` +
  `image-freeze-source: vllm/vllm-openai:v0.31.0`; its floors were checked against the released image's
  stack. If the nightly's torch moves below a floor, only the runtime torch probe would notice the build,
  not the version.
- **The client environment's `rcp-ndcg-test` wheel** (needed by the wave runner) is staged by `rc_build.sh`
  into the wheelhouse; a stage built before this lane lacks it. This was a latent post-layout-move break,
  fixed here (Files outside scope).
- **The store's location** is the wave output (`<out>/references`), so the operator reuses it by pointing
  `RCP_REFERENCE_STORE` at a downloaded copy; multi-vector outputs are hundreds of MB per recipe, which is
  why nothing is committed to the repo.

## CHANGELOG entry

Public surface:

> - **The recipe schema's `reference.known_deviations` gains `media_approximation`** (owner decision 35): a
>   recipe declaring it reports stage 2's image/video rows non-gating with the reason (the family reference's
>   media score/embed path is not wired yet; the media stage still gates placement, geometry and tokens).  The
>   exported `schema/recipe.schema.json` and `schema/family.schema.json` carry it.

Fixed:

> - **Stage 2 compares the media rows** (owner decision 35): stages 1 and 2 used to drop every image/video row,
>   so no reranker score or embedding vector of a media input was gated for any media recipe.  Stage 2 now runs
>   over the media rows too: the reference receives their `media` field, the client sends the product's Content
>   path, the same gates apply and the outputs are stored like the text rows'; a recipe declaring
>   `reference.known_deviations: [media_approximation]` reports its media rows non-gating with the reason
>   instead.  The five media families declare the approximation until their references' media score/embed paths
>   land with the E2 wave.

Changed:

> - **One reference environment per family** (owner decision 35): ... (the full entry is in `CHANGELOG.md`
>   under `## Unreleased`; it covers the lock tool, the migrated `reference.in`/`reference.lock`, the
>   per-family bootstrap venvs and `--reference-root`).
> - **Stored reference outputs** (owner decision 35): ...
> - **A wave is submitted as one GPU job per engine image** (owner decisions 38/35): ...
> - **`rc_build.sh` stages the unpublished test wheel and the families' reference wheels**: ...

## Public surface changes

- `rcp-ndcg-vllm`'s recipe schema: `reference.known_deviations` accepts `media_approximation`; the exported
  `schema/recipe.schema.json` and `schema/family.schema.json` regenerated. No CLI, exit-code or
  product-schema change.
- The recipe package data: every retrieval family's `requirements-reference.txt` is replaced by
  `reference.in` + `reference.lock`; `rcp-ndcg-vllm/requirements-reference.txt` and its `MANIFEST.in` line
  are removed; `rcp-ndcg-vllm/reference-image-v0.31.0.txt` is new.
- `rcp-ndcg-test` (unpublished): `jobs.reference_lock`, `jobs.reference_env`, `jobs.wavegroups` and
  `reference_store` are new; `jobs.reference_deps` gains `satisfies` and several wheelhouses;
  `equivalence.run`/`stage2_scores` gain `reference_store`/`reference_environment`; `run_wave` gains
  `--reference-root`/`--reference-store`; `submit.sh` groups by image; `rc_build.sh` stages the test wheel.

## Files outside scope

- `rcp-ndcg-test/src/rcp_ndcg_test/jobs/rc_build.sh` and the bootstrap's client spec (the unpublished
  `rcp-ndcg-test` wheel): a latent post-layout-move break — the client environment ran
  `python -m rcp_ndcg_test...` without the package installed. Fixed because the lane's own runner changes
  cannot run otherwise.
- `rcp-ndcg-test/src/rcp_ndcg_test/equivalence/stages.py`'s two `TemporaryDirectory(ignore_cleanup_errors=True)`:
  a scratch-cleanup race on the network-backed temp dir failed a stage; the media module already did this.

## For the next lanes

- **E2 / the media wave**: implement the five references' media score/embed paths (qwen3-vl-reranker's
  `media` decode first), remove the `media_approximation` declarations, and regenerate the affected
  goldens/DELTAS. The stage-2 wiring, the storage and the non-gating table are in place.
- **The GPU operator**: stage the flash-attn wheel in an `EXTRA_DIRS` wheelhouse; run `rc_build` from a
  commit that includes this lane (the test wheel and the family locks are staged); the RC's wheelhouse now
  carries the own-torch CUDA torch (PyPI).
- **A future family**: `reference.in` + `python -m rcp_ndcg_test.jobs.reference_lock build ...` +
  `check` (docs/how-to/add-a-model.md, "How to pin a reference").
