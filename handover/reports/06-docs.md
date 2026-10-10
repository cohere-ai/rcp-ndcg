# Lane docs06 — the final documentation (workstream 06)

**Status: DONE.** Branch `lane/docs06`, code HEAD `72b104f2` (this report is the commit after it), base
`rfc-0001` merged at `a45d4f7b` (the judge equivalence skip; the merge commit is `18f6b914`, clean). The tree
is clean
(`git status --porcelain --untracked-files=all` prints nothing) and `bin/gate lane/docs06` passes on the final
merged HEAD — the report commit; only CHANGELOG prose, this report and the merge's two test-package files
differ from the gate run on `6f3d6dc3`, whose step list the Checks section quotes.

## Commits (lane stack, in order)

| Commit | Subject |
|---|---|
| `88c70bc0` | The recipe catalog lists all 44 variants with their reference and status, pinned by a test |
| `4ce7bb25` | The final docs: the 24/44 catalog counts, the compatibility and versioning page, the judges-page fixes and the pinned README images |
| `057a6dc5` | Fold Unreleased and 0.1.0 into one 0.0.1 entry: the overview, the 24-family recipes entry and the corrections |
| `6f3d6dc3` | The CHANGELOG's historical paths and the last stale counts point at the final tree |
| `72b104f2` | The 0.0.1 entry records the final documentation pass |

## What changed (per brief item)

1. **Moved paths.** `docs/quickstart.md`'s git install gained the missing `#subdirectory=rcp-ndcg` (the
   `rcp-ndcg[hf,calibrate] @ git+...@v0.0.1` line pointed at the workspace root, which is no longer a package);
   `REPRODUCIBILITY.md` and `experiments/README.md` install `./rcp-ndcg-core ./rcp-ndcg` (the trailing `.`
   was the root, not a package). The root `README.md` is the landing card (pitch, the four distributions with
   each README linked, the install one-liners, and a Documentation section linking the site, the quickstart,
   the CHANGELOG, REPRODUCIBILITY, AGENTS and CITATION); `rcp-ndcg/README.md` is the PyPI long description
   (`rcp-ndcg/pyproject.toml` `readme = "README.md"`). `mkdocs.yml`'s nav gained the versioning page (the nav
   == pages test passes). `AGENTS.md`, `docs/how-to/release-candidates.md`, `docs/how-to/add-a-model.md`,
   `skills/rcp-ndcg/SKILL.md` and `docs/reference/rcp-ndcg-test.md` had no stale `packages/` or root-package
   paths left; every internal design id (`owner decision N`, `decision N`, `layout-move item N`, `(review X)`)
   was removed from the user-facing pages, READMEs, skill and AGENTS.
2. **The now-true sentences.** The recipe mapping (`recipe: <id>`, `--retriever/--reranker recipe:<id>`,
   `rcp-ndcg-vllm serve <id>`) and the lean dependency set (pydantic + PyYAML, `--no-deps`, `pip freeze`
   differs by one wheel) were already stated by the earlier lanes and were re-verified against the code. The
   release gates the move added (a fresh-venv wheel install of each distribution, the `--no-deps` freeze check,
   `serve --dry-run` for every recipe, `twine check`) are now stated in `AGENTS.md` "Releasing"; the node-side
   gates stay in `docs/how-to/release-candidates.md`. The catalog is documented as **24 families / 44 variants**
   (34 retrieval variants in 16 families, 10 judge variants in 8 families) everywhere it is named: the
   `rcp-ndcg-vllm` README, `docs/reference/recipes.md`, `docs/index.md`, `docs/quickstart.md`,
   `docs/how-to/serve-a-model.md` (was "31 of them"), `rcp-ndcg/README.md` (was "14 families / 20 variants")
   and `skills/rcp-ndcg/SKILL.md`.
3. **The recipe catalog page.** The `rcp-ndcg-vllm` README's table now has **44 rows** (the 10 judge variants
   were missing), a `reference` column (`paper` = the family's `reference.py` is the paper's own code path
   ported from `experiments/paper/`; `card` = the model card's published usage; `—` for a judge recipe, which
   carries no equivalence reference), and the `status` column read from each variant's own `status.state`. The
   table's rows are pinned by `rcp-ndcg-vllm/tests/test_catalog.py` (extended: all 44 recipes, the reference
   cell against the family reference's header, the status against the resolved recipe). `docs/reference/recipes.md`
   documents the columns and links the README table (the one rendered copy); its duplicated and stale
   16-family table is gone.
4. **CHANGELOG.** `## Unreleased` and the stale `## 0.1.0` draft are one `## 0.0.1 — <date at tag time>` entry
   (the release checklist fills the date): a new `### Overview` (four distributions, three published) and
   `### Highlights`; the 0.1.0 Overview/Public surface/Fixed/Changed/`Also in this release` dissolved into it
   (nothing lost except the never-released `0.1.0` heading); the recipe-family bullets folded into one
   "shipped recipes: 24 families, 44 variants" entry; the eight "tests/contract snapshots ... regenerated for
   X" bullets merged into one closing bullet; the internal review ids and the interim clauses the code moved
   past dropped; the Versioning preamble corrected (the four distributions, the 23 public modules from
   `PUBLIC_MODULES`, the extras without `local`/`vllm`) and linked to the versioning page. `rcp-ndcg-test` is
   mentioned only where it affects the published packages.
5. **The amendments** were documented by the earlier docs lanes and re-verified here: the per-row
   `ProcessingRecord` and what the harness gates (`docs/concepts/text-budgets.md`), `document_max_tokens`
   (`text-budgets.md`, `docs/api/inference.md`, `add-a-model.md`), `engine_pixel_pinning`
   (`preprocessing.md`), the messages route and the declared `add_generation_prompt` (`embeddings.md`,
   `add-a-model.md`), the media equivalence gate (`validate-a-recipe.md`), the corpus staleness gate and
   `stale.json` (`use-verified-fake-engines.md`), vLLM only (no "vLLM or SGLang"; SGLang appears only as the
   paper's history), judge recipes and `--judge <id>` (`judges.md`), `schema_version` (`add-a-model.md`,
   `serve-a-model.md`, the new versioning page), the emulators in `rcp-ndcg-test`
   (`reference/rcp-ndcg-test.md`, `use-verified-fake-engines.md`) and `rcp_ndcg.judging` (`judges.md` and the
   versioning page's public-module list).
6. **The compatibility and versioning policy page.** New `docs/reference/versioning.md`: what is public (the
   contract snapshots and the 46 exported `schemas/*.v1.json`), the `0.0.x` rules and what a patch release may
   change, the deprecation path (one minor release of warning, then removal, a CHANGELOG entry each time), the
   recipe `schema_version` contract and `RECIPE_SCHEMA_VERSIONS` refusal, the behaviour fingerprint
   (`rcp-fp/4` and when it bumps), the observation-corpus and record schema versions
   (`rcp-ndcg.observation-corpus/1`, `RECORD_SCHEMA`, `NORMALISATION_VERSION`, `rcp-ndcg.verification/1`), the
   `*.v1` artifact tags and a version table. Linked from `rcp-ndcg/README.md`, `AGENTS.md`, `docs/index.md`
   and the nav; `REPRODUCIBILITY.md`'s stale `rcp-fp/3` mention is gone.
7. **The judges page.** `docs/concepts/rubric.md` now documents the coverage refusal
   (`n_random * w >= n_units`), its hint (raise `placements_per_doc`/`random_share`, or lower `window`) and
   the shipped defaults' ~3.33-chunks-per-document limit (verified against `RubricSchedule.windows_for`);
   `docs/reference/cli.md` gains the `--seed` row (the bootstrap seed on `eval`/`sweep`, the schedule and
   offline-judge seed on the judge commands); the judge serve examples pin `vllm/vllm-openai:v0.31.0` (was
   v0.30.0 in four `runs.md` examples); the shipped `--judge` names are in the judges table; the media gate's
   scope and the one Content lowering were already documented.
8. **README images for PyPI.** `rcp-ndcg/README.md`'s light/dark `<picture>` and the pipeline figure use
   absolute `raw.githubusercontent.com` URLs pinned to the release tag (`v0.0.1`); the `<picture>` keeps
   working on GitHub. Pinned by a new `tests/docs/test_readme_pypi.py` test (red on `main`, green on the tag).

## Verification

- `rcp-ndcg-vllm/tests/test_catalog.py` (new in this lane's scope): green on the 44-row table; mutating one
  README status cell to `verified` turns it red (`FAILED test_the_readme_catalog_matches_iter_recipes`), and
  restoring it turns it green.
- `tests/docs/test_readme_pypi.py::test_readme_images_are_pinned_to_the_release_tag`: red when the image URLs
  are `main` (`AssertionError: README images must be pinned to v0.0.1`), green with `v0.0.1`.
- `uv run --no-sync pytest tests/docs tests/contract -q -p no:cacheprovider` → **307 passed, 57 skipped**.
- `uv run --no-sync mkdocs build --strict -d <scratch>/site` → built (no warnings).
- `uv run --no-sync ruff format --check .` → 612 files already formatted; `ruff check .` → clean;
  `basedpyright` → 0 errors.
- The catalog's status counts: **44 rows, 44 `unverified`** — every variant's own `status.state` in
  `family.yaml`. The E2 waves have verified five variants so far (`qwen3-embedding-4b`;
  `jina-embeddings-v5-text-small`; `octen-embedding-0.6b`/`-4b`/`-8b` per ORCHESTRATION-STATE §13), but their
  `status:` flips are the P4 operator's step ("flip each recipe's `status` to `verified` from its wave
  evidence only"), and the wave receipts live in the private results store, so this lane did not flip them:
  the catalog stays a faithful rendering of the recipe files. The page states the policy ("every variant must
  pass its end-to-end GPU validation before the tag; the tag ships none unverified").

## Checks (commands run last, and their results)

```
bin/gate lane/docs06                 rev lane/docs06 = 6f3d6dc3 (slot 3)
ruff-check exit=0, ruff-format exit=0 (612 files), basedpyright exit=0
pytest exit=0          4002 passed, 105 skipped
contract-docs exit=0   307 passed, 57 skipped
mkdocs exit=0          Documentation built
test-pkg exit=0        1111 passed, 227 skipped
recipes exit=0, vllm-pkg exit=0 (50 passed), vllm-models exit=0 (92 passed, 7 skipped)
run_all exit=0         1022 checks, 987 match, 35 known deviations, 0 failed
public-names exit=0    clean
clean exit=0
GATE: PASS
```

The gate above ran on `6f3d6dc3`; the commits after it are CHANGELOG prose, this report and the merge of
`rfc-0001` (`a45d4f7b`, the judge equivalence skip), and
`bin/gate lane/docs06` is re-run on the final merged HEAD as the acceptance gate: every step
exits 0 and the SUMMARY reads `GATE: PASS` (the run's log is in `gates/<sha>/`).

## Docs updated

- `README.md` — the landing card's Documentation section (site, quickstart, CHANGELOG, REPRODUCIBILITY,
  AGENTS, CITATION).
- `rcp-ndcg/README.md` — images pinned to `v0.0.1`; the "will be on PyPI"/"until the release is up" hedges
  gone; the 24/44 count; the versioning-page link; `retriever.yaml` for the path-2 example.
- `rcp-ndcg-vllm/README.md` — the intro (retrieval models and judges; the topk/pplx plugins), the 44-row
  catalog with the `reference` and `status` columns, the catalog caption.
- `rcp-ndcg-test/README.md` — the internal decision id removed.
- `docs/reference/versioning.md` — new.
- `docs/reference/recipes.md` — the 24/44 counts, the README link, the `reference` column, the stale family
  table replaced by the judge-family paragraph.
- `docs/reference/cli.md` — the `--seed` row.
- `docs/reference/rcp-ndcg-test.md` — the internal decision id removed.
- `docs/concepts/rubric.md` — the coverage refusal, its hint and the shipped defaults' limit.
- `docs/concepts/runs.md` — the four `v0.30.0` example images to `v0.31.0`.
- `docs/concepts/judges.md` — internal decision ids removed (the substance stays).
- `docs/how-to/serve-a-model.md` — "34 retrieval models" (was 31); the internal decision id removed.
- `docs/how-to/add-a-model.md`, `docs/how-to/validate-a-recipe.md`, `docs/how-to/release-candidates.md` —
  internal decision ids and the `layout-move item` reference removed.
- `docs/index.md` — the 24/44 count; the results-record and versioning links.
- `docs/quickstart.md` — the git install's `#subdirectory=rcp-ndcg`; the 24/44 count.
- `mkdocs.yml` — the versioning page in the nav.
- `AGENTS.md` — the release gates the move added; the versioning-page link.
- `REPRODUCIBILITY.md` — the install line; the recipe bullet in section 3.
- `experiments/README.md` — the install line.
- `skills/rcp-ndcg/SKILL.md` — the 24/44 count.
- `CHANGELOG.md` — the 0.0.1 fold.

Grep commands run (all now empty or naming only the correct counts):
`git grep -n -E "\b(13|14|16|18|19|20|30|31|34) (recipe|recipes|families|variants|sizes)"` over `docs README.md
rcp-ndcg rcp-ndcg-core rcp-ndcg-vllm rcp-ndcg-test skills examples experiments AGENTS.md REPRODUCIBILITY.md`;
`git grep -n -E "owner decision [0-9]|decision [0-9]|layout-move item|workstream [0-9]"` over the same paths;
`git grep -n "v0\.30\.0|SGLang|rcp-fp/3|packages/|mcp tools|--judge-urls"` over `docs skills README.md`;
`git grep -n "subdirectory|pip install \./rcp-ndcg-core"` over `docs experiments README.md REPRODUCIBILITY.md`.

## CHANGELOG entry

This lane writes no separate bullet: it changes no public surface (no snapshot, schema, CLI or exit code).
Its CHANGELOG work is the fold itself. The `## 0.0.1 — <date at tag time>` entry's heading and Overview are
quoted in the report's item 4 above; the lane's own documentation changes are recorded in the entry's
"documentation is reorganised" Changed bullet, extended with the versioning page, the catalog's `reference`
and `status` columns, the judge-page fixes and the README split.

## Public surface changes

None. No `tests/contract/snapshots/` or `schemas/` file changed (so the CI CHANGELOG-entry check does not
apply). Two tests changed or were added: `rcp-ndcg-vllm/tests/test_catalog.py` (extended to all 44 recipes,
the reference cell and the status) and `tests/docs/test_readme_pypi.py` (the release-tag pinning check).

## Files outside scope

- `rcp-ndcg-vllm/README.md` and `rcp-ndcg-vllm/tests/test_catalog.py` — the catalog page and its pin (the
  brief names the catalog page; the test is the pin).
- `experiments/README.md` — the stale install line.
- `tests/docs/test_readme_pypi.py` — the image-pinning test (the brief's item 8).

## Open questions

- **The recipe statuses.** The catalog reads `unverified` for all 44 because the recipe files do; the five
  variants the E2 waves have verified await the operator's `status:` flip (with the engine image, the date
  and the report). If the P4 operator flips them before the tag, the table's `status` cells change with the
  files (the test enforces it); no docs edit is needed.
- **The CHANGELOG date.** The heading is `## 0.0.1 — <date at tag time>` per the release checklist; the
  checklist sets the ISO date at tag time.
- **The recipe-family fold.** The per-family bullets (the pplx pair, the late-9b variant, embeddinggemma-2,
  the Qwen3 ladders, the harrier trio, the pplx-late-0.6b recipe, the recipe follow-ups and the five
  family-restructure bullets) are one entry now; the per-family mechanics live in each `family.yaml`'s notes
  and `sources` and in the docs pages, and the entry names every family and size the release adds.
- **`rcp-ndcg-test` in the CHANGELOG.** Its `New package` bullet and the Public surface "rcp-ndcg-test"
  paragraph stay (it is a distribution of this repository and its tooling affects the published packages);
  it is never presented as a published surface.

## For the next lanes

- **QA 07**: the public-surface freeze can point at `docs/reference/versioning.md` for the rules; the 46
  `schemas/*.v1.json` count and the 23 public modules are stated there and in `CHANGELOG.md`.
- **The release checklist**: set the CHANGELOG date; flip the verified recipes' `status` from the wave
  evidence; the README images' `v0.0.1` URLs resolve once the tag exists.
- **Any later docs lane**: the catalog table is pinned by `rcp-ndcg-vllm/tests/test_catalog.py` and the
  README images by `tests/docs/test_readme_pypi.py`; a new recipe family or a version bump updates the table
  and the tag in the URLs.
