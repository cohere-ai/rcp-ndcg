# Lane `rec-overrides`: deployment overrides at serve time and user recipe files (owner decision 36, 2026-10-09)

## 1. Status

DONE. Branch `lane/rec-overrides`, based on `lane/rfam` and merged with it three times (the last merge,
`d97a1c18`, carries rfam's own `rfc-0001` merge: workstream 08 A vLLM-only and the MRL cards). `bin/gate
lane/rec-overrides` is PASS on the final tree (section 5).

## 2. Commits

| commit | subject |
|---|---|
| `55cf3695` | The deployment surface and the serve-time overrides (decision 36): the recipe schema declares each field's role once (FIELD_ROLES: CONTENT/RUNTIME/DEPLOYMENT), serve renders exactly the DEPLOYMENT paths from --set with the max_model_len budget floor, the console gains --set and --variant, and a family directory of the operator's own loads through the same schema, unshipped and unverified, with a content-hash identity |
| `af7b30fa` | recipe: paths and the unshipped identity reach the product and the provenance |
| `6a51c725` | docs: the deployment overrides, the user recipe files and the refusals; decision 36 in the master handover; the CHANGELOG entry |
| `19e9e5b7` | merge `lane/rfam` (its golden deltas for the workstream-09 note edits) |
| `78c885f2` | round-1 verifier fixes: the identity pointer passes through, the identity hashes the template, the port and gpu_memory_utilization values are checked, an id-shaped source is the catalog's recipe first, the `~` claim, the serve docstring and the docs line |
| `1691f1c1` | the `serve.port` floor is 0, not 1 (the wave runner's ephemeral-port convention) |
| `68a52fcc` | merge `lane/rfam` (the golden guard's shrink-only check, key-unique DELTAS, operator paths out of the goldens) |
| `cef7279f` | round-2 verifier fixes: the CHANGELOG and docs state the port floor and what the provenance records; a path below a content field is refused with the variant hint; `FieldSpec` in the reference list; the add-a-model pointer |
| `1e4785ac` | round-3 verifier and gate fixes: the referenced-file check speaks before the identity reads the template; a path below a DEPLOYMENT/RUNTIME field is refused as unknown; `FIELD_ROLES` is public; the provenance and spec wording; `RecipeError` named |
| `d97a1c18` | merge `lane/rfam` (its verifier round's findings and its `rfc-0001` merge: workstream 08 A vLLM-only, the MRL cards) |

`rfc-0001` itself was not merged directly: `lane/rfam` is not merged into `rfc-0001` yet, so per the brief the
lane merged `lane/rfam`, whose last commit is a merge of `rfc-0001` at `28afb3b7` — that commit's content is in
this branch.

## 3. What changed

**1. Deployment overrides at serve time.** The recipe schema declares the deployment surface once
(`rcp_ndcg_vllm.recipe.FIELD_ROLES`, values `RecipeFieldRole.CONTENT`/`RUNTIME`/`DEPLOYMENT`, entries
`FieldSpec` with the `vllm serve` flag, the value kind and range, and the argv position).
`rcp-ndcg-vllm serve <id> --set <path>=<value>` names only a DEPLOYMENT path —
`resources.gpus`, `serve.gpu_memory_utilization`, `serve.max_num_seqs`, `serve.max_num_batched_tokens`,
`serve.host`, `serve.port`, `serve.max_model_len` — and `serve_argv(recipe, deployment=...)` renders exactly
those; the CLI never lists fields. `serve.max_model_len` is refused, with both numbers and the budget field
named, below the client's largest token budget (`max_tokens`/`query_max_tokens`/`document_max_tokens`); raising
it is allowed (the engine enforces the checkpoint's own limit at startup). A CONTENT path is refused by name
with the hint *a different revision or content is a different variant: add a variant row*;
`engine.startup_timeout_s` is refused as RUNTIME. Values are checked against their declared kind and range
(finite, `gpu_memory_utilization` strictly above 0, port 0..65535, with 0 the engine's ephemeral port).
`--dry-run` prints the argv, the identity and the applied overrides; a real serve logs the identity and the
overrides; a corpus manifest records the argv each engine was started with verbatim.

**2. User recipe files.** `rcp-ndcg-vllm serve ./family-dir/ [--variant <id>]` and
`recipe:./family-dir` / `recipe:/abs/path` (configs and the `--retriever`/`--reranker` shorthands) load a family
directory through the same schema, families included, with the `schema_version` check unchanged. Such a recipe
is unshipped: its `status` is forced to `unverified` whatever the file claims, `Recipe.shipped` says so, the
corpus provenance records `shipped`, and its identity is the content hash of its resolved form — the referenced
chat template file's bytes included, computed once at load — `unshipped:sha256:<hex>` via `recipe_digest`,
never a shipped id. The identity is what `client_config`/`expand_role_recipe` put in the config's `recipe`
field, so a recorded config reads back as it stands (a run's `status`/resume, an index reload). An id-shaped
source is the catalog's recipe first; a directory of the same name is named `./name`.

**3. Docs, snapshots, schemas.** `docs/how-to/serve-a-model.md` (a deployment section and an own-recipe-file
section, with what is refused and why), `docs/reference/recipes.md` (the console line, the public names, the
`recipe:` section), `docs/how-to/add-a-model.md` (a pointer to the file-of-your-own route), the
`rcp-ndcg-vllm` README, the module docstrings and the CLI help. `tests/contract/snapshots/{vllm_cli,python_api}.json`
regenerated (`--update-snapshots`), `rcp-ndcg-vllm/schema/recipe.schema.json` regenerated from
`recipe_json_schema()` (the `Recipe` docstring only; no field moved, and the shipped recipes' resolved
contract, serve argv and behaviour fingerprint are unmoved — the golden guard is green with no new delta).

## 4. Verification

Tests first: the new tests were written and run red before each fix. Evidence (scratch, outside the
repository): the console module failed to collect (`ImportError: cannot import name 'FIELD_ROLES'`), the
product tests failed with `no shipped recipe of that id`, the provenance test with the multi-variant refusal;
the round-1 tests failed 5/2; the gate's `recipes` step failed three zerank tests with
`Actual message: "recipe zerank-1-small-reranker: the template file ... cannot be read"` against
`match="chat_template"`.

**Round 1 — two fresh verifiers (correctness; regressions/hygiene), both FAIL.**
- Lens A: blocker — an unshipped identity pointer could not be re-read, so `run start`/`run status`/resume and
  `retrieval index`'s reload failed; major — the identity ignored the template file's bytes; minors — NaN
  accepted for `gpu_memory_utilization`, the console shadowing a shipped id with a same-named directory, `~`
  claimed as a path form but never expanded, `shipped` not named in the spec.
- Lens B: the same blocker (reproduced through `RunConfig`/`load_index`); minors — `--port` bypassing the
  declared range, NaN, `~`, a false "same three lines" docstring, the real-serve log lines untested, a
  status test that could not fail, the usage line missing `--variant`.
- Fixed: the pointer pass-through in `expand_role_recipe` (+ two round-trip tests: `expand_role_recipe` and
  `RunConfig.from_data(config.resolved())`), the template bytes in `recipe_digest` (computed once at load), the
  finite/positive check with `low_exclusive`, the caller's port through the same check (with a `--port <n>`
  label), id-first resolution, `~` dropped, the docstring, a real-serve log test, a status test that fails under
  mutation (a monkeypatched `default_recipes_root`), the usage line. Mutations of three of these fixes turned
  the matching tests red.

**Round 2 — one fresh confirmation verifier (both lenses), FAIL, minors only (all round-1 fixes confirmed).**
- Findings: the CHANGELOG still said 1..65535; the provenance sentence claimed more than any path delivers; the
  recipes reference omitted `FieldSpec`; the pointer is trusted as-is (documented, worth one explicit sentence).
- Fixed: the range text, the provenance wording, `FieldSpec` listed, the "trusted snapshot" sentence, and a
  path below a content field (`serve.hf_overrides.architectures`) refused with the variant hint.

**Round 3 — one fresh confirmation verifier (both lenses), PASS**, five minors: the nested rule mislabelled
paths below DEPLOYMENT/RUNTIME fields as CONTENT; the provenance sentence still overclaimed; `PROVENANCE_KEYS`
lacks the new `shipped` key; `RecipeError` missing from the recipes reference; `FIELD_ROLES` named in the docs
but not public. Five mutations went red on the matching tests.
- Fixed: the nested rule now consults the role (a path below a DEPLOYMENT/RUNTIME field is refused as unknown),
  the provenance sentence says exactly what is recorded (the manifest records the argv each engine was started
  with; the GPU waves serve as shipped), `FIELD_ROLES` is exported (snapshot regenerated), `RecipeError` is
  named, and the spec marks `shipped` as an extra key. `PROVENANCE_KEYS` is deliberately **not** extended: it is
  the required-key set, and every committed corpus manifest would become incomplete (a re-record on GPU).
- The gate then found the zerank template tests (network-gated, not run locally): the identity's template read
  came before the referenced-file check, so a missing template was refused with a different message. Fixed by
  ordering the check first and naming `serve.chat_template` in the identity's own refusal; a new offline test
  pins the order.

## 5. Checks

The final tree is `d97a1c18` (the merged tree); `bin/gate lane/rec-overrides`:

```
ruff-check exit=0 All checks passed!
ruff-format exit=0 529 files already formatted
basedpyright exit=0 0 errors, 0 warnings, 0 notes
pytest exit=0 3276 passed, 93 skipped
contract-docs exit=0 287 passed, 52 skipped
mkdocs exit=0 Documentation built in 1.58 seconds
test-pkg exit=0 603 passed, 231 skipped
recipes exit=0 recipes: no failure outside the baseline (0 baseline failures remain, 34 fixed; pytest exit 0)
vllm-pkg exit=0 32 passed
vllm-models exit=0 70 passed, 7 skipped
run_all exit=0 leaderboards 1022/987/35/0, human study 67/67, external LLM judges 82/82
public-names exit=0 public-names: clean (0 baselined hits remain)
clean exit=0 clean
GATE: PASS
```

Targeted runs on the same tree: `pytest rcp-ndcg-vllm/tests/test_serve_console.py` 31 passed,
`pytest tests/inference/test_recipe_reference.py` 19 passed,
`pytest rcp-ndcg-test/tests/test_deployment_overrides.py rcp-ndcg-test/tests/test_recipe.py` 28 passed,
`pytest rcp-ndcg-test/tests/recipes/test_family_goldens.py` 24 passed,
`pytest rcp-ndcg-vllm/tests --ignore=.../models` 32 passed.

## 6. Open questions

1. **The wave runner cannot pass deployment overrides.** It serves the recipes as shipped
   (`run_wave.py` builds the argv with no `deployment`), so a GPU wave cannot lower e.g.
   `gpu_memory_utilization` for a big model; the manifest still records the argv verbatim, so the record is
   honest. A wave-level `--set` (wave list or CLI) is a harness feature for a later lane.
2. **`PROVENANCE_KEYS` does not require `shipped`** (section 4); the spec now says it is an extra key. If the
   owner wants it required, the committed corpora need re-recording (GPU) or a manifest schema bump.
3. **A dumped/re-validated unshipped `Recipe` becomes `shipped`** (`_shipped` is derived state, not a field —
   a field would move every golden). No live path does this today; a future one should carry the marker.
4. **The `recipe:` pointer is a trusted snapshot**: a config carrying `unshipped:sha256:<hex>` is accepted as it
   stands (the file is not consulted again, which is the point — it may not exist on the reading machine). The
   digest cannot be re-derived from the client block alone, so tamper detection is not possible; documented.
5. **`--set serve.port` wins over `--port`** when both are given (`--port` is the default the run supplies).
6. **`recipe:` cannot select a variant of a multi-variant user family directory** (the console can, with
   `--variant`): it is refused naming the variants. A path form with a variant selector would be a small
   addition if wanted.
7. **The role vocabulary is re-declared** as `RecipeFieldRole` because the lean package must not import
   `rcp-ndcg` (the engine image installs it with `--no-deps`); the values are rcp-ndcg's
   (`content`/`runtime`) plus `deployment`. Named differently from `FieldRole` on purpose, so the contract's
   one-home check stays strict (the `Family` precedent is declared there instead).
8. **`serve.port`'s floor is 0**, not 1: port 0 is the run's documented ephemeral-port convention.

## 7. CHANGELOG entry

Under `## Unreleased` → `### Public surface`, the exact text:

```
- **Deployment overrides at serve time** (owner decision 36): `rcp-ndcg-vllm serve <id> --set <path>=<value>`
  sets the engine's resource, scheduling and address knobs without touching the recipe. The recipe schema
  declares that surface once (`rcp_ndcg_vllm.recipe.FIELD_ROLES`, whose values are the `RecipeFieldRole`
  `CONTENT`/`RUNTIME`/`DEPLOYMENT` roles), and only a DEPLOYMENT path may be named: `resources.gpus`
  (`--tensor-parallel-size`), `serve.gpu_memory_utilization`, `serve.max_num_seqs`,
  `serve.max_num_batched_tokens`, `serve.host`, `serve.port` and `serve.max_model_len` -- the last refused,
  with both numbers, below the client's largest token budget (`client.max_tokens`, `query_max_tokens` or
  `document_max_tokens`), because the engine would reject admissible prompts; raising it is allowed, up to the
  checkpoint's own limit, which the engine enforces at startup. A CONTENT path (the model, the revision,
  `serve.dtype`, the pooler config, a template, the hf overrides, a patch) is refused by name with the hint
  *a different revision or content is a different variant: add a variant row*; `engine.startup_timeout_s` is
  refused as RUNTIME (the run owns it). `FIELD_ROLES` is public (with `RecipeFieldRole` and `FieldSpec`).
  `--dry-run` prints the argv, the recipe's identity and the applied
  overrides; a real serve logs the identity and the overrides; a corpus manifest records the argv each engine
  was started with verbatim (`engine.serve_argv`), so an engine started with overrides is recorded with them
  (the GPU waves serve the recipes as shipped). A value is checked against its declared kind and range
  (`--port`/`serve.port` 0..65535, 0 being
  the engine's own ephemeral port; a finite `serve.gpu_memory_utilization` strictly above 0), and the refusal
  names the flag the operator used.
  `rcp_ndcg_vllm.recipe` gains `RecipeFieldRole`, `FieldSpec`, `deployment_fields`,
  `parse_deployment_overrides` and `recipe_digest`, `serve_argv` gains the `deployment` keyword (its `port` is
  now optional: the deployment value, else the caller's port, applies), and the `rcp-ndcg-vllm` console gains
  `--set`.
- **User recipe files** (decision 36): `rcp-ndcg-vllm serve ./family-dir/ [--variant <id>]` and
  `recipe:./family-dir` (or `recipe:/abs/path`) in `rcp-ndcg` configs and the `--retriever`/`--reranker`
  shorthands load a family directory through the same schema, families included, with the `schema_version`
  check unchanged. A name that looks like a recipe id is the catalog's recipe first (a directory of the same
  name in the working directory does not shadow it; `./name` names the file). Such a recipe is **unshipped**:
  its `status` is forced to `unverified` in every record (the verification record belongs to a shipped
  recipe), `Recipe.shipped` says so, and its identity is the content hash of its resolved form --
  `Recipe.identity`, `unshipped:sha256:<hex>` via `recipe_digest`, the referenced chat template file's bytes
  included, computed once at load -- never a shipped id, so two runs whose files differ never share a run
  identity and the path's spelling is not part of it. A config that records that identity is read back as it
  stands (a run's `status`/resume, an index reload): the pointer is recognised, so `recipe:./dir` works end to
  end. `load_recipe` gains the `variant` keyword, the console gains `--variant`, `client_config` and
  `expand_role_recipe` put that identity in the config's `recipe` field, and the corpus provenance
  (`rcp_ndcg_test.observe.provenance.recipe_facts`) records `shipped`.
```

## 8. Public surface changes

- `rcp_ndcg_vllm.recipe` `__all__` gains `FIELD_ROLES`, `FieldSpec`, `RecipeFieldRole`, `deployment_fields`,
  `parse_deployment_overrides`, `recipe_digest`; `Recipe` gains the `shipped` and `identity` properties;
  `load_recipe` gains the keyword `variant`; `serve_argv` gains the keyword `deployment` and its `port` is now
  optional. `rcp_ndcg_vllm` re-exports the same names.
- The `rcp-ndcg-vllm` console gains `--set PATH=VALUE` (repeatable) and `--variant VARIANT-ID`; `--port`'s
  default is now "the deployment value, else 8000". No exit code changes (usage 2, refusal 1, `--dry-run` 0).
- `rcp-ndcg-vllm/schema/recipe.schema.json`: the `Recipe` description only (no field change).
- The corpus manifest's `recipe` block gains the extra key `shipped` (not in `PROVENANCE_KEYS`).
- No change to the `rcp-ndcg` command tree, its flags or its schemas.

## 9. Docs updated

Pages changed, and the greps run (each over `docs/`, `README.md`, `REPRODUCIBILITY.md`, `skills/`, `examples/`,
`experiments/**/*.md`, `mkdocs.yml`, and the docstrings/help texts):

- `docs/how-to/serve-a-model.md` — the console line, a new "Deployment overrides" section, a new "Your own
  recipe file" section, the provenance wording.
- `docs/reference/recipes.md` — the console line, the public names, the `recipe:` section.
- `docs/how-to/add-a-model.md` — the pointer to serving a family directory of your own.
- `rcp-ndcg-vllm/README.md` — the console line, the deployment surface, the own-file example.
- `CHANGELOG.md`, `handover/00-MASTER.md` (decision 36), `handover/specs/observations-spec.md` (section 4).
- Docstrings and help: `rcp_ndcg_vllm/recipe.py`, `rcp_ndcg_vllm/serve.py`, `rcp_ndcg_vllm/__init__.py`,
  `rcp_ndcg/inference/recipes.py`, `rcp_ndcg_test/observe/provenance.py`, and the `--set`/`--variant`/`--port`
  help texts.

```bash
for term in "\-\-set" "recipe_digest" "deployment_fields" "parse_deployment_overrides" "RecipeFieldRole" \
            "unshipped" "shipped" "\-\-variant" "FIELD_ROLES" "low_exclusive"; do
  git grep -n -i -- "$term" -- docs/ README.md REPRODUCIBILITY.md skills/ examples/ mkdocs.yml
done
git grep -n "rcp-ndcg-vllm serve\|--dry-run" rcp-ndcg-vllm/README.md docs/ skills/ examples/ README.md
uv run --no-sync pytest tests/docs -q          # 287 passed, 52 skipped (snippets, links, wording)
uv run --no-sync mkdocs build --strict -d <scratch>/site
```

Every hit that the change made false was updated; the remaining `--set` hits are the `rcp-ndcg` CLI's own flag
and the `shipped` hits are about datasets, adapters and presets.

## 10. Files outside scope

- `handover/specs/observations-spec.md` — one sentence in section 4 (the `shipped` key is an extra one).
- `rcp-ndcg-test/tests/test_recipe.py` — one assertion whose premise this lane changed (the fixture recipes are
  not shipped, so their `client_config` pointer is the identity hash); updated with the reason.
- `handover/00-MASTER.md` — decision 36, as the brief instructs.

## 11. For the next lanes

- The judge-recipes lane: `rcp_ndcg.inference.recipes._load` is the one entry point that resolves both shipped
  ids and paths, and `expand_role_recipe` puts the identity in the config — a judge recipe needs only a role in
  `RecipeFieldRole`-terms and its own `--judge` shorthand; the loader needs no change.
- A new `serve` field must be declared in `FIELD_ROLES` (role + flag + kind + range + position) or it is
  refused as unknown for `--set`; `deployment_fields()` is what the CLI reads, and the argv order is the
  declaration order per position.
- The deployment knobs are argv-level by design: adding them as `serve` fields would move every shipped
  recipe's behaviour fingerprint and invalidate every observation corpus (they are hashed as part of the serve
  block). If a recipe ever needs a deployment default of its own, that is a deliberate fingerprint change.
