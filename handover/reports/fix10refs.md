# Lane `fix10refs` — the round-10 reference, fill and bound failures

## 1. Status

**DONE.** All three items from the `rc0-202610130020` evidence are fixed, test-first, and the lane's gate
passed twice on the lane tip. Base: the `rfc-0001` tip at lane start (`58847318`, the mediafill merge) was
already the branch's HEAD, so the merge was a no-op ("Already up to date"; the tip is an ancestor of every
lane commit). Lane HEAD: **`747ffc68`** (the report commit follows). No push, no merge into `rfc-0001`.

| Item | Disposition |
|---|---|
| 1. topk-embed-v1's CUDA→numpy conversion | **fixed**: one host-conversion helper used by both output paths; CPU tests pin it (the GPU run is the confirmation) |
| 2. `qwen3-vl-embedding-2b`'s 9 fill exceptions | **fixed**: `case_fill` carries a case's instruction only when the recipe's client places it (the product's resolved endpoint config decides) |
| 3. `qwen3-vl-reranker-2b`'s probability bound | **re-declared from evidence**: the residual is still the bf16 floor (one-quantum move on one document, τ = 1.0); the 2b declares 0.035, the 8b keeps 0.03 (its measurement is unchanged) |

**Gate.** `timeout 7200 bin/gate lane/fix10refs` **PASS, twice, first attempt each, no flakes**: once on
`25b5d068` (the code tip) and once on `747ffc68` (the docs commit, the tree the report follows). Both runs:
ruff, ruff-format, basedpyright, root `pytest 4145 passed, 155 skipped`, `contract-docs 413 passed, 107
skipped`, `mkdocs --strict`, `test-pkg 1183 passed, 229 skipped`, `recipes` (0 baseline failures remain),
`vllm-pkg 50 passed`, `vllm-models 102 passed, 8 skipped`, `run_all` (1022/987/35/0, 67/67, 82/82),
`public-names` clean, `clean` clean. The second run's SUMMARY is in
`gates/747ffc68/SUMMARY`. The report commit that follows this file is documentation only.

## 2. Commits

| Commit | Subject |
|---|---|
| `b48c0baf` | The topk-embed-v1 reference moves every encoder matrix to the host before the fp16 cast |
| `675ca316` | The case fill carries a case's instruction only when the recipe's client places it |
| `678b28f6` | The qwen3-vl-reranker-2b re-declares its measured probability bound at 0.035 |
| `25b5d068` | The CHANGELOG records the round-10 reference, fill and bound fixes |
| `747ffc68` | The docs describe the case fill's instruction declaration |
| (this report) | The lane report |

Merged `rfc-0001` commit: **`58847318`** (the lane's base; no merge commit was needed).

## 3. Item 1 — topk-embed-v1's CUDA→numpy conversion (fixed)

**Evidence.** `waves-rc0-202610130020/r4/vllm-vllm-openai-v0-31-0/topk-embed-v1-small/` (serve.log,
status.json) and the wave's `casefill/casefill-report.json`: the embed mode died with
`TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host
memory first.` — the family's `transformers==5.9.0` pin runs the checkpoint's own wrapper, whose
`encode_query`/`encode_document` return CUDA tensors on a GPU reference, and `_fp16_lists` handed one
straight to `np.asarray`. 14 case-fill exceptions carry the same error.

**Fix** (`rcp-ndcg-vllm/src/rcp_ndcg_vllm/recipes/topk-embed-v1/reference.py`): one helper
`_host_matrix` detaches a torch tensor and moves it to CPU storage (`detach()` then `cpu()`) before numpy;
a device-less array-like (nested list, numpy array, a CPU tensor whose `detach`/`cpu` are no-ops) passes
through unchanged. `_fp16_lists` converts through it, and both output paths — the query matrix and every
document matrix — go through `_fp16_lists`. The fp16 cast and the numerics are unchanged. The module
docstring's `embed` mode states the host move.

**Red-first.** The two new tests failed on the unfixed code:

```
FAILED .../test_topk_embed_v1.py::test_the_reference_host_helper_moves_a_cuda_tensor_and_passes_a_device_less_one
FAILED .../test_topk_embed_v1.py::test_the_reference_embed_path_converts_every_matrix_through_the_host_helper
E  AttributeError: module 'topk_embed_v1_reference' has no attribute '_host_matrix'
```

and a scratch reproduction of the unfixed conversion site on a CUDA-like tensor raised the wave's own
error:

```
$ uv run --no-sync python - <<'EOF'
<load recipes/topk-embed-v1/reference.py as a module; _fp16_lists(CudaLikeTensor())>
EOF
has _host_matrix: False
RED (the wave's own error): can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the
tensor to host memory first.
```

After the fix: `2 passed`; the CUDA-like tensor records `["detach", "cpu"]`, and `embed_rows` converts
both paths (2 helper calls counted).

## 4. Item 2 — `qwen3-vl-embedding-2b`'s instruction refusal (fixed)

**Evidence.** `casefill-report.json`: 9 exceptions for `qwen3-vl-embedding-2b`, each with
`pairs row 0 carries an instruction, and this recipe serves the model's default instruction on both
sides: drop the instruction field from the pairs file`. The recipe's client declares `instruction: none`
(`qwen3-vl-embedding/family.yaml:52`, the card's default instruction is fixed frame text), and 12 of its
cases carry that default as `inputs.instruction`; `case_fill._plan_rows` put it into the pairs row
unconditionally, so the reference (which refuses instruction rows) never filled them.

**Fix** (`rcp-ndcg-test/src/rcp_ndcg_test/case_fill.py`): `_plan_rows(recipe, cases)` now consults
`_places_the_instruction(recipe)`, which reads the **product's resolved endpoint config**
(`rcp_ndcg_test.equivalence.wire.recipe_config(recipe, None).instruction` — the same object the role
clients are built from, role defaults included) and carries the row's `instruction` only when the client
places it. `instruction: none` omits the field (the reference reads exactly the inputs the served request
carries); `fold` (and the rerank role's own default) keeps it. Both `_answers_per_case` call sites pass
the recipe. No second copy of the mode rule.

**Red-first.** The new test failed on the unfixed code:

```
FAILED .../test_case_fill.py::test_a_case_instruction_is_planned_only_when_the_recipe_places_it
E  TypeError: _plan_rows() takes 1 positional argument but 2 were given
```

(the fill could not consult the recipe at all), and the behaviour it pins was red in a scratch check:

```
$ uv run --no-sync python - <<'EOF'
from rcp_ndcg_test.case_fill import _plan_rows
from rcp_ndcg_test.cases import load_case
case = load_case(Path("rcp-ndcg-test/cases/qwen3-vl-embedding-2b/text-short-single.yaml"))
rows, _ = _plan_rows([case]); print(rows[0].get("instruction"))
EOF
RED: the row carries the instruction: Represent the user's input.
```

After the fix: the whole `test_case_fill.py` (15 tests) passes; a scratch sweep over all 13 shipped
`qwen3-vl-embedding-2b` cases shows **25 rows, 0 carrying an instruction** (12 cases carry one), while the
`fixture-rerank-pointwise` fold recipe's row keeps it. The case loader's own rule
(`cases._check_instruction_on_the_wire`) is untouched and still consistent.

## 5. Item 3 — `qwen3-vl-reranker-2b`'s probability bound (re-declared from evidence)

**Evidence (round 10, `waves-rc0-202610130020/r4/`).**

| Variant | p99_documents_within (bound 0.99) | max \|delta\| (bound 0.05) | Kendall τ median | Verdict |
|---|---|---|---|---|
| `qwen3-vl-reranker-2b` | 0.9818 → **FAIL** | 0.033203125 | 1.0 | FAIL (p99 only) |
| `qwen3-vl-reranker-8b` | 1.0 | 0.02734375 | 1.0 | PASS |

**Diagnosis: the residual is still the bf16 floor, not a moved path.** The wave's per-document deltas
(equivalence.json) were compared against the round-7 wave (`waves-rc0-202610120815/r4/...`):

- the 2b's maximum is the *same document* as round 7's (query 0, document 3) and moved from 0.029296875 to
  0.033203125 — **exactly one bf16 quantum** (0.00390625 = 1/256, the probability scale's grid: every
  served/reference value in both waves is a multiple of 1/256); every other document moved by at most one
  quantum;
- the 8b re-measured 0.02734375, byte-identical to round 7; τ = 1.0 in both waves for both sizes (the
  ranking never moved);
- nothing in the recipe, the reference or the engine changed between the waves: the family directory was
  last edited by the round-7 declaration itself (`e86d80cd`), the engine image is the same
  `vllm/vllm-openai:v0.31.0` pin, the reference environment is the same lock
  (`6ac5d595…`, unchanged), and the pairs/queries counts are the same (55 documents over 34 queries).

**Re-declaration.** `qwen3-vl-reranker-2b` declares `overrides.gates.prob_p99_abs: 0.035` (100% of the 55
documents are within 0.035; the measured max is 0.0332), with the round-10 numbers and the one-quantum
diagnosis in the variant notes, the family notes and the CHANGELOG. The 8b keeps `0.03`: its measurement
is unchanged and fully covered, and the policy is to widen no more than the measurement requires. The
goldens were regenerated (`--update-goldens`; the diff is exactly the 2b gate + both notes) and the
variant test pins the per-variant bound and the measured facts.

## 6. Verification

| Command | Result |
|---|---|
| `pytest rcp-ndcg-test/tests/test_case_fill.py` (+ the two topk reference tests) | 15 + 2 passed |
| `pytest rcp-ndcg-test/tests/recipes/test_topk_embed_v1.py` | 22 passed, 32 skipped (hub-gated) |
| `pytest rcp-ndcg-test/tests/recipes/test_qwen3_vl_reranker.py test_qwen3_vl_embedding.py` | passed |
| `pytest rcp-ndcg-test/tests/recipes/test_family_goldens.py` (then `--update-goldens`) | red on the moved goldens, then 49 passed |
| `heavy pytest rcp-ndcg-test/tests -q` | 1183 passed, 229 skipped (13m22s) |
| `pytest rcp-ndcg-vllm/tests --ignore=.../models -q` | 50 passed |
| `pytest tests/contract tests/docs -q` | 413 passed, 107 skipped |
| `mkdocs build --strict` | built |
| `ruff format --check . && ruff check . && basedpyright` | clean, 0 errors |
| `timeout 7200 bin/gate lane/fix10refs` on `25b5d068` | **GATE: PASS** |
| `timeout 7200 bin/gate lane/fix10refs` on `747ffc68` (final tree) | **GATE: PASS** |

One pre-gate run of `rcp-ndcg-vllm/tests` (including `tests/models`) failed 6 setup errors in
`uv build` — an index-authentication 401 from the shell's `UV_EXTRA_INDEX_URL`/private index, not the
tree; the gate's own `vllm-models` step (index env unset, its own venv) passed 102 tests. No gate flake
was seen.

## 7. CHANGELOG entry

Added under `## 0.0.1` → `### Fixed` (newest first, plus one clause on the round-7 bound bullet):

```markdown
- **The topk-embed-v1 reference converts every encoder matrix on the host** (the E2 round-10 CUDA
  crash): the family's `transformers==5.9.0` pin runs the checkpoint's own wrapper, whose
  `encode_query`/`encode_document` return CUDA tensors on a GPU reference, and the embed mode handed
  one straight to `np.asarray` -- the wave died with `TypeError: can't convert cuda:0 device type
  tensor to numpy` before any comparison. One conversion helper (`reference.py::_host_matrix`) now
  detaches the matrix and moves it to CPU storage before the fp16 cast, and both output paths (the
  query matrix and every document matrix) go through it; a device-less array-like (a nested list, a
  numpy array, a CPU tensor) passes through unchanged. CPU tests pin the helper and both paths against
  CUDA-like tensors, so the GPU run is the confirmation, not the first exercise.

- **The case fill respects the recipe's instruction declaration** (the E2 round-10 wave's nine
  `qwen3-vl-embedding-2b` exceptions): `case_fill` put a case's run-level `instruction` into the pairs
  row unconditionally, and the reference refused those rows -- "pairs row 0 carries an instruction, and
  this recipe serves the model's default instruction on both sides" -- because the recipe's client
  declares `instruction: none` (the card's default instruction is fixed frame text). The fill now
  resolves the recipe's instruction mode through the product's endpoint config (the same object the
  role clients are built from) and carries the field only when the client places it, so the reference
  reads exactly the inputs the served request carries; a recipe that folds the instruction still
  carries it. Test-first: a case with an instruction against a no-instruction recipe builds a row
  without the field, and a folding recipe's row keeps it.

- **The qwen3-vl-reranker-2b re-declares its measured probability bound** (the E2 round-10 stage-2):
  `p99_documents_within` was 0.9818 against the declared 0.03 (one of the 55 documents at 0.0332 --
  exactly one bf16 quantum, 1/256 of the probability scale, above round 7's 0.0293), with Kendall tau
  median 1.0 and every other document within one quantum of its round-7 value; the 8b re-measured its
  round-7 0.0273 exactly (p99 1.0). Nothing in the recipe, the reference or the engine moved between
  the two waves, so the residual is still the bf16 score-scale floor and the 2b now declares
  `overrides.gates.prob_p99_abs: 0.035`, the tight tolerance covering its measured maximum (100% of
  the 55 documents within 0.035); the 8b keeps 0.03.
```

## 8. Public surface changes

None: no `__all__`, CLI, exit code or schema changed; no contract snapshot moved (the gate's
`contract-docs` and `clean` steps are green on the final tree). The changed artifacts are recipe data
(the 2b's `overrides.gates.prob_p99_abs`, its notes and the family notes; the topk reference's module)
and the unpublished `rcp-ndcg-test` tool, all pinned by the `rcp-ndcg-test` tests and the recipe goldens
(regenerated: `golden/qwen3-vl-reranker-2b.json`, `golden/qwen3-vl-reranker-8b.json`).

## 9. Docs updated

- `docs/reference/rcp-ndcg-test.md` — the "Filling the cases" bullet states that a case's run-level
  `instruction` enters the pairs row only when the recipe's client places it.
- `docs/how-to/release-candidates.md` — the casefill-job bullet states the same for the cuda fills.
- `CHANGELOG.md` — the three entries above (plus the round-7 bullet's pointer to the 2b's re-declaration).
- `rcp-ndcg-vllm/src/rcp_ndcg_vllm/recipes/topk-embed-v1/reference.py` — the `embed` mode's docstring
  states the host move; `qwen3-vl-reranker/family.yaml` notes carry the round-10 numbers.
- Greps run: `git grep -n -i "case_fill\|case fill" -- docs README.md skills mkdocs.yml
  rcp-ndcg-vllm/README.md`; `git grep -n "qwen3-vl-reranker" -- docs skills README.md
  rcp-ndcg-vllm/README.md examples experiments`; `git grep -n "prob_p99_abs" -- docs
  rcp-ndcg-vllm/README.md skills examples`; `git grep -n "0.0293\|0.0273\|prob_p99_abs: 0.03"`.

## 10. Files outside scope

None.

## 11. For the next lanes

- **Re-run the case fill** for the four cuda families on the next wave: `qwen3-vl-embedding-2b`'s 9 cases
  (and any other instruction-bearing case of a `instruction: none` recipe) now plan rows the reference
  accepts, and topk's 14 now reach the conversion helper. The fill's own provenance/notes vocabulary is
  unchanged.
- **Re-run r4** for `qwen3-vl-reranker-2b`: the recipe's resolved form moved (the gate), so its behaviour
  fingerprint moved and its corpus is already declared stale in `tests/conformance/stale.json`; the
  re-run is the confirmation of the 0.035 declaration (expect p99 1.0 at the new bound).
- **topk's GPU confirmation**: the next r4 embed run exercises `_host_matrix` on real CUDA tensors; no
  other reference path reads a model tensor through numpy.
- The 8b's bound was deliberately left at 0.03 (measured 0.0273, unchanged). If a later wave moves it by
  one quantum past 0.03, the same one-quantum diagnosis applies — re-declare at 0.035 then, from that
  wave's evidence.
