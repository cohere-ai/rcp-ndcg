# Lane fix14gemma — the embeddinggemma-2 reference prepares media the way the card's processor does

**Lane:** `lane/fix14gemma` (worktree outside the operator's checkout). **Base:** `rfc-0001` at `9a9ca626`
(the tip at lane start, and the tip at every check this lane made — the branch already sat on it, so no merge
commit was needed and none was made). **Read first:** `AGENTS.md`, `COMMON.md`, this lane's brief and the
investigate report (`handover/reports/investigate.md`, Part A rows 31/32). No GPU work; nothing pushed;
`rfc-0001` was not merged into.

## 1. Status

| # | Cause (the brief) | Disposition | Evidence |
|---|---|---|---|
| 1 | the reference's embed mode feeds the card's processor the RAW image while the engine is fed the client's prepared image (the Gemma 4 fixed point) | **fixed**: the reference prepares each image at the card's fixed point before the processor sees it, with the rule read from the loaded processor's own module | CPU-proven with the checkpoint's own processor (transformers 5.19.0): the reference's prepared image is pixel-identical to the client's prepared bytes on both failing rows (272/280 tokens, the engine's recorded deltas), and the processor's own count of it is 272/280 (§3, §5) |
| 2 | the residual after the input difference is gone | **not declared**: no `gates.vec_min_cosine` bound is set. With the inputs now identical the two rows are a bf16 comparison like every other row (all ≥0.999 in the round-13 wave); the wave's re-run re-measures them, and a floor is declared only if a genuine residual survives, with its numbers and τ/rankings intact (§7) | the round-13 wave: rows 31/32 at 0.9956/0.9941 were the ONLY sub-0.999 comparisons of the variant, and their cause was the input difference, now removed |

The fix's one home: `rcp-ndcg-vllm/src/rcp_ndcg_vllm/recipes/embeddinggemma-2/reference.py` (the family's
reference subprocess). No harness, recipe-declaration, product or plugin change was needed.

## 2. Commits

| Commit | Subject |
|---|---|
| `ccac5744` | The embeddinggemma-2 reference prepares media at the card's fixed point (reference.py, the tests) |
| `15130960` | The recipe notes, the adoption guide and the CHANGELOG record the media preparation (family.yaml notes, the regenerated golden, `docs/how-to/add-a-model.md`, `CHANGELOG.md`) |

Merged `rfc-0001` commit: none needed — the lane's base is `rfc-0001`'s tip `9a9ca626`. No co-author lines,
no attribution, nothing pushed.

## 3. The cause, measured on CPU against the checkpoint's own processor

The engine's processor runs on the bytes the **client** prepares. The product prepares the Gemma 4 resize's
**fixed point** (`rcp_ndcg.data.resolution.gemma4_fixed_point`, one Pillow-BICUBIC resize from the decoded
image), a size the processor's own resize then **keeps unchanged** — so the engine's media token deltas are
the client's counts (272 for the shipped 576×4096 receipt, 280 for the 3000×20 strip). The reference's embed
mode handed the same processor the **raw** image, whose **single pass** lands on a different grid:

```
$ <reference python> -c ...   # transformers 5.19.0, the pinned revision, Gemma4ImageProcessorPil, the shipped rows
the card's own processor: transformers.models.gemma4.image_processing_pil_gemma4 Gemma4ImageProcessorPil
row 31: raw (576, 4096) -> processor single pass 264 soft + 2 = 266 tokens | prepared (288, 2160) -> processor keeps 270 soft + 2 = 272 tokens
row 32: raw (3000, 20) -> processor single pass 204 soft + 2 = 206 tokens | prepared (13344, 48) -> processor keeps 278 soft + 2 = 280 tokens
```

266/206 is the pre-fix reference's input; 272/280 is the client's count and the engine's recorded usage
delta. The two sides' patch layouts therefore differed on exactly these two rows (every other image row is at
or beside its fixed point), and the pooled vectors moved by exactly the observed amount — the round-13 wave's
0.9956 (row 31) and 0.9941 (row 32) against every other comparison ≥ 0.999. The cause is an **input
difference**, not a numeric floor, so no bound was declared (the fix13gemma decision stands).

## 4. The fix, and why this is the one home

`reference.py` now prepares every image entry at the card's fixed point before handing it to the card's
processor:

- `_aspect_ratio_rule(image_processor)` — the card's own `get_aspect_ratio_preserving_size`, read from the
  **loaded image processor's own module** (`transformers.models.gemma4.image_processing_gemma4` for the
  torchvision backend, `image_processing_pil_gemma4` for the PIL one; both define it beside the class), never
  the reference's own port and never a second implementation;
- `processor_fixed_point(image_processor, height, width, *, max_soft_tokens)` — that rule **iterated to its
  fixed point** (the Gemma 4 resize is not idempotent: the 576×4096 receipt walks 2112×288 → 2160×288, the
  3000×20 strip walks over eight passes), under the recipe's declared budget (`_image_budget`, now one home,
  shared with the media mode);
- `prepared_image(image, image_processor, *, max_soft_tokens)` — the fixed point, resized **once** with the
  processor's own declared `resample` (Pillow's BICUBIC, the client's filter), so the pixels are the client's;
- `side_input(...)` routes every image entry through it, and `mode_embed` reads the processor from the
  model's own checkpoint processor (`_model_image_processor`: ST's input module holds the `AutoProcessor` its
  `encode` calls), failing loudly if a checkpoint's pipeline carries none.

Because the prepared size is a fixed point, the processor's own resize is the identity on it — on the
reference's side exactly as on the engine — so the model consumes the same pixels the served path does. The
media mode keeps its dependency-free mirror (`gemma4_resize`/`gemma4_fixed_point`, cross-checked against the
product over a size grid by an existing test) because that mode imports no transformers; its docstring now
says so, and the embed mode takes the same rule from the processor it loads. The recipe notes and the
adoption guide state the rule for a new family's reference: prepare the image as the served path sends it.

**Measured (the same scratch venv, the real processor), after the fix:**

```
row 31: reference prepares (288, 2160) (272 tokens, sha b3d8a20a23a24d81) | client (288, 2160) (272 tokens, sha b3d8a20a23a24d81) | identical True
row 32: reference prepares (13344, 48) (280 tokens, sha fb1f1ccdd842c795) | client (13344, 48) (280 tokens, sha fb1f1ccdd842c795) | identical True
row 30: reference prepares (1056, 576) (266 tokens, sha f14b2baf80c9359e) | client (1056, 576) (266 tokens, sha f14b2baf80c9359e) | identical True
```

The digests are over the RGB buffers; "client" is the product's own `prepare_image` output, decoded.

## 5. Red-first evidence (quoted)

The new test first ran against the **unfixed** reference (its then signature), so the red is the behaviour,
not the API:

```
$ uv run --no-sync pytest rcp-ndcg-test/tests/recipes/test_embeddinggemma_2.py -q -k embed_input
FAILED rcp-ndcg-test/tests/recipes/test_embeddinggemma_2.py::test_the_reference_embed_input_is_the_image_the_client_sends
E  AssertionError: tall_receipt: the reference sends (576, 4096), the client (288, 2160)
E  assert (576, 4096) == (288, 2160)
E    At index 0 diff: 576 != 288
```

and the second new test (the rule comes from the loaded processor, not from a port) was red on the unfixed
module's signature, before the fix added the parameter:

```
FAILED rcp-ndcg-test/tests/recipes/test_embeddinggemma_2.py::test_the_reference_takes_the_image_rule_from_the_loaded_processor
E  TypeError: side_input() got an unexpected keyword argument 'image_processor'
```

Both pass after the fix; the pre-fix red is re-shown by mutation (returning the raw image from
`prepared_image` reds the first with the same line), and the second test's stand-in rule (identity) is the
mutation guard: a reference that used its own port would ignore it and prepare 2160×288 anyway.

## 6. Verification

| command | result |
|---|---|
| `uv run --no-sync ruff format --check .` / `ruff check .` | 632 files formatted / all checks passed |
| `uv run --no-sync basedpyright` | 0 errors, 0 warnings, 0 notes |
| `heavy uv run --no-sync pytest tests/ -q -n 4` | 4145 passed, 155 skipped |
| `uv run --no-sync pytest rcp-ndcg-test/tests -q` | 1199 passed, 229 skipped |
| `uv run --no-sync pytest tests/contract tests/docs -q` | 413 passed, 107 skipped |
| `uv run --no-sync pytest rcp-ndcg-test/tests/recipes -q` | 305 passed, 228 skipped |
| `uv run --no-sync mkdocs build --strict` | built (strict) |
| the CPU probes of §3/§4 (transformers 5.19.0, the shipped rows) | quoted above |
| the reference's own smoke (`--mode media` over the shipped pairs) | rows 31/32 report 288×2160 / 272 and 13344×48 / 280 |

**The gate.** `timeout 7200 bin/gate lane/fix14gemma` on `15130960` (slot 2) — **GATE: PASS**, every step
exit 0: ruff-check, ruff-format (632 files), basedpyright (0 errors), `pytest 4145 passed, 155 skipped`,
`contract-docs 413 passed, 107 skipped`, mkdocs strict, `test-pkg 1199 passed, 229 skipped`, recipes (0
baseline failures), `vllm-pkg 50 passed`, `vllm-models 110 passed, 8 skipped`, `run_all` (1022/987/35/0,
67/67, 82/82), public-names clean, clean. One earlier attempt on the same tree failed on the known `test-pkg`
SIGSEGV flake (`exit=139` at `test_wave_plugin_wheel.py::test_the_staged_plugin_wheel_is_hashed_and_recorded`
under load average 83; the file passes in isolation in 30.2 s, 7 passed), not on the tree.

## 7. The verification experiment (the exact re-run command)

The fix's confirmation is the final RC's wave. From the gated worktree, operator-side:

```bash
# 1. the RC rebuild (stages the wheels, the recipes, the pairs, the wave lists and the T4 scenarios)
bin/e2-run build <worktree> rc0-<UTC stamp>
# 2. wave0 (the node test; its report must read passed: true)
bin/e2-run wave0 <worktree> rc0-<UTC stamp>
# 3. the r1 wave: its nightly group is the embeddinggemma-2 job and lands under its own prefix
#    waves-<rc>/r1/vllm-vllm-openai-nightly-8cbd5d03006c33185f40224-b25e8a046fdb/
RCP_MAX_JOBS=8 bin/e2-run waves <worktree> rc0-<UTC stamp> r1
# (a single-recipe confirmation: stage a wave list holding only `embeddinggemma-2` and submit it as its
#  own wave name -- the grouping then submits exactly the nightly job with its own prefix)
```

What the wave must show: stage 1 PASS (112/112), the media stage PASS (the video rows' engine count equal to
the client's, 32 frames / 3936 tokens), and stage 2's **image rows 31/32 at ≥ 0.999** — the bf16 floor every
other comparison of the variant sits at — with the video rows 36/37 also at the floor. If a residual
survives on rows 31/32 after this input alignment, it is a genuine numeric difference: report it with the
measured numbers, the per-row cosines at every declared k, τ/rankings intact, and only then is a tight
`gates.vec_min_cosine` declared (the owner's policy); a bound is never the first move.

## 8. CHANGELOG entry

One bullet under `### Fixed` (the 0.0.1 section), the first of the embeddinggemma-2 entries, quoted in
`15130960`:

> **The embeddinggemma-2 reference prepares media the way the card's processor does** (the stage-2 image rows
> 31/32, cosines 0.9941–0.9963 against the 0.999 gate): the engine's processor runs on the bytes the client
> prepares — the Gemma 4 resize's fixed point (`rcp_ndcg.data.resolution.gemma4_fixed_point`, Pillow
> BICUBIC), a size the processor keeps unchanged, so the engine's own media deltas are the client's counts
> (272 tokens for the shipped 576×4096 receipt, 280 for the 3000×20 strip) — while the reference's embed mode
> handed the checkpoint's processor the RAW image, whose single pass lands on a different grid (288×2112 =
> 266 tokens; 48×9792 = 206), so the two sides' patch layouts differed and the pooled vectors moved by
> exactly the observed amount. The reference now prepares the image at the card's fixed point before the
> processor sees it (the processor's own resize is then the identity on the reference's side too, as it is on
> the engine), taking the rule from the loaded processor's own module — transformers 5.19.0's
> `get_aspect_ratio_preserving_size`, iterated to its fixed point — rather than a second implementation.
> Measured on CPU with the checkpoint's own processor: the reference's prepared image is pixel-identical to
> the client's prepared bytes (on these rows and on the rows already at their fixed point) and the
> processor's own count of it is 272/280, the client's numbers. CPU tests pin the reference's embed input
> against the product's own preparation for both rows and a 16×16 icon (the card's 768×768), and that the
> rule is the loaded processor's own; the wave's re-run re-measures the rows.

## 9. Public surface changes

None. No `__all__`, CLI command or flag, exit code, schema or contract snapshot moved: the changed code is a
recipe's reference subprocess (`rcp-ndcg-vllm` package data, unpublished harness-side) and its tests. The
recipe's golden moved in exactly one field — its `notes` (the fingerprint, the resolved `client`/`serve`
blocks and the `serve` argv are byte-identical, checked against `HEAD`'s copy), regenerated with
`pytest rcp-ndcg-test/tests/recipes/test_family_goldens.py --update-goldens`.

## 10. Files outside scope

None. The touched files are the family's `reference.py`, its test module, the regenerated golden, the
family's notes, the adoption guide's media paragraph and the CHANGELOG — all named by the brief's "one home"
and COMMON.md's documentation sweep.

## 11. Docs updated

| page | change |
|---|---|
| `rcp-ndcg-vllm/src/rcp_ndcg_vllm/recipes/embeddinggemma-2/family.yaml` (notes, "Media outputs" and the image-row paragraph) | the embed mode's image preparation (the card's rule from the loaded processor, the fixed point, the pixel-identical measurement) and the round-13/14 image-row paragraph |
| `docs/how-to/add-a-model.md` (the media-stage paragraph) | a new family's reference prepares the image as the served path sends it (the card's resize at the size the engine's processor keeps); an image the engine would resize again is a different input |
| `CHANGELOG.md` (`### Fixed`) | the entry quoted in §8 |
| `handover/reports/fix14gemma.md` | this report |

The greps behind the sweep (all over the tracked tree, `git grep`, run before the last commit):
`git grep -n "side_input\|gemma4_fixed_point\|gemma4_resize\|prepared_image\|processor_fixed_point" --
docs skills examples experiments README.md REPRODUCIBILITY.md mkdocs.yml rcp-ndcg rcp-ndcg-core`,
`git grep -rn "mode embed\|--mode embed\|embed mode" -- docs skills`,
`git grep -n "prepared\|raw bytes\|raw image\|fixed point" -- docs/how-to/add-a-model.md`, and
`git grep -n "fixed point\|fixed-point" -- docs skills examples experiments`.

## 12. For the next lanes

- **A reference's embed/score mode must prepare media the way the served path sends it.** The client
  prepares the size the engine's processor keeps (the fixed point), so a reference that hands the processor
  the raw bytes sees a different grid whenever the card's resize is not idempotent — and the mismatch shows
  up as a small, row-specific cosine drop, never as a loud failure. Take the rule from the processor the
  reference already loads, not from a copy; the media mode may keep a mirror only because it imports nothing.
- **The one check that catches this class of bug is the media stage's client-vs-engine geometry**, which
  passes while stage 2 fails: a passing media stage plus sub-0.999 cosines on the rows whose card resize is
  not a fixed point means the reference's *inputs* differ, not the numbers.
- **The reference environment's backend matters less than it looks.** Both Gemma 4 image-processor backends
  early-return when the target size equals the current one, so on a prepared (fixed-point) image the
  reference's processor resamples nothing and the pixels are the client's whatever backend the image carries.
- **A green wave is the only confirmation** of the residual question (§7); do not declare a bound before it.
