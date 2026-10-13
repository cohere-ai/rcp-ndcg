# Lane fix13gemma — the embeddinggemma-2 chain: the video pin, the template join, the media-check merge, and the image rows' cause

**Lane:** `lane/fix13gemma` (worktree outside the operator's checkout). **Base:** `rfc-0001` at `c668c439`; the
current `rfc-0001` tip `ec4fffbc` (the `fix12ctx` merge) is merged in at `fea73119`, and the gate runs on the
merged tree. **Read first:** `AGENTS.md`, `COMMON.md`, this lane's brief and the investigate report
(`handover/reports/investigate.md`, now on `rfc-0001`). No GPU work; nothing pushed; `rfc-0001` was not merged
into.

## 1. Status — per-cause disposition

| # | Cause (the brief) | Disposition | Evidence |
|---|---|---|---|
| 1 | the video pin does not reach the engine (rows 36/37: `492 = 4 x 123` against `3936 = 32 x 123`) | **fixed**: the pin's second home, `serve.mm_processor_kwargs {fps: 60, max_frames: 32}`, reaches the checkpoint's processor, which is the component that samples on the engine the pod actually runs | CPU-proven on the pod's engine image + the staged plugin wheel: 4 frames / 492 tokens without it, 32 / 3936 with it (§3.1) |
| 2 | the template join (stage 1, query row 65) | **fixed**: the media-template probe's declared side is rendered from the conversation's text parts as the engine's `openai` content format hands them to the served template (back to back), not from the client's `TEXT_JOIN` flattening (the *string* content format's join) | red first (the wave's exact heads), green after; a served template that inserts a separator still fails (§3.2) |
| 3 | the media-check arithmetic (rows 33/35, off by one) | **fixed**: the engine check subtracts the engine's own render merge, read from the served template's own render; the CPU stub engine counts the engine's way now | red first (the stub's pre-fix arithmetic fails with the +1), green after; the unpinned control still fails (§3.3) |
| 4 | the image near-misses (rows 31/32, 0.994–0.996) | **cause measured on CPU, declared, not hidden, no bound**: the client prepares at the Gemma 4 **fixed point** (`gemma4_fixed_point`: 288x2160 / 13344x48) while the card's processor resizes the **raw** image in **one** pass (288x2112 / 9792x48); the reference's embed mode runs the card's pipeline on the raw image, the engine sees the prepared bytes | the checkpoint's own processor (transformers 5.19.0, pinned revision) measured: 2,430 vs 2,376 patches (row 31) and 2,502 vs 1,836 (row 32); the engine's own usage deltas 272/280 are the client's token counts, so the media stage's count check passes on both rows (§3.4) |

## 2. Commits

| Commit | Subject |
|---|---|
| `7b1b6a35` | The harness reads the engine's render join and its media-count merge (causes 2 and 3) |
| `ea7a41f4` | The embeddinggemma-2 video pin reaches the engine's processor (cause 1) |
| `ace4305d` | The CHANGELOG and the model-adoption page record the three fixes |
| `a9b8edce` | Format the two new harness tests |
| `902e35ed` | The image rows' residual is stated in the client's token counts (cause 4) |
| `fea73119` | Merge branch `rfc-0001` into `lane/fix13gemma` (tip `ec4fffbc`) |
| `ae948061` | The fix13gemma report records the three fixes, the measured image-row cause and the re-run command |
| `2b1da7f3` | The fix13gemma report names the worktree as a placeholder, never a local path |
| `49c5e021` | The wire capture names the string content format's join as its own (the docs sweep) |

Merged `rfc-0001` commit: **`ec4fffbc`**. No co-author lines, no attribution, nothing pushed.

## 3. The fixes, the evidence, and why each is the one home

### 3.1 Cause 1 — the video pin's second home (`serve.mm_processor_kwargs`)

**The mechanism, as it actually ran (correcting the investigate report's first reading).** The wave's engine
is the **stock `vllm/vllm-openai:v0.31.0`**, not the recipe's declared nightly digest: the pod's own
environment records `VLLM_BUILD_COMMIT=db9527a468…` / `VLLM_IMAGE_TAG=vllm/vllm-openai:v0.31.0` and the
engine logs `Initializing a V1 LLM engine (v0.31.0)` (the wave's `manifest.json` `engine.env`, `serve.log`,
`status.json` `engine_version`); the node runtime is documented as "the engine version is the pod's, never
the recipe's declared image (B5)" (`rcp-ndcg-test/src/rcp_ndcg_test/jobs/bootstrap.sh:524-527`). vLLM v0.31.0
has **no `EmbeddingGemma2VideoBackend`** (vllm-project/vllm#60254 landed after the tag), so
`get_video_loader_backend_for_processor("EmbeddingGemma2VideoProcessor")` is `None` and
`--media-io-kwargs '{"video": {"fps": 60, "max_frames": 32}}'` reaches only the **generic `VideoBackend`**.
That loader samples the shipped 4.0 s / 8 fps / 32-frame clip whole and its metadata says
`do_sample_frames = len(valid) == total` = **True** (`vllm/multimodal/video.py:181-194` at the tag); the
checkpoint's processor is then free to resample the 32 decoded frames with its own defaults (fps 1,
`max_frames` 32 — `processor_config.json`), which yields `[0, 8, 16, 24]` = **4 frames**.

(The investigate report's first reading — the pinned *nightly* backend used and its `create_hf_metadata`
override left `do_sample_frames=False` — is contradicted by the measured 492: with `False`, transformers
never calls `sample_frames` and the engine would report ~3945. The pod's version settles it: v0.31.0 has no
such backend at all.)

**CPU proof on the pod's engine image + the staged plugin wheel** (the image
`vllm/vllm-openai@sha256:b25e8a04…`, the RC's `rcp_ndcg_vllm-0.0.1-py3-none-any.whl`, the shipped clip from
`rcp-ndcg-test/pairs/embeddinggemma-2.jsonl` row 36, the recipe's `--media-io-kwargs`):

```
engine vllm 0.31.0, transformers 5.17.0
resolved video processor: None
video loader backend: None
registry has EmbeddingGemma2VideoProcessor: False
decoded frames: 32
loader metadata: {'total_num_frames': 32, 'fps': 8.0, 'duration': 4.0, 'video_backend': 'opencv',
                  'do_sample_frames': True}
loader: VideoBackend
processor: EmbeddingGemma2Processor
```

and the model-side processor call (the engine's own `_apply_hf_processor_main` video branch, driven with the
loader's metadata):

| `serve.mm_processor_kwargs` | processor call kwargs | model-side frames | media tokens |
|---|---|---|---|
| `{}` (the shipped recipe) | `do_sample_frames=True, return_metadata=True` | `[4]` | **492** |
| `{fps: 60, max_frames: 32}` | `… , fps=60, max_frames=32` | `[32]` | **3936** |

`492 = 4 x 123` and `3936 = 32 x 123` are exactly the wave's engine count and the client's count. On the
declared nightly (if the image pin ever applies) the engine's own backend samples the pin and reports
`do_sample_frames=False`, so the processor keeps the backend's frames and the numbers are inert — the pin is
correct in both environments.

**Why this is the one home.** The engine has two sampling mechanisms and the recipe must pin both: the
decoder's `--media-io-kwargs` (the loader) and the processor's own call kwargs
(`serve.mm_processor_kwargs`), which `vllm/model_executor/models/embedding_gemma2.py:707-721` hands to the
checkpoint's processor. The numbers themselves are the *card's* rule (`EmbeddingGemma2VideoProcessor.sample_frames`,
reproduced by the reference's `video_processing_kwargs`), declared once in `serve.extra_args` and once in
`serve.mm_processor_kwargs` because the engine reads them through two different flags; the family notes
("Media") record both and the removal condition (a release whose registry carries the model and whose own
video backend exists makes the processor half inert).

**Files:** `rcp-ndcg-vllm/src/rcp_ndcg_vllm/recipes/embeddinggemma-2/family.yaml:48` (the new pin),
`:51-52` (the decoder pin, unchanged), the notes' "Media" paragraph (the mechanism, the CPU evidence, the
image-row cause), `rcp-ndcg-test/tests/recipes/test_embeddinggemma_2.py` (the contract pin, the argv
assertion, the new rule test), `rcp-ndcg-test/tests/recipes/golden/embeddinggemma-2.json` (regenerated).

### 3.2 Cause 2 — the media-template render check reads the parts the engine reads

**The failure, verbatim from the wave** (`equivalence.json`):

```
stage1.template_render_check: … checked 112, passed False, failures 1:
  {"shape": "query", "row": 65,
   "engine_head":   "task: search result | query: which page shows the quarterly chart",
   "declared_head": "task: search result | query: \nwhich page shows the quarterly chart"}
```

The client sends the query side as `[text("task: search result | query: "), image_url, text("which page…")]`
(the product's `Content.with_text_prefix` prepends the side's prompt as its own text part when the content
starts with media, `rcp-ndcg-core/src/rcp_ndcg_core/content.py:244-259`). The engine's `openai` content
format hands the served template that parts list and the checkpoint's own `chat_template.jinja` renders a
message's text parts **back to back**; the harness's media-template probe built its declared side from
`Capture.texts`' `TEXT_JOIN` flattening (`rcp-ndcg-test/src/rcp_ndcg_test/equivalence/wire.py:213-214`),
which is vLLM's *string* content format's join (`chat_utils.py`'s `_get_full_multimodal_text_prompt`) — the
route's rule only when the template takes a string.

**The fix** (`rcp-ndcg-test/src/rcp_ndcg_test/equivalence/stages.py`): the media probe's declared input is
built from the captured conversation's own text parts (`_conversation_text`/`_declared_texts`, the same walk
the check's engine side takes), so the declared frame is compared on the content the engine reads. The served
template stays the one home of the join: a served template that inserts a separator between the parts still
fails (the test's mutant), and a text row (one text part) is unchanged.

**Red first** (`test_stage1_messages_media_frame_compares_the_parts_the_engine_reads`, before the fix):

```
E  AssertionError: [{'shape': 'document', 'row': 1, 'engine_head': 'doc: a caption under the page',
                     'declared_head': 'doc: \na caption under the page'}]
```

and after the fix the same check passes while the separator mutant fails with `'<|SEP|>'` in
`engine_head`.

**The residual this records (not hidden).** The client's *budget* render still joins a multi-part content's
text with `TEXT_JOIN` (`rcp_ndcg.data.text_budget.fit` over `Content.text`), so a multi-part media side
over-reserves the join's tokens (≤2 here) against what the engine reads. Moving that would move the
product's cut arithmetic (`split_text_across_parts` walks the same join), so this lane records it rather
than silently changing it: the check compares the *frame*, and the join is the served template's.

### 3.3 Cause 3 — the media stage's engine check reads the render merge

The check re-posts each media request without its media parts and compares the two `usage.prompt_tokens`
(`rcp-ndcg-test/src/rcp_ndcg_test/equivalence/media.py`). The engine's prompt is its render (the served
template **with** the media placeholder) minus the placeholder plus the media expansion, while the baseline's
render carries no placeholder: a tokenizer that moves a boundary token when the placeholder stands between a
message's text parts leaves the engine one token high. Measured with the checkpoint's own tokenizer:

```
with   : 'task: search result | query: <|image|>which page shows the quarterly chart'  17 ids
without: 'task: search result | query: which page shows the quarterly chart'            15 ids
placeholder: 1 id   →  the engine's text part is 16 ids, the client's render 15: merge 1
```

**The fix:** `stages.engine_render_merge` (public, one home, beside `render_chat`) measures the merge from the
served template's own render — `tokens(render with the media) - tokens(render without) - placeholder tokens`,
the placeholders measured from the template's media-only render against the frame — and `media._engine_check`
subtracts it (and records `render_merge` in a failure). The CPU **stub engine** counted the engine's way now:
it rendered the text without the placeholders and added the expansion, which is exactly what hid the merge;
`stub_engine._chat_prompts` renders the template with the media in place (via the same helper) and counts
`text render + expansion + merge`.

**Red first** (`test_the_engine_check_reads_the_render_merge_the_placeholder_moves`): with `merge = 0`
(mutation) the check fails `engine_media_tokens: 4` against `client_media_tokens: 3` — the wave's +1 shape;
with the fix it passes, and the unpinned control still fails (`engine_media_tokens > client_media_tokens`).

### 3.4 Cause 4 — the image rows' cause, measured on CPU, declared (no bound)

The wave's two image rows: cosines 0.9956 (row 31, the 576x4096 receipt) and 0.9941 (row 32, the 3000x20
strip) at every declared width; every other comparison is ≥0.999. The media stage's engine check passes on
both rows (the engine's usage deltas 272/280 are exactly the client's counted tokens), so the residual is not
a token or sampling difference.

**The cause, measured against the checkpoint's own processor** (transformers 5.19.0, the pinned revision, in
the pinned image; the client's prepared bytes from `rcp_ndcg.data.prepare.prepare_image` under the recipe's
`image_policy {max_soft_tokens: 280}`):

| row | client prepares (the fixed point) | card processor on the raw image (one pass) |
|---|---|---|
| 31 | 288x2160 → 2,430 patches = 270 soft + 2 = **272 tokens** | 288x2112 → 2,376 patches = 264 + 2 = **266** |
| 32 | 13344x48 → 2,502 patches = 278 + 2 = **280 tokens** | 9792x48 → 1,836 patches = 204 + 2 = **206** |

The engine sees the prepared bytes (the processor's resize is the identity there — the fixed point is what
the processor keeps), and the reference's *embed* mode runs the card's own pipeline on the **raw** image, so
the two sides' patch layouts differ and the pooled vectors move by exactly the observed amount. Every other
image row is at or beside its fixed point, which is why only these two are off. The reference's *media* mode
ports the fixed point (`reference.py::gemma4_fixed_point`, used by `media_side`), which is why the media
stage's geometry check agrees with the client and does not see this.

**The decision:** no `gates.vec_min_cosine` bound is declared — a floor low enough for these rows (0.994)
would hide a same-sized difference on **every** row of the variant, and the cause is an input difference, not
a numeric floor. The named fix (the next measurement, which the wave's re-run re-measures): the reference's
embed mode should see the same bytes the engine sees (the harness's stage-2 input alignment — the client's
prepared media handed to the reference), or the client should prepare at the card's single-pass size with the
engine's own image resize pinned off (`serve.mm_processor_kwargs.images_kwargs.do_resize: false`), each of
which needs its own wave. Both keep the card's processor the definition.

**Files:** `family.yaml` notes ("Media outputs", the paragraph "The two image rows' residual …"),
`handover/reports/fix13gemma.md` (this section).

## 4. Red-first evidence (quoted)

```
$ uv run --no-sync pytest rcp-ndcg-test/tests/test_equivalence.py -q -k media_frame
FAILED …::test_stage1_messages_media_frame_compares_the_parts_the_engine_reads
E  AssertionError: [{'shape': 'document', 'row': 1, 'engine_head': 'doc: a caption under the page',
                     'declared_head': 'doc: \na caption under the page'}]

$ uv run --no-sync pytest rcp-ndcg-test/tests/test_media.py -q -k render_merge
FAILED …::test_the_engine_check_reads_the_render_merge_the_placeholder_moves
E  AssertionError: [{'row': 0, 'sides': ['document 0'], 'engine_media_tokens': 4,
                     'client_media_tokens': 3, …}]
```

(the media test's red is the pre-fix check's arithmetic, shown by mutating the check's merge back to 0; the
frame test's red is the unfixed probe). Cause 1's red is the CPU probe above: the shipped recipe's
`mm_processor_kwargs: {}` gives 4 frames / 492 tokens where the client counts 3936, and the new recipe test
pins the argv, the contract and the rule.

## 5. Verification

| command | result |
|---|---|
| `uv run --no-sync ruff format --check .` / `ruff check .` | 632 files formatted / all checks passed |
| `uv run --no-sync basedpyright` | 0 errors, 0 warnings, 0 notes |
| `heavy uv run --no-sync pytest tests/ -q -n 4` | 4145 passed, 155 skipped |
| `uv run --no-sync pytest rcp-ndcg-test/tests -q` | 1196 passed, 229 skipped |
| `uv run --no-sync pytest tests/contract tests/docs -q` | 413 passed, 107 skipped |
| `uv run --no-sync mkdocs build --strict` | built (strict) |
| the recipe suites (family goldens + embeddinggemma-2) | 66 passed, 4 skipped |
| the CPU probes of §3.1/§3.4 (the pinned image, the staged plugin wheel, the checkpoint's processor) | quoted above |

**The gate.** `timeout 7200 bin/gate lane/fix13gemma` on `49c5e021` (slot 2) — **GATE: PASS**, every step
exit 0: ruff-check, ruff-format (632 files), basedpyright (0 errors), `pytest 4145 passed, 155 skipped`,
`contract-docs 413 passed, 107 skipped`, mkdocs strict, `test-pkg 1197 passed, 229 skipped`, recipes (0
baseline failures), `vllm-pkg 50 passed`, `vllm-models 110 passed, 8 skipped`, `run_all` (1022/987/35/0,
67/67, 82/82), public-names clean, clean. Two earlier attempts on the same tree failed on environment, not
the tree: the first hit the known `test-pkg` SIGSEGV flake (`exit=139` at
`test_record_and_wave.py::test_wave_packs_engines_so_reference_gpus_remain` under load average 84; the test
passes in isolation in 13.7 s), the second the `public-names` step on this report's own local path in the
re-run command (fixed to a `<worktree>` placeholder).

## 6. The verification experiment (the exact re-run command)

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

What the wave must show: the media stage's `engine_check` passing on rows 36/37 with
`engine_media_tokens == client_media_tokens == 3936` (the engine's frame count 32), stage 2's row-36 cosine
rising from 0.9343 to the family's gate, and the stage-1 `template_render_check` passing (112/112) with the
media stage's rows 33/35 no longer +1. Rows 31/32 are re-measured by the same run; if they stay at
0.994–0.996 the cause of §3.4 stands and the decision there (the named fix, not a bound) is the follow-up.

## 7. CHANGELOG entry

The three entries are in `CHANGELOG.md` under `### Fixed` (the 0.0.1 section), quoted in the commit
`ace4305d`: "The embeddinggemma-2 video pin reaches the processor that samples the clip", "The
media-template render check reads a message's parts as the engine's chat route hands them to the template",
and "The media stage's engine check reads the render merge its subtraction baseline loses" — each with the
measured numbers, the mechanism and the one-home reasoning.

## 8. Public surface changes

None: no `__all__`, CLI, exit code, schema or snapshot moved (`rcp-ndcg-test` is unpublished; its
`equivalence.stages.__all__` gained `engine_render_merge`/`media_placeholder_tokens`, which no contract
snapshot covers). `tests/contract` and `tests/docs` pass unchanged; the recipe's golden and behaviour
fingerprint moved with the recipe (regenerated, reviewed).

## 9. Files outside scope

None. The touched files are the recipe, its test and golden, the harness modules, the stub engine, the tests,
the docs and the CHANGELOG — all named by the brief.

## 10. For the next lanes

- **The pod's engine is the job config's image, not the recipe's declared digest.** The wave's provenance
  (`VLLM_IMAGE_TAG`/`VLLM_BUILD_COMMIT`) and `serve.log` are the truth; any recipe pin that assumes the
  declared image (this family's fold, the SDPA backend, the video pin) must work on vLLM v0.31.0, or the
  node wiring must start applying the per-group image (`submit.sh` sets `env.RCP_IMAGE` only; the job's
  container image comes from the kjobs config).
- **The image rows' cause (§3.4)** is the client's fixed-point preparation against the card's single-pass
  processor. The follow-up is the stage-2 input alignment (the reference sees the prepared bytes) or the
  single-pass preparation with the engine's image resize pinned off — either needs a wave; a bound is not the
  fix.
- **A `messages` recipe's budget render joins a multi-part content with `TEXT_JOIN`**, which is the *string*
  content format's join; a template that renders the parts itself makes that an over-reserve. A declared
  per-recipe join would be the product change; this lane recorded the residual instead.
- The media stage's engine check now reads the served template's render (and the recipe's tokenizer); a
  recipe whose template cannot be resolved contributes no merge (the stage-1 template check is the one that
  fails on an unreadable template).
