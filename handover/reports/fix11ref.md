# Lane `fix11ref` — the embeddinggemma-2 reference and the corpus plan's spelling row

## 1. Status

**DONE, pending the gate.** The E2 round-11 failure was the reference subprocess (embed) dying with
`ValueError: Found 1 <image-placeholder> tokens in the text but no images were passed.` on the corpus plan's
spelling stratum: a text-only row carrying the checkpoint's image-placeholder spelling. The reference now
processes a text-only side through the card's own tokenizer path (one home: the checkpoint's own tokenizer),
and a media side keeps the card's `encode` and its validation. Test-first on CPU; the fix, the tests, the
CHANGELOG entry and the pitfall-catalogue row are committed.

| Item | Disposition |
|---|---|
| The spelling row kills the reference | **fixed**: `text_vectors` — the checkpoint's tokenizer over the prompt-prefixed text, then the card's own module stack; the file's own parse, no second spelling rule |
| Media rows | **unchanged**: still through the card's `encode` (the processor's validating path, manual-placeholder rule included) |

**Gate.** `<pending>` — `timeout 7200 bin/gate lane/fix11ref`; the merged `rfc-0001` tip is named in
section 2. The lane held its gate for `lane/fix11ctx`'s case-fill test fix (see section 5) because
`rfc-0001`'s tip `9224b700` left three `test_case_fill.py` tests red on the merged tree.

## 2. Commits

| Commit | Subject |
|---|---|
| `0e20a044` | The embeddinggemma-2 reference takes the card's tokenizer path for a text-only row |
| `12b46544` | The CHANGELOG and the pitfall catalogue record the embeddinggemma-2 text-only tokenizer path |
| `<merge>` | Merge branch `rfc-0001` into `lane/fix11ref` |
| (this report) | The lane report |

Merged `rfc-0001` commit: **`9224b700`** (the cuda reference-case fill). `<updated at the gate>`.

## 3. What changed

### The reference's text path (one home: the checkpoint's tokenizer)

`rcp-ndcg-vllm/src/rcp_ndcg_vllm/recipes/embeddinggemma-2/reference.py`

- New `text_vectors(model, texts, *, prompt)`: the card's own text path. It calls the **checkpoint's own
  tokenizer** (`model.tokenizer`, the processor's own instance) over the prompt-prefixed text with the
  processor's own text kwargs (`padding=True`, `truncation="longest_first"`, `return_tensors="pt"`; the
  tokenizer's post-processor adds BOS/EOS), then runs the **card's own module stack** (`model(features)`:
  Transformer → Pooling → Normalize) with ST's own device move (`sentence_transformers.util.batch_to_device`)
  and ST's own normalisation (`torch.nn.functional.normalize`, the call `encode(normalize_embeddings=True)`
  applies). The prompt comes from the checkpoint's own prompt table (the entry ST's `encode` resolves for the
  media sides; the table is cross-checked against the declared prompts at load).
- `mode_embed`'s `vectors` closure: a text-only batch goes through `text_vectors`; a media batch keeps
  `model.encode(..., prompt_name=..., normalize_embeddings=True, processing_kwargs=video_processing_kwargs(...))`.
- Module docstring: the embed mode's text-only sentence now names the tokenizer path and why.

Why the tokenizer path: `SentenceTransformer.encode` hands a text-only input to the checkpoint's multimodal
processor (the checkpoint's `sentence_bert_config.json` declares the `message` modality), and the processor's
`validate_inputs` counts the checkpoint's placeholder spellings in the text and refuses a text-only row that
carries one. The engine has no such check — its tokenizer matches the spelling as the checkpoint's own added
token (id 258880) and the model reads that token. The reference now produces exactly the card's own ids and
vectors for a text-only row (the checkpoint's template renders a text-only user turn to exactly its content
and the prompt as its system turn, so `prompt + text` is the same string the message path tokenizes), minus a
media-count check that cannot apply to a row with no media. No spelling rule of its own: the file's own parse
is what the checkpoint's tokenizer does.

### The tests (test-first)

`rcp-ndcg-test/tests/recipes/test_embeddinggemma_2.py`

- New `test_the_reference_takes_the_card_tokenizer_path_for_a_text_only_row`: the spelling row renders; the
  checkpoint's tokenizer receives the prompt-prefixed texts with the processor's text kwargs; the card's
  module stack runs per side; the vectors are the card's own normalisation (`encode`'s
  `normalize_embeddings=True`); and a media row (a text with the spelling beside an image) still goes through
  `encode` — the card's own validating path. The fake `encode` mirrors the real processor's refusal (the
  wave's exact message), so the unfixed code is red with the wave's own error.
- `test_the_embed_video_pin_is_the_engines_rule_not_the_generic_uniform_one`: its fake now carries the prompt
  table, the tokenizer and the module call the text path needs; its tail asserts the media call is the only
  `encode` call ("the text-only query takes the tokenizer path, never the validating encode").

### Docs

- `CHANGELOG.md`: the round-11 entry under `### Fixed` (quoted in section 7).
- `docs/how-to/adopt-a-model.md`: one row in the pitfall catalogue — "the reference dies on a text-only row
  carrying the checkpoint's placeholder spelling" → give the reference the card's own tokenizer path for
  text-only sides, never a spelling rule of its own.

## 4. Verification

### Red-first (the unfixed code, the wave's own error)

```
$ uv run --no-sync pytest rcp-ndcg-test/tests/recipes/test_embeddinggemma_2.py -q -p no:cacheprovider \
      -k "tokenizer_path or video_pin"
FAILED ...::test_the_embed_video_pin_is_the_engines_rule_not_the_generic_uniform_one
FAILED ...::test_the_reference_takes_the_card_tokenizer_path_for_a_text_only_row
E   ValueError: Found 1 <image-placeholder> tokens in the text but no images were passed.
    (the fake mirrors the real processor's refusal; the reference called encode for a text-only side)
E   AssertionError: the text-only query takes the tokenizer path, never the validating encode
2 failed, 18 deselected
```

### The real checkpoint's own objects on CPU (transformers 5.19.0, the pinned tokenizer, the real processor)

A scratch probe (a scratch directory outside the repository) loads the checkpoint's tokenizer and processor
from local files and runs the fixed `text_vectors` against them:

```
transformers 5.19.0
tokenizer GemmaTokenizer
image token id 258880 occurrences in the file's own parse 1
split parse occurrences 0 widths (1, 19) (1, 21)
processor EmbeddingGemma2Processor image_token '<|image|>'
REFUSAL: Found 1 <image-placeholder> tokens in the text but no images were passed.
CHAT-TEMPLATE PATH REFUSAL: Found 1 <image-placeholder> tokens in the text but no images were passed.
  frames: ['probe.py:48:<module>', 'processing_utils.py:2250:apply_chat_template',
           'processing_utils.py:669:__call__', 'processing_embedding_gemma2.py:210:validate_inputs']
text_vectors ids equal the checkpoint tokenizer's: True
vectors (1, 4) norm 1.0
```

The frame chain is the wave's own (`processing_utils.py:669` → `validate_inputs`), reached through
`apply_chat_template` — the call ST's message path makes. The probe also shows the file's own parse matches
the spelling as the added token exactly once (the split parse matches 0, a different id sequence), which is
why the reference must use the checkpoint's tokenizer and not a parse of its own.

### After the fix

```
$ uv run --no-sync pytest rcp-ndcg-test/tests/recipes/test_embeddinggemma_2.py -q -p no:cacheprovider
16 passed, 4 skipped
```

### Checks on the merged tree

```
$ uv run --no-sync ruff check .            All checks passed!
$ uv run --no-sync ruff format --check .   630 files already formatted
$ uv run --no-sync basedpyright            0 errors, 0 warnings, 0 notes
$ heavy uv run --no-sync pytest tests/ -q -n 4 -p no:cacheprovider
                                           4145 passed, 155 skipped
$ uv run --no-sync pytest tests/contract tests/docs -q -p no:cacheprovider
                                           413 passed, 107 skipped
$ uv run --no-sync mkdocs build --strict -d <scratch>/site
                                           built (strict)
$ bin/public-names-step                   public-names: clean (0 baselined hits remain)
```

### The gate

`<pending>` — `timeout 7200 bin/gate lane/fix11ref` on the merged tree.

## 5. The merged tip and the case-fill failures (integration, not this lane's change)

Merging `rfc-0001`'s tip `9224b700` (the cuda reference-case fill) left **three pre-existing failures** in
`rcp-ndcg-test/tests/test_case_fill.py`, reproduced with this lane's changes present and absent:

- `test_a_media_case_is_no_longer_skipped_and_a_kind_none_case_still_is` → `assert 1 == 10`
  (`report.pending_before`: the shipped cases are the fill's outputs now);
- `test_an_embed_media_case_fills_and_carries_its_own_tolerance`;
- `test_the_retirement_vocabulary_covers_every_shipped_generated_note`.

The fix is `lane/fix11ctx`'s `3bd22c08` ("The cuda-filled cases' tests and notes read the filled state"), which
its own report lists under **Files outside scope**. Per the supervisor's decision this lane held its gate for
that commit to land on `rfc-0001`, re-merged, and gated; nothing of `3bd22c08` was taken into this branch.

## 6. Open questions

- **The engine's own read of the spelling row is a wave observation, not a CPU test.** The engine tokenizes
  the rendered content with the file's own parse and, with no media in the request, the placeholder id goes
  through the embedding layer as an ordinary token; this lane verified the *reference* side's ids and the
  processor's refusal on CPU, and the round-11 wave's stage 2 is the engine-side confirmation.
- **A checkpoint whose prompt table omits the name** (`model.prompts` has no `SearchQuery`/`Document`) now
  fails with a `KeyError` in the text path where the media path would raise ST's own `ValueError`. The
  shipped checkpoint declares both (the reference cross-checks them against the declared prompts at load);
  a checkpoint without the table is a loud failure either way.

## 7. CHANGELOG entry

```
- **The embeddinggemma-2 reference takes the card's tokenizer path for a text-only row** (the E2 round-11
  reference failure): `SentenceTransformer.encode` hands a text-only input to the checkpoint's multimodal
  processor (its `sentence_bert_config.json` declares the `message` modality), and the processor's
  `validate_inputs` counts the checkpoint's placeholder spellings in the text and refuses a text-only row
  that carries one -- `ValueError: Found 1 ... tokens in the text but no images were passed.` at
  transformers' `EmbeddingGemma2Processor.validate_inputs`, reached through `processing_utils.py`'s
  `apply_chat_template` -- so the harness's spelling stratum killed the reference subprocess (embed) before
  any comparison. The engine has no such check: its tokenizer matches the spelling as the checkpoint's own
  added token (id 258880) and the model reads that token. A text-only side now goes through the card's own
  tokenizer path (`reference.py::text_vectors`): the checkpoint's own tokenizer (`model.tokenizer`, the
  processor's own instance) over the prompt-prefixed text, with the processor's text kwargs (padding,
  longest-first truncation) and the tokenizer's post-processor, and then the card's own module stack
  (`Transformer -> Pooling -> Normalize`) with ST's own device move and normalisation -- the file's own
  parse, never a second spelling rule; a media side keeps the card's `encode` and its validation. CPU tests
  pin the text path (a fake mirrors the processor's refusal, red on the unfixed code with the wave's exact
  message), the module stack, the normalisation, and that a media row (a text with the spelling beside an
  image) still goes through `encode`.
```

## 8. Public surface changes

None: no `__all__`, CLI, exit code, schema or snapshot moved. `tests/contract` and `tests/docs` pass
unchanged.

## 9. Files outside scope

None.

## 10. For the next lanes

- The round-11 wave's stage 2 for `embeddinggemma-2` is the confirmation: the spelling stratum's rows should
  now compare like any other text row, and a media row's manual-placeholder text still rides the processor's
  validation.
- A family whose card's own code refuses a row the engine accepts needs a *declared* text path (this lane's
  `text_vectors`) rather than a spelling detector: the detector would be a second tokenization rule, and the
  card's own tokenizer is the only authority on the parse.
