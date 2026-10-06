# Preprocessing and chunking

What a judge reads decides what its judgements mean, and what a served model reads decides what its vectors and
scores mean. This page describes the one place that decides both: the preprocessing policy of a judging pass, for
text, page images and video, and the text budget that fits every served role's requests into a model's input.
Nothing is cut silently: every cut is either declared policy or a per-window budget cut, and both are recorded.

## Text

A judging pass declares what happens to a document longer than a token cap. The declaration is the `text` field of
the pass's `preprocessing` (in a run config, or `--set preprocessing.text.on_overflow=...` on `rcp-ndcg judge`), and
it is the only place that decides how much of a document the judge sees at load. It is recorded in the run manifest
and folded into the judging identity and the judgement family key. Two passes that showed the judge different text
therefore never share cached judgements or pool observations.

```yaml
# in a run config
dataset: beir:data/my_corpus
preprocessing:
  text: {on_overflow: truncate, max_tokens: 20000}    # or chunk, see below; unset: 32768
```

### The judge's tokenizer

Every text limit counts tokens of the judge's own tokenizer, which the judge config names:

```yaml
# in a judge config
tokenizer: Qwen/Qwen3.5-397B-A17B-FP8           # a Hugging Face repository id, optionally @revision
# tokenizer: /models/my-judge/tokenizer.json    # or a local tokenizer.json, or the directory holding it
```

On the command line it is `--set judge.tokenizer=...`. The client reads the repository's `tokenizer.json` with the
`tokenizers` library (no torch), from the `[hf]` extra (`pip install "rcp-ndcg[hf]"`), once per process. The shipped
configs of self-served judges name their model's repository. The hosted `gpt5_hosted` config names none, because
OpenAI publishes no tokenizer on the Hub.

The tokenizer's identity is the SHA-256 of its `tokenizer.json`. The name and the hash are recorded in the pass's
preprocessing identity, and the hash is part of the judgement family key, so passes counted with different
tokenizers never pool. The tokenizer belongs to the judge, like its model. So the rubric key, which lets several
judges pool with a severity term, leaves it out.

Without a tokenizer no text is cut, because there is no character fallback. `keep` runs, and `truncate`, `chunk` and
`fail` are refused with a `ConfigError` whose hint is to set `judge.tokenizer`.

### `on_overflow`

| `on_overflow` | Over-cap document | What is recorded |
|---|---|---|
| `keep` (default) | kept whole | nothing is cut |
| `truncate` | cut to `max_tokens` | every cut (id, tokens and characters before and after) in the text census |
| `fail` | refused with `DocumentOverCapError` (exit 12) | the offending document id |
| `chunk` | split into overlapping chunks, each judged | the `chunk_mapping` on every judged row |

`max_tokens` defaults to 32,768 (2^15) for `truncate` and `fail`, and `chunk` takes it from the chunk geometry. A store judged under an earlier default (20,000) keeps its recorded policy: an unset cap that re-judges after this change resolves differently and the store refuses the mixed instrument, so pin `max_tokens: 20000` explicitly when a config must keep the old cap's identity.

### Where a cut falls

Every cut is made in the original text at a token boundary. The boundary is located with the tokenizer's offset
mapping, the character span of each token, and tokens are never decoded back into text. So a truncated document is a
verbatim prefix of the document, and a chunk is a verbatim slice of it, with the whitespace, the punctuation and every
character as stored. The kept text is counted again as the judge reads it. A word cut in the middle, or an escaped
character, can take more tokens on its own than it did in place, so the cut moves back to the longest prefix that
fits. The census records the tokens before and after each cut, and the characters for information.

```python
from tokenizers import Tokenizer, models, pre_tokenizers

from rcp_ndcg.data import TextTokenizer
from rcp_ndcg.data.preprocess import token_prefix

# A word-level tokenizer built in memory; a judge's own comes from rcp_ndcg.data.load_tokenizer.
backend = Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
backend.pre_tokenizer = pre_tokenizers.Whitespace()
words = TextTokenizer.from_backend(backend, name="words")

print(repr(token_prefix("Keep  the\nwhitespace, exactly.", 4, words)))  # 'Keep  the\nwhitespace,'
```

### The window budget

Apart from the load-time policy, each judging window has a text budget, counted in tokens with the judge's tokenizer.
It starts from the judge's `context_tokens` and subtracts, in order:

1. the prompt's own tokens: the stage's template rendered with the query and empty documents;
2. a fixed reserve of 256 tokens for what the tokenizer file does not describe and no media occupies (the chat
   template's role markers, its generation prompt, and a system message that some templates insert);
3. the documents' media charge: each image and each sampled video frame its vision block (the processor's
   vision start and end markers plus its patch tokens), a video container its temporal grid, and the
   template's media marker per media part, counted with the judge's tokenizer;
4. the completion reserve, `max_output_tokens`, at most half of what is left.

The rest is shared equally by the documents of the window. Each document's text, as the prompt carries it (stripped
and XML-escaped), is cut to its share at a token boundary, and the cut is recorded under `window_budget` in
`preprocessing.jsonl` beside the judgement store.

A judge config without `context_tokens` sends documents whole. So does a judge without a tokenizer, which logs a
warning when it declares a context: the endpoint then refuses a window that does not fit, and that window is recorded
as invalid. `chunk` keeps chunks within `max_tokens`. Choose a cap below the window budget when every token must reach
the judge.

### Estimates

`rcp_ndcg.estimate` walks the same windows and budgets as the pass. With a tokenizer it counts every prompt's tokens
exactly (plus the 256-token reserve), and the estimate says `input_token_count: exact`. Without one it approximates
2.0 characters per token, says `input_token_count: approximate`, and lists the approximation under `assumptions`.
Nothing is ever cut by that approximation.

### `chunk`

```yaml
preprocessing:
  text:
    on_overflow: chunk
    chunk:
      max_tokens: 4000      # split trigger and the largest chunk (tokens)
      overlap_tokens: 250   # tokens repeated from the previous chunk
```

A document of at most `max_tokens` tokens is judged whole. A longer document
is split where it meets its query, before judging:

1. **Chunks.** The chunks are windows over the document's own tokens. Chunk `k + 1`
   starts `overlap_tokens` tokens before chunk `k` ends, and each chunk holds at most
   `max_tokens` tokens. The first chunk starts at the start of the document and the last
   ends at its end. Each chunk is the verbatim slice of the document between its first
   and its last token. A slice that takes more tokens on its own than it did in place
   ends one token earlier until it fits. The split is deterministic for a given tokenizer,
   and the tokenizer's hash is in the family key.
2. **Identity.** Chunk `k` of document `d` has the id `d#k` (`k` from 0), and the
   row's `chunk_mapping` maps every candidate to its document (an unsplit
   document maps to itself). The mapping, never the id, is what later stages
   read. First-stage scores and score vectors are copied onto each chunk;
   `qrels` stay keyed by document.
3. **Candidate cutoff.** The candidate depth counts documents: a document inside
   the cutoff is judged with all of its chunks.
4. **Aggregation.** Judgements return to the document by max-pooling. In a
   rubric window a document passes criterion `Ck` when any of its chunks in that
   window passes it; each window stays a separate placement in the 2PL
   likelihood. Tournament scores take the best chunk per document.

Documents that carry media (page images, frames) are never split: their text
is not what a vision judge reads.

Chunking costs more judge calls: a query whose candidates grow from `n`
documents to `n + extra` chunks gets proportionally more windows.

### In code

```python
from tokenizers import Tokenizer, models, pre_tokenizers

from rcp_ndcg.data import TextTokenizer
from rcp_ndcg.data.preprocess import ChunkPolicy, split_into_chunks

backend = Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
backend.pre_tokenizer = pre_tokenizers.Whitespace()
words = TextTokenizer.from_backend(backend, name="words")

pieces = split_into_chunks("one two three four five six seven", ChunkPolicy(max_tokens=3, overlap_tokens=1), words)
print(pieces)  # ['one two three', 'three four five', 'five six seven']
```

The same policy in Python is `rcp_ndcg.Preprocessing`, which `rcp_ndcg.judge(..., preprocessing=...)` and
`rcp_ndcg.estimate` take:

```python
from rcp_ndcg import Preprocessing

policy = Preprocessing.model_validate(
    {"text": {"on_overflow": "chunk", "chunk": {"max_tokens": 4000, "overlap_tokens": 250}}}
)
print(policy.chunk)  # the chunk geometry the judging pass applies
```

## Text budgets for served roles

A served model (an embedder, a reranker) reads its requests through a fixed frame: a system turn, instruction
prefixes, role markers, and the end-of-turn token its score is pooled from. That frame is the **anchor** rule this
package implements everywhere text is cut: a model reads its output from fixed positions of its template, so a cut
must apply to the content spans only, inside a budget computed after reserving every fixed template token, and the
template must be re-attached after the cut. A right cut of a whole rendered prompt drops tail anchors, a left cut
drops head anchors -- which is why no role config declares an engine-truncation field (`truncate_prompt_tokens`
`--allow-auto-truncate` and their relatives are absent from the role configs by design), and why the fixed overhead
is measured rather than guessed.

One mechanism does this for every role: the declared :class:`~rcp_ndcg.data.TextBudget` and
:func:`~rcp_ndcg.data.preprocess.fit`. The role configs carry no engine-truncation field
(`truncate_prompt_tokens` and `truncation_side` are absent from them by design): the package cuts the content
itself, and an engine-side truncation of a rendered prompt is exactly the cut this rule forbids.

### The template as data

A `TemplateSpec` declares, per request shape (an embedder's `query` and `document`, a reranker's `pair`), an
ordered list of segments -- a fixed piece of frame text or a content span:

```python
from rcp_ndcg.data import Segment, TemplateSpec

pair = TemplateSpec(
    pair=(
        Segment(fixed="<instruct>: judge the pair\\n<query>: "),
        Segment(content="query"),
        Segment(fixed="\\n<document>: "),
        Segment(content="document"),
        Segment(fixed="{special:end_turn}"),
    ),
    anchor="last",
)
print(pair.shapes())
```

The `pair` shape orders query and document per model -- the reranker above reads the query first; a
reranker whose template puts the document first makes the query block an anchor, and the segments say so. A
special token is written by name (`{special:end_turn}`) and resolved at render time from the tokenizer's added
tokens; specials are never typed literally, so a template outlives tokenizer rewrites. The template also declares
`anchor` (`last`, `first`, `mean` or `marker`, with `anchor_markers` naming the specials) -- the position the
model reads its output from -- and `add_special_tokens` per shape: what the engine does to the rendered string
for that route (vLLM's pooling and scoring routes append the tokenizer's post-processor tokens; the chat-embed
form does not). The budget reserves those tokens too: they are part of the measured overhead.

### The budget and the fit

The budget names the tokenizer, the `max_tokens` (the model's whole input sequence, in that tokenizer's tokens),
optionally `query_max_tokens` (the query's share of a pair budget: when a pair overflows, the query is cut to it
first and the document gets the rest -- an input under budget goes out unchanged, so within `fit` the share
binds on overflow), `on_overflow` (`cut` by default, `chunk` or `fail` opt-in), the chunk geometry, and
`aggregation: max`. `fit` then, per input:

1. measures the fixed overhead once per (template, shape): the template rendered with every content span empty,
   counted as the engine reads it (the shape's `add_special_tokens` flag included);
2. cuts only the content spans, at token boundaries, verified against the *assembled* render so a byte-level
   merge across a span join cannot push the request over the budget, and re-attaches the template;
3. on `chunk`, splits the document into verbatim chunks and renders **every chunk with the full template** --
   engine-side chunking of a framed render keeps the frame only on the first and last chunk, so chunking is
   always client-side here;
4. records every cut in the census under the `text_budget` mechanism.

An input under budget comes back byte-identical to the uncut render -- within `fit`, which settles a pair's
query span per pair. The rerank wire carries one query per request, so the rerank client settles the shared
query span once per call (`fit`'s own rules, on a probe pair): whenever the query exceeds its declared share it
ships at it, recorded once in the census under the doc id `<query>`, and every document span is verified
against the span that ships. The function returns the rendered strings
(the wire routes take text; tokenising once here to measure and cut is the same work either way), the cut content
per span (for routes the engine renders the template on), the output ids, and the chunk mapping.

```python
from tokenizers import Tokenizer, models, pre_tokenizers, processors

from rcp_ndcg.data import Segment, TemplateSpec, TextBudget, TextTokenizer
from rcp_ndcg.data.preprocess import fit

backend = Tokenizer(
    models.WordLevel({"[UNK]": 0, "the": 1, "query": 2, "document": 3, "evidence": 4, "page": 5}, unk_token="[UNK]")
)
backend.pre_tokenizer = pre_tokenizers.Whitespace()
backend.add_special_tokens(["<|end_turn|>", "<|end_of_text|>"])
backend.post_processor = processors.TemplateProcessing(
    single="$A <|end_of_text|>",
    pair="$A $B <|end_of_text|>",
    special_tokens=[("<|end_of_text|>", backend.token_to_id("<|end_of_text|>"))],
)
tokenizer = TextTokenizer.from_backend(backend, name="docs/word-level")

document_template = TemplateSpec(
    document=(Segment(fixed="<doc> "), Segment(content="document"), Segment(fixed=" {special:end_turn}"))
)
budget = TextBudget(tokenizer="docs/word-level", max_tokens=16, template=document_template, on_overflow="cut")
result = fit(["the document with evidence " * 10], shape="document", budget=budget, tokenizer=tokenizer)
print(result.texts[0])          # the frame re-attached around the cut: '<doc> the document with evidence ...'
print(result.cuts[0].as_row())  # the cut, mechanism 'text_budget', budget_source 'tokenizer'
```

### Chunked documents pool by maximum

A document chunked under `on_overflow: chunk` is scored once per chunk and its document score is **the maximum
over its chunks** -- a document is as relevant as its best chunk, the same rule as
`max_pool_scores_by_document` (the aggregation the judging pass applies to its own chunks). It is declared on
the config (`aggregation: max`, the only value for now), returned on every `FitResult` of a chunked call, and
named on every census row of a chunked document, so a reader of `preprocessing.jsonl` never has to guess how the
numbers were pooled. `FitResult.chunk_mapping` carries each chunk id (`<id>#<k>`) back to its input, ready for
that pooling function.

### Explicit budgets, and hosted vendor profiles

A self-hosted role config (`api: openai_embeddings`, `vllm_pooling` or `rerank`) must declare `tokenizer` and
`max_tokens` -- the package cuts the content itself, so it must know both; without them the config is refused
with a hint naming the two fields. A hosted vendor profile (cohere, voyage, gemini) without a tokenizer sends
its content uncut: the vendor's documented limit is declared as `max_tokens`, recorded as the effective budget
(`budget_source: vendor` in the census -- one row per corpus and budget per census -- and one warning per
corpus per process), and nothing is measured or cut client-side. With a
tokenizer, a vendor profile follows the same rule as self-hosted.

The role configs also declare `template` (the `TemplateSpec` above), `empty_doc` (`send`, `omit_zero` --
never sent and scored `0.0` -- or `send_text` with its `empty_doc_text` placeholder; every role client
consumes it, for an empty text document and for one whose every media item the budget dropped),
`request_shape` (`text`, `messages` or `token_ids`; the adapters implement it), and the reranker's
`instruction` gains a `system` value (the instruction as a system message).

Media are never cut. Every served request goes through one preparation call
(`rcp_ndcg.data.prepare.prepare_request`) -- the same path the judge's images take -- which sizes every image
and video frame exactly as the declared `image_processor` would under the role's `image_policy`, counts the
request's media tokens exactly, and reserves them whole out of `max_tokens` before the content is cut. When
media alone fill the budget the declared `on_overflow` decides: `cut` (the default) shrinks to the policy's
minimum, then drops whole items most expensive first -- every drop recorded in the media census with
`dropped=True` --, `fail` refuses the request, and `chunk` is refused (media are not chunkable: a vision
block is atomic, the engine sees a whole item or none of it). A document whose every media item was dropped
is an empty document, and follows `empty_doc`. A role with an `image_processor` also declares
`max_images`/`max_videos` (the per-request gates, refused before sending), and its startup probe runs the
engine media check: one prepared probe image, the engine's reported prompt tokens compared with the counted
ones -- a mismatch is refused, a reply without usage is recorded `not_checked`, never silent.

## Page images and video

The client prepares every image and every video frame before it is sent, and records what it sent. A judge
endpoint therefore only has to be OpenAI-compatible: vLLM and SGLang run stock, with no media flags, and a hosted API
receives the same bytes.

How a judge sees page images and video is a judging setting, the `image` and `video` fields of the pass's
`preprocessing` (in a run config, or `--set preprocessing.image.max_px=...` on `rcp-ndcg judge`):

```yaml
preprocessing:
  image: {min_px: 65536, max_px: 1310720}   # the pixel budget of every page image and video frame
  video: {num_frames: 8, wire: frames}      # frames per clip, sampled by the client
```

### Processor families

Which resize applies is a property of the judge's model, so the judge config names it as `image_processor`:

| `image_processor` | Models | Factor (patch x merge) | Budget a stock engine keeps (pixels) |
|---|---|---|---|
| `qwen2_vl` | Qwen2-VL | 28 (14 x 2) | 3,136 to 1,003,520 |
| `qwen2_5_vl` | Qwen2.5-VL | 28 (14 x 2) | 3,136 to 12,845,056 |
| `qwen3_vl` | Qwen3-VL, Qwen3.5-397B, Qwen3.6-27B | 32 (16 x 2) | 65,536 to 16,777,216 |

The shipped Qwen3.5-397B and Qwen3.6-27B judge configs declare `qwen3_vl`. The resize is the processor's own `smart_resize`,
as transformers' Qwen-VL image processor implements it and as vLLM and SGLang both run it for these models. Both
edges are rounded to a multiple of the factor. If the area then lies outside `[min_px, max_px]`, the image is scaled
by the square root of the ratio and each edge is floored (when shrinking) or ceiled (when growing) to the factor.
Aspect ratio is kept to within one factor.

Each image is prepared the same way:

1. It is decoded and rotated by its EXIF orientation.
2. It is converted to RGB, with any transparency composited onto white, as vLLM loads an image.
3. It is resized to its target size with the BICUBIC filter, the processor's own. An image already at that size is
   not resampled.
4. It is encoded as PNG, so the engine decodes exactly the resized pixels.

A judge whose config names no `image_processor` (a hosted API, or a family not in the table) is sent every image
unchanged, its stored bytes, and the processor on the far side decides. The client never guesses a family, and the
recorded policy says `processor: null`.

### No server flags

The judging pass checks the pixel budget against the budget a stock engine keeps for the judge's family, and refuses
(`ConfigError`) a budget outside it. Inside it, a prepared image is a fixed point of the engine's resize: the engine's
`smart_resize` returns the same size, and an equal-size resize leaves the pixels unchanged. So the engine keeps what
the client sent, and no `--mm-processor-kwargs`, `SGLANG_IMAGE_MAX_PIXELS` or other media flag is needed. The one
engine setting a multimodal judge still needs is the per-request media count (`max_images`, `max_videos`).

The test suite checks this over a grid of image sizes and budgets for every family, against copies of the
transformers and SGLang resize functions under each engine's default settings. An image whose prepared size a stock
engine would still change or refuse is refused by name (`DataError`) rather than sent. This happens only to an
extreme strip under a budget at the edge of the range, where flooring to the factor leaves the engine's range or an
aspect ratio above 200.

### Video

`video` sets the number of frames shown per clip, `num_frames`, and how they are sent, `wire`:

- **`wire: frames` (the default and the exact one).** The client samples `num_frames` frames from the clip's
  pre-extracted frames (the `frames` reader) at `np.linspace(0, total - 1, num_frames)` truncated to integers.
  vLLM's video loader and SGLang's Qwen-VL video preprocessing apply the same rule to a decoded container.
  Each frame is prepared as an image under the image policy and sent as a standard `image_url` part, so the
  frames counted are the frames sent.
- **`wire: video_url` (opt-in, for models with a native video encoder).** The container is sent unchanged and
  the engine decodes and samples it with its own video loader. A stock engine samples its own default number
  of frames (32 on vLLM), which would make the counted tokens and the recorded instrument describe frames
  nobody chose, so the policy refuses `video_url` unless `engine_video_pinning: true` declares the engine
  pinned to the same frame count -- `--media-io-kwargs '{"video": {"num_frames": N}}'` on vLLM,
  `--mm-process-config` on SGLang (see [serving](serving.md)) -- and refuses a single-frame container (the
  declared instrument merges frames in time, which needs a temporal pair; a single frame is an image). Run
  `engine_media_check` once against a prepared probe when a serving setup changes (below); a mismatch says
  the engine's media handling is not the one the counted tokens describe.

### What a container costs

A container is counted as a stock engine's video accounting counts it, never at the client's declared image
budget -- the container is sent unchanged, so the client's pixel budget never reaches the engine. It is
patchified in time: 8 frames merge into `ceil(8 / 2) = 4` per-frame token runs, not 8. The frames' geometry is
the family's own video budget (`PROCESSORS`):

- The Qwen2-VL families size each frame independently by the checkpoint's per-frame budget -- stock vLLM's
  accounting, which passes the checkpoint's image-processor size for videos -- under one vision block for the
  whole clip: 8 frames of 720x1280 cost 4,786 tokens. SGLang's video path caps per-frame pixels lower
  (602,112 px, clip-dependently), so there the count differs; the pinning declaration ties the frame count,
  and `engine_media_check` compares the engine's actual count at run time.
- `qwen3_vl` budgets the whole clip together (4,096 to 25,165,824 px), which shrinks the per-frame resolution
  as the frame count grows -- 8 frames of 720x1280 cost 3,520 patch tokens, 128 frames 11,520 -- and renders
  one timestamp line (`<0.0 seconds>`, a declared bound of 10 tokens, covering every timestamp a clip of up
  to 27.8 hours can carry) and one vision block per temporal group.

An odd frame count is padded by repeating its last frame, as the processors do.

A clip with fewer frames than `num_frames` is refused, and so is a clip longer than an optional `max_duration_s`.

### What is recorded

The effective image policy, the declared budget plus the judge's processor family, is part of the preprocessing
record and of the judgement family key, and so is the video policy. Two passes with different budgets or different
processors therefore never pool. Every prepared image, frame and container is also written once per judgement
store to `preprocessing.jsonl`, beside the text cuts, as a `media` row. The row holds the stored size and hash, the
sent size, hash and MIME type, the processor, whether the image was resized, and who sampled the frames.

The judge's window text budget subtracts the documents' media charge: each image and each sampled video frame
its vision block, `(height / factor) x (width / factor)` patch tokens plus the processor's vision start and end
markers, a video container its temporal grid, and -- a declared reserve, not an engine count -- the template's
media marker per media part, counted with the judge's tokenizer. The payload builder replaces each marker with
the media part, so the charge errs a few tokens high per media part, never low. They cannot be counted when
there is no pixel budget or no `image_processor`. A judge with a text budget (`context_tokens` and a
`tokenizer`) is then refused (`ConfigError`) instead of budgeted on a guess. A judge without one, such as a
hosted API with no public
tokenizer, is sent the images as stored. For such a judge, `estimate` assumes 1,000 tokens per image or video
frame shown, roughly one document page at a hosted API's high detail, and says so in its `assumptions`. The
value is an assumption, not a bound, and no text is ever cut on it.

A prompt with images or video is refused (`CapabilityError`) unless the judge config declares that the model reads
them (`max_images`, `max_videos` above 0).

```python
from rcp_ndcg.data import ImagePolicy

policy = ImagePolicy(min_px=65536, max_px=1280 * 32 * 32).for_processor("qwen3_vl")
print(policy.target_size(2200, 1700))  # (1280, 992): the size the judge's processor keeps
print(policy.image_tokens(2200, 1700))  # 1240 patch tokens; the vision block costs two more
print(policy.max_image_tokens)  # 1280: the bound when a size was never recorded
```

### Retrieval roles: one preparation path, and what to send when media do not fit

The same policy objects size the media of the retrieval roles (encoders, rerankers). A role config declares
the judge's `image_processor`, `max_images`, `max_videos` and the optional image and video policies (the same
`ImagePolicy` / `VideoPolicy` types, content fields), and its clients prepare one request's contents with
`rcp_ndcg.data.prepare.prepare_request`, which returns the prepared contents and the request's exact media
token counts -- so the role's text budget can subtract them, never cut them:

```python
from pathlib import Path

from PIL import Image
from rcp_ndcg_core.content import Content, ImagePart, MediaRef

from rcp_ndcg.data import ImagePolicy
from rcp_ndcg.data.prepare import MediaCensus, fit_media_to_budget, prepare_request

image_policy = ImagePolicy(min_px=65536, max_px=1280 * 32 * 32).for_processor("qwen3_vl")
Image.new("RGB", (2560, 2560), (30, 30, 30)).save("page.png")
page = Content.from_parts([ImagePart(ref=MediaRef(uri="page.png", mime="image/png"))])

prepared = prepare_request([page], image_policy, None)
print(prepared.tokens)  # MediaTokenCount(tokens=1227, bounded=0): what the media cost the prompt

# When the media alone exceed the text budget, a vision block is never cut; the declared rule is
# shrink to the policy's minimum, then drop whole items, recorded in the census:
fit = fit_media_to_budget(prepared.media, image=image_policy, video=None, text_budget_tokens=800)
print(fit.tokens, len(fit.dropped))  # 66 0: shrunk to the 65,536px floor, nothing dropped
MediaCensus(sink="preprocessing.jsonl").record(corpus="c", doc_id="d1", media=fit.dropped, dropped=True)
```

Run `rcp_ndcg.data.resolution.engine_media_check` against a prepared probe whenever a serving setup changes:
send one prepared image, count its prompt exactly (`content_media_tokens` plus the template tokens the probe
request carries), and compare the engine's `usage.prompt_tokens` against it. A returned mismatch
(`EngineMediaMismatch`) says the served engine's media handling is not what the counted tokens describe -- a
reconfigured engine or a mis-declared `image_processor` -- and every later count is suspect: record or raise
it instead of judging around it. The runtime call site is wired: a role client with an `image_processor`
exposes `probe()` and `check_engine_media()` -- the startup probe sends one prepared probe image and refuses
on a mismatch; a reply without usage is recorded `not_checked` (never silent).

## Identity

`rcp_ndcg.Preprocessing` is the effective policy of one judging pass: the text policy, the chunk geometry, the
image policy (with the judge's processor family) and the video policy. It is part of the judging identity and of the
judgement family key, and the judge's tokenizer is recorded beside it: its name and hash in the preprocessing
identity, its hash in the family key. Two passes that showed the judge different text or pixels therefore never share
cached judgements or pool observations.
