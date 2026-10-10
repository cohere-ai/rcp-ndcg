# Preprocessing and chunking

What a judge reads decides what its judgements mean. This page is the judge's side: the preprocessing policy of a
judging pass -- text, page images and video. What a served model reads is decided by its
[text budget](text-budgets.md), the other half of the same machinery. Nothing is cut silently: every cut is either
declared policy or a per-window budget cut, and both are recorded.

The machinery has one home per concept: the text policy and the chunking in `rcp_ndcg.data.text_policy`, the cut
record and the census in `rcp_ndcg.data.census`, the served roles' fit in `rcp_ndcg.data.text_budget`, the census
files' record I/O in `rcp_ndcg.storage.census`. `rcp_ndcg.data.preprocess` re-exports the names of those homes
(the module's docstring lists them exactly; the postprocess names it does not re-export come from
`rcp_ndcg.data.postprocess` and the Matryoshka head from `rcp_ndcg.data.mrl`), and every snippet below imports
from it.

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
2. a fixed reserve of 256 tokens for what the tokenizer file does not describe and no media occupies (the
   chat-template role markers, its generation prompt, and a system message that some templates insert);
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

## Page images and video

The client prepares every image and every video frame before it is sent, and records what it sent. A judge
endpoint therefore only has to be OpenAI-compatible: vLLM runs stock, with no media flags, and a hosted API
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
| `qwen2_vl` | Qwen2-VL | 28 (14 x 2) | 3,136 to 12,845,056 |
| `qwen2_5_vl` | Qwen2.5-VL | 28 (14 x 2) | 3,136 to 12,845,056 |
| `qwen3_vl` | Qwen3-VL, Qwen3.5-397B, Qwen3.6-27B | 32 (16 x 2) | 65,536 to 16,777,216 |

The shipped Qwen3.5-397B and Qwen3.6-27B judge configs declare `qwen3_vl`. The resize is the processor's own `smart_resize`,
as vLLM runs it for these models. Both
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

### One lowering, and the part order

Every wire block a content becomes is built by the one lowering,
`rcp_ndcg.data.media.content_parts_payload`: text, `image_url` and `video_url` blocks in the content's own part
order, so a caption after its page stays after it -- the order of an item's parts is information the model reads.
The judge's wire adds its two guards as hooks and changes no block: an image or frame must be prepared (the
preparation above), and a container must be under the inlined-size cap (`RCP_NDCG_MAX_VIDEO_BYTES`, 64 MiB by
default). The lowering's declared mechanisms, in one place: an empty text part lowers to nothing; a video part
carrying both frames and a container lowers to its frames (the sampling the policy chose), the container only when
there are no frames; an already-inlined image's `data:` URI is sent as it is, and any other reference's bytes are
read through the media resolver and inlined.

### No server flags

The judging pass checks the pixel budget against the budget a stock engine keeps for the judge's family, and refuses
(`ConfigError`) a budget outside it. Inside it, a prepared image is a fixed point of the engine's resize: the engine's
`smart_resize` returns the same size, and an equal-size resize leaves the pixels unchanged. So the engine keeps what
the client sent, and no `--mm-processor-kwargs` or other media flag is needed. The one
engine setting a multimodal judge still needs is the per-request media count (`max_images`, `max_videos`).

A model whose own budget lies outside the stock range (the Qwen3-VL-Embedding card resizes to 4,096..1,843,200 px,
below `qwen3_vl`'s stock floor) is served with the engine pinned to that budget instead, and the policy declares
it: `engine_pixel_pinning: true` admits the budget and makes it the one the engine keeps, so a prepared image is a
fixed point of the pinned resize. A serving recipe pins vLLM with `serve.mm_processor_kwargs: {images_kwargs:
{min_pixels: ..., max_pixels: ...}}` and refuses to load unless both sides carry the same numbers.

The test suite checks this over a grid of image sizes and budgets for every family, against a copy of the
transformers resize function under the engine's default settings. An image whose prepared size a stock
engine would still change or refuse is refused by name (`DataError`) rather than sent. This happens only to an
extreme strip under a budget at the edge of the range, where flooring to the factor leaves the engine's range or an
aspect ratio above 200.

### Video

`video` sets the frames shown per clip and how they are sent, `wire`; under `wire: video_url` exactly one
sampling rule is declared -- a uniform `num_frames` or the engine's own `fps`:

- **`wire: frames` (the default and the exact one).** The client samples `num_frames` frames from the clip's
  pre-extracted frames (the `frames` reader) at `np.linspace(0, total - 1, num_frames)` truncated to integers.
  vLLM's default video loader applies the same rule to a decoded container. Each frame is prepared as an image
  under the image policy and sent as a standard `image_url` part, so the frames counted are the frames sent.
- **`wire: video_url` (opt-in, for models with a native video encoder).** The container is sent unchanged and
  the engine decodes and samples it with its own video loader. A stock engine samples its own default -- the
  checkpoint's processor fps on the Qwen3-VL backend, whose loader is fps-driven and ignores `num_frames`;
  the default vLLM loader's 32 frames elsewhere -- which would make the counted tokens and the recorded
  instrument describe frames nobody chose, so the policy refuses `video_url` unless
  `engine_video_pinning: true` declares the engine
  pinned -- to a uniform `num_frames` (`--media-io-kwargs '{"video": {"num_frames": N}}'` on vLLM) or to
  the engine's own rate, `fps` (`--media-io-kwargs '{"video": {"fps": N}}'`; see [judges](judges.md)).
  The two are different measurements, so exactly one is declared. The engine's Qwen3-VL video backend samples
  by fps and ignores `num_frames`; `qwen3_vl_video_frame_indices` ports its rule
  (`int(total_frames / original_fps * fps)`, clamped to its 30 fps ceiling and its 4..768 frame bounds), and
  the client counts each clip's frames from its recorded frame count and rate. On that family a pinned
  `num_frames` policy is refused at count time: the backend would ignore the pin, so the declared policy
  would name a layout the engine never renders (declare `fps`). The policy refuses a
  single-frame container (the declared instrument merges frames in time, which needs a temporal pair; a
  single frame is an image), including a clip whose fps sampling realises one frame. Run
  `engine_media_check` once against a prepared probe when a serving setup
  changes (below); a mismatch says the engine's media handling is not the one the counted tokens describe.
  The engine's video-token pruning (`--video-pruning-rate`, with `--video-pruning-method` `evs` or
  `vidcom2`) changes that layout: the retained tokens render in the first temporal group and the others
  carry none. Declare the same rate and method on the policy (`engine_video_pruning`,
  `engine_video_pruning_method`); the recipe loader refuses a serve flag the client has not declared (and a
  declaration the serve args do not carry), and the client counts the engine's own retention formula for the
  `qwen3_vl` family (a per-frame family's flat pruned run is not ported, so the pair is refused).

### What a container costs

A container is counted as a stock engine's video accounting counts it, never at the client's declared image
budget -- the container is sent unchanged, so the client's pixel budget never reaches the engine. It is
patchified in time: 8 frames merge into `ceil(8 / 2) = 4` per-frame token runs, not 8. The frames' geometry is
the family's own video budget (`PROCESSORS`):

- The Qwen2-VL families size each frame independently by the checkpoint's per-frame budget -- stock vLLM's
  accounting, which passes the checkpoint's image-processor size for videos -- under one vision block for the
  whole clip: 8 frames of 720x1280 cost 4,786 tokens. `engine_media_check` compares the engine's actual count
  at run time.
- `qwen3_vl` budgets the whole clip together (4,096 to 25,165,824 px), which shrinks the per-frame resolution
  as the frame count grows -- 8 frames of 720x1280 cost 3,520 patch tokens, 128 frames 11,520 -- and renders
  one timestamp line (`<0.0 seconds>`) and one vision block per temporal group inside the chat template's own
  vision pair. The timestamp line's token count is tokenizer-dependent; the client counts it exactly with its
  loaded tokenizer when the policy samples at `fps`, and falls back to the family's declared 10-token bound
  (which covers every timestamp a clip of up to 27.8 hours can carry) when it has none.

An odd frame count is padded by repeating its last frame, as the processors do.

A clip with fewer frames than a declared `num_frames` is refused, and so is a container longer than an
optional `max_duration_s` (the limit is a container limit: a frame set carries no duration to check); under
the engine's `fps` rule the realised count is per clip, and a clip whose recorded frame
count or rate is missing is refused (ingest containers with `hash_media=True`).

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

## Identity

`rcp_ndcg.Preprocessing` is the effective policy of one judging pass: the text policy, the chunk geometry, the
image policy (with the judge's processor family) and the video policy. It is part of the judging identity and of the
judgement family key, and the judge's tokenizer is recorded beside it: its name and hash in the preprocessing
identity, its hash in the family key. Two passes that showed the judge different text or pixels therefore never share
cached judgements or pool observations.

What a served role reads is decided by its own policy over the same machinery:
[text budgets for served roles](text-budgets.md).
