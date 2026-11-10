# Text budgets for served roles

What the judge reads is a judging policy ([preprocessing and chunking](preprocessing.md)). This page is the
other half. It describes what a served role reads: an embedder, a multi-vector pooler or a reranker. It also
describes the one mechanism that fits every request into the model's input.

The mechanism has one home. The declared budget and the fit live in `rcp_ndcg.data.text_budget`. The judge's
side of the machinery is `rcp_ndcg.data.text_policy`. The cut record and census are in `rcp_ndcg.data.census`
, and the census files' record I/O is in `rcp_ndcg.storage.census`. `rcp_ndcg.data.preprocess` re-exports
them, and every snippet below imports from it.

## The anchor rule

A served model (an embedder or a reranker) reads its requests through a fixed frame. The frame holds a system
turn, instruction prefixes, role markers, and the end-of-turn token that its score is pooled from. That frame
is the **template anchor** rule, and this package implements it everywhere text is cut. A model reads its
output from fixed positions of its template. A cut must therefore apply to the content spans only. It must sit
inside a budget computed after reserving every fixed template token. The template must be re-attached after
the cut. A right cut of a whole rendered prompt drops tail anchors, and a left cut drops head anchors. No role
config therefore declares an engine-truncation field. `truncate_prompt_tokens`, `--allow-auto-truncate` and
their relatives are absent from the role configs by design. The fixed overhead is measured rather than
guessed.

One mechanism does this for every role: the declared :class:`~rcp_ndcg.data.TextBudget` and
:func:`~rcp_ndcg.data.preprocess.fit`. The role configs carry no engine-truncation field:
`truncate_prompt_tokens` and `truncation_side` are absent from them by design. The package cuts the content
itself. An engine-side truncation of a rendered prompt is exactly the cut that this rule forbids.

## The template as data

A `TemplateSpec` declares an ordered list of segments per request shape. The shapes are an embedder's `query`
and `document`, and a reranker's `pair`. A segment is a fixed piece of frame text or a content span:

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

The `pair` shape orders query and document per model. The reranker above reads the query first. A reranker
whose template puts the document first makes the query block an anchor, and the segments say so. A special
token is written by name (`{special:end_turn}`), and it is resolved at render time from the tokenizer's added
tokens. Specials are never typed literally, so a template outlives tokenizer rewrites. Without a template, a
single-sequence shape is the content itself. A `pair` is counted through the tokenizer's own pair encoding
(`count_pair`): the engine's pair post-processor and the two spans' join. It is never counted as the single
sequence `query + document`, which sees one separator fewer and merges the join. The template also declares
`anchor`: `last`, `first`, `last_content`, `mean` or `marker`, with `anchor_markers` naming the specials.
The anchor is the position that the model reads its output from. `last_content` is for a model that pools the
last real token of raw text, such as jina-embeddings-v5. No fixed tail exists there, and the shape may end on
a content span, which `last` refuses. The fixed segments, such as a head marker, are still reserved by the
budget. The prefix that a cut keeps always carries the last content token. The template also declares
`add_special_tokens` per shape. That is what the engine does to the rendered string for that route. vLLM's
pooling and scoring routes append the tokenizer's post-processor tokens, and the chat-embed form does not. The
budget reserves those tokens too, because they are part of the measured overhead.

The template can also declare a content **normalisation** per shape, in `normalize`: `"strip"` or
`"lowercase"`, in the declared order. `fit` applies it to the shape's content spans before measuring, so the
reference and the engine see the same text. The topk wrapper strips the query text and the whole document. A
checkpoint trained on lowercased text lowercases its input. The census rows keep the input as given on their
original side, because normalisation is declared policy, not a cut. A declared normalisation beside a media
content with several text parts is refused (`ConfigError`, before anything is measured or recorded). `fit`
normalises the joined text, so its cut span is not a prefix of the raw parts. The later parts would then be
hoisted into the first slot. That is the interleaved-part bug. A single-text-part media content and a
text-only content are unaffected.

## The budget and the fit

The budget names the tokenizer and the `max_tokens`. `max_tokens` is the model's whole input sequence, in
that tokenizer's tokens. The budget can also name `query_max_tokens`, `on_overflow` (`cut` by default,
`chunk` or `fail` opt-in), the chunk geometry, and `aggregation: max`. `query_max_tokens` has a meaning per
role. On a reranker's `pair` budget it is the query's share. When a pair overflows, the query is cut to it
first, and the document gets the rest. An input under budget goes out unchanged, so within `fit` the share
binds on overflow. On an embedder's or a late-interaction encoder's `query` shape it is that shape's WHOLE
budget. These are per-shape budgets, for the asymmetric and late-interaction embedders that cap queries and
documents differently. topk-embed-v1-small reads 1024 tokens of query and 8192 of document. `max_tokens` keeps
capping the `document` shape. A query budget above `max_tokens` is refused. On a reranker, a budget at or over
`max_tokens` is refused, because the document would keep nothing. A reranker whose checkpoint cuts each
document itself declares `document_max_tokens` beside the pair budget. jina-reranker-v3 reads 2048 document
tokens and 512 query tokens. Every document over it is cut to it, also in a pair that the budget would take
whole. The frame is re-attached and the cut recorded (`cause: document_share`), before the pair is fitted. The
value must be below `max_tokens`, and it is refused beside `on_overflow: chunk`. `fit` then, per input:

1. measures the fixed overhead once per (template, shape). The template is rendered with every content span empty,
   and it is counted as the engine reads it, with the shape's `add_special_tokens` flag included;
2. cuts only the content spans, at token boundaries. The cut is verified against the *assembled* render. A byte-level
   merge across a span join therefore cannot push the request over the budget. It re-attaches the template. A content
   with several text parts keeps every part in its own place around its media. An interleaved `[text, image, text]`
   sends its second text after the image, never hoisted into the first slot. The cut of the joined text is distributed
   over the parts where they stand. A part that the cut shortened gets its own census row (below);
3. on `chunk`, splits the document into verbatim chunks and renders **every chunk with the full template**.
   Engine-side chunking of a framed render keeps the frame only on the first and last chunk, so chunking is always
   client-side here;
4. records every cut in the census under the `text_budget` mechanism. Each row names the shape's own budget in
   `budget_tokens`: the query rows a declared `query_max_tokens`, and the document rows `max_tokens`. The row also names
   why the input changed, in `cause`: `budget_cut`, `query_share` or `document_share`. It names the uncut request's
   whole size as the engine would read it, in `original_request_tokens`. That size is the frame, its specials, the
   content and the reserved media. A content with several text parts records one row per part that the cut shortened.
   Each row's original and kept counts are the part's own, and the request's totals repeat on every row. A part that the
   cut kept whole records nothing. A single-part content, and an input whose no part changed, keep their one row.
   `original_tokens` counts the content alone. A request that the frame pushed over the budget therefore has a content
   count under it. Whether an input was changed is read from the row, never from that count.

Every role client also keeps one `ProcessingRecord` per input row that it changed (`client.processing`). That
record is the output of the one declared preparation pipeline. Its `STAGES` are normalise, empty, media,
render, budget and lower, in one order for every role. It is read from those census rows, and from the media
fit's decisions and the empty-document policy's decisions. It holds the row's id in its call: its position, or
`<query>` for a reranker's shared query. A chunk's row is grouped under its input through the fit's own
`chunk_mapping`. An input id that itself contains `#` is therefore never mis-split. The record holds each
change by its mechanism: `empty_doc`, `media_resize`, `media_drop`, `document_share`, `query_share`,
`budget_cut`. It also holds the postprocess `skip_unapplied`, when a pooled document's declared skip list
could not apply to a media item. No such row is written when the served plugin applies a rule engine-side
(`document_skip_engine_side`). It is also not written when a media allowlist (`media_keep_token_ids`) makes
the engine keep only its positions. The engine did apply the rule in that case. The record holds the uncut and
kept request totals. A row without a record was sent as given, and the equivalence harness gates exactly
those.

An input under budget comes back byte-identical to the uncut render, within `fit`. `fit` settles a pair's query span per
pair. The rerank wire carries one query per request. The rerank client therefore settles the shared query span once per
call. It uses `fit` 's own rules on a probe pair. Whenever the query exceeds its declared share, it ships at it. That is
recorded once in the census under the doc id `<query>`, with `cause: query_share`. It is recorded even when every pair
would fit whole. Every document span is verified against the span that ships. The function returns the rendered strings.
The wire routes take text, and tokenising once here to measure and cut is the same work either way. It returns the cut
content per span, for routes where the engine renders the template on. It returns the output ids and the chunk mapping.

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

## Chunked documents pool by maximum

A document chunked under `on_overflow: chunk` is scored once per chunk. Its document score is **the maximum
over its chunks**: a document is as relevant as its best chunk. That is the same rule as
`max_pool_scores_by_document`, the aggregation that the judging pass applies to its own chunks. The rule is
declared on the config (`aggregation: max`, the only value for now). It is returned on every `FitResult` of a
chunked call. It is named on every census row of a chunked document. A reader of `preprocessing.jsonl`
therefore never has to guess how the numbers were pooled. `FitResult.chunk_mapping` carries each chunk id
(`<id>#<k>`) back to its input, ready for that pooling function.

## Explicit budgets, and hosted vendor profiles

A self-hosted role config (`api: openai_embeddings`, `vllm_pooling` or `rerank`) must declare `tokenizer` and
`max_tokens`. The package cuts the content itself, so it must know both. Without them the config is refused,
with a hint naming the two fields. A hosted vendor profile (cohere, voyage, gemini) without a tokenizer sends
its content uncut. The vendor's documented limit is declared as `max_tokens`. It is recorded as the effective
budget, with `budget_source: vendor` in the census. That is one row per corpus and budget per census, and one
warning per corpus per process. Nothing is measured or cut client-side. With a tokenizer, a vendor profile
follows the same rule as self-hosted.

The role configs also declare `template`, the `TemplateSpec` above. They declare `empty_doc`, with four
values. `send` sends the document. `omit_zero` never sends it and scores it `0.0`. `omit_zero_blank` is the
paper's blank rule: whitespace-only text is empty too. `send_text` sends its `empty_doc_text` placeholder.
Every role client consumes the value. It applies to an empty text document, and to one whose every media item
the budget dropped. It decides on the content before the side's prompt and the template frame it. The
placeholder is then prompted and framed like any content. The configs declare `request_shape`: `text`,
`messages` or `token_ids`. The served embedding and pooling wires, the `openai_embeddings` and `vllm_pooling`
adapters, implement all three. The messages route is the chat-style embeddings input, and `token_ids` is the
ids that the fit tokenised. The hosted embed profiles speak text only. A rerank config that declares anything
but `text` is refused, because the rerank wires send rendered text today. The configs also declare the
`instruction` field, which places the TASK instruction (`Dataset.task_instruction`: one per task, subset or
domain).

A reranker declares `fold`, `field` or `none`. `system` is refused at the config. No shipped rerank wire has
a system-message slot. A mode that the wire cannot carry would silently drop the instruction. An embedder or
pooler declares `fold` or `none`. `fold` is the generic default, `Task: <instruction>\nQuery: <text>` on the
query side. A template with an `instruction` span places it instead, and the template's own placement wins,
never both. On a rerank wire, the ENGINE renders the span from the request's `instruction` field. A wire
without that field, such as a hosted profile, therefore refuses the combination at construction.
`instruction: none` beside a span is refused too, because the span would render empty. A
`request_shape: messages` recipe with a span is refused for the same reason. The engine's chat template frames
the content and cannot render the span. `instruction: field` with a template that renders no `instruction`
span still sends the instruction. The engine's own chat template places it. The client therefore reserves its
tokens in the fixed overhead before cutting anything. Otherwise the measured render would be smaller than the
prompt that the engine reads. For an embedder or pooler, `None` is the default, and it means UNDECLARED. A
request that carries a task instruction is then refused, naming `fold` /`none`. A recipe that declares nothing
therefore never has its text changed by a dataset that it never met. A dataset without a task instruction
needs no declaration.

The PER-QUERY instruction (`Query.instruction`, mteb's InstructionRetrieval data) is the data's own. It is
appended to the query text exactly as mteb's dataloader appends it (`query + " " + instruction`). It is never
folded as a task instruction. The reranker also declares `empty_query`, which is `refuse` by default. An
empty query is then refused with a typed error naming the query id. The refusal comes before any task
instruction is folded around it. A frame around nothing is still an empty query. `send` keeps the empty
string. Every role config declares `media_sides`, which names the sides that may carry media, both by
default. Media on a side that it does not name is refused with the error naming the field. `title`
(`None`/`join` or `separate`) says how a document's title reaches the model. The value is MTEB's join
`(title + " " + body).strip()`, or the body alone without a title. It can also be the title as its own
leading part ([data](../data.md)). The sparse (BM25) path is the one exception. It follows mteb's own BM25,
`title + "\n" + body` with no task instruction ([retrieval](retrieval.md)).

Media are never cut. Every served request goes through one preparation call
(`rcp_ndcg.data.prepare.prepare_request`), which is the same path that the judge's images take. It sizes every
image and video frame exactly as the declared `image_processor` would under the role's `image_policy`. It
counts the request's media tokens exactly, and it reserves them whole out of `max_tokens` before the content
is cut. The declared `on_overflow` decides when media alone fill the budget. `cut` is the default. It shrinks
to the policy's minimum, then drops whole items most expensive first. Every drop is recorded in the media
census with `dropped=True`. `fail` refuses the request. `chunk` is refused, because media are not chunkable.
A vision block is atomic, and the engine sees a whole item or none of it. A video part's prepared frames are
one unit. A clip is kept or dropped whole, never shortened below its declared `num_frames`. A document whose
every media item was dropped is an empty document, and it follows `empty_doc`. A role with an
`image_processor` also declares `max_images` / `max_videos`, the per-request gates, refused before sending.
The pool and rerank roles' startup probe runs the engine media check. It sends one prepared probe image beside
its no-media baseline. It compares the DELTA of the engine's two prompt-token reports with the counted media
tokens. The template and the text cancel in that delta. A mismatch is refused. A passing check is recorded
`engine_media_check:ok`, and a reply without usage is recorded `not_checked`, never silent.

## Retrieval roles: one preparation path, and what to send when media do not fit

The same policy objects size the media of the retrieval roles: encoders and rerankers. A role config declares
the judge's `image_processor`, `max_images`, `max_videos` and the optional image and video policies. Those
are the same `ImagePolicy` / `VideoPolicy` types, and they are content fields. Its clients prepare one
request's contents with `rcp_ndcg.data.prepare.prepare_request`. That call returns the prepared contents and
the request's exact media token counts. The role's text budget can therefore subtract them, never cut them:

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

Run `rcp_ndcg.data.resolution.engine_media_check` against a prepared probe whenever a serving setup changes.
Send one prepared image, and send the same request without its media. Count the media block exactly, with
`content_media_tokens` of the prepared reference. Compare the engine's media DELTA against that count: its
`usage.prompt_tokens` with the image, minus its report without it. A returned mismatch (`EngineMediaMismatch`)
says that the served engine's media handling is not what the counted tokens describe. The cause is a
reconfigured engine or a mis-declared `image_processor`. Every later count is then suspect, so record or
raise the mismatch instead of judging around it. The runtime call site is wired. A pool or rerank client with
an `image_processor` runs `check_engine_media()` from its `probe()`. The startup probe sends the prepared
image and its baseline, and it refuses on a delta mismatch. A passing check is recorded
`engine_media_check:ok`, and a reply without usage is recorded `not_checked`, never silent.

## The tokenizer's digest

The tokenizer's name is runtime, and its digest is content. `Endpoint.identity_extra()` returns the SHA-256 of
the named `tokenizer.json`, under the one key `{"tokenizer_sha256": ...}`. It comes from the one helper in
`rcp_ndcg.data.tokenizer`. Every role config with a `tokenizer` carries the digest under this one key. That
covers the judge's config, and the embedding, pooling and rerank configs. The `retrieve` /`rerank` step
identities and the index identity key on it. Two passes that counted with different tokenizers therefore never
share an index or a cache.
