# Text budgets for served roles

What the judge reads is a judging policy ([preprocessing and chunking](preprocessing.md)); this page is the other
half: what a served role -- an embedder, a multi-vector pooler, a reranker -- reads, and the one mechanism that
fits every request into the model's input.

The mechanism has one home: the declared budget and the fit live in `rcp_ndcg.data.text_budget` (the judge's side
of the machinery is `rcp_ndcg.data.text_policy`, the cut record and census `rcp_ndcg.data.census`, the census
files' record I/O `rcp_ndcg.storage.census`); `rcp_ndcg.data.preprocess` re-exports them, and every snippet below
imports from it.

## The anchor rule

A served model (an embedder, a reranker) reads its requests through a fixed frame: a system turn, instruction
prefixes, role markers, and the end-of-turn token its score is pooled from. That frame is the **template anchor**
rule this
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

## The template as data

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
`anchor` (`last`, `first`, `last_content`, `mean` or `marker`, with `anchor_markers` naming the specials) --
the position the model reads its output from. `last_content` is for a model that pools the last real token of raw
text (jina-embeddings-v5): no fixed tail exists and the shape may end on a content span -- which `last` refuses --
while the fixed segments (a head marker) are still reserved by the budget, and the prefix a cut keeps always
carries the last content token. It also declares `add_special_tokens` per shape: what the engine does to the
rendered string for that route (vLLM's pooling and scoring routes append the tokenizer's post-processor tokens;
the chat-embed form does not). The budget reserves those tokens too: they are part of the measured overhead.

The template can also declare, per shape, a content **normalisation** (`normalize`: `"strip"`, `"lowercase"`,
in the declared order): `fit` applies it to the shape's content spans before measuring, so the reference and the
engine see the same text -- the topk wrapper strips the query text and the whole document, Cobble checkpoints
lowercase their input. The census rows keep the input as given on their original side: normalisation is declared
policy, not a cut. A declared normalisation beside a media content with several text parts is refused
(`ConfigError`, before anything is measured or recorded): `fit` normalises the joined text, so its cut span is
not a prefix of the raw parts and the later parts would be hoisted into the first slot (the interleaved-part
bug); a single-text-part media content and a text-only content are unaffected.

## The budget and the fit

The budget names the tokenizer, the `max_tokens` (the model's whole input sequence, in that tokenizer's tokens),
optionally `query_max_tokens`, `on_overflow` (`cut` by default, `chunk` or `fail` opt-in), the chunk geometry, and
`aggregation: max`. `query_max_tokens` has a meaning per role: on a reranker's `pair` budget it is the query's
share (when a pair overflows, the query is cut to it first and the document gets the rest -- an input under budget
goes out unchanged, so within `fit` the share binds on overflow); on an embedder's or a late-interaction encoder's
`query` shape it is that shape's WHOLE budget -- per-shape budgets, for the asymmetric and late-interaction
embedders that cap queries and documents differently (topk-embed-v1-small reads 1024 tokens of query, 8192 of
document) -- while `max_tokens` keeps capping the `document` shape. A query budget above `max_tokens` is refused
(on a reranker, one at or over it is refused: the document would keep nothing). A reranker whose checkpoint cuts
each document itself declares `document_max_tokens` beside the pair budget (jina-reranker-v3 reads 2048
document tokens and 512 query tokens): every document over it is cut to it, also in a pair the budget would take
whole, the frame re-attached and the cut recorded (`cause: document_share`), before the pair is fitted; it must
be below `max_tokens` and is refused beside `on_overflow: chunk`. `fit` then, per input:

1. measures the fixed overhead once per (template, shape): the template rendered with every content span empty,
   counted as the engine reads it (the shape's `add_special_tokens` flag included);
2. cuts only the content spans, at token boundaries, verified against the *assembled* render so a byte-level
   merge across a span join cannot push the request over the budget, and re-attaches the template. A content
   with several text parts keeps every part in its own place around its media (an interleaved
   `[text, image, text]` sends its second text after the image, never hoisted into the first slot): the cut
   of the joined text is distributed over the parts where they stand, and a part the cut shortened gets its
   own census row (below);
3. on `chunk`, splits the document into verbatim chunks and renders **every chunk with the full template** --
   engine-side chunking of a framed render keeps the frame only on the first and last chunk, so chunking is
   always client-side here;
4. records every cut in the census under the `text_budget` mechanism, each row naming the shape's own budget
   (`budget_tokens`: the query rows a declared `query_max_tokens`, the document rows `max_tokens`), why the
   input changed (`cause`: `budget_cut`, `query_share` or `document_share`) and the uncut request's whole size as
   the engine would read it (`original_request_tokens`: the frame, its specials, the content and the reserved
   media). A content with several text parts records one row per part the cut shortened -- each row's original
   and kept counts are the part's own, the request's totals repeat on every row, and a part the cut kept whole
   records nothing (a single-part content, and an input whose no part changed, keep their one row).
   `original_tokens` counts the content alone, so a request the frame pushed over the budget has a
   content count under it: whether an input was changed is read from the row, never from that count.

Every role client also keeps, per input row it changed, one `ProcessingRecord` (`client.processing`) -- the one
declared preparation pipeline's output (`STAGES`: normalise, empty, media, render, budget, lower; one order for
every role) -- read from those census rows and from the media fit's and the empty-document policy's decisions:
the row's id in its call (its position, or `<query>` for a reranker's shared query; a chunk's row is grouped
under its input through the fit's own `chunk_mapping`, so an input id that itself contains `#` is never
mis-split), each change by its mechanism
(`empty_doc`, `media_resize`, `media_drop`, `document_share`, `query_share`, `budget_cut`, and the postprocess
`skip_unapplied` when a pooled document's declared skip list could not be applied to a media item), and the uncut
and kept request totals. A row without a record was sent as given -- the equivalence harness gates exactly those.

An input under budget comes back byte-identical to the uncut render -- within `fit`, which settles a pair's
query span per pair. The rerank wire carries one query per request, so the rerank client settles the shared
query span once per call (`fit`'s own rules, on a probe pair): whenever the query exceeds its declared share it
ships at it, recorded once in the census under the doc id `<query>` (`cause: query_share`, also when every pair
would fit whole), and every document span is verified
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

## Chunked documents pool by maximum

A document chunked under `on_overflow: chunk` is scored once per chunk and its document score is **the maximum
over its chunks** -- a document is as relevant as its best chunk, the same rule as
`max_pool_scores_by_document` (the aggregation the judging pass applies to its own chunks). It is declared on
the config (`aggregation: max`, the only value for now), returned on every `FitResult` of a chunked call, and
named on every census row of a chunked document, so a reader of `preprocessing.jsonl` never has to guess how the
numbers were pooled. `FitResult.chunk_mapping` carries each chunk id (`<id>#<k>`) back to its input, ready for
that pooling function.

## Explicit budgets, and hosted vendor profiles

A self-hosted role config (`api: openai_embeddings`, `vllm_pooling` or `rerank`) must declare `tokenizer` and
`max_tokens` -- the package cuts the content itself, so it must know both; without them the config is refused
with a hint naming the two fields. A hosted vendor profile (cohere, voyage, gemini) without a tokenizer sends
its content uncut: the vendor's documented limit is declared as `max_tokens`, recorded as the effective budget
(`budget_source: vendor` in the census -- one row per corpus and budget per census -- and one warning per
corpus per process), and nothing is measured or cut client-side. With a
tokenizer, a vendor profile follows the same rule as self-hosted.

The role configs also declare `template` (the `TemplateSpec` above), `empty_doc` (`send`, `omit_zero` --
never sent and scored `0.0` -- `omit_zero_blank` -- the paper's blank rule: whitespace-only text is empty
too -- or `send_text` with its `empty_doc_text` placeholder; every role client
consumes it, for an empty text document and for one whose every media item the budget dropped, deciding on
the content before the side's prompt and the template frame it -- the placeholder is then prompted and framed
like any content),
`request_shape` (`text`, `messages` or `token_ids`; the served embedding and pooling wires -- the
`openai_embeddings` and `vllm_pooling` adapters -- implement all three, the messages route being the
chat-style embeddings input and `token_ids` the ids the fit tokenised, while the hosted embed profiles speak
text only, and a rerank config that declares anything but `text` is refused -- the rerank wires send
rendered text today),
and the
`instruction` field, which places the TASK instruction (`Dataset.task_instruction`: one per task, subset or
domain). A reranker declares `fold`, `field` or `none` (`system` is refused at the config -- no shipped rerank
wire has a system-message slot, and a mode the wire cannot carry would silently drop the instruction); an
embedder or pooler declares `fold` or `none`. `fold` is the generic default,
`Task: <instruction>\nQuery: <text>` on the query side, and a template with an `instruction` span places it
instead -- the template's own placement wins, never both (on a rerank wire the span is rendered by the ENGINE
from the request's `instruction` field, so a wire without that field -- a hosted profile -- refuses the
combination at construction, and `instruction: none` beside a span is refused too: the span would render
empty; a `request_shape: messages` recipe with a span is refused for the same reason -- the engine's chat
template frames the content and cannot render the span). `instruction: field` with a template that renders no
`instruction` span still sends the instruction (the engine's own chat template places it), so the client
reserves its tokens in the fixed overhead before cutting anything -- otherwise the measured render would be
smaller than the prompt the engine reads. For an embedder or pooler `None` (the default) means UNDECLARED: a
request that
carries a task instruction is refused, naming `fold`/`none`, so a recipe that declares nothing never has its text
changed by a dataset it never met; a dataset without a task instruction needs no declaration. The PER-QUERY
instruction (`Query.instruction`, mteb's
InstructionRetrieval data) is the data's own: it is appended to the query text exactly as mteb's dataloader
appends it (`query + " " + instruction`) and is never folded as a task instruction. The reranker also declares
`empty_query` (`refuse` by default -- an empty query is refused with a typed error naming the query id,
before any task instruction is folded around it, so a frame around nothing is still an empty query;
`send` keeps the empty string), and every role config
declares `media_sides`, which names the sides that may carry media (both by default; media on a side it
does not name is refused with the error naming the field). `title` (`None`/`join` or `separate`) says how a
document's title reaches the model: MTEB's join `(title + " " + body).strip()` (the body alone without a
title), or the title as its own leading part ([data](../data.md)); the sparse (BM25) path is the one exception --
it follows mteb's own BM25, `title + "\n" + body` with no task instruction ([retrieval](retrieval.md)).

Media are never cut. Every served request goes through one preparation call
(`rcp_ndcg.data.prepare.prepare_request`) -- the same path the judge's images take -- which sizes every image
and video frame exactly as the declared `image_processor` would under the role's `image_policy`, counts the
request's media tokens exactly, and reserves them whole out of `max_tokens` before the content is cut. When
media alone fill the budget the declared `on_overflow` decides: `cut` (the default) shrinks to the policy's
minimum, then drops whole items most expensive first -- every drop recorded in the media census with
`dropped=True` --, `fail` refuses the request, and `chunk` is refused (media are not chunkable: a vision
block is atomic, the engine sees a whole item or none of it). A document whose every media item was dropped
is an empty document, and follows `empty_doc`. A role with an `image_processor` also declares
`max_images`/`max_videos` (the per-request gates, refused before sending), and the pool and rerank roles'
startup probe runs the engine media check: one prepared probe image beside its no-media baseline, the DELTA
of the engine's two prompt-token reports (the template and the text cancel) compared with the counted media
tokens -- a mismatch is refused, a reply without usage is recorded `not_checked`, never silent.

## Retrieval roles: one preparation path, and what to send when media do not fit

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
send one prepared image and the same request without its media, count the media block exactly
(`content_media_tokens` of the prepared reference), and compare the engine's media DELTA (its
`usage.prompt_tokens` with the image minus its report without it) against it. A returned mismatch
(`EngineMediaMismatch`) says the served engine's media handling is not what the counted tokens describe -- a
reconfigured engine or a mis-declared `image_processor` -- and every later count is suspect: record or raise
it instead of judging around it. The runtime call site is wired: a pool or rerank client with an
`image_processor` runs `check_engine_media()` from its `probe()` -- the startup probe sends the prepared image
and its baseline and refuses on a delta mismatch; a reply without usage is recorded `not_checked` (never
silent).

## The tokenizer's digest

The tokenizer's name is runtime and its digest is content: `Endpoint.identity_extra()` returns the SHA-256 of the
named `tokenizer.json` under the one key `{"tokenizer_sha256": ...}`, from the one helper in
`rcp_ndcg.data.tokenizer`. Every role config with a `tokenizer` -- the judge's, the embedding, pooling and rerank
configs -- carries the digest under this one key, so the `retrieve`/`rerank` step identities and the index identity
key on it, and two passes that counted with different tokenizers never share an index or a cache.

