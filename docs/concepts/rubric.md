# Stage B: the rubric

The rubric gives every document of a query a set of binary answers against an absolute standard. A binary
answer has two values, yes or no. The judge reads windows of documents. A window is one request that shows the
judge several documents together. The judge rates each document on its own, on five criteria. The criteria
mean the same on every query. Their answers are therefore what [the calibration](calibration.md) uses to put
all queries on one scale.

## The five criteria

Stage B has exactly five criteria, C1 to C5, defined in the shipped rubric prompt:

| ID | Criterion | Diagnostic question |
|---|---|---|
| C1 | Topical Relevance | Is its topic clearly related to the query's information need? |
| C2 | Information Utility | Does it contain specific, useful information to address the query? |
| C3 | Entity/Detail Match | Does it name and discuss the particular entity or detail in the query? |
| C4 | Direct Answer | Does it explicitly and directly answer the query's primary question? |
| C5 | Thorough Treatment | Does it treat the topic in depth rather than superficially? |

Document $d_i$ can appear in more than one window of query $q_j$. Each appearance yields five answers $Y_{ijc}
\in \{0, 1\}$, one per criterion. The criteria target general-purpose retrieval. Other applications may need
other criteria.

The prompt is part of the judgement family. A judgement family is one set of judgements that a calibration may
pool. A changed prompt is a different family, and so is a custom prompt passed as `RubricSchedule(prompt=...)`
. Its judgements never pool with those of the shipped rubric. `rcp_ndcg.judging.load_prompt` returns the
shipped prompts. They are `rubric`, `rubric_vision` and `rubric_video`, and the matching tournament prompts.

## The window schedule

`rcp_ndcg.judging.RubricSchedule` holds the schedule. The schedule is specified in placements per document. A
placement is one appearance of a document in a window. `placements_per_doc` is how often a document is shown
on average. Its default is $p = 100 \cdot 10 / 150 \approx 6.67$. A pool of $n$ documents gets $\max(\lceil n
/ w \rceil, \operatorname{round}(p\,n / w))$ windows of $w$ documents. Here $w$ = `min(window, n)`, and
`window` is 10 by default. The schedule has no reversed copies. Every document is therefore seen, and the
calls scale with the pool. The balanced random phase is the only phase that guarantees coverage. The schedule
must hold $n_{\text{random}} \cdot w \ge n_{\text{units}}$. A pass whose settings cannot show every document
is refused. The same holds for a pass that cannot show every chunk of a chunked document. The refusal names
that precondition, and it comes before a call. A planned pass with `windows=` is exempt, because the plan
decides what is shown. The refusal names the query and the counts. The counts are `n_random`, `window` and
`n_units`, plus how many units would be shown in no window. Its hint names the fix. Raise
`placements_per_doc` or `random_share`, or lower `window`. The shipped defaults are `placements_per_doc`
6.67, `window` 10 and `random_share` 0.5. With these defaults, the precondition holds up to about 3.33 chunks
per document. A pool of 150 with 3 chunks per document holds, and 4 is refused. A more finely chunked corpus
therefore needs one of the three moved. The placements per document hold for any pool. A pool of 4 gets 7
windows of all 4 documents, each in its own order. At the paper's pool of 150, this is exactly its 100 windows
of 10.

1. **Balanced random windows**. The first $\operatorname{round}(s \cdot \text{windows})$ windows use `random_share` $s =
   0.5$. Each takes the documents shown least often so far. At least one window is used. Random tie-breaks decide
   between equal counts. All documents thus appear about equally often, and they meet many different partners. At a pool
   of 150: 50.
2. **Stratified windows**. A Rasch model, $P(Y_{ic} = 1) = \sigma(\theta_i - \beta_c)$, gives each document a
   preliminary ability. That ability comes from the answers so far. The remaining windows (50 at a pool of 150) group
   documents with similar estimates. The random phase can produce no valid answer. Then there is nothing to stratify.
   The phase is dropped, and the pass warns about it. It is recorded as a `phase_dropped` row in the store's
   `preprocessing.jsonl`, so the shortfall is never silent.

Rounding is to the nearest integer, and ties round to even. Each query draws from its own random stream,
seeded by `seed` (42 by default). Page-image corpora use `RubricSchedule.for_modality("image")`: windows of
8, which gives 125 windows and 62 random at a pool of 150. Video corpora use `for_modality("video")`: windows
of 5, which gives 200 windows and 100 random. Both keep the placements per document. A schedule can name only
some fields, such as a `seed` or a placement. It then keeps those windows for the corpus's modality, and the
fields it leaves unset take `for_modality` 's values. A re-judged subset of a pool (`docs=`) gets the windows
its own size gives.

Each placement is one observation of the calibration, with five answers.

## Parsing

The tournament's tolerant decoder reads the answer's JSON object ([parsing an answer](tournament.md#parsing-an-answer)). An answer is accepted only under three conditions. Every document
of the window appears exactly once. The answer gives exactly C1 to C5. Each criterion has 0 or 1, and the
strings `"0"` and `"1"` are accepted too. Booleans and other numbers are rejected. So are missing or extra
criteria, and unknown or repeated documents. An answer equal to the prompt's own worked example is refused
too, as in the tournament. A rejected answer is asked again, up to three attempts in total. After that it is
recorded as an invalid judgement with its category. The fit skips invalid judgements, and `coverage.json`
counts them.

## Output

Each window becomes one `Judgement` record in `<store>/rubric.jsonl`. In that record, every placement carries
its document's answers `{"C1": 0 or 1, ..., "C5": 0 or 1}`. The same answers give the Count-nDCG gain
directly ([the metric](metric.md#count-ndcg)).

## A custom rubric

The shipped rubric is the paper's, and it has exactly the five criteria above. A research rubric with other
criteria is a custom prompt. Pass it as `RubricSchedule(prompt="my_rubric.txt")`, or as
`rubric: {prompt: my_rubric.txt}` in a run config. Its criteria are the labels `C1` to `CK` that the prompt
text names. Those labels must form a contiguous ladder. They become the judgement family's `criteria`. The
parser then requires exactly those `K` answers per document, and the calibration fits `K` criteria. A custom
prompt must contain the `{query_placeholder}` and `{passages_placeholder}` slots of the shipped prompts. A
tournament prompt may also use `{num_documents_placeholder}`. A prompt without these slots is refused
(`ConfigError`) when it loads. Its judgements never pool with the shipped rubric's.

Judgements made elsewhere enter `rcp_ndcg.calibration.calibrate` as a `JudgementSet`. Its rubric
`JudgementFamily` declares its criteria, named `C1..CK` (`rcp_ndcg_core.schemas.criterion_labels(K)`). Every
placement must answer exactly the declared criteria. A missing, unknown or renamed criterion raises
`DataError`, which names the record, the document and the criteria. No verdict is ever defaulted.

## Running it

<!-- snippet: skip (needs a judge endpoint and a loaded dataset) -->
```python
from rcp_ndcg.judging import JudgeConfig, RubricSchedule, judge

judge_cfg = JudgeConfig.load("recipe:gpt-oss-120b")
judge(dataset, None, judge_cfg, stage="rubric", out="judgements/", schedule=RubricSchedule(seed=7))
```

On the command line, `rcp-ndcg judge rubric` runs this stage, with `--estimate` as for the tournament. To
re-judge only some documents of a query, make the same call with `docs={query_id: [doc_id, ...]}`
([primitives](primitives.md)).
