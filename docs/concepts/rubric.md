# Stage B: the rubric

The rubric gives every document of a query a set of binary answers against an absolute standard. The judge reads
windows of documents, but rates each document on its own on five criteria. Because the criteria mean the same on
every query, their answers are what [the calibration](calibration.md) uses to put all queries on one scale.

## The five criteria

Stage B has exactly five criteria, C1 to C5, defined in the shipped rubric prompt:

| ID | Criterion | Diagnostic question |
|---|---|---|
| C1 | Topical Relevance | Is its topic clearly related to the query's information need? |
| C2 | Information Utility | Does it contain specific, useful information to address the query? |
| C3 | Entity/Detail Match | Does it name and discuss the particular entity or detail in the query? |
| C4 | Direct Answer | Does it explicitly and directly answer the query's primary question? |
| C5 | Thorough Treatment | Does it treat the topic in depth rather than superficially? |

Each appearance of document $d_i$ in a window of query $q_j$ yields five answers
$Y_{ijc} \in \{0, 1\}$, one per criterion. The criteria target general-purpose retrieval; other applications may
need other criteria.

The prompt is part of the judgement family. A changed prompt, or a custom one passed as `RubricSchedule(prompt=...)`,
is a different family, and its judgements never pool with those of the shipped rubric. `rcp_ndcg.llm.load_prompt`
returns the shipped prompts: `rubric`, `rubric_vision` and `rubric_video`, and the matching tournament prompts.

## The window schedule

`rcp_ndcg.llm.RubricSchedule` holds the schedule. It is specified in placements per document: `placements_per_doc`
($p = 100 \cdot 10 / 150 \approx 6.67$ by default) is how often a document is shown on average. A pool of $n$
documents gets $\max(\lceil n / w \rceil, \operatorname{round}(p\,n / w))$ windows of $w$ = `min(window, n)`
documents (`window`, 10 by default), with no reversed copies, so every document is seen and the calls scale with the
pool. The placements per document hold for any pool: a pool of 4 gets 7 windows of all 4 documents, each in its own
order. At the paper's pool of 150 this is exactly its 100 windows of 10.

1. **Balanced random windows.** The first $\operatorname{round}(s \cdot \text{windows})$ windows (`random_share`
   $s = 0.5$; at least one) each take the documents shown least often so far, with random tie-breaks. All documents
   thus appear about equally often and meet many different partners. At a pool of 150: 50.
2. **Stratified windows.** A Rasch model, $P(Y_{ic} = 1) = \sigma(\theta_i - \beta_c)$, gives each document a
   preliminary ability from the answers so far. The remaining windows (50 at a pool of 150) group documents with
   similar estimates.

Rounding is to the nearest integer, ties to even. Each query draws from its own random stream, seeded by `seed` (42
by default). Page-image corpora use `RubricSchedule.for_modality("image")` (windows of 8: 125 windows, 62 random, at
a pool of 150), and video corpora `for_modality("video")` (windows of 5: 200 windows, 100 random); both keep the
placements per document. A re-judged subset of a pool (`docs=`) gets the windows its own size gives.

Each placement is one observation of the calibration, with five answers.

## Parsing

The answer's JSON object is read by the tournament's tolerant decoder ([parsing an
answer](tournament.md#parsing-an-answer)). An answer is accepted only when every document of the window appears exactly
once and answers exactly C1 to C5, each with 0 or 1 (the strings `"0"` and `"1"` are accepted too). Booleans, other
numbers, missing or extra criteria, and unknown or repeated documents are rejected. A rejected answer is asked again, up
to three attempts in total, and is otherwise recorded as an invalid judgement with its category. The fit skips invalid
judgements and `coverage.json` counts them.

## Output

Each window becomes one `Judgement` record in `<store>/rubric.jsonl`, in which every placement carries its
document's answers `{"C1": 0 or 1, ..., "C5": 0 or 1}`. The same answers give the Count-nDCG gain directly
([the metric](metric.md#count-ndcg)).

## A custom rubric

The shipped rubric, the paper's, has exactly the five criteria above. A research rubric with other criteria is a
custom prompt, passed as `RubricSchedule(prompt="my_rubric.txt")` (or `rubric: {prompt: my_rubric.txt}` in a run
config). Its criteria are the labels `C1` to `CK` the prompt text names, which must form a contiguous ladder. They
become the judgement family's `criteria`: the parser then requires exactly those `K` answers per document, and the
calibration fits `K` criteria. A custom prompt must contain the `{query_placeholder}` and `{passages_placeholder}`
slots of the shipped prompts (a tournament prompt may also use `{num_documents_placeholder}`), or it is refused
(`ConfigError`) when it loads. Its judgements never pool with the shipped rubric's.

Judgements made elsewhere enter `rcp_ndcg.calibration.calibrate` as a `JudgementSet` whose rubric `Family` declares
its criteria, named `C1..CK` (`rcp_ndcg_core.schemas.criterion_labels(K)`). Every placement must answer exactly the
declared criteria: a missing, unknown or renamed criterion raises `DataError` naming the record, the document and
the criteria. No verdict is ever defaulted.

## Running it

<!-- snippet: skip (needs a judge endpoint and a loaded dataset) -->
```python
from rcp_ndcg.llm import JudgeConfig, RubricSchedule, judge

judge_cfg = JudgeConfig.load("gpt_oss_120b")
judge(dataset, None, judge_cfg, stage="rubric", out="judgements/", schedule=RubricSchedule(seed=7))
```

On the command line, `rcp-ndcg judge rubric` runs this stage, with `--estimate` as for the tournament. Re-judging only some documents of a query is the same call with `docs={query_id: [doc_id, ...]}`
([primitives](primitives.md)).
