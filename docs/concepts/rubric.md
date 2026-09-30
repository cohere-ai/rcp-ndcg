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

`rcp_ndcg.llm.RubricSchedule` holds the schedule, and its defaults are the paper's: 100 windows of 10 documents per
query, with no reversed copies.

1. **Balanced random windows.** The first 50 windows each take the documents shown least often so far, with random
   tie-breaks. All documents thus appear about equally often and meet many different partners.
2. **Stratified windows.** A Rasch model, $P(Y_{ic} = 1) = \sigma(\theta_i - \beta_c)$, gives each document a
   preliminary ability from the answers so far. The remaining 50 windows group documents with similar estimates.

The window count is raised to $\lceil n / w \rceil$ when a pool of $n$ documents needs more windows to show
every document once. Each query draws from its own random stream, seeded by `seed` (42 by default). Page-image
corpora use `RubricSchedule.for_modality("image")` (125 windows of 8, 62 random), and video corpora
`for_modality("video")` (200 windows of 5, 100 random).

A pool of 150 documents with 100 windows of 10 places each document in about 6.7 windows. Each placement is one
observation of the calibration, with five answers.

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

On the command line, `rcp-ndcg judge rubric` runs this stage, with `--estimate` and `--budget-usd` as for the
tournament. Re-judging only some documents of a query is the same call with `docs={query_id: [doc_id, ...]}`
([primitives](primitives.md)).
