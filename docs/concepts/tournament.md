# Stage A: the tournament

The tournament orders each query's candidate pool finely. An LLM judge reranks many windows of the pool, and a
Bradley-Terry model turns its answers into one score per document. These scores order the documents within a
query, but each query has a scale of its own. [The calibration](calibration.md) later maps them onto one scale that
all queries share.

## One window

A window holds $w = 10$ documents of the query's pool. The judge assigns every document of the window a
relevance score $s_i \in [-5, +5]$, which the prompt describes as a logit scale. Only the pairwise soft
preferences

$$
p_{ii'} = \sigma(s_i - s_{i'})
$$

are kept, for all $w(w-1)/2$ pairs of the window. The $w$ scores fix only $w - 1$ independent differences,
so each pair enters the fit with weight $2/w$. A window then carries the evidence of $w - 1$ comparisons, and
every document of the window is still compared with every other.

An answer is accepted only when its ranking is a permutation of the window and every document has one finite score.
An answer that fails this check is asked again, up to three attempts in total. After that the window is recorded as
an invalid judgement, with the reason and its category, which the fit skips and the calibration's `coverage.json`
counts. Nothing is completed or filled in.

## Parsing an answer

The judge's JSON object is read from the answer with a tolerant decoder, which the rubric shares. The decoder removes
`<think>...</think>` blocks and an orphaned `</think>` that sits before the object (one that trails it is merely what surrounds it), removes one code fence around the answer, and decodes
the first complete JSON object from the first `{`, ignoring any text after it (prose, a stray `}`, a second object).
When a string of that object holds an invalid escape, such as LaTeX (`\pi`, `\{`) in the reasoning, it doubles that
backslash, so the string keeps the backslash as text, and decodes again (at most 1000 times per answer). This changes
the content of that string only. If the object still does not decode, it retries once with a stray quote after a number
removed (`"doc_6": -4.5"}`). It does nothing else: it never reads numbers out of prose and never completes a missing
document, so every ranking and score of a judgement is a value of the judge's own JSON object.

An invalid judgement records one category: `truncated` (the answer hit the token limit), `no_json` (no JSON object),
`invalid_json` (the object does not decode), `schema` (a missing key, a repeated key, a wrong type, an unknown or
repeated document), `incomplete` (a document is missing) or `refused` (the endpoint rejected the request, so there is no
answer).

A judge configured with `decoding: json_schema` has each request carry the stage's answer schema as the
OpenAI-standard `response_format` (`json_schema`), which vLLM, SGLang and the OpenAI API enforce. The judgement
family records this as `decoding: json_schema`, and as `decoding: free` otherwise, so the two never pool. The parse
version is part of the family too: `rcp-ndcg judge reparse` reads a store's stored answers again with the current
parser, into a new store, without calling the judge ([the judgement store](judges.md#the-judgement-store)).

## The Bradley-Terry fit

The Bradley-Terry model gives the probability that document $i$ is preferred to $i'$ as
$\sigma(\theta_i - \theta_{i'})$. The fit minimises the weighted mean cross-entropy between the soft preferences
$p_{ii'}$ and the predicted ones, plus a ridge penalty of $10^{-4}$ on the scores, with L-BFGS (at most 250
iterations). The scores $\hat\theta^{\mathrm{BT}}_i$ have mean zero within the query. They are identified only up
to this additive constant, and their spread depends on how decisive the judge was on that query.

## The window schedule

The schedule has three *schedule phases* (`random`, `stratified`, `adaptive`; unrelated to the job phases of a
run on a cluster, [runs](runs.md#phases)). `rcp_ndcg.judging.TournamentSchedule` holds it, specified in placements per document: a
phase with $p$ placements asks $\operatorname{round}(p\,n / w)$ windows of $w$ documents of a pool of $n$ (ties to
even), so the calls scale with the pool. The window is the effective one, $w$ = `min(window, n)`, so the
placements per document hold for any pool: a pool of 4 gets $\operatorname{round}(3.53) = 4$ random windows of all
4 documents, each in its own order. Its defaults are the paper's, and give exactly its counts at a pool of 150:

| Phase | Placements per document | Windows at a pool of 150 | Calls at a pool of 150 |
|---|---|---|---|
| Random (`random_placements`) | $53 \cdot 10 / 150 \approx 3.53$ | 53 balanced random windows of 10: every document is drawn about equally often | 106 (each also sent reversed) |
| Stratified (`stratified_placements`) | 1.8 | 27 windows of documents that share a tier of a preliminary Bradley-Terry fit | 54 (each also sent reversed) |
| Adaptive (`adaptive_placements`, over all batches) | $7 \cdot 8 \cdot 10 / 150 \approx 3.73$ | 7 batches of 8 windows of 10 consecutive documents over the top 150 of the current order, with a Bradley-Terry refit before each batch | 56 |

A query of 150 candidates thus costs 216 calls. The adaptive windows per batch are
$\operatorname{round}(p\,n / (w_a \cdot \text{batches}))$ with the effective adaptive window
$w_a$ = `min(adaptive_window, n)`. A pool no larger than the adaptive window gets one adaptive window in total
(one batch, whatever `adaptive_batches` says): every adaptive window of such a pool holds the whole pool, so a
further batch asks only comparisons the first window's answers already cover. `estimate` counts exactly these
windows. A re-judged subset of a pool (`docs=`)
gets the windows its own size gives; a new document is judged against opponents chosen from the calibration with
`windows=` ([primitives](primitives.md)). Reversed copies (`mirror=True`) cancel the judge's position bias.
Stratified windows keep documents of similar quality together, which keeps a dominant document from lowering the
scores of mid-ranked ones.

The adaptive phase spends its calls where the order is least certain. Let
$p_r = \sigma\bigl(\hat\theta_{(r)} - \hat\theta_{(r+1)}\bigr)$ be the predicted preference between the documents
at ranks $r$ and $r + 1$ of the current order. The boundary between them has the value

$$
v_r = p_r\,(1 - p_r) \cdot \frac{1}{\log_2(r+1)} \cdot \frac{1}{1 + n_r},
$$

where $n_r$ counts the windows that have already held both documents. The first factor peaks when the two
documents are a coin flip, the second is the nDCG rank discount, and the third favours pairs the judge has rarely
compared. A batch greedily picks its windows (8 at a pool of 150) with the largest sum of boundary values. After each pick, the values
of the boundaries it covers are multiplied by `overlap_discount` (0.3). Adaptive windows are shown once, in the
current order.

Every query draws its windows from its own random stream, seeded by `seed` (42 by default). Page-image and video
corpora use smaller windows with the same placements: `TournamentSchedule.for_modality("image")` and
`for_modality("video")` give windows of 5, so at a pool of 150 106 random and 54 stratified windows, and 7 adaptive
batches of 16 windows.

## Output

Each window becomes one `Judgement` record in `<store>/tournament.jsonl`: its placements with the judge's score for
each, the judge's ranking, and the raw answer. The store is append-only. The per-query Bradley-Terry scores are
refitted from these records when the calibration runs, and when a document is inserted.

The refit reads every stored window by one rule. A valid window scores every document it showed and contributes the
soft pairs of its scores, each at the weight $2/w$ of a window of $w$ documents. A window without scores for all its
documents is invalid and contributes nothing. See [the judgement store](judges.md#the-judgement-store) for resuming and identity.

## Running it

<!-- snippet: skip (needs a judge endpoint and a loaded dataset) -->
```python
from rcp_ndcg.judging import JudgeConfig, TournamentSchedule, estimate, judge

judge_cfg = JudgeConfig.load("gpt_oss_120b")
print(estimate(dataset, None, judge_cfg, stages=["tournament"]))  # calls, tokens, wall time
judge(dataset, None, judge_cfg, stage="tournament", out="judgements/", schedule=TournamentSchedule())
```

On the command line, `rcp-ndcg judge tournament` runs this stage, with `--estimate` to count its calls and
tokens first. A [run](runs.md#runs-and-job-runners) runs it as its `tournament` step.
