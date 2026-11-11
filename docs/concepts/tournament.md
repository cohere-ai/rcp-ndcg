# Stage A: the tournament

The tournament orders each query's candidate pool finely. An LLM judge reranks many windows of the pool. A
Bradley-Terry model turns its answers into one score per document. Bradley-Terry is a model that turns
pairwise preferences into one score per item. These scores order the documents within a query. Each query,
however, has a scale of its own. [The calibration](calibration.md) later maps them onto one scale that all
queries share.

## One window

A window holds $w = 10$ documents of the query's pool. The judge assigns every document of the window a
relevance score $s_i \in [-5, +5]$. The prompt describes this score as a logit scale. Only the pairwise soft
preferences are kept:

$$
p_{ii'} = \sigma(s_i - s_{i'})
$$

They are kept for all $w(w-1)/2$ pairs of the window. The $w$ scores fix only $w - 1$ independent differences.
Each pair therefore enters the fit with weight $2/w$. A window then carries the evidence of $w - 1$
comparisons. Every document of the window is still compared with every other.

An answer is accepted only under two conditions. Its ranking must be a permutation of the window, and every
document must have one finite score. An integer too large for a float is refused with the `schema` category.
It never becomes an overflow out of the parser. An answer that fails this check is asked again, up to three
attempts in total. After that the window is recorded as an invalid judgement, with the reason and its
category. The fit skips it, and the calibration's `coverage.json` counts it. Nothing is completed or filled
in.

## Parsing an answer

A tolerant decoder reads the judge's JSON object from the answer. The rubric shares this decoder. The answer
is the text after the last `</think>`. Everything before it is reasoning: ` thinking...` blocks, or reasoning
whose opening tag the engine dropped. An object drafted there is never read. An object that only precedes a
`</think>` is refused. An object that opens inside a ` thinking` that never closes is a draft too. The judge
was cut while thinking. Such an object is refused, as `truncated` when the answer hit the token limit. The
decoder then removes one code fence around the answer. It decodes the first complete JSON object from the
first `{`. It ignores any text after that object: prose, a stray `}`, or a second object. The text after the
last `</think>` is never rewritten. No tag is removed from inside it. An answer equal to the prompt's own
worked example is refused too. That example is the fenced JSON block the prompt shows. A model that echoes the
example must not become an observation. The example can happen to fit a window.

A string of that object can hold an invalid escape, such as LaTeX (`\pi`, `\{`) in the reasoning. The decoder
then doubles that backslash, so the string keeps the backslash as text. It decodes again, at most 1000 times
per answer. This changes the content of that string only. The object can still fail to decode. The decoder
then retries once, with a stray quote after a number removed (`"doc_6": -4.5"}`). It does nothing else. It
never reads numbers out of prose, and it never completes a missing document. Every ranking and score of a
judgement is therefore a value of the judge's own JSON object.

An invalid judgement records one category. `truncated` means the answer hit the token limit. `no_json` means
no JSON object. `invalid_json` means the object does not decode. `schema` means a missing key, a repeated key,
a wrong type, an unknown or repeated document. `incomplete` means a document is missing. `refused` means the
endpoint rejected the request, so there is no answer. `superseded` means a resumed pass refitted under an
answer that the first fit was missing. This window of the first fit is then retired.

A judge configured with `decoding: json_schema` has each request carry the stage's answer schema. The request
carries it as the OpenAI-standard `response_format` (`json_schema`), which vLLM and the OpenAI API enforce.
The judgement family records this as `decoding: json_schema`. It records `decoding: free` otherwise, so the
two never pool. The parse version is part of the family too. `rcp judge reparse` reads a store's stored
answers again with the current parser. It writes them into a new store, without calling the judge ([the judgement store](judges.md#the-judgement-store)).

## The Bradley-Terry fit

The Bradley-Terry model gives the probability that document $i$ is preferred to $i'$ as $\sigma(\theta_i -
\theta_{i'})$. The fit minimises the weighted mean cross-entropy between the soft preferences $p_{ii'}$ and
the predicted ones. It adds a ridge penalty of $10^{-4}$ on the scores, and it uses L-BFGS (at most 250
iterations). The scores $\hat\theta^{\mathrm{BT}}_i$ have mean zero within the query. They are identified only
up to this additive constant. Their spread depends on how decisive the judge was on that query. A document
that no readable window compared gets no score. The fit would only return its prior, so the calibration
reports it as uncalibrated instead ([calibration](calibration.md#documents-without-a-tournament-ability)).

## The window schedule

The schedule has three *schedule phases*: `random`, `stratified` and `adaptive`. They are not the job phases
of a run on a cluster ([runs](runs.md#phases)). `rcp.judging.TournamentSchedule` holds it. The schedule
is specified in placements per document. A phase with $p$ placements asks $\operatorname{round}(p\,n / w)$
windows of $w$ documents of a pool of $n$. Ties round to even. The calls therefore scale with the pool. The
window is the effective one, $w$ = `min(window, n)`. The placements per document therefore hold for any pool.
A pool of 4 gets $\operatorname{round}(3.53) = 4$ random windows of all 4 documents, each in its own order.
Its defaults are the paper's, and they give exactly its counts at a pool of 150:

| Phase | Placements per document | Windows at a pool of 150 | Calls at a pool of 150 |
|---|---|---|---|
| Random (`random_placements`) | $53 \cdot 10 / 150 \approx 3.53$ | 53 balanced random windows of 10: every document is drawn about equally often | 106 (each also sent reversed) |
| Stratified (`stratified_placements`) | 1.8 | 27 windows of documents that share a tier of a preliminary Bradley-Terry fit | 54 (each also sent reversed) |
| Adaptive (`adaptive_placements`, over all batches) | $7 \cdot 8 \cdot 10 / 150 \approx 3.73$ | 7 batches of 8 windows of 10 consecutive documents over the top 150 of the current order, with a Bradley-Terry refit before each batch | 56 |

A query of 150 candidates thus costs 216 calls. The adaptive windows per batch are $\operatorname{round}(p\,n
/ (w_a \cdot \text{batches}))$, with the effective adaptive window $w_a$ = `min(adaptive_window, n)`. A pool
no larger than the adaptive window gets one adaptive window in total. That is one batch, whatever
`adaptive_batches` says. Every adaptive window of such a pool holds the whole pool. A further batch would ask
only comparisons that the first window's answers already cover. `estimate` counts exactly these windows. A
re-judged subset of a pool (`docs=`) gets the windows its own size gives. A new document is judged against
opponents chosen from the calibration with `windows=` ([primitives](primitives.md)). Reversed copies
(`mirror=True`) cancel the judge's position bias. Stratified windows keep documents of similar quality
together. That keeps a dominant document from lowering the scores of mid-ranked ones.

The adaptive phase spends its calls where the order is least certain. Let $p_r = \sigma\bigl(\hat\theta_{(r)}
- \hat\theta_{(r+1)}\bigr)$. This value is the predicted preference between the documents at ranks $r$ and $r
+ 1$ of the current order. The boundary between them has the value

$$
v_r = p_r\,(1 - p_r) \cdot \frac{1}{\log_2(r+1)} \cdot \frac{1}{1 + n_r},
$$

where $n_r$ counts the windows that have already held both documents. The first factor peaks when the two
documents are equally likely to be preferred. The second factor is the nDCG rank discount. The third factor
favours pairs that the judge has rarely compared. A batch greedily picks its windows with the largest sum of
boundary values (8 at a pool of 150). After each pick, the values of the boundaries it covers are multiplied
by `overlap_discount` (0.3). Adaptive windows are shown once, in the current order.

Every query draws its windows from its own random stream, seeded by `seed` (42 by default). Page-image and
video corpora use smaller windows with the same placements. `TournamentSchedule.for_modality("image")` and
`for_modality("video")` give windows of 5. At a pool of 150 that gives 106 random and 54 stratified windows,
and 7 adaptive batches of 16 windows. A schedule can name only some fields, such as a `seed` or a placement.
It then keeps those windows for the corpus's modality, and the fields it leaves unset take `for_modality` 's
values. Naming the seed of a page-image corpus therefore does not silently turn its windows into the text
defaults.

## Output

Each window becomes one `Judgement` record in `<store>/tournament.jsonl`. The record holds its placements
with the judge's score for each, the judge's ranking, and the raw answer. The store is append-only. The
per-query Bradley-Terry scores are refitted from these records when the calibration runs, and when a document
is inserted.

The refit reads every stored window by one rule. A valid window scores every document it showed. It
contributes the soft pairs of its scores, each at the weight $2/w$ of a window of $w$ documents. A window
without scores for all its documents is invalid, and it contributes nothing. See [the judgement store](judges.md#the-judgement-store) for resuming and identity.

## Running it

<!-- snippet: skip (needs a judge endpoint and a loaded dataset) -->
```python
from rcp.judging import JudgeConfig, TournamentSchedule, estimate, judge

judge_cfg = JudgeConfig.load("recipe:gpt-oss-120b")
print(estimate(dataset, None, judge_cfg, stages=["tournament"]))  # calls, tokens, wall time
judge(dataset, None, judge_cfg, stage="tournament", out="judgements/", schedule=TournamentSchedule())
```

On the command line, `rcp judge tournament` runs this stage. Add `--estimate` to count its calls and
tokens first. A [run](runs.md#runs-and-job-runners) runs it as its `tournament` step.
