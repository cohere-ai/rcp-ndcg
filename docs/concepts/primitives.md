# Primitives: re-annotation, insertion and several judges

A calibration is published. Its thetas are in `thetas.parquet`. Papers, leaderboards or dashboards already
quote them. Then something changes:

- the judging of some documents **did not work** (a failed window, a document that could not be loaded), so
  they have no theta;
- the candidate pool **grew**: a new document must be ranked against the old ones;
- a **second judge** answered the same rubric, and its answers should count.

Refitting everything would move every published score, including the scores of documents that nobody
re-judged. The functions below add the new evidence to the published calibration. They re-estimate nothing
that is already in it.

| Situation | Judge again | Then | Estimator |
|---|---|---|---|
| documents without a theta | Stage B on those documents | `score_documents` | fixed-item EAP (items frozen) |
| a new document in a tournament calibration | Stage A windows pairing it with existing documents (`select_opponents`) | `insert_documents` | conditional MAP (every existing theta frozen, the fit's ridge in the objective) |
| a second judge, same rubric | Stage B with the second judge | `calibrate(..., judges="pooled")` | one 2PL with a severity offset per judge |

All three live in `rcp.calibration`. `score_documents` and `insert_documents` are also top-level
functions of `rcp`. The command line takes the judge, dataset and judgement store of the calibration:

```bash
# documents without a theta: judge them on the rubric into the same store, then score them
rcp judge rubric --dataset <uri> --judge <judge> --docs q2:q2-d10 --docs q2:q2-d11 --out <store>
rcp calibration score --calibration <calibration> --judgements <store> --out <extended-calibration>

# a new document: let the calibration plan its windows, judge exactly those, then insert
rcp calibration insert --calibration <calibration> --judgements <store> --dry-run --query q1 --doc q1-new \
    --n 36 --out plan.json --json
rcp judge tournament --dataset <uri> --judge <judge> --plan plan.json --out <store> --estimate
rcp judge tournament --dataset <uri> --judge <judge> --plan plan.json --out <store>
rcp calibration insert --calibration <calibration> --judgements <store> --out <extended-calibration>

# a second judge: one pooled fit over both stores
rcp calibration fit --judgements <store> --judgements <second-store> --judges pooled --out <calibration>
```

In the second procedure, `calibration insert --dry-run` picks `--n` opponents across the query's ability
range. It splits them into windows of the store's `schedule.window`, read from its `identity.json`. Each
window holds the new document. `--out` writes the plan, and `data.plan.calls` says how many judge calls it
takes. Each window is asked twice when the schedule mirrors. `judge tournament --plan` asks exactly those
windows, with the store's schedule. It writes into the store, which must be the calibration's own tournament
store, judged by the same judge. Anything else is exit 11. `calibration insert` then reads that store: the
windows the calibration was fitted on and the new ones. The new document must be in the dataset's corpus.
Appending it to a local dataset is fine. Each opponent is compared once per window it shares with the new
document. The evidence therefore grows with `--n`. The new document's standard error can miss `--se-target`
(0.5 logits). The insertion then refuses (exit 12). Plan more opponents and judge the new plan, or accept a
larger `--se-target`. The new plan's windows are new, and the ones judged before still count. A plan holds at
most the query's other documents. `data.plan.capped` says when `--n` asked for more, and `data.plan.opponents`
says how many it holds. On such a small pool, only a larger `--se-target` helps.
`examples/05_insert_documents.py` does the same in Python on a smaller pool.

## A tiny example you can run offline

`rcp.testing.build_tiny_world` builds a small world, judged by a deterministic fake judge. The world
holds two queries, a tournament calibration (`tiny/calibration`) and two documents of `q2`. Those two
documents are left out of the fit and re-judged on Stage B, appended to `tiny/judgements`. The world also
holds a new document `q1-new`, judged in Stage A windows against opponents that the calibration picked
(`tiny/insertion`). It holds the rubric answers of a second, more lenient judge (`tiny/lenient`).

```python
from rcp.calibration import (
    Calibration, calibrate, insert_documents, read_judgements, score_documents, select_opponents,
)
from rcp.testing import build_tiny_world

build_tiny_world("tiny")
calibration = Calibration.load("tiny/calibration")
```

With a real judge you produce the judgements the same way. `judge(...)` with `docs={query_id: [doc_id, ...]}`
judges only those documents of a query. It uses the same prompt and judge as the calibration. The judgement
store is append-only. Re-judging a subset therefore writes into the same store, and it asks only for the new
windows.

### Score documents the calibration lacks

```python
extension = score_documents(calibration, read_judgements("tiny/judgements"))
for record in extension.records:
    estimate = record.estimate  # theta and se (logits), information, flags
    print(record.doc_id, estimate.theta, estimate.se, estimate.flags.degenerate)
calibration.extended(extension).save("tiny/extended")
```

Each document is scored alone, from its own criterion answers. The calibration's item parameters are frozen.
The score is the posterior mean (EAP) of

$$P(C_c = 1 \mid \theta) = \sigma\big(\gamma_c(\theta - \beta_c)\big)$$

under a normal prior. Its posterior standard deviation is `theta_se`. A rubric-only calibration uses the
fit's own ability prior (`diagnostics.fit.ability_prior`). A document scored from exactly its fit answers
therefore gets its fit ability back. A tournament calibration has no rubric prior of its own. Its prior is the
mean and standard deviation of the calibration's thetas. The item logits are the calibration's, so the score
is on the calibration's scale. In a pooled calibration, a document answered by several judges gets one score
from all their answers. Each judge's logit is shifted by its severity $s_j$. A document can fail every
criterion in every placement, or pass every one. It then has no interior maximum of the likelihood. Its score
is prior-bounded and flagged: `flags.degenerate` is `all_fail` or `all_pass`. A document whose own evidence
misses the standard-error target is flagged `low_information`. Pass counts are counts. A NaN or fractional
count is refused where it is scored, not read as a share.

The judgements must come from the calibration's rubric family: the same prompt, judge and parse. Otherwise the
call raises `IdentityError`. A document that the calibration already scores keeps its published theta, and it
is listed in `extension.skipped`.

### Insert a new document into a tournament calibration

Choose the opponents of the new document. Judge exactly those windows on Stage A, with `windows=` and the
calibration's tournament schedule. Then insert, and pass the windows that the calibration was fitted on
together with the new ones:

```python
windows = select_opponents(calibration, "q1", "q1-new", n=9, window=5)   # 3 windows: 4 + 4 + 1 opponents
# judge(dataset, None, judge_cfg, stage="tournament", out="tiny/judgements", windows={"q1": windows})
inserted = insert_documents(calibration, read_judgements("tiny/judgements", "tiny/insertion"))
```

Opponents are scored against the new document's provisional ability. That ability is `provisional_theta=`,
which is the query's median fitted ability by default. Pass your own guess when it is far from the median. The
informative opponents are then the ones near it. The guess is the EAP of `score_documents`, in logits on the
calibration's scale. The call maps it onto the query's Bradley-Terry scale. A query with no other calibrated
document has no window to offer. `select_opponents` refuses it with a `DataError`. `[[doc_id]]` is not a
window, and `judge(windows=...)` would refuse it later.

The new document's Bradley-Terry ability is estimated from its own comparisons. Every existing ability is held
fixed. The query's own transform, $\theta = \tau_j \theta_{BT} + \alpha_j$, maps the result onto the
calibrated scale. The evidence can fail to identify the document. The call then refuses (`DataError`) rather
than answers. There are three cases. The document has fewer than `min_opponents` (5) distinct opponents. Its
comparison graph is in more than one piece. Its Fisher information is below what `se_target` (0.5 logits) asks
for (`1 / se_target^2`). It also refuses (`IdentityError`) when the fitted windows no longer reproduce the
calibration's abilities.

### Pool two judges

```python
import warnings

# The stores also hold the re-judged documents, so the pooled fit warns about them (the warning is expected):
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    pooled = calibrate(read_judgements("tiny/judgements", "tiny/lenient"), judges="pooled")
assert {getattr(w.message, "code", None) for w in caught} == {"UNCALIBRATED_DOCUMENTS"}
print(pooled.judge_severity)   # one offset per judge, in logits
```

Both judges share one set of item parameters. Each judge gets one severity offset $s_j$ on the logit. The
offsets are centred so that $\sum_j n_j s_j = 0$, where $n_j$ is the judge's number of observations. A more
lenient judge has a negative offset. The offsets are lower bounds on the judges' true difference, because the
free abilities absorb part of it. Judges must share documents. Every pair of judges must have judged at least
one common document of a query on the rubric. Otherwise the fit refuses with a `DataError` that names the
judges and how many documents each pair shares. The judges must also have answered the same prompt. With the
default `judges="single"`, judgements of two judges are refused.

## What is written

`score_documents` and `insert_documents` never modify the calibration. They return an `Extension` (schema
`rcp.extension.v1`). `calibration.extended(extension)` is a new, complete calibration, and `save` writes
it like any other. `rcp eval score --calibration` reads it:

| File | Content |
|---|---|
| `items.json`, `queries.parquet` | the calibration's item and query parameters, unchanged |
| `thetas.parquet` | the calibration's thetas plus every added document, each with its `source` (`fit`, `scored` or `inserted`) |
| `extensions.jsonl` | one record per added document: its `estimate` on the calibrated scale (`theta`, `se`, `information`, `flags`), and provenance (source, calibration fingerprint, judgements digest, `record_key`) |
| `coverage.json`, `diagnostics.json`, `identity.json` | as the calibration's |

Scoring recomputes no row of the calibration, so nothing already in it can move. An insertion carries an
**anchor report**. This is the *insertion anchor report*, and it differs from the template anchors of the
[text budgets](text-budgets.md). The calibration's own tournament windows are refitted. Every document of the
calibration is then compared with its published ability, in theta and in gain. The call fails if any gain
moved more than `max_gain_shift` (0.01 by default). The windows passed are then not the ones the calibration
was fitted on.
