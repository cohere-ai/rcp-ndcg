# Primitives: re-annotation, insertion and several judges

A calibration is published: its thetas are in `thetas.parquet` and papers,
leaderboards or dashboards already quote them. Then something changes:

- some documents' judging **did not work** (a failed window, a document that could not be loaded), so
  they have no theta;
- the candidate pool **grew**: a new document must be ranked against the old ones;
- a **second judge** answered the same rubric, and its answers should count.

Refitting everything would move every published score, including the scores of
documents nobody re-judged. The functions below add the new evidence to the
published calibration without re-estimating anything already in it.

| Situation | Judge again | Then | Estimator |
|---|---|---|---|
| documents without a theta | Stage B on those documents | `score_documents` | fixed-item EAP (items frozen) |
| a new document in a tournament calibration | Stage A windows pairing it with existing documents (`select_opponents`) | `insert_documents` | conditional MLE (every existing theta frozen) |
| a second judge, same rubric | Stage B with the second judge | `calibrate(..., judges="pooled")` | one 2PL with a severity offset per judge |

All three live in `rcp_ndcg.calibration` (`score_documents` and `insert_documents`
are also top-level functions of `rcp_ndcg`). On the command line, with the judge,
dataset and judgement store of the calibration:

```bash
# documents without a theta: judge them on the rubric into the same store, then score them
rcp-ndcg judge rubric --dataset <uri> --judge <judge> --docs q2:q2-d10 --docs q2:q2-d11 --out <store>
rcp-ndcg calibration score --calibration <calibration> --judgements <store> --out <extended-calibration>

# a new document: let the calibration plan its windows, judge exactly those, then insert
rcp-ndcg calibration insert --calibration <calibration> --judgements <store> --plan --query q1 --doc q1-new \
    --n 36 --out plan.json --json
rcp-ndcg judge tournament --dataset <uri> --judge <judge> --plan plan.json --out <store> --estimate
rcp-ndcg judge tournament --dataset <uri> --judge <judge> --plan plan.json --out <store>
rcp-ndcg calibration insert --calibration <calibration> --judgements <store> --out <extended-calibration>

# a second judge: one pooled fit over both stores
rcp-ndcg calibration fit --judgements <store> --judgements <second-store> --judges pooled --out <calibration>
```

In the second recipe, `--plan` picks `--n` opponents across the query's ability range and splits them into
windows of the store's `schedule.window` (read from its `identity.json`), each holding the new document; `--out`
writes the plan, and `data.plan.calls` says how many judge calls it takes (each window twice when the schedule
mirrors). `judge tournament --plan` asks exactly those windows, with the store's schedule and into the store,
which must be the calibration's own tournament store judged by the same judge (anything else is exit 11).
`calibration insert` then reads that store: the windows the calibration was fitted on and the new ones. The new
document must be in the dataset's corpus; appending it to a local dataset is fine. Each opponent is compared once
per window it shares with the new document, so the evidence grows with `--n`: when the new document's standard
error misses `--se-target` (0.5 logits), the insertion refuses (exit 12). Plan more opponents and judge the new
windows (the judged ones are reused), or accept a larger `--se-target`. `examples/05_insert_documents.py` does the same in Python on a smaller pool.

## A tiny example you can run offline

`rcp_ndcg.testing.build_tiny_world` builds a small world judged by a deterministic
fake judge: two queries, a tournament calibration (`tiny/calibration`), two
documents of `q2` left out of the fit and re-judged on Stage B (appended to
`tiny/judgements`), a new document `q1-new` judged in Stage A windows against
opponents the calibration picked (`tiny/insertion`), and a second, more lenient
judge's rubric answers (`tiny/lenient`).

```python
from rcp_ndcg.calibration import (
    Calibration, calibrate, insert_documents, read_judgements, score_documents, select_opponents,
)
from rcp_ndcg.testing import build_tiny_world

build_tiny_world("tiny")
calibration = Calibration.load("tiny/calibration")
```

With a real judge you produce the judgements the same way: `judge(...)` with
`docs={query_id: [doc_id, ...]}` judges only those documents of a query, with the
same prompt and judge as the calibration. The judgement store is append-only, so
re-judging a subset writes into the same store and asks only for the new windows.

### Score documents the calibration lacks

```python
extension = score_documents(calibration, read_judgements("tiny/judgements"))
for record in extension.records:
    estimate = record.estimate  # theta and se (logits), information, flags
    print(record.doc_id, estimate.theta, estimate.se, estimate.flags.degenerate)
calibration.extended(extension).save("tiny/extended")
```

Each document is scored alone, from its own criterion answers, with the
calibration's item parameters frozen: the posterior mean (EAP) of

$$P(C_c = 1 \mid \theta) = \sigma\big(\gamma_c(\theta - \beta_c)\big)$$

under a normal prior with the mean and standard deviation of the calibration's own
thetas, and its posterior standard deviation as `theta_se`. The item logits are the
calibration's, so the score is on the calibration's scale. In a pooled calibration a
document answered by several judges gets one score from all their answers, each
judge's logit shifted by its severity $s_j$. A document that failed
(or passed) every criterion in every placement has no interior maximum of the
likelihood; its score is prior-bounded and flagged (`flags.degenerate` is `all_fail`
or `all_pass`); a document whose own evidence misses the standard-error target is
flagged `low_information`.

The judgements must come from the calibration's rubric family (the same prompt,
judge and parse), or the call raises `IdentityError`. A document the calibration
already scores keeps its published theta and is listed in `extension.skipped`.

### Insert a new document into a tournament calibration

Choose whom to judge the new document against, judge exactly those windows on Stage A
(`windows=`, with the calibration's tournament schedule), then insert, passing the
windows the calibration was fitted on together with the new ones:

```python
windows = select_opponents(calibration, "q1", "q1-new", n=9, window=5)   # two windows of the new one and 4 opponents
# judge(dataset, None, judge_cfg, stage="tournament", out="tiny/judgements", windows={"q1": windows})
inserted = insert_documents(calibration, read_judgements("tiny/judgements", "tiny/insertion"))
```

The new document's Bradley-Terry ability is estimated from its own comparisons with
every existing ability held fixed, and mapped onto the calibrated scale by the
query's own transform, $\theta = \tau_j \theta_{BT} + \alpha_j$. The call refuses
(`DataError`) rather than answers when the evidence cannot identify the document:
fewer than `min_opponents` (5) distinct opponents, a comparison graph in more than
one piece, or less Fisher information than `se_target` (0.5 logits) asks for
(`1 / se_target^2`). It also
refuses (`IdentityError`) when the fitted windows no longer reproduce the
calibration's abilities.

### Pool two judges

```python
pooled = calibrate(read_judgements("tiny/judgements", "tiny/lenient"), judges="pooled")
print(pooled.judge_severity)   # one offset per judge, in logits
```

Both judges share one set of item parameters; each judge gets one severity offset
$s_j$ on the logit, centred so that $\sum_j n_j s_j = 0$ ($n_j$: the judge's number
of observations). A more lenient judge has a negative one. The offsets are lower
bounds on the judges' true difference, because the free abilities absorb part of
it. Judges must share documents: every pair of judges must have judged at least one
common document of a query on the rubric, or the fit refuses with a `DataError` that
names the judges and how many documents each pair shares. They must also have answered
the same prompt; with the default `judges="single"`, judgements of two judges are
refused.

## What is written

`score_documents` and `insert_documents` never modify the calibration. They return
an `Extension` (schema `rcp-ndcg.extension.v1`); `calibration.extended(extension)`
is a new, complete calibration that `save` writes like any other
(`rcp-ndcg eval score --calibration` reads it):

| File | Content |
|---|---|
| `items.json`, `queries.parquet` | the calibration's item and query parameters, unchanged |
| `thetas.parquet` | the calibration's thetas plus every added document, each with its `source` (`fit`, `scored` or `inserted`) |
| `extensions.jsonl` | one record per added document: its `estimate` on the calibrated scale (`theta`, `se`, `information`, `flags`), and provenance (source, calibration fingerprint, judgements digest, `record_key`) |
| `coverage.json`, `diagnostics.json`, `identity.json` | as the calibration's |

Scoring recomputes no row of the calibration, so nothing already in it can move. An
insertion carries an **anchor report**: the calibration's own tournament windows are
refitted, and every document of the calibration is compared with its published ability,
in theta and in gain. The call fails if any gain moved more than `max_gain_shift` (0.01
by default), because then the windows passed are not the ones the calibration was
fitted on.
