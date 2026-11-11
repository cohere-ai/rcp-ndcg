# The calibration

The tournament orders the documents of each query. Its Bradley-Terry scores, however, have no common scale.
Each query has its own free additive constant. Its spread depends on how decisive the judge was on that query.
The rubric answers mean the same on every query, but five binary answers cannot order documents finely. The
calibration combines the two. It fits one two-parameter logistic (2PL) item-response model to all rubric
answers of a suite. The tournament scores are the abilities in that fit. The calibration then maps every
query's scores onto one shared scale.

## The model

Each criterion $c$ has a discrimination $\gamma_c > 0$ and a difficulty $\beta_c$, shared across queries. A
discrimination is how sharply a criterion separates documents. A difficulty is how hard a criterion is to
pass. Each query $q_j$ has a scale $\tau_j > 0$ and an offset $\alpha_j$. The Bradley-Terry scores
$\hat\theta^{\mathrm{BT}}_i$ of Stage A are held fixed. Then a rubric answer of document $d_i$ on query $q_j$
passes criterion $c$ with probability

$$
P(Y_{ijc} = 1) = \sigma\Bigl(\gamma_c\,\bigl(\tau_j\,\hat\theta^{\mathrm{BT}}_i + \alpha_j - \beta_c\bigr)\Bigr),
\qquad \sigma(x) = \frac{1}{1 + e^{-x}}.
$$

The calibrated ability of the document is

$$
\tilde\theta_{ij} = \tau_j\,\hat\theta^{\mathrm{BT}}_i + \alpha_j .
$$

The offset $\alpha_j$ fixes the Bradley-Terry constant of query $j$ on the shared scale. The scale $\tau_j$
sets its spread, so that a one-logit difference in ability has the same meaning on every query. Because
$\tau_j > 0$, sorting a query's documents by $\tilde\theta_{ij}$ reproduces their Stage A order. A document
with $\tilde\theta_{ij} = \beta_c$ passes criterion $c$ with probability one half on every query. The gain of
RCP-nDCG is defined on this scale ([the metric](metric.md)): $g(\tilde\theta) = \sum_c
\gamma_c\,\sigma(\gamma_c(\tilde\theta - \beta_c)) / \sum_c \gamma_c$.

## Identification and fit

Shifting or stretching all abilities leaves every probability unchanged, if $\beta_c$ and $\gamma_c$ change to
match. The fit therefore fixes the unit with $\sum_c \gamma_c = C$, and the origin with $\operatorname{mean}_c
\beta_c = 0$. It parameterises $\tau_j = \operatorname{softplus}(t_j)$, $\gamma_c = C\,e^{u_c} / \sum_{c'}
e^{u_{c'}}$ and $\beta_c = v_c - \operatorname{mean}_{c'} v_{c'}$. It then minimises

$$
\frac{1}{NC} \biggl[ \sum_{n=1}^{N} \sum_{c=1}^{C} \ell_{nc}
 + \sum_j \Bigl( \frac{(\tau_j - 1)^2}{2\sigma_\tau^2} + \frac{\alpha_j^2}{2\sigma_\alpha^2} \Bigr) \biggr]
 + \frac{\lambda}{2} \sum_c \bigl( u_c^2 + v_c^2 \bigr),
$$

Here $N$ is the number of rubric placements. Each placement answers all $C$ criteria. The term $\ell_{nc}$ is
the binary cross-entropy of placement $n$ on criterion $c$. The priors are $\tau_j \sim \mathcal{N}(1, 1)$ and
$\alpha_j \sim \mathcal{N}(0, 2^2)$. The ridge is $\lambda = 10^{-4}$. The fit is thus the maximum a
posteriori estimate. It starts from $\tau_j = 1$, $\alpha_j = 0$ and equal criterion parameters. It runs
L-BFGS for at most 500 iterations on the CPU. `rcp.calibration.Priors` holds these settings (`sigma_tau`,
`sigma_alpha`, `l2_gamma`, `l2_beta`). It also holds `bt_l2`, the L2 penalty ($10^{-4}$) of the
Bradley-Terry abilities that the fit refits from the tournament's windows. The defaults, `Priors()`, are the
paper's. The tournament's live fit uses the default `bt_l2` while judging, and the judgement store records it.
A fit with another `bt_l2` refits abilities that the adaptive windows were not chosen from.
`rcp calibration fit` and a run's calibrate step fit it anyway, and they record a `BT_L2_MISMATCH`
warning. `calibration show` lists that warning.

The fit is deterministic. Identical judgements give identical parameters, whatever the order of the records or
the number of threads.

## Two modes

`calibrate(judgements, mode="auto")` picks the mode from the judgements:

| | `mode="tournament"` | `mode="rubric_only"` |
|---|---|---|
| Needs Stage A | yes | no |
| Ability of a document | $\tau_j \hat\theta^{\mathrm{BT}}_i + \alpha_j$ | posterior mean under $\theta \sim \mathcal{N}(0, 1)$ (`Priors.ability_sd`) |
| Item parameters | MAP estimate above | marginal maximum likelihood |
| Order within a query | fine (Bradley-Terry) | coarse: documents with equal pass counts tie |

`auto` picks `tournament` when every query has tournament judgements. It picks `rubric_only` when no query has
them. Mixed coverage raises `DataError` with the per-query counts. It does not fit some queries one way and
some the other. Both modes report item parameters with $\sum_c \gamma_c = C$ and $\operatorname{mean}_c
\beta_c = 0$. The gain therefore applies to either. A rubric-only fit can have discriminations that all
saturate. It can also have documents that almost never pass a criterion. Such a fit is flagged `ordinal_only`
in `diagnostics.json`. Its order is usable, but its gains are not calibrated probabilities.

A tournament query without rubric answers has no fitted $\tau_j$ and $\alpha_j$. It therefore has no
calibrated abilities. It is listed under `uncalibrated_queries` in `coverage.json`, and its raw Bradley-Terry
scores are never written as abilities.

## Several judges

Two or more judges who answered the same rubric pool enter one fit with `judges="pooled"`. They share the
item parameters. Each judge gets one severity offset on the logit, reported as `judge_severity`
([primitives](primitives.md#pool-two-judges)). With the default `judges="single"`, judgements of two judges
are refused. Judgements of different families never pool. A family differs by its prompt, parse version,
decoding or preprocessing.

## Artifacts

A `Calibration` is immutable. `calibration.save(dir)` writes one layout. `Calibration.load(dir)` reads that
layout, and a run's `calibrate` step writes it to `runs/<run_id>/calibration/`:

```text
calibration/
├── items.json          # mode, criteria, gamma, beta, judge severity, the families fitted
├── queries.parquet     # dataset, query_id, tau, alpha (tournament mode)
├── thetas.parquet      # dataset, query_id, doc_id, theta, theta_se, gain, source (fit | scored | inserted)
├── coverage.json       # queries per stage, uncalibrated, no-evidence and flagged queries, invalid windows, degenerate documents
├── diagnostics.json    # fit summary, reliability (ECE, Brier score) per family, warnings
├── extensions.jsonl    # every document added later by scoring or insertion, with its provenance
└── identity.json       # what was fitted: the judgements, the families, the switches and the priors
```

`theta_se` is a standard error in logits, when one is known. A row with none leaves the column empty. A
rubric-only document's standard error is the posterior standard deviation of its ability. A tournament
document's standard error is its Bradley-Terry standard error, mapped onto the calibrated scale by the query's
`tau`. The Bradley-Terry one is the **diagonal approximation**:

`1 / sqrt(sum of weight * p * (1 - p) over the document's own comparisons + bt_l2 * W)`

Here `W` is the fit's total comparison weight. The approximation is a per-document information referent that
ignores the covariance between documents. It is therefore comparable across documents of one fit, but it is
not a full-information standard error. A document that a fitted query never compared carries no information of
its own. It therefore has no ability and no SE. `coverage.json` lists those documents under
`no_tournament_evidence_documents`.

`diagnostics.json` reports how well the predicted pass probabilities match the observed answers. The expected
calibration error (ECE) is the size-weighted mean gap between predicted probabilities and observed pass rates.
The gap is measured in bins of the prediction, and 0 is perfect. The Brier score is the mean squared error of
the predicted probabilities. `coverage.json` reports how much of each pool has evidence behind it. Together
they tell whether a calibration can be trusted.

## Invalid windows

A window whose answer could not be read contributes nothing to the fit ([the tournament](tournament.md#parsing-an-answer)). `coverage.json` counts the invalid windows of every query, by
stage, by schedule phase (`random`, `stratified`, `adaptive`) and by category. A query is flagged in two
cases. The first case is that more than 5% of its windows in a stage the fit reads are invalid. The second
case is that any of its adaptive tournament windows is invalid. The adaptive windows are the few that order
the top of the ranking. The flags are an `INVALID_WINDOWS` warning. That warning is kept in the calibration's
diagnostics. `rcp calibration show` shows it, and so do the warnings of an evaluation scored with the
calibration's gains. `calibrate(..., strict=True)` and `rcp calibration fit --strict` refuse a flagged
fit with `DataError` (exit
12) instead. What clears a flag depends on the failure. A window that the endpoint refused (no answer) is asked
again by the same judging call or command. An unparseable answer is the judge's answer, and a resumed pass
keeps it. Re-reading the stored answers with a newer parser (`rcp judge reparse`) can recover it.
Otherwise the windows are judged again only into a new store.

## Documents without a tournament ability

In tournament mode, a document's rubric verdicts enter the fit through its Bradley-Terry ability. A document
that the tournament did not judge therefore gets no ability from the fit. One example is a document added to
the pool after the tournament. A document also gets no ability when the tournament showed it only in windows
whose answers could not be read. No comparison informs it, and its prior is never reported as an ability. The
fit lists such documents in `coverage.json` (`uncalibrated_documents`) and warns with `UNCALIBRATED_DOCUMENTS`
. Score them from their rubric verdicts with `score_documents` (`rcp calibration score`), or insert them
into the tournament ([primitives](primitives.md)).

The query can have at least one valid window. Such a document is then named by the windows that did not parse,
but no comparison involves it. The Bradley-Terry model would only give it its prior, the query's mean ability.
That is no evidence. The fit gives it no ability, and a document judged in chunks pools only its compared
chunks. The fit lists it in `coverage.json` (`no_tournament_evidence_documents`), beside
`uncalibrated_documents` when it has rubric verdicts. It warns with `NO_VALID_TOURNAMENT_EVIDENCE`, and
`calibrate(..., strict=True)` (`rcp calibration fit --strict`) refuses it. Judge a valid window for it,
and an insertion plan's windows are the way in. Or leave it out of the pool. Every window of a query can be
invalid. Then nothing of the query is fitted at all. It is listed under `uncalibrated_queries`, and its
rubric-judged documents under `uncalibrated_documents`. No ability is written for them. A pure-tournament
document of such a query appears in neither list, since the fit only sees rubric verdicts. A fit whose every
query is uncalibrated has nothing to fit.

## Repeated judgements

The unit of evidence is the window. Each window has a `record_id`. That id is a hash of its judgement family,
query, stage and the dataset's identity key. Two corpora that share query and document ids therefore never
share a window. The hash also covers the window's position in the schedule. For a planned window, it covers
the schedule that the window was asked under. It covers the documents that the window showed, in order. The
judgements passed to `calibrate`, or read from several stores by `read_judgements`, can hold one window more
than once. The window then counts once. Its latest valid judgement (by `recorded_at`) is used. An invalid
copy is used only when the window has no valid one. No window is counted twice.

Re-judging documents asks new windows, at other positions or with other companions. Those windows have new
record ids. A refit over the stores counts them together with the documents' earlier windows. The command is
`rcp calibration fit --judgements <store> --judgements <re-judged store>`. The re-judged documents are
therefore refit on all their evidence, and so is every calibrated parameter. `calibration score` differs. It
adds documents that have no theta, and it leaves a document that has one as it is
([primitives](primitives.md)).

## In code

The package's offline test world judges two small queries with a deterministic fake judge. That world is
enough to run the calibration end to end:

```python
import warnings

from rcp.calibration import Calibration, calibrate, read_judgements
from rcp.testing import build_tiny_world

build_tiny_world("tiny")  # judgements of a fake judge under tiny/judgements
# The store also holds two rubric-only documents (the re-judged q2-d10, q2-d11 above), so the fit warns about
# them -- the warning is expected here, not an accident:
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    calibration = calibrate(read_judgements("tiny/judgements"))  # mode="auto" picks "tournament" here
assert {getattr(w.message, "code", None) for w in caught} == {"UNCALIBRATED_DOCUMENTS"}
print(calibration.mode, calibration.items)
calibration.save("tiny/my_calibration")

gains = Calibration.load("tiny/my_calibration").gains()  # {query_id: {doc_id: gain}}
print(sorted(gains["q1"].items(), key=lambda kv: -kv[1])[:3])
```

On the command line, `rcp calibration fit` fits judgements into a calibration directory.
`rcp calibration show` prints its parameters, coverage and diagnostics.
