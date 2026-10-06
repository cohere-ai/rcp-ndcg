"""The tournament-anchored 2PL calibration engine.

Frozen Bradley-Terry theta from the tournament enters the likelihood as a
covariate, with per-query affine transforms (tau, alpha) mapping it onto the
common scale: ``sigma(gamma_k * (tau_j * theta_bt - beta_k + alpha_j))``. The
rubric-only calibration (no tournament) is ``Criteria2PL``; both are reached
through :func:`rcp_ndcg_core.irt.fit_calibration`.

Multi-judge: every observation may carry a
``judge_id``. When two or more judges are tagged, the fit gains one additive
**severity** per judge on the logit,

    logit P(Y = 1) = gamma_k * (theta_eff - beta_k) - s_judge(row),

centred so ``sum_j n_j * s_j = 0`` exactly (n_j = the judge's observation
count) -- the location the mean-zero constraint would otherwise absorb is
pinned by construction, and a pass-rate gap between judges is absorbed instead
of biasing the shared item parameters. A single tagged judge is structurally identical to no severity at
all (the centre is the only judge), so the fit stays bit-identical to the
untagged fit; mixing tagged and untagged observations is refused.
"""

from __future__ import annotations

import math
from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from rcp_ndcg_core.irt._calibration import MIN_SHARED_ITEMS, JudgeOverlapError, verdicts


class Tournament2PLCalibrator(nn.Module):
    """Offline 2PL IRT calibrator with per-query affine transforms on frozen BT theta.

    Fits the model::

        P(Y_{ijk} = 1) = sigma(gamma_k * (tau_j * theta_{ij} - beta_k + alpha_j))

    where:

    * ``theta_{ij}`` is a frozen BT score (per query ``j``, document ``i``),
    * ``gamma_k`` is per-criterion discrimination (shared across queries),
    * ``beta_k`` is per-criterion difficulty (shared across queries),
    * ``tau_j`` is a per-query scale absorbing BT's arbitrary scale,
    * ``alpha_j`` is a per-query offset absorbing BT's arbitrary location.

    Usage::

        cal = Tournament2PLCalibrator(num_criteria=5)
        for query_id, doc_id, theta_bt, criteria_dict in observations:
            cal.add_observation(query_id, doc_id, theta_bt, criteria_dict)
        cal.finalize()
        cal.fit()
    """

    _GAMMA_RAW_CAP: float = 10.0

    def __init__(
        self,
        num_criteria: int = 5,
        sigma_tau: float = 1.0,
        sigma_alpha: float = 2.0,
        l2_gamma: float = 1e-4,
        l2_beta: float = 1e-4,
    ):
        super().__init__()
        self.num_criteria = num_criteria
        self.sigma_tau = sigma_tau
        self.sigma_alpha = sigma_alpha
        self.l2_gamma = l2_gamma
        self.l2_beta = l2_beta

        # gamma_raw lives in log-space: gamma = exp(gamma_raw), so gamma_raw=0 → gamma=1.
        # L2 on gamma_raw is a log-normal prior centred at gamma=1 (Rasch default).
        self.gamma_raw = nn.Parameter(torch.zeros(num_criteria))
        self.beta_raw = nn.Parameter(torch.zeros(num_criteria))

        self.tau_raw: nn.Parameter | None = None
        self.alpha_param: nn.Parameter | None = None

        self._query_to_idx: dict[str, int] = {}
        self._obs_rows: list[tuple[int, float, list[int]]] = []
        self._obs_keys: list[tuple[str, str]] = []
        self._judge_ids: list[str | None] = []
        self._finalized = False

        # Multi-judge state. Allocated at finalize() only
        # when >= 2 distinct judges were tagged; with 0 or 1 tagged judges these
        # stay None and the fit is exactly the single-judge model.
        self.severity_raw: nn.Parameter | None = None
        self._row_judge: torch.Tensor | None = None
        self._judge_list: list[str] = []
        self._judge_counts: torch.Tensor | None = None

        self._query_indices: torch.Tensor | None = None
        self._theta_bt: torch.Tensor | None = None
        self._Y: torch.Tensor | None = None

    @property
    def gamma(self) -> torch.Tensor:
        """Positive discrimination, normalised so ``sum == K``.

        ``gamma_raw`` is in log-space (``gamma = exp(gamma_raw)``), so the
        gradient ``d gamma / d gamma_raw = gamma`` is always positive and
        never vanishes -- unlike ``softplus`` which has a dead zone for large
        negative inputs that can trap L-BFGS.

        An upper cap on ``gamma_raw`` prevents ``exp`` overflow; it is far
        above any physically meaningful discrimination value.
        """
        g = torch.exp(self.gamma_raw.clamp(max=self._GAMMA_RAW_CAP))
        return g * (self.num_criteria / g.sum())

    @property
    def beta(self) -> torch.Tensor:
        """Zero-mean difficulty."""
        return self.beta_raw - self.beta_raw.mean()

    @property
    def tau(self) -> torch.Tensor:
        """Positive per-query scale."""
        if self.tau_raw is None:
            raise ValueError("tau exists only after finalize()")
        return F.softplus(self.tau_raw)

    @property
    def alpha(self) -> torch.Tensor:
        """Per-query offset."""
        if self.alpha_param is None:
            raise ValueError("alpha exists only after finalize()")
        return self.alpha_param

    def add_observation(
        self,
        query_id: str,
        doc_id: str,
        theta_bt: float,
        criteria_dict: dict[str, int],
        judge_id: str | None = None,
    ) -> None:
        """Register one ``(query, doc)`` observation with frozen theta and binary criteria.

        ``judge_id`` tags the judge that produced this observation (multi-judge
        pooling). Either every observation carries one or
        none does: a mixed record set has no defined severity group for the
        untagged rows and is refused at ``finalize``.
        """
        if self._finalized:
            raise RuntimeError("Cannot add observations after finalize()")
        if not math.isfinite(theta_bt):
            raise ValueError(
                f"theta_bt must be finite, got {theta_bt!r}: a missing Bradley-Terry score is a dropped "
                "observation, not a zero, and a NaN one would silently NaN the whole fit"
            )
        if query_id not in self._query_to_idx:
            self._query_to_idx[query_id] = len(self._query_to_idx)
        q_idx = self._query_to_idx[query_id]
        vec = verdicts(criteria_dict, self.num_criteria)
        self._obs_rows.append((q_idx, float(theta_bt), vec))
        self._obs_keys.append((query_id, doc_id))
        self._judge_ids.append(judge_id)

    def _canonicalize(self) -> None:
        """Put queries and observations in one canonical order.

        The fit sums float32 terms and stops after a fixed iteration budget, so
        the order in which identical observations were added (dict or file order
        upstream) would otherwise move the result. Queries are indexed in sorted
        id order and rows sorted by ``(query, doc, criteria, theta, judge)``.
        """
        index_to_query = {index: query_id for query_id, index in self._query_to_idx.items()}
        self._query_to_idx = {query_id: index for index, query_id in enumerate(sorted(self._query_to_idx))}
        rows = sorted(
            zip(self._obs_keys, self._obs_rows, self._judge_ids, strict=True),
            key=lambda row: (row[0], row[1][2], row[1][1], row[2] or ""),
        )
        self._obs_keys = [key for key, _, _ in rows]
        self._obs_rows = [(self._query_to_idx[index_to_query[q]], theta, vec) for _, (q, theta, vec), _ in rows]
        self._judge_ids = [judge for _, _, judge in rows]

    def finalize(self) -> None:
        """Allocate per-query parameters and build observation tensors."""
        Q = len(self._query_to_idx)
        if Q == 0:
            raise ValueError("No observations registered")
        self._canonicalize()

        judge_ids = self._judge_ids
        n_tagged = sum(1 for j in judge_ids if j is not None)
        if 0 < n_tagged < len(judge_ids):
            raise ValueError(
                f"{n_tagged} of {len(judge_ids)} observations carry a judge_id: mixing tagged and "
                "untagged observations has no defined severity group for the untagged rows"
            )
        judges = sorted({j for j in judge_ids if j is not None})
        if len(judges) >= 2:
            self._judge_list = judges
            judge_index = {j: i for i, j in enumerate(judges)}
            keys = self._obs_keys
            self.severity_raw = nn.Parameter(torch.zeros(len(judges)))
            # Every id is a str here: the mixed tagged/untagged case raised above.
            self._row_judge = torch.tensor([judge_index[cast("str", j)] for j in judge_ids], dtype=torch.long)
            counts = torch.zeros(len(judges))
            counts.scatter_add_(0, self._row_judge, torch.ones(len(keys)))
            self._judge_counts = counts

        # tau initialized so softplus(raw) ≈ 1.0; inverse softplus(1) = log(e-1) ≈ 0.541.
        self.tau_raw = nn.Parameter(torch.full((Q,), math.log(math.e - 1)))
        self.alpha_param = nn.Parameter(torch.zeros(Q))
        query_indices, thetas, criteria_vecs = zip(*self._obs_rows, strict=False)
        self._query_indices = torch.tensor(query_indices, dtype=torch.long)
        self._theta_bt = torch.tensor(thetas, dtype=torch.float32)
        self._Y = torch.tensor(criteria_vecs, dtype=torch.float32)
        self._finalized = True

    def _compute_logits(self) -> torch.Tensor:
        """Compute the likelihood logits ``gamma_k * (tau_j * theta_ij - beta_k + alpha_j)``."""
        if not self._finalized or self._query_indices is None or self._theta_bt is None:
            raise ValueError("finalize() allocates the per-query parameters; call it before fitting or scoring")
        tau_obs = self.tau[self._query_indices]  # (N,)
        alpha_obs = self.alpha[self._query_indices]  # (N,)
        effective_theta = tau_obs * self._theta_bt + alpha_obs  # (N,)
        logits = self.gamma[None, :] * (effective_theta[:, None] - self.beta[None, :])  # (N, K)
        if self.severity_raw is not None:
            logits = logits - self._severity_centered()[self._row_judge][:, None]
        return logits

    def _severity_centered(self) -> torch.Tensor:
        """Per-judge severity, centred so ``sum_j n_j * s_j = 0`` exactly.

        The centre of the severity scale is not identified (a constant added to
        every judge's logit is absorbable by the item difficulties); pinning it
        to the observation-mass-weighted mean keeps the constraint an invariant
        of the parameterization rather than a reporting convention.
        """
        if self.severity_raw is None or self._judge_counts is None:
            raise ValueError("there is no severity to centre: the fit has fewer than two tagged judges")
        counts = self._judge_counts
        mean = (counts * self.severity_raw).sum() / counts.sum()
        return self.severity_raw - mean

    def get_judge_severity(self) -> dict[str, float]:
        """Fitted per-judge severity offsets ``{judge_id: s_j}``, centred.

        Empty unless two or more judges were tagged: a single tagged judge is
        structurally the no-severity model (nothing was absorbed, so there is
        no severity to report), and an untagged fit has no judge dimension at
        all. Required output of any pooled multi-judge fit -- a consumer must
        be able to see which judge's offset was absorbed into the shared scale.
        """
        if self.severity_raw is None:
            return {}
        centred = self._severity_centered().detach().cpu()
        return {judge: float(centred[i].item()) for i, judge in enumerate(self._judge_list)}

    def check_judge_overlap(self, min_shared: int = MIN_SHARED_ITEMS) -> dict:
        """Judge overlap: does every pair of tagged judges share at least ``min_shared`` ``(query, doc)`` items?

        A judge with no shared items contributes severity but no linking
        information -- the shared scale would be identified only by the prior.
        Returns the pairwise shared-item counts; refuses (:class:`JudgeOverlapError`)
        when a pair shares fewer, because a pooled fit in that state is not the
        bigger fit it looks like.
        """
        keys, judge_ids = self._obs_keys, self._judge_ids
        judges = self._judge_list or sorted({j for j in judge_ids if j is not None})
        if len(judges) < 2:
            raise ValueError(f"the judge-overlap check needs >= 2 tagged judges, got {judges or 'none'}")
        by_judge: dict[str, set[tuple[str, str]]] = {j: set() for j in judges}
        for key, judge in zip(keys, judge_ids, strict=True):
            if judge is not None:
                by_judge[judge].add(key)
        pairs: dict[str, int] = {}
        ok = True
        for i, a in enumerate(judges):
            for b in judges[i + 1 :]:
                shared = len(by_judge[a] & by_judge[b])
                pairs[f"{a}|{b}"] = shared
                ok = ok and shared >= min_shared
        report = {
            "min_shared": min_shared,
            "pairs": pairs,
            "items_per_judge": {j: len(v) for j, v in by_judge.items()},
            "overlap_ok": ok,
        }
        if not ok:
            raise JudgeOverlapError(
                f"judges overlap too little: a pair shares fewer than {min_shared} (query, doc) items "
                f"(shared per pair: {pairs}; items per judge: {report['items_per_judge']})",
                report,
            )
        return report

    def model_loss(self) -> torch.Tensor:
        if not self._finalized or not self._obs_rows:
            return torch.tensor(0.0)
        if self._Y is None:  # pragma: no cover - finalize() builds it with every observation
            raise ValueError("finalize() allocates the observation tensor; call it before fitting")
        logits = self._compute_logits()
        return F.binary_cross_entropy_with_logits(logits, self._Y, reduction="mean")

    def regularization_loss(self) -> torch.Tensor:
        reg = torch.tensor(0.0)

        # Item parameters (gamma, beta): only K values each.  Regularisation
        # is NOT scaled by observation count -- the model loss is already a
        # mean, so the prior should be commensurate.
        reg = reg + 0.5 * self.l2_gamma * torch.sum(self.gamma_raw**2)
        reg = reg + 0.5 * self.l2_beta * torch.sum(self.beta_raw**2)

        N, K = self._Y.shape  # type: ignore[union-attr]  # _Y exists: model_loss guards, or finalize ran
        # Per-query parameters (tau, alpha): Q values, one per query.  Scale
        # by 1/(N*K) so the per-parameter penalty is proportional to each
        # query's share of the total observations.
        query_scale = 1.0 / (N * K)
        if self.tau_raw is not None:
            reg = reg + query_scale * (1.0 / (2.0 * self.sigma_tau**2)) * torch.sum((self.tau - 1.0) ** 2)
        if self.alpha_param is not None:
            reg = reg + query_scale * (1.0 / (2.0 * self.sigma_alpha**2)) * torch.sum(self.alpha_param**2)

        return reg

    def fit(self) -> None:
        """Joint MAP estimation via L-BFGS.

        Deterministic: the initialisation is fixed, observations are in canonical
        order (:meth:`finalize`) and the optimiser runs on one intra-op thread, so
        identical observations give bit-identical parameters.
        """
        if not self._finalized:
            self.finalize()

        optimizer = optim.LBFGS(
            [p for p in self.parameters() if p.requires_grad],
            lr=1.0,
            max_iter=500,
            tolerance_grad=1e-13,
            tolerance_change=1e-15,
            history_size=100,
            line_search_fn="strong_wolfe",
        )

        def closure():
            optimizer.zero_grad()
            loss = self.model_loss() + self.regularization_loss()
            loss.backward()
            return loss

        # Intra-op threads split float32 reductions differently per thread count;
        # one thread makes identical inputs give identical fits on any machine.
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            optimizer.step(closure)
        finally:
            torch.set_num_threads(threads)

    def get_item_params(self) -> dict:
        """Return calibrated item parameters as a plain dict."""
        return {
            "gamma": self.gamma.detach().cpu().tolist(),
            "beta": self.beta.detach().cpu().tolist(),
            "num_criteria": self.num_criteria,
        }

    def get_query_params(self) -> dict[str, dict[str, float]]:
        """Return per-query affine parameters ``{query_id: {tau, alpha}}``."""
        tau = self.tau.detach().cpu()
        alpha = self.alpha.detach().cpu()
        idx_to_query = {v: k for k, v in self._query_to_idx.items()}
        return {
            idx_to_query[i]: {"tau": float(tau[i]), "alpha": float(alpha[i])} for i in range(len(self._query_to_idx))
        }


def calibrate_2pl_from_results(
    bt_scores: dict[str, dict[str, float]],
    rubric_observations: dict[str, list[tuple[str, dict[str, int]]]],
    num_criteria: int = 5,
    **calibrator_kwargs,
) -> Tournament2PLCalibrator:
    """Build and fit a :class:`Tournament2PLCalibrator` from tournament + rubric results.

    Args:
        bt_scores: ``{query_id: {doc_id: theta_bt}}`` -- BT scores from the
            listwise tournament (Stage A).
        rubric_observations: ``{query_id: [(doc_id, criteria_dict), ...]}`` (or
            ``(doc_id, criteria_dict, judge_id)`` rows for a pooled multi-judge fit)
            -- binary criteria observations from the rubric pass (Stage B).
        num_criteria: Number of binary criteria (default 5, matching the prompt).
        **calibrator_kwargs: Extra kwargs for :class:`Tournament2PLCalibrator`
            (e.g. ``sigma_tau``, ``sigma_alpha``, ``l2_gamma``, ``l2_beta``).

    Returns:
        A fitted :class:`Tournament2PLCalibrator`. Rows whose document has no Bradley-Terry score, and every
        row of a query ``bt_scores`` lacks, are skipped: the caller reports them through
        ``FitDiagnostics.skipped_observations`` / ``skipped_queries``, never silently.

    Raises:
        ValueError: No Stage B observation has a Bradley-Terry theta.
        JudgeOverlapError: Two tagged judges share fewer than :data:`MIN_SHARED_ITEMS` ``(query, doc)`` items.
    """
    calibrator = Tournament2PLCalibrator(num_criteria=num_criteria, **calibrator_kwargs)

    def _split(row: tuple) -> tuple[str, dict[str, int], str | None]:
        """Accept ``(doc_id, criteria)`` (single-judge) or ``(doc_id, criteria, judge_id)``."""
        if len(row) == 2:
            return row[0], row[1], None
        if len(row) == 3:
            return row[0], row[1], row[2]
        raise ValueError(
            f"rubric observation rows must be (doc_id, criteria) or (doc_id, criteria, judge_id), got length {len(row)}"
        )

    n_obs = 0
    n_skipped = 0
    for query_id, obs_list in rubric_observations.items():
        query_bt = bt_scores.get(query_id)
        if query_bt is None:
            n_skipped += len(obs_list)
            continue
        for row in obs_list:
            doc_id, criteria_dict, judge_id = _split(row)
            theta = query_bt.get(doc_id)
            if theta is None:
                n_skipped += 1
                continue
            calibrator.add_observation(query_id, doc_id, theta, criteria_dict, judge_id=judge_id)
            n_obs += 1

    if n_obs == 0:
        raise ValueError(
            f"No matching observations found between BT scores and rubric observations "
            f"({n_skipped} skipped due to missing BT theta)"
        )
    if len({judge for judge in calibrator._judge_ids if judge is not None}) >= 2:
        calibrator.check_judge_overlap()

    calibrator.finalize()
    calibrator.fit()
    return calibrator
