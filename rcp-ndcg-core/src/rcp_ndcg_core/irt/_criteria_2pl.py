"""The rubric-only 2PL: item parameters and abilities from binary criteria alone.

Given, per document, ``K`` binary criteria observed over ``n`` judge placements,
the sufficient statistic is ``(n, s_1..s_K)`` where ``s_k`` counts how often
the document passed criterion ``k``. :meth:`Criteria2PL.fit` estimates the items
by marginal maximum likelihood under a Gaussian ability prior,

    P(s_k | theta) = Binom(n, sigmoid(gamma_k * (theta - beta_k))),
    theta ~ N(0, prior_sd),

on the exact gradient (so the fit does not depend on the CPU's rounding), and
:meth:`Criteria2PL.eap` / :meth:`Criteria2PL.score_document` give each
document's posterior-mean ability with the items held fixed.

Two degeneracies are guarded:

1. **One item runs away.** A near-constant criterion has an unidentified
   discrimination that an unbounded optimiser drives to thousands, so it takes
   all of the gain weight. ``fit`` bounds gamma and raises :class:`ValueError`
   when one item exceeds :data:`DEFAULT_MAX_ITEM_WEIGHT` of the total.
2. **Every item saturates together.** With very few positive observations every
   gamma pins at the upper bound and the gain becomes effectively binary. A
   max-weight check cannot see this (equal gammas give weight 1/K), so the fit
   flags it in :attr:`Criteria2PLDiagnostics.ordinal_only` instead of raising:
   the ordering stays usable, the probability reading does not.

Numpy only at import; ``fit`` imports scipy's bounded L-BFGS-B lazily.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rcp_ndcg_core.gain import sigmoid
from rcp_ndcg_core.schemas import DocumentEstimate, EstimateFlags

#: Quadrature grid for the marginal likelihood and the posterior integrals:
#: 561 points on [-18, 10] (spacing 0.05). It covers the calibrated-ability scale
#: of tournament fits and item difficulties well above 5 logits.
QUAD = np.linspace(-18.0, 10.0, 561)

#: Prior standard deviation on ability (logits).
DEFAULT_PRIOR_SD = 1.0

DEFAULT_GAMMA_BOUNDS = (0.05, 20.0)
DEFAULT_MAX_ITEM_WEIGHT = 0.90

#: Below this share of documents passing any criterion, the 2PL is not
#: identifiable and the gain is not a calibrated probability.
MIN_FRAC_PASSING_ANY = 0.02

#: Default standard error (logits) a stand-alone or inserted document's
#: evidence must reach: its own Fisher information >= ``1 / DEFAULT_SE_TARGET**2``.
DEFAULT_SE_TARGET = 0.5


def _moments(posterior: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean and standard deviation of each row of a ``(D, len(QUAD))`` posterior over the grid :data:`QUAD`."""
    mean = posterior @ QUAD
    sd = np.sqrt(np.einsum("dq,dq->d", posterior, (QUAD[None, :] - mean[:, None]) ** 2))
    return mean, sd


@dataclass(frozen=True)
class Criteria2PLDiagnostics:
    """Fit health. All fields are derived from the fit itself -- no human labels.

    Consumers must read ``ordinal_only`` before treating gains as probabilities.
    A dataset can pass the item-weight guard and still be degenerate: the two
    failure modes are independent.
    """

    n_documents: int
    mean_placements: float
    max_item_weight: float
    gamma_saturated: bool
    frac_passing_any: float
    ordinal_only: bool
    reason: str | None


def _validate_observations(n: np.ndarray, S: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = np.asarray(n, dtype=float)
    S = np.asarray(S, dtype=float)
    if not (np.isfinite(n).all() and np.isfinite(S).all()):
        raise ValueError(
            "placement and pass counts must be finite: a NaN or infinite count is missing data, "
            "not a zero, and would silently produce a NaN ability"
        )
    if n.ndim != 1 or n.size == 0:
        raise ValueError(f"n must be a non-empty 1-D array, got shape {n.shape}")
    if S.ndim != 2 or S.shape[0] != n.size:
        raise ValueError(f"S must be (len(n), K), got {S.shape} for n of shape {n.shape}")
    if S.shape[1] < 2:
        raise ValueError(f"need at least 2 criteria to identify a 2PL, got K={S.shape[1]}")
    if (n != np.round(n)).any():
        raise ValueError(f"every placement count must be a whole number, got {n.tolist()}")
    if (S != np.round(S)).any():
        raise ValueError(f"every pass count s_k must be a whole number with 0 <= s_k <= n, got {S.tolist()}")
    if (S < 0).any() or (S > n[:, None]).any():
        raise ValueError("every s_k must satisfy 0 <= s_k <= n")
    if (n < 1).any():
        raise ValueError("every document needs at least one placement")
    return n, S


class Criteria2PL:
    """A fitted 2PL over binary criteria, with gains derived from it.

    Construct directly from known item parameters (e.g. to reproduce published
    gains), or via :meth:`fit` to estimate them from sufficient statistics.
    """

    def __init__(
        self,
        gamma: np.ndarray,
        beta: np.ndarray,
        *,
        prior_sd: float = DEFAULT_PRIOR_SD,
        diagnostics: Criteria2PLDiagnostics | None = None,
    ) -> None:
        self.gamma = np.asarray(gamma, dtype=float)
        self.beta = np.asarray(beta, dtype=float)
        if self.gamma.ndim != 1 or self.gamma.shape != self.beta.shape or self.gamma.size < 2:
            raise ValueError("gamma and beta must be equal-length 1-D arrays of >= 2 criteria")
        if (self.gamma <= 0).any():
            raise ValueError("every discrimination gamma_k must be positive")
        self.prior_sd = float(prior_sd)
        self.diagnostics = diagnostics

    # -- construction ------------------------------------------------------

    @classmethod
    def fit(
        cls,
        n: np.ndarray,
        S: np.ndarray,
        *,
        prior_sd: float = DEFAULT_PRIOR_SD,
    ) -> Criteria2PL:
        """Fit by marginal ML on the sufficient statistics.

        The discriminations are bounded to :data:`DEFAULT_GAMMA_BOUNDS`, and the fit
        raises when one criterion takes more than :data:`DEFAULT_MAX_ITEM_WEIGHT` of
        the gain weight (degenerate fit, failure mode 1).

        Args:
            n: ``(D,)`` placement counts per document.
            S: ``(D, K)`` per-criterion pass counts, ``0 <= s_k <= n``.
            prior_sd: Gaussian prior sd on ability (1.0 is empirically best).

        Returns:
            A fitted :class:`Criteria2PL`. Check ``.diagnostics.ordinal_only``
            before treating its gains as calibrated probabilities.
        """
        try:
            from scipy.optimize import minimize
        except ImportError as exc:  # pragma: no cover - exercised only without scipy
            raise ImportError(
                "Criteria2PL.fit requires scipy (bounded L-BFGS-B). "
                "Install it directly, or via the package extra: pip install 'rcp-ndcg-core[irt]'"
            ) from exc

        n_arr, S_arr = _validate_observations(n, S)
        n_items = S_arr.shape[1]
        lo, hi = float(np.log(DEFAULT_GAMMA_BOUNDS[0])), float(np.log(DEFAULT_GAMMA_BOUNDS[1]))

        prior = np.exp(-0.5 * (QUAD / prior_sd) ** 2)
        prior /= prior.sum()

        # Collapsing to unique (n, s) patterns makes the likelihood cheap and exact.
        uniq, counts = np.unique(np.concatenate([n_arr[:, None], S_arr], axis=1), axis=0, return_counts=True)
        n_u, s_u = uniq[:, 0], uniq[:, 1:]

        weighted_s = counts[:, None] * s_u  # (U, K): c_u * s_uk
        weighted_n = counts * n_u  # (U,): c_u * n_u

        def neg_log_lik(params: np.ndarray) -> tuple[float, np.ndarray]:
            """Negative marginal log-likelihood and its exact gradient in (log gamma, beta).

            With ``z_kq = gamma_k (q - beta_k)``, ``p_kq = sigmoid(z_kq)`` clipped to
            ``[1e-9, 1 - 1e-9]`` and ``w_uq`` the posterior weight of grid point ``q``
            for pattern ``u``, the derivative of the log-likelihood in ``z_kq`` is
            ``sum_u c_u w_uq (s_uk - n_u p_kq)`` (zero where the clip binds), so
            ``d/d log gamma_k = sum_q G_kq z_kq`` and ``d/d beta_k = -gamma_k sum_q G_kq``.
            An analytic gradient keeps the optimiser's path independent of the CPU's
            rounding, which a finite-difference gradient amplifies.
            """
            gamma = np.exp(params[:n_items])
            beta = params[n_items:]
            z = gamma[:, None] * (QUAD[None, :] - beta[:, None])
            raw = sigmoid(z)
            p = np.clip(raw, 1e-9, 1 - 1e-9)
            ll = (s_u[:, :, None] * np.log(p)[None] + (n_u[:, None, None] - s_u[:, :, None]) * np.log(1 - p)[None]).sum(
                1
            )
            m = ll.max(1, keepdims=True)
            unnormalised = np.exp(ll - m) * prior[None]
            total = unnormalised.sum(1)
            marginal = np.log(total) + m[:, 0]
            weights = unnormalised / total[:, None]  # (U, Q) posterior over the grid
            g = weighted_s.T @ weights - p * (weighted_n @ weights)[None, :]  # (K, Q)
            g = np.where((raw > 1e-9) & (raw < 1 - 1e-9), g, 0.0)
            grad = np.concatenate([-(g * z).sum(1), gamma * g.sum(1)])
            return -float((counts * marginal).sum()), grad

        x0 = np.concatenate([np.zeros(n_items), np.linspace(1.0, 4.0, n_items)])
        bounds = [(lo, hi)] * n_items + [(-10.0, 10.0)] * n_items
        result = minimize(neg_log_lik, x0, jac=True, method="L-BFGS-B", bounds=bounds, options={"maxiter": 800})
        gamma = np.exp(result.x[:n_items])
        beta = result.x[n_items:]

        weights = gamma / gamma.sum()
        max_weight = float(weights.max())
        if max_weight > DEFAULT_MAX_ITEM_WEIGHT:
            worst = int(weights.argmax()) + 1
            raise ValueError(
                f"degenerate 2PL fit: criterion C{worst} holds {max_weight:.4f} of the gain weight "
                f"(gamma={gamma.round(3).tolist()}). Its pass rate is probably too low to identify a "
                f"discrimination; drop the criterion or widen the input."
            )

        fitted = cls(gamma, beta, prior_sd=prior_sd)

        gamma_saturated = bool((gamma >= DEFAULT_GAMMA_BOUNDS[1] * 0.999).all())
        frac_passing = float((S_arr.sum(axis=1) > 0).mean())

        reasons: list[str] = []
        if gamma_saturated:
            reasons.append(
                "every discrimination saturated at the bound: too few positive observations "
                "to identify the 2PL, the gain is effectively binary and NOT a calibrated probability"
            )
        if frac_passing < MIN_FRAC_PASSING_ANY:
            reasons.append(
                f"only {frac_passing:.4f} of documents pass any criterion "
                f"(identifiability floor {MIN_FRAC_PASSING_ANY})"
            )
        ordinal_only = bool(reasons)

        fitted.diagnostics = Criteria2PLDiagnostics(
            n_documents=int(n_arr.size),
            mean_placements=float(n_arr.mean()),
            max_item_weight=max_weight,
            gamma_saturated=gamma_saturated,
            frac_passing_any=frac_passing,
            ordinal_only=ordinal_only,
            reason="; ".join(reasons) if reasons else None,
        )
        return fitted

    # -- fitted surface ----------------------------------------------------

    @property
    def item_params(self) -> dict:
        """``{"gamma", "beta", "num_criteria"}`` as plain lists."""
        return {
            "gamma": [float(g) for g in self.gamma],
            "beta": [float(b) for b in self.beta],
            "num_criteria": int(self.gamma.size),
        }

    def posterior(self, n: np.ndarray, S: np.ndarray) -> np.ndarray:
        """``(D, len(QUAD))`` posterior over ability for each document."""
        n_arr, S_arr = _validate_observations(n, S)
        return self._posterior(n_arr[:, None], S_arr[:, None, :], np.zeros(1))

    def _posterior(self, n: np.ndarray, S: np.ndarray, offsets: np.ndarray) -> np.ndarray:
        """``(D, len(QUAD))`` posterior of each document from groups of evidence, each with a logit offset.

        ``n`` is ``(D, G)`` placements, ``S`` is ``(D, G, K)`` pass counts and ``offsets`` is ``(G,)``: in
        group ``g`` the logit of criterion ``k`` is ``gamma_k * (theta - beta_k) - offsets[g]``.
        """
        prior = np.exp(-0.5 * (QUAD / self.prior_sd) ** 2)
        prior /= prior.sum()
        logits = self.gamma[None, :, None] * (QUAD[None, None, :] - self.beta[None, :, None]) - offsets[:, None, None]
        p = np.clip(sigmoid(logits), 1e-9, 1 - 1e-9)
        terms = S[..., None] * np.log(p)[None] + (n[:, :, None, None] - S[..., None]) * np.log(1 - p)[None]
        ll = terms.sum(2).sum(1)  # criteria first, then groups: one group sums exactly as a plain posterior
        m = ll.max(1, keepdims=True)
        posterior = np.exp(ll - m) * prior[None]
        posterior /= posterior.sum(1, keepdims=True)
        return posterior

    def eap(self, n: np.ndarray, S: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``(theta, sd)`` per document: the posterior mean ability and the posterior standard deviation."""
        return _moments(self.posterior(n, S))

    def score_document(self, n, S, *, se_target: float = DEFAULT_SE_TARGET, offsets=None) -> DocumentEstimate:
        """Score ONE document from its own criterion responses, items held fixed.

        The ability is estimated by EAP under this instance's prior with
        ``(gamma, beta)`` frozen, so the estimate lands on the same ability scale
        the fit's calibrated thetas use -- the scale cannot move because nothing
        is re-estimated. The evidence may come in groups, one per judge of a
        pooled fit: in group ``g`` every logit is ``gamma_k * (theta - beta_k)
        - offsets[g]`` (the judge's severity), and the groups multiply into one
        likelihood. The identifiability predicates are checked, not assumed:

        * degeneracy: ``0 < total passes < total observations``. When it fails
          the EAP stays finite (prior-bounded) and ``flags.degenerate`` names the
          boundary instead of refusing, because the prior-driven value is exactly
          what EAP is for.
        * precision: ``information >= 1 / se_target**2``, else ``flags.low_information``.

        ``theta`` is the posterior mean, ``se`` the posterior standard deviation (prior
        included: the SE of the number reported), and ``information`` the observed Fisher
        information of the document's own responses at ``theta``, prior excluded.

        Args:
            n: Placements (each answers every criterion): an int, or with ``offsets`` one int per group.
            S: Per criterion, the placements it passed: ``(K,)``, or with ``offsets`` ``(G, K)``;
                ``0 <= s_k <= n``.
            se_target: the SE the caller wants the evidence to certify.
            offsets: ``(G,)`` logit offsets, one per group; ``None`` is one group without an offset.

        Returns:
            The :class:`~rcp_ndcg_core.schemas.DocumentEstimate`.
        """
        shift = np.zeros(1) if offsets is None else np.asarray(offsets, dtype=float)
        counts = np.asarray(n, dtype=float).reshape(-1) if offsets is not None else np.asarray([n], dtype=float)
        passes = np.asarray(S, dtype=float)
        if offsets is None:
            if counts.shape != (1,):
                raise ValueError(f"n counts placements and must be one whole number, got {n}")
            passes = passes[None, :] if passes.ndim == 1 else passes
        if shift.ndim != 1 or counts.shape != shift.shape or passes.shape != (shift.size, self.gamma.size):
            raise ValueError(
                f"n must hold one placement count per group and S one (K,) pass-count row per group (K="
                f"{self.gamma.size}); got n of shape {counts.shape}, S of shape {passes.shape}, {shift.size} offsets"
            )
        if (counts != np.round(counts)).any():
            raise ValueError(f"n counts placements and must be an integer, got {n}")
        counts, passes = _validate_observations(counts, passes)

        n_placements = int(counts.sum())
        total_passes = float(passes.sum())
        total_obs = float(n_placements * self.gamma.size)
        if total_passes == 0.0:
            degenerate = "all_fail"
        elif total_passes == total_obs:
            degenerate = "all_pass"
        else:
            degenerate = None

        means, sds = _moments(self._posterior(counts[None, :], passes[None, :, :], shift))
        theta, se = float(means[0]), float(sds[0])

        # Observed Fisher information of this document's own data at the EAP,
        # prior excluded: sum over placements and criteria of gamma_k^2 p(1-p).
        # Bernoulli information does not depend on the observed value, so each
        # group's placement count scales its per-placement information linearly.
        p_at_theta = sigmoid(self.gamma[None, :] * (theta - self.beta[None, :]) - shift[:, None])
        info_per_group = (self.gamma[None, :] ** 2 * p_at_theta * (1 - p_at_theta)).sum(1)
        information = float((info_per_group * counts).sum())
        return DocumentEstimate(
            theta=theta,
            se=se,
            information=information,
            flags=EstimateFlags(degenerate=degenerate, low_information=information < 1.0 / (se_target**2)),
        )
