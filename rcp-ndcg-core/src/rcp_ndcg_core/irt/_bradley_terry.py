"""Bradley-Terry: pairwise soft comparisons -> an ability per document (logits, mean zero)."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from rcp_ndcg_core.irt._base import BaseScoredModel


class BradleyTerryEstimator(BaseScoredModel):
    """Bradley-Terry over weighted soft comparisons.

    Observations are tuples ``(winner_idx, loser_idx, weight, soft_label)``;
    ``soft_label`` is the probability that the winner is better (1.0 = a hard
    win). The loss is the weighted binary cross-entropy of
    ``sigmoid(theta_winner - theta_loser)`` against the label, normalised by the
    total weight.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.observations: list[tuple[int, int, float, float]] = []
        self._observation_tensors: None | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] = None

    def add_comparison(self, winner: str, loser: str, weight: float = 1.0, soft_label: float | None = None):
        """Record one weighted soft comparison.

        Nothing is dropped: an observation the estimator cannot attach (an unknown id, a
        self-pair) or that is malformed (a non-positive weight, a ``soft_label`` outside
        ``[0, 1]``) is refused, because a silently dropped comparison fits a weaker model
        and reports standard errors from less information than the data held.

        Raises:
            ValueError: a comparison names a document outside ``doc_ids``, compares a document
                with itself, or carries a non-finite/non-positive ``weight`` or a ``soft_label``
                outside ``[0, 1]``.
        """
        unknown = [doc for doc in (winner, loser) if doc not in self.doc_to_idx]
        if unknown:
            raise ValueError(
                f"comparison ({winner!r}, {loser!r}) names unknown document(s) {unknown}: a comparison the "
                "estimator cannot attach is silently dropped evidence. An id the fit does not know (an "
                "id-format mismatch such as chunk id vs document id) must be named in doc_ids"
            )
        if winner == loser:
            raise ValueError(
                f"comparison ({winner!r}, {loser!r}) compares a document with itself: it carries no "
                "evidence about any difference and is refused rather than dropped"
            )
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError(f"comparison weight must be finite and > 0, got {weight!r}")
        label = 1.0 if soft_label is None else float(soft_label)
        if not math.isfinite(label) or not 0.0 <= label <= 1.0:
            raise ValueError(f"soft_label must be a probability in [0, 1], got {soft_label!r}")
        self.observations.append((self.doc_to_idx[winner], self.doc_to_idx[loser], float(weight), label))
        self._observation_tensors = None

    def _prepare_observation_tensors(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Materialise (and cache) the observations as tensors.

        Returns them rather than only filling the cache, so callers hold a value
        that is known to exist instead of re-reading an optional attribute.
        """
        if self._observation_tensors is not None:
            return self._observation_tensors
        if not self.observations:
            self._observation_tensors = (
                torch.empty(0, dtype=torch.long),
                torch.empty(0, dtype=torch.long),
                torch.empty(0, dtype=torch.float32),
                torch.empty(0, dtype=torch.float32),
            )
            return self._observation_tensors
        winners, losers, weights, targets = zip(*self.observations, strict=False)
        self._observation_tensors = (
            torch.tensor(winners, dtype=torch.long),
            torch.tensor(losers, dtype=torch.long),
            torch.tensor(weights, dtype=torch.float32),
            torch.tensor(targets, dtype=torch.float32),
        )
        return self._observation_tensors

    def model_loss(self) -> torch.Tensor:
        if not self.observations:
            return torch.tensor(0.0)
        winners, losers, weights, targets = self._prepare_observation_tensors()
        diff = self.theta[winners] - self.theta[losers]
        return F.binary_cross_entropy_with_logits(diff, targets, weight=weights, reduction="sum") / torch.clamp(
            weights.sum(), min=1e-9
        )

    def _compute_standard_errors(self):
        if not self.observations:
            return
        winners, losers, weights, _targets = self._prepare_observation_tensors()
        p = torch.sigmoid(self.theta[winners] - self.theta[losers])
        contrib = weights * p * (1 - p)
        h_diag = torch.zeros(self.n_docs)
        h_diag.index_add_(0, winners, contrib)
        h_diag.index_add_(0, losers, contrib)
        # Per-document information referent: a document's SE is
        # ``1 / sqrt(sum of weight*p*(1-p) over the comparisons it took part in + l2_reg)``,
        # independent of how many unrelated comparisons the fit holds.
        h_diag = h_diag + self.l2_reg
        self.theta_se = 1.0 / torch.sqrt(torch.clamp(h_diag, min=1e-9))


__all__ = ["BradleyTerryEstimator"]
