"""Bradley-Terry: pairwise soft comparisons -> an ability per document (logits, mean zero)."""

from __future__ import annotations

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
        if winner in self.doc_to_idx and loser in self.doc_to_idx and winner != loser:
            label = float(soft_label) if soft_label is not None else 1.0
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
