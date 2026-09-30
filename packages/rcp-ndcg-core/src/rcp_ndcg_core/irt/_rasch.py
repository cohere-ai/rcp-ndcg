"""Rasch: the rubric schedule's preliminary ability, used only to stratify windows."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from rcp_ndcg_core.irt._base import BaseScoredModel
from rcp_ndcg_core.irt._calibration import verdicts


class RaschEstimator(BaseScoredModel):
    """Rasch Item Response Theory model for pointwise relevance assessment.

    Each document ``d`` has an ability parameter ``theta_d`` and each of K
    binary criteria has a difficulty parameter ``beta_k``.  The model is::

        P(Y_{dk} = 1) = sigma(theta_d - beta_k)

    The total score ``T_d = sum_k Y_{dk}`` is a sufficient statistic for
    ``theta_d``.  Final scores are mapped to ``[0, 100]`` via the Test
    Characteristic Curve (TCC)::

        Score_d = (100 / K) * sum_k sigma(theta_d - beta_k)
    """

    def __init__(
        self,
        doc_ids: list[str],
        num_criteria: int = 5,
        l2_reg: float = 1e-4,
    ):
        super().__init__(doc_ids=doc_ids, l2_reg=l2_reg)
        self.num_criteria = num_criteria
        self.beta_raw = nn.Parameter(torch.zeros(num_criteria))
        self.observations: list[tuple[int, list[int]]] = []
        self._observation_tensors: None | tuple[torch.Tensor, torch.Tensor] = None

    @property
    def theta(self) -> torch.Tensor:
        return self.theta_raw

    @property
    def beta(self) -> torch.Tensor:
        return self.beta_raw - self.beta_raw.mean()

    def add_criteria(self, doc_id: str, criteria: dict[str, int]) -> None:
        """Add a single observation: one document's K binary criteria responses."""
        if doc_id not in self.doc_to_idx:
            return
        vec = verdicts(criteria, self.num_criteria)
        self.observations.append((self.doc_to_idx[doc_id], vec))
        self._observation_tensors = None

    def _prepare_observation_tensors(self) -> None:
        if self._observation_tensors is not None:
            return
        if not self.observations:
            self._observation_tensors = (
                torch.empty(0, dtype=torch.long),
                torch.empty((0, self.num_criteria), dtype=torch.float32),
            )
            return
        doc_indices, criteria_vecs = zip(*self.observations, strict=False)
        self._observation_tensors = (
            torch.tensor(doc_indices, dtype=torch.long),
            torch.tensor(criteria_vecs, dtype=torch.float32),
        )

    def model_loss(self) -> torch.Tensor:
        if not self.observations:
            return torch.tensor(0.0)
        self._prepare_observation_tensors()
        assert self._observation_tensors is not None
        doc_indices, Y = self._observation_tensors  # (M,), (M, K)
        theta_obs = self.theta[doc_indices]  # (M,)
        logits = theta_obs[:, None] - self.beta[None, :]  # (M, K)
        return F.binary_cross_entropy_with_logits(logits, Y, reduction="mean")

    def regularization_loss(self) -> torch.Tensor:
        return 0.5 * self.l2_reg * (torch.sum(self.theta_raw**2) + torch.sum(self.beta_raw**2))

    def get_scores(self) -> dict[str, float] | None:
        """Return TCC-mapped ``[0, 100]`` scores for all documents."""
        if not self.observations:
            return None
        theta = self.theta.detach()
        beta = self.beta.detach()
        K = self.num_criteria
        scores: dict[str, float] = {}
        for d, idx in self.doc_to_idx.items():
            tcc = float((100.0 / K) * torch.sigmoid(theta[idx] - beta).sum().item())
            scores[d] = tcc
        return scores


__all__ = ["RaschEstimator"]
