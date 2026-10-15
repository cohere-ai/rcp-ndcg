"""Shared machinery of the per-query estimators: a mean-centred ability vector fitted by L-BFGS."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim


class BaseScoredModel(nn.Module):
    """An ability ``theta`` per document (zero-mean), an L2 penalty, and an L-BFGS fit.

    ``l2_reg`` defaults to the paper's Stage A Bradley-Terry ridge (1e-4, ``Priors.bt_l2``); a caller
    with a fitted calibration's own penalty passes it explicitly. ``doc_ids`` must be unique: the index
    map collapses a duplicate to one slot while the ability vector still counts it.

    Subclasses define :meth:`model_loss` and may override the ability
    parameterisation and :meth:`_compute_standard_errors`.
    """

    def __init__(self, doc_ids: list[str], l2_reg: float = 1e-4):
        super().__init__()
        if not math.isfinite(l2_reg) or l2_reg < 0:
            raise ValueError(f"l2_reg must be finite and >= 0, got {l2_reg!r}")
        seen: set[str] = set()
        duplicates: list[str] = []
        for doc_id in doc_ids:
            if doc_id in seen and doc_id not in duplicates:
                duplicates.append(doc_id)
            seen.add(doc_id)
        if duplicates:
            raise ValueError(
                f"doc_ids contains duplicate document IDs: {duplicates}. A duplicate collapses to one "
                "ability while the fit still counts it, so an orphan slot pinned at 0 shifts every theta"
            )
        self.doc_ids = list(doc_ids)
        self.n_docs = len(self.doc_ids)
        self.doc_to_idx = {d: i for i, d in enumerate(self.doc_ids)}
        self.l2_reg = l2_reg
        self.theta_raw = nn.Parameter(torch.zeros(self.n_docs))
        self.theta_se: None | torch.Tensor = None
        self.observations: list[Any] = []

    @property
    def theta(self) -> torch.Tensor:
        return self.theta_raw - self.theta_raw.mean()

    def get_scores(self) -> dict[str, float] | None:
        theta = self.theta.detach()
        return {d: float(theta[self.doc_to_idx[d]].item()) for d in self.doc_to_idx}

    def model_loss(self) -> torch.Tensor:
        raise NotImplementedError

    def regularization_loss(self) -> torch.Tensor:
        return 0.5 * self.l2_reg * torch.sum(self.theta_raw**2)

    def _has_data(self) -> bool:
        return len(self.observations) > 0

    def fit_lbfgs(self) -> None:
        """Fit by L-BFGS from the current parameters (a warm start after new observations)."""
        if not self._has_data():
            return
        optimizer = optim.LBFGS(
            [p for p in self.parameters() if p.requires_grad],
            lr=1.0,
            max_iter=250,
            tolerance_grad=1e-7,
            tolerance_change=1e-9,
            history_size=100,
            line_search_fn="strong_wolfe",
        )

        def closure():
            optimizer.zero_grad()
            total = self.model_loss() + self.regularization_loss()
            total.backward()
            return total

        # Intra-op threads split float32 reductions differently per thread count; one thread makes
        # identical inputs give identical fits on any machine (the same pin as the tournament fit).
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            optimizer.step(closure)
        finally:
            torch.set_num_threads(threads)
        self._compute_standard_errors()

    def _compute_standard_errors(self) -> None:
        self.theta_se = None

    def get_scores_with_se(self, doc_ids: list[str]) -> dict[str, tuple[float, float | None]]:
        """``{doc_id: (theta, standard error or None)}`` for the requested documents."""
        if self.theta_se is None:
            self._compute_standard_errors()
        theta = self.theta.detach()
        out: dict[str, tuple[float, float | None]] = {}
        for d in doc_ids:
            if d not in self.doc_to_idx:
                continue
            idx = self.doc_to_idx[d]
            se = None if self.theta_se is None else float(self.theta_se[idx].item())
            out[d] = (float(theta[idx].item()), se)
        return out
