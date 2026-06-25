"""Linear probing classifier (FR-SM-002, FR-SM-004).

Implements a small linear classifier in pure torch — no sklearn dependency.
Trains with SGD on a single linear layer (no hidden); good enough to
demonstrate whether a semantic attribute is linearly recoverable from
hidden states, which is the canonical probing protocol.
"""

from __future__ import annotations

import torch
from torch import nn

from latent_agents.schemas import Verdict
from latent_agents.verifier import BaseVerifier, register


class ProbingClassifier:
    """Linear probe over latent representations."""

    def __init__(self, num_classes: int, lr: float = 1e-1) -> None:
        if num_classes < 2:
            raise ValueError("num_classes must be >= 2")
        self.num_classes = num_classes
        self.lr = lr
        self._head: nn.Linear | None = None

    def fit(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        *,
        epochs: int = 50,
    ) -> ProbingClassifier:
        if x.ndim != 2:
            raise ValueError(f"x must be 2D [n, dim], got {tuple(x.shape)}")
        if x.shape[0] != y.shape[0]:
            raise ValueError("x and y must share batch dim")
        head = nn.Linear(x.shape[1], self.num_classes)
        opt = torch.optim.SGD(head.parameters(), lr=self.lr)
        loss_fn = nn.CrossEntropyLoss()
        x_in = x.float()
        y_in = y.long()
        for _ in range(epochs):
            opt.zero_grad()
            logits = head(x_in)
            loss = loss_fn(logits, y_in)
            loss.backward()
            opt.step()
        self._head = head
        return self

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        if self._head is None:
            raise RuntimeError("call fit() before predict()")
        with torch.no_grad():
            return torch.argmax(self._head(x.float()), dim=1)

    def score(self, x: torch.Tensor, y: torch.Tensor) -> float:
        pred = self.predict(x)
        return float((pred == y.long()).float().mean().item())


@register
class ProbingAccuracyVerifier(BaseVerifier[tuple[torch.Tensor, torch.Tensor]]):
    """FR-SM-002: probing accuracy on a semantic attribute must be >= min_accuracy."""

    test_id = "FR-SM-002"

    def __init__(
        self,
        min_accuracy: float = 0.85,
        epochs: int = 50,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        self.threshold = min_accuracy
        self.epochs = epochs
        self.num_classes = num_classes

    def _score(self, target: tuple[torch.Tensor, torch.Tensor]) -> float:
        x, y = target
        clf = ProbingClassifier(num_classes=self.num_classes)
        clf.fit(x, y, epochs=self.epochs)
        return clf.score(x, y)

    def _evaluate(self, target: tuple[torch.Tensor, torch.Tensor]) -> tuple[Verdict, str | None]:
        acc = self._score(target)
        if acc < float(self.threshold):
            return Verdict.FAIL, f"probing accuracy {acc:.4f} < {self.threshold}"
        return Verdict.PASS, None

    def _measured(self, target: tuple[torch.Tensor, torch.Tensor]) -> float:
        return self._score(target)
