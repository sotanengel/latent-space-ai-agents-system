"""Tests for HiddenStateTap.

We use a tiny stub model rather than a real Transformer so the unit tests
stay fast and CPU-only. The integration test in tests/integration covers
the real-model path.
"""

from __future__ import annotations

import torch
from torch import nn

from latent_agents.capture import HiddenStateTap


class _StubModelOutput:
    def __init__(self, hidden_states: tuple[torch.Tensor, ...]) -> None:
        self.hidden_states = hidden_states


class _StubModel(nn.Module):
    """Emits hidden_states only when `output_hidden_states=True` is set."""

    def __init__(self, n_layers: int = 3, hidden_dim: int = 8) -> None:
        super().__init__()
        self.config = type("C", (), {"hidden_size": hidden_dim, "num_hidden_layers": n_layers})()
        self._n = n_layers
        self._h = hidden_dim

    def forward(
        self,
        input_ids: torch.Tensor,
        output_hidden_states: bool = False,
        **kwargs: object,
    ) -> _StubModelOutput:
        b, s = input_ids.shape
        if not output_hidden_states:
            return _StubModelOutput(hidden_states=())
        torch.manual_seed(int(input_ids.sum().item()))
        layers = tuple(torch.randn(b, s, self._h) for _ in range(self._n + 1))
        return _StubModelOutput(hidden_states=layers)


class TestHiddenStateTap:
    def test_captures_all_layers(self) -> None:
        model = _StubModel(n_layers=3, hidden_dim=8)
        ids = torch.tensor([[1, 2, 3, 4]])
        with HiddenStateTap(model, agent_id="agent-a") as tap:
            model(ids, output_hidden_states=True)
        assert len(tap.snapshots) == 4  # embeddings + 3 layers

    def test_each_snapshot_has_correct_shape_metadata(self) -> None:
        model = _StubModel(n_layers=2, hidden_dim=8)
        ids = torch.tensor([[1, 2, 3, 4]])
        with HiddenStateTap(model, agent_id="agent-a") as tap:
            model(ids, output_hidden_states=True)
        for snap in tap.snapshots:
            assert snap.tensor.shape == (1, 4, 8)
            assert snap.tensor.dtype == "fp32"

    def test_reproducibility_same_input(self) -> None:
        """FR-LG-007: identical inputs yield identical hidden_states."""
        model = _StubModel(n_layers=2, hidden_dim=8)
        ids = torch.tensor([[7, 7, 7]])
        with HiddenStateTap(model) as t1:
            model(ids, output_hidden_states=True)
        with HiddenStateTap(model) as t2:
            model(ids, output_hidden_states=True)
        for a, b in zip(t1.tensors, t2.tensors, strict=True):
            assert torch.equal(a, b)

    def test_detects_nan_in_hidden_states(self) -> None:
        """FR-LG-004: NaN must be flagged in numerical properties."""
        model = _StubModel(n_layers=1, hidden_dim=4)
        # Monkey-patch model to emit NaN-laden hidden states.
        bad = torch.full((1, 2, 4), float("nan"))
        with HiddenStateTap(model, agent_id="a") as tap:
            tap._on_hidden_states((bad, bad))  # type: ignore[attr-defined]
        assert all(s.numerical_properties.has_nan for s in tap.snapshots)

    def test_observer_does_not_mutate_outputs(self) -> None:
        model = _StubModel(n_layers=2, hidden_dim=4)
        ids = torch.tensor([[1, 2, 3]])
        with HiddenStateTap(model) as tap:
            out = model(ids, output_hidden_states=True)
        # The tap copies tensors, so mutating captured tensors must not
        # touch the live model outputs.
        tap.tensors[0].zero_()
        assert not torch.all(out.hidden_states[0] == 0)
