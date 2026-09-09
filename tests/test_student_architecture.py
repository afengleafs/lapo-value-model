from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from lapo_value_model.checkpoint_init import transplant_non_value_state
from lapo_value_model.student import (
    ARCHITECTURE_VERSION,
    MLPHead,
    QwenValueModel,
    VALUE_PATH_DIRECT,
    VALUE_PATH_LATENT_HIDDEN,
    architecture_metadata,
    checkpoint_value_path,
    student_loss,
)


class _TinyBackbone(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(32, hidden_size)

    def forward(self, *, input_ids, **kwargs):
        return SimpleNamespace(hidden_states=[self.embedding(input_ids)])


def _tiny_model(value_path: str) -> QwenValueModel:
    model = QwenValueModel.__new__(QwenValueModel)
    nn.Module.__init__(model)
    hidden_size = 16
    model.backbone = _TinyBackbone(hidden_size)
    model.value_path = value_path
    model.value_feature_dim = 128 if value_path == VALUE_PATH_LATENT_HIDDEN else hidden_size
    model.value_head = MLPHead([model.value_feature_dim, 12, 8, 7], dropout=0.0)
    model.latent_head = MLPHead([hidden_size, 10, 128, 4], dropout=0.0)
    model.register_buffer("value_bin_centers", torch.linspace(-1.0, 0.0, 7))
    return model


def _inputs() -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.tensor([[1, 2, 0], [3, 4, 5]]),
        "attention_mask": torch.tensor([[1, 1, 0], [1, 1, 1]]),
    }


def test_mlp_head_exposes_penultimate_without_changing_forward() -> None:
    torch.manual_seed(3)
    head = MLPHead([6, 5, 4], dropout=0.0)
    value = torch.randn(2, 6)
    output, hidden = head.forward_with_penultimate(value)
    assert hidden.shape == (2, 5)
    torch.testing.assert_close(output, head(value))
    torch.testing.assert_close(output, head.project(head.penultimate(value)))


def test_direct_and_latent_hidden_value_paths_have_stable_output_contract() -> None:
    for value_path in (VALUE_PATH_DIRECT, VALUE_PATH_LATENT_HIDDEN):
        model = _tiny_model(value_path)
        output = model(**_inputs())
        assert output["value_logits"].shape == (2, 7)
        assert output["value"].shape == (2,)
        assert output["latent"].shape == (2, 4)
        assert torch.all(output["value"] >= -1.0)
        assert torch.all(output["value"] <= 0.0)


def test_latent_hidden_value_gradient_stops_before_32d_projection() -> None:
    torch.manual_seed(7)
    model = _tiny_model(VALUE_PATH_LATENT_HIDDEN)
    targets = {
        "value_bin": torch.tensor([1, 5]),
        "latent": torch.randn(2, 4),
    }

    output = model(**_inputs())
    loss, _ = student_loss(output, targets, lambda_latent=0.0)
    loss.backward()
    assert model.latent_head.network[1].weight.grad is not None
    assert model.latent_head.network[4].weight.grad is not None
    assert model.latent_head.network[7].weight.grad is None

    model.zero_grad(set_to_none=True)
    output = model(**_inputs())
    loss, _ = student_loss(output, targets, lambda_latent=0.1)
    loss.backward()
    projection_grad = model.latent_head.network[7].weight.grad
    assert projection_grad is not None
    assert torch.isfinite(projection_grad).all()
    assert projection_grad.abs().sum() > 0


def test_direct_value_gradient_does_not_update_latent_head() -> None:
    model = _tiny_model(VALUE_PATH_DIRECT)
    output = model(**_inputs())
    loss, _ = student_loss(
        output, {"value_bin": torch.tensor([2, 4])}, lambda_latent=0.0
    )
    loss.backward()
    assert all(parameter.grad is None for parameter in model.latent_head.parameters())


def test_checkpoint_architecture_defaults_legacy_to_direct() -> None:
    assert checkpoint_value_path({"trainable_model": {}}) == VALUE_PATH_DIRECT
    assert checkpoint_value_path(
        {"architecture": {"value_path": VALUE_PATH_LATENT_HIDDEN}}
    ) == VALUE_PATH_LATENT_HIDDEN
    model = _tiny_model(VALUE_PATH_LATENT_HIDDEN)
    metadata = architecture_metadata(model)
    assert metadata == {
        "version": ARCHITECTURE_VERSION,
        "value_path": VALUE_PATH_LATENT_HIDDEN,
        "value_feature_dim": 128,
        "latent_dim": 4,
    }


def test_transplant_copies_everything_except_value_head() -> None:
    torch.manual_seed(11)
    source = _tiny_model(VALUE_PATH_DIRECT)
    target = _tiny_model(VALUE_PATH_LATENT_HIDDEN)
    source_state = {
        name: torch.full_like(parameter, 0.25)
        for name, parameter in source.named_parameters()
        if parameter.requires_grad
    }
    original_value = {
        name: parameter.detach().clone()
        for name, parameter in target.named_parameters()
        if name.startswith("value_head.")
    }
    copied, reset = transplant_non_value_state(target, source_state)
    assert copied and reset
    assert all(not name.startswith("value_head.") for name in copied)
    assert all(name.startswith("value_head.") for name in reset)
    for name, parameter in target.named_parameters():
        if name in copied:
            torch.testing.assert_close(parameter, torch.full_like(parameter, 0.25))
        elif name in reset:
            torch.testing.assert_close(parameter, original_value[name])
