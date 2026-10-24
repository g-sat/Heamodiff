"""Evidential vacuity, critic-only Opacus, and the DiT privacy boundary."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from haemodiff.losses.evidential import dirichlet_loss
from haemodiff.models.evidential import EvidentialCritic, split_critic_output, vacuity
from haemodiff.privacy.dp import make_private_critic
from haemodiff.privacy.mia import loss_threshold_auc, shadow_attack_advantage


def test_untrained_head_is_more_vacuous_than_a_fitted_head():
    torch.manual_seed(0)
    layer = nn.Linear(2, 2)
    nn.init.zeros_(layer.weight)
    nn.init.constant_(layer.bias, -4.0)
    features = torch.tensor([[-1.0, -1.0], [1.0, 1.0], [-1.2, -0.8], [0.8, 1.1]])
    labels = torch.tensor([0, 1, 0, 1])
    untrained = vacuity(
        split_critic_output(torch.cat([layer(features).detach(), torch.zeros(4, 6)], dim=1))["severe_alpha"]
    ).mean()
    optimizer = torch.optim.Adam(layer.parameters(), lr=0.2)
    for _ in range(80):
        optimizer.zero_grad()
        alpha = torch.nn.functional.softplus(layer(features)) + 1
        loss = dirichlet_loss(alpha, labels)
        loss.backward()
        optimizer.step()
    trained = vacuity(torch.nn.functional.softplus(layer(features)).detach() + 1).mean()
    assert float(trained) < float(untrained)


def test_membership_inference_on_toy_losses():
    memorized = loss_threshold_auc(np.zeros(40), np.full(40, 5.0))
    assert memorized > 0.9
    chance = loss_threshold_auc(np.linspace(0, 1, 40), np.linspace(0, 1, 40))
    assert 0.4 < chance < 0.6
    advantage = shadow_attack_advantage(np.zeros(30), np.ones(30), np.zeros(30), np.ones(30))
    assert advantage > 0.3


def test_opacus_steps_a_critic_and_not_the_dit_module():
    torch.manual_seed(0)
    model = EvidentialCritic(in_channels=16, base_channels=32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    data = TensorDataset(
        torch.randn(4, 16, 32, 32),
        torch.rand(4) * 60,
        torch.randint(0, 2, (4,)).float(),
        torch.randint(0, 2, (4,)).float(),
    )
    loader = DataLoader(data, batch_size=2)
    model, optimizer, loader, engine = make_private_critic(model, optimizer, loader, noise_multiplier=0.5, max_grad_norm=1.0)
    batch = next(iter(loader))
    loss = model(batch[0]).sum()
    loss.backward()
    optimizer.step()
    assert engine.get_epsilon(1 / 4) > 0
    source = (Path(__file__).resolve().parents[1] / "src" / "haemodiff" / "train" / "train_dit.py").read_text(encoding="utf-8")
    assert "import opacus" not in source.lower()
    assert "privacyengine" not in source.lower()
    for name in list(sys.modules):
        if name == "haemodiff.train.train_dit" or name.startswith("haemodiff.train.train_dit."):
            del sys.modules[name]
    sys.modules.pop("opacus", None)
    import haemodiff.train.train_dit as dit_module

    assert "opacus" not in sys.modules
    assert not hasattr(dit_module, "PrivacyEngine")
