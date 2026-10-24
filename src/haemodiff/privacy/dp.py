"""Critic-only differential privacy. The VAE and DiT never call this module."""

from __future__ import annotations


def make_private_critic(model, optimizer, data_loader, noise_multiplier: float, max_grad_norm: float):
    """Wrap a critic with Opacus. Raises with the rejected module names on failure."""
    from opacus import PrivacyEngine
    from opacus.validators import ModuleValidator

    errors = ModuleValidator.validate(model, strict=False)
    if errors:
        names = []
        for error in errors:
            module = getattr(error, "module_name", None) or error.__class__.__name__
            names.append(str(module))
        raise RuntimeError("Opacus rejected the critic: " + ", ".join(names))
    engine = PrivacyEngine()
    model, optimizer, data_loader = engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=data_loader,
        noise_multiplier=float(noise_multiplier),
        max_grad_norm=float(max_grad_norm),
    )
    return model, optimizer, data_loader, engine


def epsilon_at(engine, n_examples: int) -> float:
    delta = 1.0 / max(int(n_examples), 1)
    return float(engine.get_epsilon(delta))
