"""Train a conditioned video DiT with AdamW. Privacy accounting is not used here."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from haemodiff.config import load_config
from haemodiff.losses.diffusion import DiffusionSchedule, diffusion_mse
from haemodiff.losses.physiology import PhysiologyBalancer, physiology_terms
from haemodiff.models.dit import VideoDiT, apply_condition_mode
from haemodiff.models.ef_net import EFNet
from haemodiff.models.vae import FrameVAE
from haemodiff.train.engine import (
    load_checkpoint,
    load_records,
    make_loader,
    make_split_dataset,
    prepare,
    require_full_dit_train,
    run_stage,
    save_checkpoint,
)
from haemodiff.config import stage_complete


def _load_frozen(path: Path, module: torch.nn.Module, device: torch.device) -> torch.nn.Module:
    payload = load_checkpoint(path, device)
    module.load_state_dict(payload["model"])
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    return module


def train(cfg: dict, mode: str = "full", stage: str | None = None, fraction: float = 1.0) -> None:
    if mode not in {"unconditional", "clinical", "mask", "full"}:
        raise ValueError(mode)
    run_dir, device, batch_size = prepare(cfg)
    stage = stage or f"dit_{mode}"
    if stage_complete(run_dir, stage):
        print(f"Skipping completed stage {stage}")
        return
    records, groups = load_records(cfg, run_dir)
    train_ds = make_split_dataset(records, groups, cfg, "TRAIN", "dit")
    val_ds = make_split_dataset(records, groups, cfg, "VAL", "dit")
    if fraction < 1:
        keep = max(1, int(len(train_ds) * fraction))
        held_out = [row["filename"] for row in train_ds.records[keep:]]
        train_ds.records = train_ds.records[:keep]
        train_ds.clinical = train_ds.clinical[:keep]
        (run_dir / f"{stage}.heldout.json").write_text(json.dumps(held_out), encoding="utf-8")
    else:
        require_full_dit_train(records, train_ds)
    workers = int(cfg["optim"]["num_workers"])
    train_loader = make_loader(train_ds, batch_size, True, workers)
    val_loader = make_loader(val_ds, batch_size, False, workers)
    vae_cfg = cfg["model"]["vae"]
    dit_cfg = cfg["model"]["dit"]
    vae = _load_frozen(
        run_dir / "vae.best.pt",
        FrameVAE(int(vae_cfg["latent_channels"]), int(vae_cfg["base_channels"])),
        device,
    )
    ef_net = _load_frozen(
        run_dir / "ef.best.pt",
        EFNet(int(cfg["model"]["ef_net"]["base_channels"])),
        device,
    )
    model = VideoDiT(
        latent_channels=int(vae_cfg["latent_channels"]),
        hidden=int(dit_cfg["hidden"]),
        depth=int(dit_cfg["depth"]),
        heads=int(dit_cfg["heads"]),
        patch=int(dit_cfg["patch"]),
        mlp_ratio=float(dit_cfg["mlp_ratio"]),
        grad_checkpoint=True,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["optim"]["lr"]), weight_decay=float(cfg["optim"]["weight_decay"]))
    schedule = DiffusionSchedule(
        int(cfg["diffusion"]["timesteps"]),
        float(cfg["diffusion"]["beta_start"]),
        float(cfg["diffusion"]["beta_end"]),
    ).to(device)
    phys_cfg = cfg["phys"]
    balancer = PhysiologyBalancer(
        warmup_steps=int(phys_cfg["warmup_steps"]),
        ramp_steps=int(phys_cfg["ramp_steps"]),
        max_weight=float(phys_cfg["max_weight"]),
        patience=int(phys_cfg["patience"]),
        ema_decay=float(phys_cfg["ema_decay"]),
    )
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    use_ef = mode in {"clinical", "full"}
    use_mask = mode in {"mask", "full"}
    active = tuple(key for key, flag in (("ef", use_ef), ("dice", use_mask), ("tv", True)) if flag)
    t_phys = int(cfg["diffusion"]["t_phys"])
    start_epoch = 0
    state = {"best": float("inf"), "bad": 0}
    last_path = run_dir / f"{stage}.last.pt"
    if last_path.is_file():
        payload = load_checkpoint(last_path, device)
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        balancer.load_state_dict(payload["balancer"])
        start_epoch = int(payload["epoch"]) + 1
        state = {"best": payload["best"], "bad": payload["bad"]}

    def forward_loss(batch, train_mode: bool) -> torch.Tensor:
        video = batch["video"].to(device)
        with torch.no_grad():
            latent = vae.encode_video(video, sample=False)
        clinical, ef, mask, valid = apply_condition_mode(
            batch["clinical"].to(device),
            batch["ef"].to(device),
            batch["mask"].to(device),
            batch["valid_mask"].to(device),
            mode,
        )
        noise = torch.randn_like(latent)
        timesteps = torch.randint(0, schedule.timesteps, (latent.shape[0],), device=device)
        noisy = schedule.q_sample(latent, timesteps, noise)
        with torch.set_grad_enabled(train_mode):
            with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                predicted = model(noisy, timesteps, clinical, ef, mask, valid)
                denoising = diffusion_mse(predicted, noise)
                low = timesteps < t_phys
                if train_mode and bool(low.any()):
                    x0 = schedule.predict_x0(noisy, timesteps, predicted)[low]
                    decoded = vae.decode_video(x0)
                    pred_ef, pred_mask = ef_net(decoded)
                    terms = physiology_terms(
                        decoded,
                        pred_mask,
                        pred_ef,
                        ef[low],
                        mask[low],
                        valid[low],
                        batch["pad"].to(device)[low],
                        use_ef,
                        use_mask,
                    )
                    phys = balancer(denoising, terms, active)
                    return denoising + phys
                return denoising

    def train_one_epoch(epoch: int) -> float:
        model.train()
        total = 0.0
        count = 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = forward_loss(batch, True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total += float(loss.detach())
            count += 1
            if count % int(cfg["optim"]["log_every"]) == 0:
                print(
                    f"  {stage} step {count} phys_fraction {balancer.last_fraction:.4f} "
                    f"terms {balancer.last_contributions}"
                )
        return total / max(count, 1)

    @torch.no_grad()
    def validate() -> float:
        model.eval()
        total = 0.0
        count = 0
        for batch in val_loader:
            total += float(forward_loss(batch, False).detach())
            count += 1
        return total / max(count, 1)

    def save_state(epoch: int, best: float, improved: bool) -> None:
        payload = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "balancer": balancer.state_dict(),
            "epoch": epoch,
            "best": best,
            "bad": 0 if improved else state["bad"],
            "mode": mode,
        }
        state["best"] = best
        state["bad"] = 0 if improved else state["bad"] + 1
        save_checkpoint(last_path, payload)
        if improved:
            save_checkpoint(run_dir / f"{stage}.best.pt", payload)

    run_stage(
        run_dir=run_dir,
        stage=stage,
        epochs=int(cfg["train"]["dit_epochs"]),
        patience=int(cfg["train"]["early_stop_patience"]),
        start_epoch=start_epoch,
        state=state,
        train_one_epoch=train_one_epoch,
        validate=validate,
        save_state=save_state,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/kaggle.yaml")
    parser.add_argument("--mode", default="full")
    parser.add_argument("--stage", default=None)
    parser.add_argument("--fraction", type=float, default=1.0)
    args = parser.parse_args()
    train(load_config(Path(args.config)), mode=args.mode, stage=args.stage, fraction=args.fraction)


if __name__ == "__main__":
    main()
