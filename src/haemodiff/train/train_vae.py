"""Train the per-frame VAE with ordinary AdamW."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from haemodiff.config import load_config, stage_complete
from haemodiff.models.vae import FrameVAE, vae_loss
from haemodiff.train.engine import (
    load_checkpoint,
    load_records,
    make_loader,
    make_split_dataset,
    prepare,
    run_stage,
    save_checkpoint,
)


def train(cfg: dict) -> None:
    run_dir, device, batch_size = prepare(cfg)
    stage = "vae"
    if stage_complete(run_dir, stage):
        print(f"Skipping completed stage {stage}")
        return
    records, groups = load_records(cfg, run_dir)
    train_ds = make_split_dataset(records, groups, cfg, "TRAIN", "dit")
    val_ds = make_split_dataset(records, groups, cfg, "VAL", "dit")
    from haemodiff.train.engine import require_full_dit_train

    require_full_dit_train(records, train_ds)
    workers = int(cfg["optim"]["num_workers"])
    train_loader = make_loader(train_ds, batch_size, True, workers)
    val_loader = make_loader(val_ds, batch_size, False, workers)
    model_cfg = cfg["model"]["vae"]
    model = FrameVAE(int(model_cfg["latent_channels"]), int(model_cfg["base_channels"])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["optim"]["lr"]), weight_decay=float(cfg["optim"]["weight_decay"]))
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    kl_weight = float(cfg["train"]["vae_kl_weight"])
    start_epoch = 0
    state = {"best": float("inf"), "bad": 0}
    last_path = run_dir / f"{stage}.last.pt"
    if last_path.is_file():
        payload = load_checkpoint(last_path, device)
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        start_epoch = int(payload["epoch"]) + 1
        state = {"best": payload["best"], "bad": payload["bad"]}

    def batches(loader, train_mode: bool):
        total = 0.0
        count = 0
        for batch in loader:
            video = batch["video"].to(device)
            b, t, _, h, w = video.shape
            frames = video.reshape(b * t, 1, h, w)
            with torch.set_grad_enabled(train_mode):
                with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                    recon, mu, logvar = model(frames)
                    loss, _ = vae_loss(recon, frames, mu, logvar, kl_weight)
            if train_mode:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            total += float(loss.detach())
            count += 1
        return total / max(count, 1)

    def train_one_epoch(epoch: int) -> float:
        model.train()
        return batches(train_loader, True)

    @torch.no_grad()
    def validate() -> float:
        model.eval()
        return batches(val_loader, False)

    def save_state(epoch: int, best: float, improved: bool) -> None:
        payload = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best": best,
            "bad": 0 if improved else state["bad"],
        }
        state["best"] = best
        state["bad"] = 0 if improved else state["bad"] + 1
        save_checkpoint(last_path, payload)
        if improved:
            save_checkpoint(run_dir / f"{stage}.best.pt", payload)

    run_stage(
        run_dir=run_dir,
        stage=stage,
        epochs=int(cfg["train"]["vae_epochs"]),
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
    train(load_config(Path(parser.parse_args().config)))


if __name__ == "__main__":
    main()
