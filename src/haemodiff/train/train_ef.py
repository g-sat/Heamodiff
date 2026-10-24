"""Train the ejection-fraction and mask network on real EchoNet clips."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from haemodiff.config import load_config, stage_complete
from haemodiff.losses.physiology import masked_bce, soft_dice_loss
from haemodiff.models.ef_net import EFNet
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
    stage = "ef"
    if stage_complete(run_dir, stage):
        print(f"Skipping completed stage {stage}")
        return
    records, groups = load_records(cfg, run_dir)
    train_ds = make_split_dataset(records, groups, cfg, "TRAIN", "critic")
    val_ds = make_split_dataset(records, groups, cfg, "VAL", "critic")
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise RuntimeError("EF training needs non-empty TRAIN and VAL critic splits.")
    workers = int(cfg["optim"]["num_workers"])
    train_loader = make_loader(train_ds, batch_size, True, workers)
    val_loader = make_loader(val_ds, batch_size, False, workers)
    model = EFNet(int(cfg["model"]["ef_net"]["base_channels"])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["optim"]["lr"]), weight_decay=float(cfg["optim"]["weight_decay"]))
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    start_epoch = 0
    state = {"best": float("inf"), "bad": 0}
    last_path = run_dir / f"{stage}.last.pt"
    if last_path.is_file():
        payload = load_checkpoint(last_path, device)
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        start_epoch = int(payload["epoch"]) + 1
        state = {"best": payload["best"], "bad": payload["bad"]}

    def step_batch(batch, train_mode: bool) -> torch.Tensor:
        video = batch["video"].to(device)
        target_ef = batch["ef"].to(device)
        mask = batch["mask"].to(device)
        valid = batch["valid_mask"].to(device)
        with torch.set_grad_enabled(train_mode):
            with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                pred_ef, pred_mask = model(video)
                loss = F.l1_loss(pred_ef, target_ef)
                loss = loss + soft_dice_loss(pred_mask, mask, valid) + masked_bce(pred_mask, mask, valid)
        return loss

    def train_one_epoch(epoch: int) -> float:
        model.train()
        total = 0.0
        count = 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = step_batch(batch, True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total += float(loss.detach())
            count += 1
        return total / max(count, 1)

    @torch.no_grad()
    def validate() -> float:
        model.eval()
        total = 0.0
        count = 0
        for batch in val_loader:
            pred_ef, _ = model(batch["video"].to(device))
            total += float((pred_ef - batch["ef"].to(device)).abs().mean())
            count += 1
        return total / max(count, 1)

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
        epochs=int(cfg["train"]["ef_epochs"]),
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
