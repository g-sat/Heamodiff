"""Train the ejection-fraction and mask network on real EchoNet clips."""

from __future__ import annotations

import argparse
import math
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

EF_ARCH = "ef-temporal-v2"


def _adamw(model: torch.nn.Module, lr: float, weight_decay: float) -> torch.optim.AdamW:
    decay, plain = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim == 1 or name.endswith(".bias"):
            plain.append(param)
        else:
            decay.append(param)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": plain, "weight_decay": 0.0},
        ],
        lr=lr,
    )


def _cosine_warmup(optimizer: torch.optim.Optimizer, epochs: int, warmup_epochs: int = 1):
    def scale(epoch: int) -> float:
        warm = max(warmup_epochs, 1)
        if epoch < warm:
            return float(epoch + 1) / float(warm + 1)
        progress = (epoch - warmup_epochs) / float(max(epochs - warmup_epochs, 1))
        progress = min(1.0, max(0.0, progress))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return 0.05 + 0.95 * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


def train(cfg: dict) -> None:
    run_dir, device, batch_size = prepare(cfg)
    stage = "ef"
    ef_cfg = cfg["model"]["ef_net"]
    best_path = run_dir / f"{stage}.best.pt"
    if stage_complete(run_dir, stage) and best_path.is_file():
        finished = load_checkpoint(best_path, device)
        if finished.get("arch") == EF_ARCH:
            print(f"Skipping completed stage {stage}")
            return
        print("EF checkpoint is from the previous network. Training the stronger one.")
    records, groups = load_records(cfg, run_dir)
    train_ds = make_split_dataset(records, groups, cfg, "TRAIN", "critic", cover_annotation=True)
    val_ds = make_split_dataset(records, groups, cfg, "VAL", "critic", cover_annotation=True)
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise RuntimeError("EF training needs non-empty TRAIN and VAL critic splits.")
    workers = int(cfg["optim"]["num_workers"])
    ef_batch = int(ef_cfg.get("batch_size", batch_size))
    train_loader = make_loader(train_ds, ef_batch, True, workers)
    val_loader = make_loader(val_ds, ef_batch, False, workers)
    model = EFNet(int(ef_cfg["base_channels"])).to(device)
    lr = float(ef_cfg.get("lr", cfg["optim"]["lr"]))
    optimizer = _adamw(model, lr, float(cfg["optim"]["weight_decay"]))
    epochs = int(cfg["train"]["ef_epochs"])
    scheduler = _cosine_warmup(optimizer, epochs)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    grad_clip = float(ef_cfg.get("grad_clip", 1.0))
    mask_weight = float(ef_cfg.get("mask_weight", 10.0))
    start_epoch = 0
    state = {"best": float("inf"), "bad": 0}
    last_path = run_dir / f"{stage}.last.pt"
    if last_path.is_file():
        payload = load_checkpoint(last_path, device)
        if payload.get("arch") == EF_ARCH:
            model.load_state_dict(payload["model"])
            optimizer.load_state_dict(payload["optimizer"])
            if "scheduler" in payload:
                scheduler.load_state_dict(payload["scheduler"])
            start_epoch = int(payload["epoch"]) + 1
            state = {"best": payload["best"], "bad": payload["bad"]}
        else:
            print("Ignoring the in-progress EF checkpoint from the previous network.")
    print(f"ef batch_size {ef_batch} lr {lr:.6f} mask_weight {mask_weight}")

    def step_batch(batch, train_mode: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        video = batch["video"].to(device, non_blocking=True)
        target_ef = batch["ef"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        valid = batch["valid_mask"].to(device, non_blocking=True)
        with torch.set_grad_enabled(train_mode):
            with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                pred_ef, pred_mask = model(video)
                ef_loss = F.l1_loss(pred_ef, target_ef)
                mask_loss = soft_dice_loss(pred_mask, mask, valid) + masked_bce(pred_mask, mask, valid)
                loss = ef_loss + mask_weight * mask_loss
        return loss, ef_loss.detach(), mask_loss.detach()

    def train_one_epoch(epoch: int) -> float:
        model.train()
        total = 0.0
        ef_total = 0.0
        mask_total = 0.0
        count = 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss, ef_loss, mask_loss = step_batch(batch, True)
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            total += float(loss.detach())
            ef_total += float(ef_loss)
            mask_total += float(mask_loss)
            count += 1
        print(
            f"  ef parts l1 {ef_total / max(count, 1):.3f} "
            f"mask {mask_total / max(count, 1):.3f} "
            f"lr {optimizer.param_groups[0]['lr']:.6f}"
        )
        scheduler.step()
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
            "arch": EF_ARCH,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
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
