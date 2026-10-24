"""Train the evidential critic, with an optional Opacus wrapper."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset, Subset

from haemodiff.config import load_config, stage_complete
from haemodiff.data.dataset import EchoClipDataset, assert_no_test_leak
from haemodiff.losses.evidential import critic_loss
from haemodiff.models.evidential import EvidentialCritic, split_critic_output
from haemodiff.train.engine import (
    load_checkpoint,
    load_records,
    make_loader,
    make_split_dataset,
    prepare,
    run_stage,
    save_checkpoint,
)


class SyntheticClipDataset(torch.utils.data.Dataset):
    def __init__(self, path: Path, severe_ef: float, reduced_ef: float) -> None:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        self.video = payload["video"].float()
        self.ef = payload["label_ef"].float()
        self.clinical = payload["clinical"].float()
        self.severe_ef = severe_ef
        self.reduced_ef = reduced_ef
        self.filenames = list(payload["source_filename"])
        self.clip_len = int(self.video.shape[1])
        self.size = int(self.video.shape[-1])

    def __len__(self) -> int:
        return int(self.video.shape[0])

    def __getitem__(self, index: int) -> dict:
        ef = float(self.ef[index])
        frames = self.video.shape[1]
        size = self.video.shape[-1]
        return {
            "video": self.video[index],
            "mask": torch.zeros(frames, 1, size, size),
            "valid_mask": torch.zeros(frames),
            "pad": torch.zeros(frames, dtype=torch.bool),
            "ef": torch.tensor(ef, dtype=torch.float32),
            "severe": torch.tensor(float(ef < self.severe_ef), dtype=torch.float32),
            "reduced": torch.tensor(float(ef < self.reduced_ef), dtype=torch.float32),
            "clinical": self.clinical[index],
            "ed_index": -1,
            "es_index": -1,
            "filename": f"synthetic:{self.filenames[index]}",
            "split": "SYN",
        }


def majority_indices(labels: np.ndarray, ratio: str, seed: int) -> np.ndarray:
    """Keep every minority row and subsample the majority class.

    ``1:5`` means one minority sample for every five majority samples.
    """
    labels = np.asarray(labels).astype(int)
    indices = np.arange(len(labels))
    if ratio == "natural":
        return indices
    per_minority = {"1:1": 1, "1:5": 5, "1:10": 10, "1:20": 20}[ratio]
    minority = indices[labels == 1]
    majority = indices[labels == 0]
    if len(minority) == 0:
        raise RuntimeError(f"No minority samples available for ratio {ratio}.")
    keep_majority = min(len(majority), per_minority * len(minority))
    if keep_majority == 0:
        return minority
    rng = np.random.default_rng(seed)
    chosen = rng.choice(majority, size=keep_majority, replace=False)
    return np.concatenate([minority, chosen])


def _severe_labels(dataset) -> np.ndarray:
    if isinstance(dataset, SyntheticClipDataset):
        return (dataset.ef.numpy() < dataset.severe_ef).astype(np.int64)
    if isinstance(dataset, EchoClipDataset):
        return np.asarray([float(row["ef"]) < dataset.severe_ef for row in dataset.records], dtype=np.int64)
    if isinstance(dataset, ConcatDataset):
        return np.concatenate([_severe_labels(part) for part in dataset.datasets])
    if isinstance(dataset, Subset):
        parent = _severe_labels(dataset.dataset)
        return parent[np.asarray(dataset.indices)]
    raise TypeError(type(dataset))


def build_critic_dataset(cfg: dict, records, groups, arm: str, ratio: str, fraction: float, seed: int):
    data = cfg["data"]
    real = make_split_dataset(records, groups, cfg, "TRAIN", "critic")
    if fraction < 1:
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(real))
        keep = max(1, int(round(len(real) * fraction)))
        held_out = order[keep:]
        real = Subset(real, order[:keep].tolist())
        held_out_names = [real.dataset.records[int(i)]["filename"] for i in held_out]
    else:
        held_out_names = []
    parts = [real]
    if arm == "real_uncond":
        parts.append(SyntheticClipDataset(Path(cfg["_run_dir"]) / "synthetic_severe_unconditional.pt", float(data["severe_ef"]), float(data["reduced_ef"])))
    elif arm == "real_haemodiff":
        parts.append(SyntheticClipDataset(Path(cfg["_run_dir"]) / "synthetic_severe_full.pt", float(data["severe_ef"]), float(data["reduced_ef"])))
    elif arm != "real":
        raise ValueError(arm)
    if arm != "real":
        test_ids = {row["filename"] for row in records if row["split"] == "TEST"}
        assert_no_test_leak(parts[-1].filenames, test_ids)
    dataset = parts[0] if len(parts) == 1 else ConcatDataset(parts)
    labels = _severe_labels(dataset)
    chosen = majority_indices(labels, ratio, seed)
    return Subset(dataset, chosen.tolist()), held_out_names


def train(cfg: dict, stage: str = "critic_real_natural", dp: bool = False, arm: str = "real", ratio: str = "natural", fraction: float = 1.0) -> None:
    run_dir, device, batch_size = prepare(cfg)
    cfg = dict(cfg)
    cfg["_run_dir"] = str(run_dir)
    if stage_complete(run_dir, stage):
        print(f"Skipping completed stage {stage}")
        return
    records, groups = load_records(cfg, run_dir)
    train_ds, held_out_names = build_critic_dataset(cfg, records, groups, arm, ratio, fraction, int(cfg["seed"]) + 3)
    val_ds = make_split_dataset(records, groups, cfg, "VAL", "critic")
    workers = 0 if __import__("os").name == "nt" else int(cfg["optim"]["num_workers"])
    train_loader = make_loader(train_ds, batch_size, not dp, workers)
    val_loader = make_loader(val_ds, batch_size, False, workers)
    model = EvidentialCritic(in_channels=int(cfg["data"]["clip_len"]), base_channels=int(cfg["model"]["critic"]["base_channels"])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["optim"]["lr"]), weight_decay=float(cfg["optim"]["weight_decay"]))
    engine = None
    if dp:
        from haemodiff.privacy.dp import epsilon_at, make_private_critic

        model, optimizer, train_loader, engine = make_private_critic(
            model,
            optimizer,
            train_loader,
            noise_multiplier=float(cfg["privacy"]["noise_multiplier"]),
            max_grad_norm=float(cfg["privacy"]["max_grad_norm"]),
        )
    start_epoch = 0
    state = {"best": float("inf"), "bad": 0}
    last_path = run_dir / f"{stage}.last.pt"
    if last_path.is_file():
        payload = load_checkpoint(last_path, device)
        raw_model = getattr(model, "_module", model)
        raw_model.load_state_dict(payload["model"])
        start_epoch = int(payload["epoch"]) + 1
        state = {"best": payload["best"], "bad": payload["bad"]}
    if held_out_names:
        (run_dir / f"{stage}.heldout.json").write_text(json.dumps(held_out_names), encoding="utf-8")

    def loss_on(batch) -> torch.Tensor:
        video = batch["video"].to(device).requires_grad_(True)
        parts = split_critic_output(model(video))
        return critic_loss(parts, batch["ef"].to(device), batch["severe"].to(device), batch["reduced"].to(device))

    def train_one_epoch(epoch: int) -> float:
        model.train()
        total = 0.0
        count = 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            if batch["video"].shape[0] == 0:
                empty = batch["video"].to(device).requires_grad_(True)
                model(empty).sum().backward()
                optimizer.step()
                continue
            loss = loss_on(batch)
            loss.backward()
            optimizer.step()
            total += float(loss.detach())
            count += 1
        if engine is not None:
            epsilon = epsilon_at(engine, max(len(train_ds), 1))
            curve_path = run_dir / f"{stage}.epsilon.json"
            curve = []
            if curve_path.is_file():
                curve = json.loads(curve_path.read_text(encoding="utf-8"))
            curve.append({"epoch": epoch, "epsilon": epsilon, "delta": 1.0 / max(len(train_ds), 1)})
            curve_path.write_text(json.dumps(curve), encoding="utf-8")
            print(f"  {stage} epsilon {epsilon:.4f}")
        return total / max(count, 1)

    @torch.no_grad()
    def validate() -> float:
        model.eval()
        total = 0.0
        count = 0
        for batch in val_loader:
            total += float(loss_on(batch).detach())
            count += 1
        return total / max(count, 1)

    def save_state(epoch: int, best: float, improved: bool) -> None:
        state["best"] = best
        state["bad"] = 0 if improved else int(state["bad"]) + 1
        raw_model = getattr(model, "_module", model)
        payload = {
            "model": raw_model.state_dict(),
            "epoch": epoch,
            "best": best,
            "bad": state["bad"],
            "arm": arm,
            "ratio": ratio,
            "dp": dp,
        }
        save_checkpoint(last_path, payload)
        if improved:
            save_checkpoint(run_dir / f"{stage}.best.pt", payload)

    epochs = int(cfg["train"]["dp_epochs"] if dp else cfg["train"]["critic_epochs"])
    run_stage(
        run_dir=run_dir,
        stage=stage,
        epochs=epochs,
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
    parser.add_argument("--stage", default="critic_real_natural")
    parser.add_argument("--arm", default="real")
    parser.add_argument("--ratio", default="natural")
    parser.add_argument("--dp", action="store_true")
    parser.add_argument("--fraction", type=float, default=1.0)
    args = parser.parse_args()
    train(load_config(Path(args.config)), stage=args.stage, dp=args.dp, arm=args.arm, ratio=args.ratio, fraction=args.fraction)


if __name__ == "__main__":
    main()
