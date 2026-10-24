"""Draw full synthetic cohorts from a trained DiT. No step cap."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from haemodiff.config import load_config, stage_complete, write_complete
from haemodiff.data.dataset import assert_no_test_leak
from haemodiff.losses.diffusion import DiffusionSchedule
from haemodiff.models.dit import VideoDiT, apply_condition_mode
from haemodiff.models.vae import FrameVAE
from haemodiff.train.engine import load_checkpoint, load_records, make_split_dataset, prepare


@torch.no_grad()
def _ddim(model, schedule, x, clinical, ef, mask, valid, steps: int) -> torch.Tensor:
    times = torch.linspace(schedule.timesteps - 1, 0, steps, device=x.device).long()
    for index, current in enumerate(times):
        t_batch = torch.full((x.shape[0],), int(current), device=x.device, dtype=torch.long)
        eps = model(x, t_batch, clinical, ef, mask, valid)
        alpha = schedule.alpha_bar[int(current)]
        x0 = (x - (1 - alpha).sqrt() * eps) / alpha.sqrt().clamp_min(1e-8)
        if index + 1 == len(times):
            return x0
        previous = schedule.alpha_bar[int(times[index + 1])]
        x = previous.sqrt() * x0 + (1 - previous).sqrt() * eps
    return x


def _load_frozen(path: Path, module: torch.nn.Module, device: torch.device) -> torch.nn.Module:
    payload = load_checkpoint(path, device)
    module.load_state_dict(payload["model"])
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    return module


def generate(cfg: dict) -> None:
    run_dir, device, batch_size = prepare(cfg)
    records, groups = load_records(cfg, run_dir)
    dataset = make_split_dataset(records, groups, cfg, "TRAIN", "dit")
    severe_level = float(cfg["data"]["severe_ef"])
    severe_ids = [i for i, row in enumerate(dataset.records) if float(row["ef"]) < severe_level]
    if not severe_ids:
        raise RuntimeError("The training split has no severe-EF clips to size the synthetic cohort.")
    test_ids = {row["filename"] for row in records if row["split"] == "TEST"}
    vae_cfg = cfg["model"]["vae"]
    dit_cfg = cfg["model"]["dit"]
    vae = _load_frozen(run_dir / "vae.best.pt", FrameVAE(int(vae_cfg["latent_channels"]), int(vae_cfg["base_channels"])), device)
    schedule = DiffusionSchedule(int(cfg["diffusion"]["timesteps"]), float(cfg["diffusion"]["beta_start"]), float(cfg["diffusion"]["beta_end"])).to(device)
    rng = np.random.default_rng(int(cfg["seed"]))
    matched = rng.choice(len(dataset), size=len(severe_ids), replace=True)
    severe = rng.choice(severe_ids, size=len(severe_ids), replace=True)
    for mode in cfg["conditionings"]:
        model = _load_frozen(
            run_dir / f"dit_{mode}.best.pt",
            VideoDiT(
                latent_channels=int(vae_cfg["latent_channels"]),
                hidden=int(dit_cfg["hidden"]),
                depth=int(dit_cfg["depth"]),
                heads=int(dit_cfg["heads"]),
                patch=int(dit_cfg["patch"]),
                mlp_ratio=float(dit_cfg["mlp_ratio"]),
                grad_checkpoint=False,
            ),
            device,
        )
        for cohort, choices in (("matched", matched), ("severe", severe)):
            destination = run_dir / f"synthetic_{cohort}_{mode}.pt"
            stage = f"sample_{cohort}_{mode}"
            if stage_complete(run_dir, stage) and destination.is_file():
                print(f"Skipping completed {stage}")
                continue
            _generate_one(dataset, choices, mode, cohort, vae, model, schedule, device, batch_size, int(cfg["diffusion"]["sample_steps"]), destination, test_ids)
            write_complete(run_dir, stage, {"path": str(destination), "count": int(len(choices))})


def _generate_one(dataset, choices, mode, cohort, vae, model, schedule, device, batch_size, steps, destination, test_ids) -> None:
    videos = []
    labels = []
    conditions = []
    clinicals = []
    sources = []
    ed_masks = []
    es_masks = []
    ed_indices = []
    es_indices = []
    for start in range(0, len(choices), batch_size):
        batch_ids = [int(i) for i in choices[start : start + batch_size]]
        items = [dataset[i] for i in batch_ids]
        video = torch.stack([item["video"] for item in items]).to(device)
        clinical, ef, mask, valid = apply_condition_mode(
            torch.stack([item["clinical"] for item in items]).to(device),
            torch.stack([item["ef"] for item in items]).to(device),
            torch.stack([item["mask"] for item in items]).to(device),
            torch.stack([item["valid_mask"] for item in items]).to(device),
            mode,
        )
        with torch.no_grad():
            latent = torch.randn(vae.encode_video(video).shape, device=device)
            sampled = _ddim(model, schedule, latent, clinical, ef, mask, valid, steps)
            generated = vae.decode_video(sampled).clamp(0, 1)
        source_ef = torch.stack([item["ef"] for item in items]).float()
        videos.append(generated.cpu().half())
        labels.append(source_ef)
        conditions.append(source_ef)
        clinicals.append(torch.stack([item["clinical"] for item in items]).cpu().float())
        sources.extend(item["filename"] for item in items)
        for item in items:
            ed = int(item["ed_index"])
            es = int(item["es_index"])
            ed_masks.append(item["mask"][ed] if ed >= 0 else torch.zeros_like(item["mask"][0]))
            es_masks.append(item["mask"][es] if es >= 0 else torch.zeros_like(item["mask"][0]))
            ed_indices.append(ed)
            es_indices.append(es)
        print(f"  sampled {cohort} {mode}: {min(start + batch_size, len(choices))}/{len(choices)}")
    assert_no_test_leak(sources, test_ids)
    torch.save(
        {
            "video": torch.cat(videos),
            "label_ef": torch.cat(labels),
            "condition_ef": torch.cat(conditions),
            "clinical": torch.cat(clinicals),
            "source_filename": sources,
            "ed_mask": torch.stack(ed_masks).half(),
            "es_mask": torch.stack(es_masks).half(),
            "ed_index": torch.tensor(ed_indices),
            "es_index": torch.tensor(es_indices),
            "mode": mode,
            "cohort": cohort,
        },
        destination,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/kaggle.yaml")
    generate(load_config(Path(parser.parse_args().config)))


if __name__ == "__main__":
    main()
