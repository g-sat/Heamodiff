"""Four-pillar evaluation and the result tables for a Kaggle run."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from haemodiff.config import load_config
from haemodiff.data.dataset import collate_clips
from haemodiff.eval.metrics import (
    auprc,
    auroc,
    binary_f1,
    expected_calibration_error,
    frame_fid,
    ks_two_sample,
    laplacian_variance,
    relative_change,
    spearman,
    temporal_ms_ssim,
    wasserstein,
)
from haemodiff.losses.diffusion import DiffusionSchedule
from haemodiff.losses.evidential import critic_loss_per_sample
from haemodiff.models.dit import VideoDiT, apply_condition_mode
from haemodiff.models.ef_net import EFNet
from haemodiff.models.evidential import (
    EvidentialCritic,
    aleatoric_variance,
    class_probability,
    split_critic_output,
    vacuity,
)
from haemodiff.models.vae import FrameVAE
from haemodiff.privacy.mia import loss_threshold_auc, shadow_attack_advantage
from haemodiff.train.engine import load_checkpoint, load_records, make_split_dataset, prepare
from haemodiff.train.train_critic import train as train_critic
from haemodiff.train.train_dit import train as train_dit


def _loader(dataset, batch_size: int) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_clips)


def _load_ef(run_dir: Path, cfg: dict, device: torch.device) -> EFNet:
    model = EFNet(int(cfg["model"]["ef_net"]["base_channels"])).to(device)
    model.load_state_dict(load_checkpoint(run_dir / "ef.best.pt", device)["model"])
    model.eval()
    return model


def _load_vae(run_dir: Path, cfg: dict, device: torch.device) -> FrameVAE:
    model = FrameVAE(int(cfg["model"]["vae"]["latent_channels"]), int(cfg["model"]["vae"]["base_channels"])).to(device)
    model.load_state_dict(load_checkpoint(run_dir / "vae.best.pt", device)["model"])
    model.eval()
    return model


def _load_dit(path: Path, cfg: dict, device: torch.device) -> VideoDiT:
    dit_cfg = cfg["model"]["dit"]
    model = VideoDiT(
        latent_channels=int(cfg["model"]["vae"]["latent_channels"]),
        hidden=int(dit_cfg["hidden"]),
        depth=int(dit_cfg["depth"]),
        heads=int(dit_cfg["heads"]),
        patch=int(dit_cfg["patch"]),
        mlp_ratio=float(dit_cfg["mlp_ratio"]),
        grad_checkpoint=False,
    ).to(device)
    model.load_state_dict(load_checkpoint(path, device)["model"])
    model.eval()
    return model


def _load_critic(path: Path, cfg: dict, device: torch.device) -> EvidentialCritic:
    model = EvidentialCritic(int(cfg["data"]["clip_len"]), int(cfg["model"]["critic"]["base_channels"])).to(device)
    model.load_state_dict(load_checkpoint(path, device)["model"])
    model.eval()
    return model


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _flag(value: float, target: float, higher: bool) -> bool:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return False
    return bool(value >= target) if higher else bool(value <= target)


def train_utility_models(cfg: dict) -> None:
    for arm in ("real", "real_uncond", "real_haemodiff"):
        for ratio in cfg["utility"]["ratios"]:
            stage = f"critic_{arm}_{ratio.replace(':', '-')}"
            train_critic(cfg, stage=stage, arm=arm, ratio=ratio, fraction=1.0)


def train_privacy_models(cfg: dict) -> None:
    train_critic(cfg, stage="critic_dp", dp=True, arm="real", ratio="natural", fraction=1.0)
    train_critic(cfg, stage="critic_shadow", dp=False, arm="real", ratio="natural", fraction=0.5)
    train_dit(cfg, mode="full", stage="dit_shadow", fraction=0.5)


@torch.no_grad()
def _critic_outputs(model, loader, device: torch.device) -> dict[str, np.ndarray]:
    store = {key: [] for key in ("p_severe", "p_reduced", "vacuity", "aleatoric", "severe", "reduced", "ef", "pred_ef", "loss", "filename")}
    for batch in loader:
        parts = split_critic_output(model(batch["video"].to(device)))
        store["p_severe"].append(class_probability(parts["severe_alpha"])[:, 1].cpu().numpy())
        store["p_reduced"].append(class_probability(parts["reduced_alpha"])[:, 1].cpu().numpy())
        store["vacuity"].append(vacuity(parts["severe_alpha"]).cpu().numpy())
        store["aleatoric"].append(aleatoric_variance(parts["alpha"], parts["beta"]).cpu().numpy())
        store["severe"].append(batch["severe"].numpy())
        store["reduced"].append(batch["reduced"].numpy())
        store["ef"].append(batch["ef"].numpy())
        store["pred_ef"].append((parts["gamma"] * 100).cpu().numpy())
        store["loss"].append(
            critic_loss_per_sample(parts, batch["ef"].to(device), batch["severe"].to(device), batch["reduced"].to(device)).cpu().numpy()
        )
        store["filename"].extend(batch["filename"])
    stacked = {key: np.concatenate(value) if key != "filename" else value for key, value in store.items()}
    return stacked


def _endpoint_row(arm: str, ratio: str, endpoint: str, labels: np.ndarray, scores: np.ndarray) -> dict:
    return {
        "arm": arm,
        "ratio": ratio,
        "majority_per_minority": "" if ratio == "natural" else ratio.split(":")[1],
        "endpoint": endpoint,
        "auroc": auroc(labels, scores),
        "auprc": auprc(labels, scores),
        "minority_f1": binary_f1(labels, scores),
        "n_test_positive": int(labels.sum()),
        "n_test": int(len(labels)),
    }


def write_utility(cfg: dict, run_dir: Path, device: torch.device, batch_size: int) -> None:
    records, groups = load_records(cfg, run_dir)
    test_ds = make_split_dataset(records, groups, cfg, "TEST", "critic")
    test_loader = _loader(test_ds, batch_size)
    rows = []
    for arm in ("real", "real_uncond", "real_haemodiff"):
        for ratio in cfg["utility"]["ratios"]:
            stage = f"critic_{arm}_{ratio.replace(':', '-')}"
            model = _load_critic(run_dir / f"{stage}.best.pt", cfg, device)
            outputs = _critic_outputs(model, test_loader, device)
            rows.append(_endpoint_row(arm, ratio, "ef_under_30", outputs["severe"].astype(int), outputs["p_severe"]))
            rows.append(_endpoint_row(arm, ratio, "ef_under_40", outputs["reduced"].astype(int), outputs["p_reduced"]))
    baseline = {(row["ratio"], row["endpoint"]): row["minority_f1"] for row in rows if row["arm"] == "real"}
    for row in rows:
        base = baseline[(row["ratio"], row["endpoint"])]
        row["real_only_f1"] = base
        row["relative_f1_change"] = 0.0 if row["arm"] == "real" else relative_change(row["minority_f1"], base)
    _write_csv(run_dir / "tables" / "table2_utility.csv", rows)


def _batch_videos(videos: torch.Tensor, batch_size: int):
    for start in range(0, len(videos), batch_size):
        yield videos[start : start + batch_size]


@torch.no_grad()
def _predicted_ef_and_masks(ef_net: EFNet, videos: torch.Tensor, device: torch.device, batch_size: int) -> tuple[np.ndarray, torch.Tensor]:
    efs = []
    masks = []
    for batch in _batch_videos(videos, batch_size):
        pred_ef, pred_mask = ef_net(batch.to(device))
        efs.append(pred_ef.cpu().numpy())
        masks.append(torch.sigmoid(pred_mask).cpu())
    return np.concatenate(efs), torch.cat(masks)


def _mean_dice(pred: torch.Tensor, target: torch.Tensor) -> float:
    if len(pred) == 0:
        return float("nan")
    binary = (pred >= 0.5).float()
    keep = target.flatten(1).sum(dim=1) > 0
    binary = binary[keep]
    target = target[keep]
    if len(binary) == 0:
        return float("nan")
    intersection = (binary * target).sum(dim=(1, 2, 3))
    denom = binary.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return float((2 * intersection / denom.clamp_min(1e-8)).mean())


def write_fidelity(cfg: dict, run_dir: Path, device: torch.device, batch_size: int) -> None:
    records, groups = load_records(cfg, run_dir)
    real_ef = np.asarray(
        [float(row["ef"]) for row in records if row["split"] == "TRAIN" and row["status"] == "dit"],
        dtype=np.float64,
    )
    val_ds = make_split_dataset(records, groups, cfg, "VAL", "dit")
    ef_net = _load_ef(run_dir, cfg, device)
    real_videos = []
    for index in range(min(len(val_ds), 256)):
        real_videos.append(val_ds[index]["video"])
    real_stack = torch.stack(real_videos) if real_videos else None
    real_ms = temporal_ms_ssim(real_stack) if real_stack is not None else float("nan")
    rows = []
    for mode in cfg["conditionings"]:
        payload = torch.load(run_dir / f"synthetic_matched_{mode}.pt", map_location="cpu", weights_only=False)
        videos = payload["video"].float()
        generated_ms = temporal_ms_ssim(videos[: min(len(videos), 256)])
        pred_ef, pred_masks = _predicted_ef_and_masks(ef_net, videos, device, batch_size)
        has_ef = mode in {"clinical", "full"}
        has_mask = mode in {"mask", "full"}
        ef_mae = float(np.mean(np.abs(pred_ef - payload["condition_ef"].numpy()))) if has_ef else float("nan")
        if has_mask:
            ed_index = payload["ed_index"].numpy()
            es_index = payload["es_index"].numpy()
            ed_ok = ed_index >= 0
            es_ok = es_index >= 0
            ed_pred = torch.stack([pred_masks[i, int(ed_index[i])] for i in np.where(ed_ok)[0]]) if ed_ok.any() else torch.zeros(0, 1, 1, 1)
            es_pred = torch.stack([pred_masks[i, int(es_index[i])] for i in np.where(es_ok)[0]]) if es_ok.any() else torch.zeros(0, 1, 1, 1)
            ed_dice = _mean_dice(ed_pred, payload["ed_mask"].float()[ed_ok])
            es_dice = _mean_dice(es_pred, payload["es_mask"].float()[es_ok])
            dice_mean = float(np.nanmean([ed_dice, es_dice]))
        else:
            ed_dice = es_dice = dice_mean = float("nan")
        distance = wasserstein(real_ef, pred_ef)
        ks_stat, ks_p = ks_two_sample(real_ef, pred_ef)
        fid = None
        if real_stack is not None:
            fid = frame_fid(real_stack[:, real_stack.shape[1] // 2], videos[: len(real_stack), videos.shape[1] // 2], device)
        rows.append(
            {
                "conditioning": mode,
                "temporal_ms_ssim_generated": generated_ms,
                "temporal_ms_ssim_real": real_ms,
                "ms_ssim_target": 0.85,
                "ms_ssim_meets_target": _flag(generated_ms, 0.85, True) if mode == "full" else "",
                "ef_mae": ef_mae,
                "ef_mae_target": 5.0,
                "ef_mae_tight_target": 3.0,
                "ef_mae_meets_target": _flag(ef_mae, 5.0, False) if has_ef else "",
                "ed_dice": ed_dice,
                "es_dice": es_dice,
                "dice_mean": dice_mean,
                "dice_target": 0.75,
                "dice_tight_target": 0.80,
                "dice_meets_target": _flag(dice_mean, 0.75, True) if has_mask else "",
                "wasserstein": distance,
                "ks_statistic": ks_stat,
                "ks_pvalue": ks_p,
                "ks_n_real": int(len(real_ef)),
                "ks_n_generated": int(len(pred_ef)),
                "ks_target": 0.05,
                "ks_meets_target": _flag(ks_p, 0.05, True),
                "frame_fid": "" if fid is None else fid,
                "frame_fid_auxiliary": True,
            }
        )
    _write_csv(run_dir / "tables" / "table1_fidelity.csv", rows)


def _corrupt(videos: torch.Tensor, kind: str, strength: float) -> torch.Tensor:
    if kind == "gaussian":
        return (videos + strength * torch.randn_like(videos)).clamp(0, 1)
    kernel = max(3, int(strength))
    weight = torch.ones(1, 1, 1, kernel, device=videos.device, dtype=videos.dtype) / kernel
    n, t, _, h, w = videos.shape
    blurred = F.conv2d(videos.reshape(n * t, 1, h, w), weight, padding=(0, kernel // 2))
    return blurred.reshape(n, t, 1, h, w).clamp(0, 1)


def write_uncertainty(cfg: dict, run_dir: Path, device: torch.device, batch_size: int) -> None:
    records, groups = load_records(cfg, run_dir)
    test_ds = make_split_dataset(records, groups, cfg, "TEST", "critic")
    model = _load_critic(run_dir / "critic_real_natural.best.pt", cfg, device)
    outputs = _critic_outputs(model, _loader(test_ds, batch_size), device)
    labels = outputs["severe"].astype(int)
    predicted = (outputs["p_severe"] >= 0.5).astype(int)
    ece, bins = expected_calibration_error(outputs["p_severe"], labels, n_bins=15)
    errors = (predicted != labels).astype(int)
    error_auroc = auroc(errors, outputs["vacuity"])
    error_auprc = auprc(errors, outputs["vacuity"])
    clean = []
    for index in range(min(len(test_ds), 128)):
        clean.append(test_ds[index]["video"])
    clean_videos = torch.stack(clean)
    motion_aleatoric, motion_blur, motion_vacuity = [], [], []
    noise_aleatoric, noise_level = [], []
    corrupted_flags = []
    corrupted_vacuity = []
    for kind, strengths in (("gaussian", (0.01, 0.05, 0.1)), ("motion", (3, 7, 11))):
        for strength in strengths:
            corrupted = _corrupt(clean_videos, kind, float(strength))
            batch_loader = _loader(_TensorVideos(corrupted, test_ds, len(corrupted)), batch_size)
            corrupt_out = _critic_outputs(model, batch_loader, device)
            corrupted_flags.append(np.ones(len(corrupted), dtype=np.int64))
            corrupted_vacuity.append(corrupt_out["vacuity"])
            if kind == "motion":
                motion_aleatoric.append(corrupt_out["aleatoric"])
                motion_blur.append(-laplacian_variance(corrupted))
                motion_vacuity.append(corrupt_out["vacuity"])
            else:
                noise_aleatoric.append(corrupt_out["aleatoric"])
                noise_level.append(np.full(len(corrupted), float(strength)))
    clean_out = _critic_outputs(model, _loader(_TensorVideos(clean_videos, test_ds, len(clean_videos)), batch_size), device)
    aleatoric_blur = spearman(np.concatenate(motion_aleatoric), np.concatenate(motion_blur))
    epistemic_blur = spearman(np.concatenate(motion_vacuity), np.concatenate(motion_blur))
    aleatoric_noise = spearman(np.concatenate(noise_aleatoric), np.concatenate(noise_level))
    epistemic_error = spearman(outputs["vacuity"], errors.astype(np.float64))
    detect_labels = np.concatenate([np.zeros(len(clean_videos), dtype=np.int64), np.concatenate(corrupted_flags)])
    detect_scores = np.concatenate([clean_out["vacuity"], np.concatenate(corrupted_vacuity)])
    payload = {
        "ece": {"value": ece, "target": 0.05, "meets_target": _flag(ece, 0.05, False), "bins": bins},
        "epistemic_error_detection": {"auroc": error_auroc, "auprc": error_auprc},
        "epistemic_corruption_detection": {"auroc": auroc(detect_labels, detect_scores), "auprc": auprc(detect_labels, detect_scores)},
        "separation": {
            "aleatoric_blur_spearman": aleatoric_blur,
            "epistemic_blur_spearman": epistemic_blur,
            "aleatoric_noise_spearman": aleatoric_noise,
            "epistemic_error_spearman": epistemic_error,
            "meets_target": bool(aleatoric_blur > 0 and abs(epistemic_blur) < abs(epistemic_error)),
        },
    }
    path = run_dir / "tables" / "figure3_calibration.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


class _TensorVideos(torch.utils.data.Dataset):
    def __init__(self, videos: torch.Tensor, template, count: int) -> None:
        self.videos = videos
        self.template = template
        self.count = count

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, index: int) -> dict:
        item = self.template[index]
        item = dict(item)
        item["video"] = self.videos[index]
        return item


@torch.no_grad()
def _dit_losses(model, vae, schedule, dataset, device, mode: str, batch_size: int) -> np.ndarray:
    import zlib

    losses = []
    loader = _loader(dataset, batch_size)
    for batch in loader:
        video = batch["video"].to(device)
        latent = vae.encode_video(video)
        clinical, ef, mask, valid = apply_condition_mode(
            batch["clinical"].to(device),
            batch["ef"].to(device),
            batch["mask"].to(device),
            batch["valid_mask"].to(device),
            mode,
        )
        noise = []
        for name in batch["filename"]:
            generator = torch.Generator()
            generator.manual_seed(zlib.adler32(str(name).encode()) % (2**31 - 1))
            noise.append(torch.randn(latent.shape[1:], generator=generator))
        noise_tensor = torch.stack(noise).to(device)
        timesteps = torch.full((latent.shape[0],), schedule.timesteps // 2, device=device, dtype=torch.long)
        noisy = schedule.q_sample(latent, timesteps, noise_tensor)
        predicted = model(noisy, timesteps, clinical, ef, mask, valid)
        losses.append(((predicted - noise_tensor) ** 2).mean(dim=(1, 2, 3, 4)).cpu().numpy())
    return np.concatenate(losses)


def _name_subset(dataset, names: set[str]) -> Subset:
    indices = [i for i, row in enumerate(dataset.records) if row["filename"] in names]
    return Subset(dataset, indices)


def write_privacy(cfg: dict, run_dir: Path, device: torch.device, batch_size: int) -> None:
    records, groups = load_records(cfg, run_dir)
    train_ds = make_split_dataset(records, groups, cfg, "TRAIN", "critic")
    val_ds = make_split_dataset(records, groups, cfg, "VAL", "critic")
    rows = []
    curve_path = run_dir / "critic_dp.epsilon.json"
    curve = json.loads(curve_path.read_text(encoding="utf-8")) if curve_path.is_file() else []
    for point in curve:
        rows.append(
            {
                "row_type": "epsilon_epoch",
                "model": "critic_dp",
                "epoch": point["epoch"],
                "epsilon": point["epsilon"],
                "delta": point["delta"],
                "mia_auc": "",
                "mia_auc_target": "",
                "mia_meets_target": "",
                "shadow_advantage": "",
                "shadow_advantage_target": "",
                "shadow_meets_target": "",
                "notes": "Critic-only Opacus. No epsilon is reported for the VAE or the DiT.",
            }
        )
    nonprivate = _load_critic(run_dir / "critic_real_natural.best.pt", cfg, device)
    private = _load_critic(run_dir / "critic_dp.best.pt", cfg, device)
    shadow = _load_critic(run_dir / "critic_shadow.best.pt", cfg, device)
    target_member = _critic_outputs(nonprivate, _loader(train_ds, batch_size), device)["loss"]
    target_nonmember = _critic_outputs(nonprivate, _loader(val_ds, batch_size), device)["loss"]
    private_member = _critic_outputs(private, _loader(train_ds, batch_size), device)["loss"]
    private_nonmember = _critic_outputs(private, _loader(val_ds, batch_size), device)["loss"]
    held_out = set(json.loads((run_dir / "critic_shadow.heldout.json").read_text(encoding="utf-8")))
    shadow_non_ds = _name_subset(train_ds, held_out)
    shadow_member_ds = _name_subset(train_ds, {row["filename"] for row in train_ds.records} - held_out)
    shadow_member_loss = _critic_outputs(shadow, _loader(shadow_member_ds, batch_size), device)["loss"]
    shadow_non_loss = _critic_outputs(shadow, _loader(shadow_non_ds, batch_size), device)["loss"]
    rows.extend(
        [
            _attack_row("critic_nonprivate", None, loss_threshold_auc(target_member, target_nonmember), shadow_attack_advantage(shadow_member_loss, shadow_non_loss, target_member, target_nonmember), False, "Non-private critic memorization audit."),
            _attack_row("critic_dp", curve[-1]["epsilon"] if curve else "", loss_threshold_auc(private_member, private_nonmember), shadow_attack_advantage(shadow_member_loss, shadow_non_loss, private_member, private_nonmember), True, "Differential privacy claim applies only to this critic."),
        ]
    )
    vae = _load_vae(run_dir, cfg, device)
    schedule = DiffusionSchedule(int(cfg["diffusion"]["timesteps"]), float(cfg["diffusion"]["beta_start"]), float(cfg["diffusion"]["beta_end"])).to(device)
    dit_train = make_split_dataset(records, groups, cfg, "TRAIN", "dit")
    dit_val = make_split_dataset(records, groups, cfg, "VAL", "dit")
    full = _load_dit(run_dir / "dit_full.best.pt", cfg, device)
    shadow_dit = _load_dit(run_dir / "dit_shadow.best.pt", cfg, device)
    dit_held = set(json.loads((run_dir / "dit_shadow.heldout.json").read_text(encoding="utf-8")))
    dit_member_names = {row["filename"] for row in dit_train.records} - dit_held
    full_member = _dit_losses(full, vae, schedule, dit_train, device, "full", batch_size)
    full_nonmember = _dit_losses(full, vae, schedule, dit_val, device, "full", batch_size)
    shadow_member = _dit_losses(shadow_dit, vae, schedule, _name_subset(dit_train, dit_member_names), device, "full", batch_size)
    shadow_non = _dit_losses(shadow_dit, vae, schedule, _name_subset(dit_train, dit_held), device, "full", batch_size)
    rows.append(
        _attack_row(
            "dit_full",
            "",
            loss_threshold_auc(full_member, full_nonmember),
            shadow_attack_advantage(shadow_member, shadow_non, full_member, full_nonmember),
            False,
            "Non-private generator memorization audit. This row is not a DP claim.",
        )
    )
    _write_csv(run_dir / "tables" / "table3_privacy.csv", rows)


def _attack_row(model: str, epsilon, auc: float, advantage: float, claim: bool, notes: str) -> dict:
    return {
        "row_type": "attack",
        "model": model,
        "epoch": "",
        "epsilon": epsilon,
        "delta": "",
        "mia_auc": auc,
        "mia_auc_target": 0.55 if claim else "",
        "mia_meets_target": _flag(auc, 0.55, False) if claim else "",
        "shadow_advantage": advantage,
        "shadow_advantage_target": 0.05 if claim else "",
        "shadow_meets_target": _flag(advantage, 0.05, False) if claim else "",
        "notes": notes,
    }


def write_tables(cfg: dict) -> None:
    run_dir, device, batch_size = prepare(cfg)
    write_fidelity(cfg, run_dir, device, batch_size)
    write_utility(cfg, run_dir, device, batch_size)
    write_uncertainty(cfg, run_dir, device, batch_size)
    write_privacy(cfg, run_dir, device, batch_size)
    print(f"Wrote tables under {run_dir / 'tables'}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/kaggle.yaml")
    write_tables(load_config(parser.parse_args().config))


if __name__ == "__main__":
    main()
