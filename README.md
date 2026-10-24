# HaemoDiff-Risk

Uncertainty-calibrated latent diffusion for rare low-ejection-fraction echocardiography. The Kaggle notebook trains on the full EchoNet-Dynamic cohort and writes four result tables. Passing the local developer checks is not a finished experiment.

## What this study is

EchoNet-Dynamic apical four-chamber videos are the imaging data. Supervised labels are the dataset's ejection fraction, reduced EF (under 40%), and severe dysfunction (under 30%). A video diffusion transformer is conditioned on an LV tracing mask and on simulated tabular covariates. Those covariates use a `simulated_` prefix. They are drawn from fixed marginals and are not measurements from the EchoNet patients, and they are not a hemophilia registry.

The downstream critic is an evidential model. It reports a Dirichlet vacuity score and a normal-inverse-gamma noise term. The regression head is trained on ejection fraction divided by 100 so that term does not swamp the classifiers, and reported EF is scaled back to percentage points. Differentially private SGD wraps that critic only. The VAE and the diffusion transformer train with AdamW. Membership inference reports attack AUC and advantage over chance. An epsilon from Opacus is an accounting result for the critic, not a proof that a patient cannot be identified, and not a regulatory clearance.

This code does not generate hemophilic arthropathy, does not enforce joint-space biomechanics, and is not a clinical device.

## Kaggle run

1. Open a Kaggle notebook with the GPU accelerator and internet enabled.
2. Attach EchoNet-Dynamic so `FileList.csv` and `VolumeTracings.csv` are under `/kaggle/input`. Do not commit or reupload the videos from this repository.
3. Put this repository on the machine (upload it, or add it as a private dataset) so `pyproject.toml` is available.
4. Run `notebooks/HaemoDiff_Risk_Kaggle.ipynb` from the top. The session proxy URL is a short-lived credential and is not stored here.
5. Checkpoints and tables are written to `/kaggle/working/haemodiff-runs/haemodiff`. If the session expires, start the notebook again. Completed stages are skipped. A previous saved output can be reattached under `/kaggle/input` and is copied back into the working directory when this session is empty.

The notebook does not call pytest and does not train on a toy subset. Stage budgets are in `configs/kaggle.yaml`: 20 epochs for the EF network, 30 for the VAE, 50 for each diffusion conditioning, and 20 for each critic, with early stopping patience 5. Batch size is 1 clip on a 16 GB T4 and 2 when at least 24 GB is free.

Training order: EF network, VAE, four diffusion conditionings (unconditional, clinical, mask, full), matched and severe synthetic cohorts, utility critics at natural prevalence and at 1:1, 1:5, 1:10, and 1:20, the DP critic, the shadow critic, the shadow diffusion model, then the tables.

A ratio of 1:5 keeps every severe (`EF < 30`) training row and subsamples the majority class to five majority rows per minority row. The official test split is never subsampled, and synthetic source ids are rejected if they appear in that split.

## Result files

The run is finished only when these files exist:

- `tables/table1_fidelity.csv` — temporal MS-SSIM, EF mean absolute error, ED/ES Dice, Wasserstein distance, Kolmogorov-Smirnov statistic and p-value with sample size, auxiliary frame FID
- `tables/table2_utility.csv` — AUROC, AUPRC, minority F1, and relative F1 change for EF under 30% and under 40%
- `tables/figure3_calibration.json` — ECE, reliability bins, epistemic detection, and aleatoric/epistemic blur correlations
- `tables/table3_privacy.csv` — critic epsilon curve, membership-inference AUC, and shadow-model advantage

Missing a publication target sets `meets_target` to false and does not delete checkpoints. The Kolmogorov-Smirnov p-value is reported with its sample size. A large synthetic set can reject tiny distribution differences, so the p-value is not treated as proof of equivalence.

## Tracing and loss safeguards

Mask conditioning is applied only on the end-diastole and end-systole frames that survive the audit. A valid-mask flag of 0 means the frame has no annotation. Clips shorter than 8 frames are dropped. Clips of 8 to 15 frames are repeat-padded, and temporal variation ignores padded steps. End-diastole and end-systole more than 16 frames apart are kept for the EF critic and excluded from diffusion training. The audit is `tracing_audit.csv`.

The physiological term is off for the warmup steps, then ramps to at most 0.1 of the diffusion loss. An exponential moving average rescales the EF, Dice, and temporal terms. If the physiological loss stays above the diffusion loss, the multiplier stops increasing.

## Developer checks

`pytest` runs without `ECHONET_ROOT`. It checks the tracing rules, the physiology balancer, evidential vacuity, the critic-only Opacus step, and the diffusion module's lack of an Opacus import. Install with `pip install -e ".[dev]"`. Kaggle already provides the CUDA build of PyTorch; the project does not pin a CPU wheel over it.
