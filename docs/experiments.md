# Experiment log

One entry per run that matters for the report. Every run writes
`outputs/runs/<run_name>/run.json` (config, commit, environment, per-epoch losses and
validation IoU, best epoch, timings, peak GPU memory); checkpoints go to DVC.

## E1. Laptop overfit sanity check (2026-11-08)

**Purpose:** the plan's Phase 2 check: overfit 8 tiles until the loss nearly reaches
zero, to catch label, flip and normalisation bugs before server hours are spent.

**Setup:** `make overfit` (`configs/train_overfit.yaml`): fused 8-channel U-Net
(ResNet-34, ImageNet encoder), optical `global`, SAR `db_lee7`; the 8 working-subset
train tiles with the most buildings (`data/splits/overfit8.txt`), one fixed 512 px crop
each, no flips; batch 4, AdamW lr 1e-3 with cosine decay, 150 epochs; scored on the
same crops.

**Result:** IoU **0.9875** (best epoch 144), loss 1.89 -> 0.03, validation loss tracking
training loss (0.030 vs 0.031). 15.6 min on the GTX 1650 in fp32, peak GPU memory
2.2 GiB. Predictions match the masks with no flip or offset (checked visually).

**Found on the way:**
- fp16 autocast gives non-finite outputs on the GTX 1650 (erratic with input size and
  mode, a known GTX 16-series problem), so the first attempt trained on NaN from
  epoch 0. `amp: auto` now uses fp32 on GTX 16-series cards and probes other GPUs.
- YAML 1.1 read the config key `on:` as the boolean `true`, so validation silently ran
  on whole tiles (IoU 0.80, mostly unseen pixels). The key is now `target`, and
  configs with unknown or non-string keys are rejected.
