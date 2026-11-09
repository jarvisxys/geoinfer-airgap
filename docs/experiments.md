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

## E1b. Overfit check with look-direction alignment (2026-11-09)

Same as E1 with `align_orientation: true` (decision D5); 6 of the 8 tiles come from
look-direction-1 strips and are flipped north-south. IoU **0.9869** (best epoch 138),
loss 1.99 -> 0.04, 16 min, peak GPU 2.2 GiB: flipping image, mask and valid together
does not hurt learning. `run.json` records commit `0c7ea6b-dirty` (configs not yet
committed when it ran).

## E2. S3 first server runs: optical, SAR, fused (2026-11-09)

**Setup:** college server, GPU 1 (RTX PRO 6000 Blackwell, fp16), the three models trained
in parallel with `scripts/server/remote_train.sh`: full train split (1798 tiles), tune
split for model selection, 30 epochs, batch 16, 512 px crops, east-west flips, 24 loader
workers each; optical `global`, SAR `db_lee7`; look-direction alignment for SAR and
fused (D5); ImageNet encoder for all (D6). Evaluation (`scripts/evaluate_run.py`):
threshold with the best pixel IoU on tune (sweep 0.20-0.80), then the val split (637
tiles, whole tiles, map orientation); object F1 = SpaceNet matching, IoU >= 0.5,
buildings under 20 m2 ignored, per tile against the tile label polygons.

| Model | Best epoch | Tune IoU | Thr | Val pixel IoU raw / clean | Val pixel F1 | Val object F1 | Train time |
|---|---|---|---|---|---|---|---|
| optical | 11 | 0.773 | 0.55 | 0.682 / 0.682 | 0.811 | 0.508 | 31 min |
| SAR | 20 | 0.558 | 0.20 | 0.479 / 0.466 | 0.648 | 0.244 | 60 min |
| fused | 21 | 0.754 | 0.40 | **0.688** / 0.685 | **0.815** | **0.519** | 74 min |

Val pixel IoU (raw) by look direction of the tile (353 / 284 tiles):

| Model | Direction 0 | Direction 1 |
|---|---|---|
| optical | 0.610 | 0.721 |
| SAR | 0.372 | 0.536 |
| fused | 0.628 | 0.718 |

**Reading:**
- Fused is slightly ahead of optical on the held-out val split (+0.005 pixel IoU, +0.011
  object F1) although it was behind on the tune split; the gain is on direction-0
  tiles. SAR alone reaches about 70 % of the optical IoU and half its object F1.
- The thresholds are interior optima of a 0.05-0.95 sweep (SAR peaks at 0.15-0.20: its
  probabilities are less confident).
- Clean-up (open 3, close 3, < 20 m2) leaves optical unchanged and costs SAR 0.013
  IoU: SAR masks are fragmented and opening removes true positives. Object F1 lags
  pixel F1 mostly through missed small buildings and merged neighbours.
- SAR reaches 61 % of the optical IoU on direction 0 and 74 % on direction 1, the
  direction with the smaller and more consistent SAR/optical offsets in D2. With the
  fused gain this small, D2's per-strip shift correction is worth testing next.
- Runs, checkpoints and logs: `models/s3/<model>/` (DVC, `models/s3.dvc`). `run.json`
  shows `git_commit: unknown` for these runs (server copy without `.git`); since then
  `push_code.sh` writes `REVISION`. All three ran on server commit 5cd3a2d (`eval.json`
  on 3a68fd0, which only widened the threshold sweep).
- Loader-bound: SAR and fused epochs (~100-190 s) are dominated by the per-sample Lee
  filter on a busy CPU; the GPU step itself is 69 ms per batch of 16.

## E3. S1 stitching with the trained optical model on D4 (2026-11-09)

**Setup:** `make s1-optical` (`scripts/s1_stitching.py --model run`): the E2 optical
model (`models/s3/optical`, threshold 0.55 from its `eval.json`) on the D4 optical mosaic
(held-out val tiles, 2621 x 15451 px), T 512, O 64 for crop / Hann / Tukey; all
strategies scored on the union of the O 0 and O 64 seam lines; seam score skips pixels
within 4 px of strip joins. Laptop GTX 1650, fp32.

| Strategy | Scene IoU | Border-band IoU | Seam score | Object F1 | Tile + stitch |
|---|---|---|---|---|---|
| naive (O 0) | 0.698 | 0.704 | 2.79 | 0.502 | 15 s |
| crop | 0.701 | 0.709 | 1.48 | 0.509 | 19 s |
| Hann | 0.701 | 0.709 | 0.99 | 0.510 | 16 s |
| Tukey | 0.701 | 0.709 | 0.98 | 0.509 | 16 s |

**Reading:** with a real network the accuracy gain from overlap is small (+0.005
border-band IoU, +0.008 object F1): the U-Net degrades little at tile edges, much less
than the synthetic edge-defect model of the dry run (`s1_d4_synthetic`). The visible
effect is the seam score: naive pasting leaves hard steps (2.79), crop halves them
(1.48), Hann and Tukey blending removes them (about 1.0 = no visible seams);
`docs/figures/s1_d4_optical.png` shows a building cut at a tile corner. Blending costs
no accuracy and no measurable time, so Hann (or Tukey) stays the default.
