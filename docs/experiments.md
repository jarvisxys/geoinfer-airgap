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

## E4. S2 normalisation arms (2026-11-09)

**Setup:** as E2 (server GPU 1, full train split, tune for model selection, 30 epochs,
batch 16), five runs in parallel from `scripts/server/remote_s2.sh` with the
Lee-filtered SAR cache (SAR epochs ~65 s instead of 100-190 s in E2); the E2 runs
`optical` (global) and `sar` (Lee 7, global) are the reference arms. One training run
per arm. Started on commit 9daef09 (`run.json` shows 100c767: the commit was read at the
end of training after newer code was pushed; fixed, it is now read at the start).
Checkpoints: `models/s2/` (DVC).

| Arm | Tune IoU | Thr | Val pixel IoU raw / clean | Val pixel F1 | Val object F1 | IoU dir 0 / 1 |
|---|---|---|---|---|---|---|
| optical, per-tile min-max | 0.775 | 0.40 | 0.681 / 0.678 | 0.810 | 0.513 | 0.633 / 0.706 |
| optical, global 2-98 % (E2) | 0.773 | 0.55 | 0.682 / 0.682 | 0.811 | 0.508 | 0.610 / 0.721 |
| SAR, linear | 0.441 | 0.15 | 0.403 / 0.384 | 0.575 | 0.148 | 0.305 / 0.458 |
| SAR, dB | 0.559 | 0.20 | 0.470 / 0.453 | 0.639 | 0.257 | 0.385 / 0.515 |
| SAR, dB + Lee 5 | 0.543 | 0.30 | 0.462 / 0.452 | 0.632 | 0.227 | 0.350 / 0.527 |
| SAR, dB + Lee 7 (E2) | 0.551 | 0.20 | 0.479 / 0.466 | 0.648 | 0.244 | 0.372 / 0.536 |
| SAR, dB + Lee 7 per strip (D4) | 0.574 | 0.25 | 0.472 / 0.463 | 0.642 | 0.261 | 0.379 / 0.522 |

**Reading:**
- The clear S2 effect is the dB conversion: +0.067 val pixel IoU and +0.11 object F1
  over raw linear intensity, whose z-scores are dominated by a heavy bright tail.
- Among the dB arms the differences are about 0.01 or less and not consistent: Lee 7
  global has the best pixel IoU, per-strip normalisation the best object F1 and by far
  the best tune IoU (0.574) but not the best val IoU. With one run per arm these
  differences are within run-to-run noise; ranking them needs 2-3 seeds per arm.
- Optical: per-tile min-max matches the global stretch on tile IoU (0.681 vs 0.682); its
  cost is visible as seams when tiles are stitched (S2 figure: seam score 2.60 vs 0.92),
  not in per-tile scores.
## E5. ONNX export and parity (2026-11-09)

`make export` (`scripts/export_run.py`): each S3 model's `best.pt` exported with opset 18
to `models/s3/<run>/model.onnx` (input `image` (batch, C, h, w), output `logits`
(batch, 1, h, w); batch, h and w dynamic), then PyTorch fp32 vs ONNX Runtime fp32 on 4
random inputs (batch 1 at 512 x 512 and batch 3 at 256 x 384), GTX 1650 / CUDA EP.

| Model | ONNX size | Max abs logit diff | Max abs probability diff |
|---|---|---|---|
| optical | 93.2 MiB | 1.0e-5 | 6.0e-8 |
| SAR | 93.2 MiB | 1.8e-5 | 1.4e-7 |
| fused | 93.3 MiB | 2.4e-5 | 2.4e-7 |

All within the 1e-3 tolerance (`configs/export.yaml`); details in `export.json` per run.

## E6. S3 robustness to clouds (2026-11-09)

**Setup:** synthetic clouds (`preprocess.clouds`: fractal noise, beta 2.2, soft edges,
train-split p98 white point, per-tile seeds so every model sees identical clouds) blended
into the raw optical bands of the val split at 0-80 % cover
(`scripts/cloud_curve.py`, `remote_cloud.sh`); each model at its tuned threshold. Clean-
trained models = E2. Cloud-trained models: same configs plus
`data.cloud_aug = {p: 0.5, max_coverage: 0.8}` (half the training samples get clouds of
uniform random cover in [0, 0.8)), commit 100c767, `models/s3_clouds/` (DVC).

| Val pixel IoU at cloud cover | 0 % | 20 % | 40 % | 60 % | 80 % |
|---|---|---|---|---|---|
| optical, clean-trained | 0.682 | 0.308 | 0.130 | 0.042 | 0.008 |
| fused, clean-trained | 0.688 | 0.382 | 0.189 | 0.078 | 0.025 |
| SAR only | 0.479 | 0.479 | 0.479 | 0.479 | 0.479 |
| optical, cloud-trained | 0.686 | 0.654 | 0.623 | 0.578 | 0.502 |
| fused, cloud-trained | 0.680 | 0.644 | 0.614 | 0.577 | **0.527** |

Cloud-trained models on clean val (threshold from tune): optical 0.686 pixel IoU / 0.505
object F1 (threshold 0.50), fused 0.680 / 0.507 (0.40).

**Reading** (`docs/figures/s3_clouds_all.png`):
- Trained without clouds, both optical models collapse: at 20 % cover they fall below SAR
  alone, and the fused model does not fall back on its SAR input.
- Cloud augmentation is what buys robustness: at 40 % cover IoU goes from 0.13-0.19 to
  0.61-0.62, and clean accuracy is unchanged (0.686 / 0.680 vs 0.682 / 0.688).
- With cloud training, optical and fused stay within 0.01 up to 60 % cover; at 80 % the
  fused model leads (0.527 vs 0.502) and is the only model above SAR alone. SAR's value
  in fusion is as a backstop under heavy cloud, not on clear or lightly clouded images.

## E7. S2 SAR arms over three seeds (2026-11-09)

**Setup:** seeds 1 and 2 of the four SAR dB arms (`scripts/server/remote_s2_seeds.sh`,
commit f57cf22, 8 runs in parallel, 10 loader workers each), seed 0 = the E4 runs (Lee 7
global = E2 `sar`). Same evaluation as E4. Runs: `models/s2/*_seed{1,2}` (DVC).

| SAR arm | Val pixel IoU (mean +- sd) | Seeds | Pixel F1 | Object F1 (mean +- sd) |
|---|---|---|---|---|
| dB | 0.478 +- 0.009 | 0.470 / 0.488 / 0.477 | 0.647 | 0.256 +- 0.002 |
| dB + Lee 5 | 0.469 +- 0.007 | 0.462 / 0.475 / 0.471 | 0.639 | 0.232 +- 0.004 |
| dB + Lee 7 | 0.474 +- 0.004 | 0.479 / 0.471 / 0.472 | 0.643 | 0.248 +- 0.007 |
| dB + Lee 7 per strip | **0.482 +- 0.009** | 0.472 / 0.483 / 0.490 | **0.650** | **0.264 +- 0.005** |

(raw linear intensity, one seed: 0.403 pixel IoU, 0.148 object F1)

**Reading:**
- Pixel IoU does not separate the dB arms: the spread across arms (0.013) is about one to
  two seed standard deviations.
- Object F1 does: Lee 5 is clearly worst (0.024 below plain dB, more than 4 sd), and the
  per-strip arm is the best (+0.016 over the same Lee 7 filter with global statistics,
  about 2.5 sd).
- The Lee filter does not help the CNN: plain dB matches or beats Lee 5 and Lee 7; the
  network learns its own speckle smoothing, while the Lee filter blurs building edges
  (edge-preservation index 0.32-0.36, `s2_sar_patch.png`) and that costs whole buildings.
  What helps is per-strip normalisation (decision D4).
- S2 conclusion for SAR: dB conversion is the large, certain gain (+0.07 IoU, +0.11
  object F1 over linear); per-strip normalisation adds a small object-level gain; speckle
  filtering before a CNN is not worth it here.

## E8. Morphology ablation: object F1 before and after clean-up (2026-11-09)

`score_predictions` now also scores the raw thresholded mask at object level
(`object_raw` in `eval.json`); the S3 models were re-evaluated (val split, tuned
thresholds). Clean-up = opening 3, closing 3, components < 20 m2 removed.

| Model | Object F1 raw -> clean | FP raw -> clean | FN raw -> clean | Pixel IoU raw -> clean |
|---|---|---|---|---|
| optical | 0.513 -> 0.508 | 7968 -> 7913 | 16206 -> 16378 | 0.682 -> 0.682 |
| SAR | **0.223 -> 0.244** | 10251 -> 10352 | 24028 -> 23480 | 0.479 -> 0.466 |
| fused | 0.517 -> 0.519 | 8042 -> 7686 | 16056 -> 16104 | 0.688 -> 0.685 |
| optical, cloud-trained | 0.509 -> 0.505 | 8491 -> 8324 | 16166 -> 16337 | 0.686 -> 0.685 |
| fused, cloud-trained | 0.505 -> 0.507 | 7816 -> 7445 | 16537 -> 16588 | 0.680 -> 0.676 |

**Reading:** the clean-up helps SAR at object level (+0.021 F1): closing joins the
fragmented SAR detections of one building, so fewer buildings are missed (FN -548). For
optical and fused it is neutral (within 0.005): their masks are already clean, opening
removes a few false positives and closing merges a few neighbours. It always costs a
little pixel IoU (up to 0.013 for SAR). Keep it for SAR and fused outputs; for optical
it is optional.

## E9. S1 on the official Val extent (D3-val) (2026-11-09)

`scripts/build_d3val.py`: the D3 PS-RGBNIR TrainVal mosaic cut to the official Val
extent (7444 x 16200 px, 121 Mpx, 99.3 % valid) with the official All-Val labels (8396
buildings, 14.3 % of valid pixels); no D2 tile lies in this extent (D1), so this is the
held-out scene-level test. Same harness and model as E3 (optical, threshold 0.55), on the
server GPU.

| Strategy | Scene IoU | Border-band IoU | Seam score | Object F1 | Tile + stitch |
|---|---|---|---|---|---|
| naive (O 0) | 0.702 | 0.700 | 3.23 | 0.539 | 28 s |
| crop | 0.709 | 0.709 | 1.65 | 0.548 | 32 s |
| Hann | 0.709 | 0.709 | 0.995 | 0.549 | 29 s |
| Tukey | 0.709 | 0.709 | 0.993 | 0.549 | 29 s |

**Reading:** the same picture as D4 on a scene three times larger with the official
labels, and a slightly larger accuracy gain from overlap (+0.009 border-band IoU, +0.010
object F1). Seams go from 3.23 (naive) to 1.65 (crop) to 0.99 (blending). The close-ups
(`s1_d3val_optical.png`, `s1_d4_optical.png`) now sit at the seam crossing with the most
labelled buildings; on D3-val two tiles disagree about a large structure, which naive
pasting cuts with a hard edge and blending turns into a gradient.
