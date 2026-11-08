# Decisions and deviations from the R&D plan

## D1. Geographic splits are cut inside the D2 tile coverage (2026-11-08)

**Plan:** train = official Train extent minus a 15% eastern strip (tune); val = official
Val extent; D4 (large SAR scene) built from test-region tiles of D2.

**Finding:** the SN6 train tarball (D2, 3401 tiles) covers only x 590.1-596.7 km, the
western 85% of the Train extent. No tile reaches the planned tune strip
(596.6-597.7 km), the Val extent (597.7-601.4 km) or the Test extent. Applied as
written, the plan gives train 3316, tune 0, val 0. The `test_public` tarball is SAR-only
with no labels. Official-Val SAR exists only as raw SAR-MAG-POL strips (0.25 m,
another product; about 40 GB for the Val extent).

**Decision:**

- Tile splits are cut west to east over the covered x-range, inside the Train extent:
  train | tune 15% | val 20%. A tile joins a split only if its footprint lies fully
  inside (0.5 m tolerance); boundary-crossing tiles are dropped.
  Result: train 1798, tune 349, val 637, dropped 617. Every split draws on 187-204
  collection strips. See `data/splits/splits.json` and `docs/figures/splits_map.png`.
- The official Val extent is kept for scene-level optical evaluation on the PS-RGBNIR
  TrainVal mosaic (D3) with the All-Val labels; this also exercises the stitcher on a
  real large scene.
- D4 is built from val-split tiles of a single collection strip, so it has exact
  optical co-registration and labels.
- Official-Val SAR from SAR-MAG-POL strips is a "Could" item, only if time allows.

**Cost:** a smaller held-out set than the official Val, and val sits next to train
instead of in a separate district. The dropped boundary tiles keep the two apart.

## D2. SAR/optical offset is logged, not corrected (2026-11-08)

**Finding (S2, pipeline stage 2):** `make coreg` on the 200-tile subset (masked NCC of
smoothed edge maps, `docs/figures/coreg_shifts.{csv,png}`) finds a real SAR-to-optical
offset that depends on the SAR look direction (`SAR_orientations.txt`):

| Look direction | Tiles | Median dx, dy (px, 0.5 m) | Per-strip behaviour |
|---|---|---|---|
| 1 | 88 | -3.0, -2.1 | consistent, 2-4 px spread within a strip |
| 0 | 112 | +0.5, +7.4 | strip medians -2 to +16 px dy, 6-11 px spread |

The range-direction (dy) offsets have opposite signs for the two look directions, which
fits building layover; the asymmetry suggests a smaller common offset as well.
SAR and optical tile grids themselves agree to 2e-6 m.

**Decision:** no correction for now. Tiles are used on their native shared grid.

**Revisit in S3:** if the fused model gains little over optical alone, test a per-strip
shift correction (strip median from `make coreg`) and report fused IoU with and without
it, split by look direction.

## D3. D4 is a val-tile mosaic, not a single acquisition (2026-11-08)

**Plan:** D4 = one large held-out SAR scene (3-6 GiB) with exact optical co-registration
and labels, built from D2 tiles with `gdalbuildvrt` + COG.

**Finding:** no such scene exists in D2. Each strip is a band of about 900 m x 5 km
(median 18 tiles); its val part is about 1800 x 1800 px. The val block is covered by
tiles of about 200 overlapping strips, one look direction covers only 55 % of it, and
each strip has its own sub-pixel grid origin (up to 0.5 px off a common 0.5 m lattice).

**Decision:** `make d4` (`scripts/build_d4.py`) mosaics all 637 val tiles onto one 0.5 m
grid (2621 x 15451 px, 40 Mpx):

- nearest whole-pixel paste (values unchanged; positions move up to 0.5 px per axis,
  the same for SAR and its optical twin, so SAR / optical co-registration is exact);
- look direction 1 strips first, then 0, each in time order; first valid pixel wins;
- labels: 46920 tile fragments merged by `Building_ID` into 4791 buildings
  (`d4_buildings.geojson`) and rasterised as an instance raster (`d4_labels.tif`);
- `d4_strips.tif` records which of the 125 contributing strips owns each pixel.

Result: 91.8 % of the grid has valid SAR (49 % look direction 1, 43 % direction 0);
outputs in `/mnt/d/geo/processed/sn6/d4/` with `d4.json`; quick-look
`docs/figures/d4_overview.png`.

**Caveat for S1:** SN6 SAR-Intensity is not radiometrically consistent between strips,
so the D4 SAR input has brightness steps at strip joins (the optical mosaic is
seamless). S1 seam metrics on D4 must exclude pixels near strip-ID changes, or they
would count acquisition seams as stitching seams. D3 (369 Mpx optical) stays the
large-scale memory and throughput test.

## D4. S2 gets a per-strip SAR normalisation arm (2026-11-08)

**Finding:** the Otsu baseline on D4 (`make otsu`, `docs/figures/otsu_d4.json`) shows
the median SAR dB of the 132 strips in D4 ranging from 15.3 to 44.0 dB (29 dB). Per-strip
Otsu thresholds beat one scene threshold by 0.03-0.05 IoU. The plan's S2 SAR arms use one
global percentile clip and z-score, which keeps those offsets, so a network would have to
learn each acquisition's calibration.

**Decision:** S2 gets one extra SAR arm, `db_lee7_strip` (configs/preprocess.yaml): Lee
7 x 7 on linear intensity, dB, clip to the strip's own 1st-99th percentiles, z-score with
the strip's own clipped mean and std. Lee 7 because it was the best Otsu arm; the arm
is compared against the global `db_lee7` arm in S2.

- Statistics: `make strip-stats` (`scripts/compute_strip_stats.py`) streams all D2 SAR
  tiles of each strip, from every split, into `data/processed/stats/strip_stats.json`
  (DVC). They use input pixels only, never labels: this is what a deployed system can
  compute from a new scene itself, so no label leakage.
- Strips with fewer than 1,000,000 valid pixels fall back to the global `db_lee7` stats
  (`select_arm_stats`); `apply_sar_arm` applies either.

**Cost:** one more SAR-only training run in Phase 2. Per-tile normalisation was not
added: a 450 m tile is often mostly water or fields, so its own statistics are less
stable than a strip's.
