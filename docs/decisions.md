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
