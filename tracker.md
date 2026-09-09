# caltrust — tracker

**One command against an existing calibration returns a defensible statement of metric accuracy in task units.**

The target sentence, which is the whole product:

> Your measurement of a feature at 800 mm has an expected error of 0.41 mm, with a 95% interval of ±1.12 mm. Your reported 0.24 px reprojection error understates out-of-sample error by 2.9x. The dominant cause is insufficient depth variation across your 18 views: focal length and distance are correlated at 0.94. Adding six views at 400 mm and 1200 mm would reduce the interval to approximately ±0.35 mm.

Everything in this repo exists to make that paragraph true and checkable.

---

## The five failures being fixed

| # | Failure | Milestone that fixes it |
|---|---|---|
| 1 | Reprojection RMS is an in-sample fit statistic | M3 — held-out reprojection error |
| 2 | No usable uncertainty on the parameters | **M2 — full covariance, not marginal stddevs** |
| 3 | No diagnosis of the cause | M3 — the eight-cause diagnostic set |
| 4 | No translation into task space | M4 — mm at a working distance |
| 5 | No staleness detection | M6 — drift monitor |

---

## Milestones

| M | Scope | State |
|---|---|---|
| **M1** | **Ingest** — images + target (checkerboard / ChArUco / circle grid), or existing calibration + detections. Pinhole/Brown-Conrady and fisheye/Kannala-Brandt. Optional robot poses for hand-eye. | in progress |
| **M2** | **Refit with instrumentation** — full parameter covariance, per-view residual distributions, per-corner residuals, condition number of the normal equations, parameter correlation matrix. | in progress |
| M3 | Out-of-sample error + the cause diagnostics | not started |
| M4 | Task-space translation (mm at working distance) | not started |
| M5 | Report generation, one command | not started |
| M6 | Staleness / drift detection | not started |

Current goal is **M1 + M2 only**, built so M3–M6 drop in without rework.

---

## Non-negotiables carried through every milestone

- Ships as a pip package **and** as a single PyInstaller binary. No dynamic plugin discovery, no `__file__` data loading without frozen-path handling, stdlib argparse, three runtime deps (`numpy`, `opencv-contrib-python`, `pyyaml`).
- Every numeric claim has a test that could fail. Covariance is checked against Monte Carlo, Jacobians against finite differences, the Schur complement against a dense inverse.
- Docstrings are the documentation. Google style, `pdoc` generates the API reference from them with no hand-written duplication.
- Comments explain *why*, never *what*. No paragraph headers over obvious code.

---

## M1 — Ingest · task list

- [ ] Core types with no OpenCV dependency: camera models, detections, target specs, session
- [ ] `TargetSpec` hierarchy: `Checkerboard`, `CharucoBoard`, `CircleGrid`; canonical object points in mm; per-view point IDs so ChArUco's variable subsets are first class
- [ ] Detectors behind one protocol, statically registered
- [ ] Image-set ingest: directory walk, detection, failure accounting
- [ ] Calibration file readers: OpenCV FileStorage, ROS `camera_info`, Kalibr, native JSON
- [ ] Detection file readers: native, generic JSON/NPZ
- [ ] Robot pose ingest with SE(3) validation, for hand-eye
- [ ] Session persistence: single `.npz`, JSON manifest inside, round-trip exact
- [ ] `caltrust ingest` CLI

## M2 — Refit with instrumentation · task list

- [ ] Synthetic rig generator (truth-known cameras, poses, targets, noise) — the test backbone
- [ ] Analytic Jacobian blocks from `projectPoints`, layout verified empirically and against finite differences
- [ ] Fisheye Jacobian
- [ ] Normal equations by view blocks; Schur complement for the intrinsic covariance
- [ ] σ² estimation with the correct DOF count
- [ ] Full covariance: intrinsics, extrinsics, cross terms
- [ ] Correlation matrix and both condition numbers (raw, and Jacobi-scaled — the meaningful one)
- [ ] Rank-deficiency handling: no silent `inv()`, report the null space
- [ ] Per-view residual distributions, per-corner residuals, radial binning
- [ ] Evaluate-without-refit path, for auditing a calibration the user already has
- [ ] `caltrust refit` CLI, `caltrust show`

---

## Verification log

Filled in as tests land. A claim without a row here is not a claim.

| Claim | How it is checked | State |
|---|---|---|
| Analytic Jacobian is correct | central-difference agreement, pinhole and fisheye | pending |
| Covariance is calibrated | Monte Carlo refits vs predicted covariance | pending |
| Marginal stddevs match OpenCV | vs `calibrateCameraExtended` | pending |
| Schur complement is exact | vs dense inverse on a small rig | pending |
| Degeneracy is detected | frontoparallel constant-depth rig shows fx/tz correlation near 1 | pending |
