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

Current goal was **M1 + M2 only**, built so M3–M6 drop in without rework.
Both are complete: 635 tests, 98% statement coverage, and the package runs both
as a pip install and as a 63 MB PyInstaller binary with no Python present.

---

## Non-negotiables carried through every milestone

- Ships as a pip package **and** as a single PyInstaller binary. No dynamic plugin discovery, no `__file__` data loading without frozen-path handling, stdlib argparse, three runtime deps (`numpy`, `opencv-contrib-python`, `pyyaml`).
- Every numeric claim has a test that could fail. Covariance is checked against Monte Carlo, Jacobians against finite differences, the Schur complement against a dense inverse.
- Docstrings are the documentation. Google style, `pdoc` generates the API reference from them with no hand-written duplication.
- Comments explain *why*, never *what*. No paragraph headers over obvious code.

---

## M1 — Ingest · task list

- [x] Core types with no OpenCV dependency: camera models, detections, target specs, session
- [x] `TargetSpec` hierarchy: `Checkerboard`, `CharucoBoard`, `CircleGrid`; canonical object points in mm; per-view point IDs so ChArUco's variable subsets are first class
- [x] Detectors behind one protocol, statically registered
- [x] Image-set ingest: directory walk, detection, failure accounting
- [x] Calibration file readers: OpenCV FileStorage, ROS `camera_info`, Kalibr, native JSON
- [x] Detection file readers: native, generic JSON/NPZ
- [x] Robot pose ingest with SE(3) validation, for hand-eye
- [x] Session persistence: single `.npz`, JSON manifest inside, round-trip exact
- [x] `caltrust ingest` CLI

## M2 — Refit with instrumentation · task list

- [x] Synthetic rig generator (truth-known cameras, poses, targets, noise) — the test backbone
- [x] Analytic Jacobian blocks from `projectPoints`, layout verified empirically and against finite differences
- [x] Fisheye Jacobian
- [x] Normal equations by view blocks; Schur complement for the intrinsic covariance
- [x] σ² estimation with the correct DOF count
- [x] Full covariance: intrinsics, extrinsics, cross terms
- [x] Correlation matrix and both condition numbers (raw, and Jacobi-scaled — the meaningful one)
- [x] Rank-deficiency handling: no silent `inv()`, report the null space
- [x] Per-view residual distributions, per-corner residuals, radial binning
- [x] Evaluate-without-refit path, for auditing a calibration the user already has
- [x] `caltrust refit` CLI, `caltrust show`

Added along the way, because M2 was wrong without them:

- [x] **Identifiability, above the standard deviations.** A pseudo-inverse gives a
  rank-deficient direction *zero* variance, so a degenerate fit reports false
  confidence rather than a wide interval. Weak directions come from the null
  space of the Jacobi-scaled Schur complement, and the report leads with them.
- [x] **At-optimum test** via the relative Newton decrement, which is scale
  invariant where a raw gradient norm is not. Matters when instrumenting a
  calibration produced elsewhere: those are frequently not the optimum of their
  own detections.
- [x] **OpenCV flag compatibility by name** (`refit/cv_compat.py`). OpenCV 4 and 5
  disagree about both where the fisheye flags live and what they are worth.
- [x] **Fisheye initialisation ladder.** `cv2.fisheye.calibrate` raises rather than
  degrading when its own initialiser fails, which it does on ordinary captures.

---

## Verification log

Filled in as tests land. A claim without a row here is not a claim.

| Claim | How it is checked | State |
|---|---|---|
| Analytic Jacobian is correct | central differences, 6 cameras × 3 poses, both models | **1e-9 relative** |
| Covariance is calibrated | 250 Monte Carlo refits vs predicted covariance | **sd ratio 0.98–1.02, mean Mahalanobis 9.37 vs 9** |
| Predicted correlations are right | same Monte Carlo, correlation matrices | **max difference 0.12** |
| Marginal stddevs match OpenCV | vs `calibrateCameraExtended` | **6 significant figures** |
| Schur complement is exact | vs dense `σ²(JᵀJ)⁻¹` | **6e-10 relative** |
| Degeneracy is detected | frontoparallel rig | **`identifiable=False`, weak direction `-0.71*fx -0.71*fy`** |
| Rodrigues is correct | vs `cv2.Rodrigues`, 60k rotations incl. the π singularity | **4e-8 worst round-trip** |
| Detectors find real patterns | rendered images, 3 targets × 2 camera models | **0.09–0.4 px mean localisation** |
| The frozen binary works | `dist/caltrust` in a stripped environment | **full pipeline, fx 899.564 ± 0.751** |

### The correction this milestone forced

The problem statement lists "insufficient depth variation" as the cause of the
focal-length ambiguity, with `corr(fx, tz) = 0.94` as its signature. Measuring it
showed that framing is not right, and getting it wrong would produce the wrong
recommendation. From `examples/degeneracy_demo.py`, 18 views, 0.2 px noise:

| rig | RMS | fx error | identifiable |
|---|---|---|---|
| flat, 1 depth | 0.2753 px | +3333.5 px | no |
| flat, 3 depths | 0.2761 px | −139.1 px | **no** |
| 20° tilt, 1 depth | 0.2757 px | +2.3 px | yes |
| 40° tilt, 3 depths | 0.2759 px | +1.3 px | yes |

**Tilt is what makes the focal length identifiable; depth variation alone does
not**, because every added view brings its own free translation and absorbs a
global rescale of focal length and depth together. Depth spread is a genuine
second-order gain — sd(fx) 2.24 → 1.19 — once tilt exists.

And `corr(fx, tz)` is a trap on its own: it reads **0.93 on the good capture**
and **≈0 on the fully degenerate one**, because a cut direction carries no
variance to correlate. M3 must therefore check identifiability first and reach
for the correlation only when the system is non-singular. Recorded so M3
diagnoses tilt before depth.
