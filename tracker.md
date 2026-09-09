# caltrust — tracker

**One command against an existing calibration returns a defensible statement of metric accuracy in task units.**

The target sentence, which is the whole product:

> Your measurement of a feature at 800 mm has an expected error of 0.41 mm, with a 95% interval of ±1.12 mm. Your reported 0.24 px reprojection error understates out-of-sample error by 2.9x. The dominant cause is insufficient depth variation across your 18 views: focal length and distance are correlated at 0.94. Adding six views at 400 mm and 1200 mm would reduce the interval to approximately ±0.35 mm.

Everything in this repo exists to make that paragraph true and checkable.

---

## The five failures being fixed

| # | Failure | Milestone that fixes it |
|---|---|---|
| 1 | Reprojection RMS is an in-sample fit statistic | **M3 — held-out reprojection error** |
| 2 | No usable uncertainty on the parameters | **M2 — full covariance, not marginal stddevs** |
| 3 | No diagnosis of the cause | **M4 — the named-cause diagnostic set** |
| 4 | No translation into task space | M5 — mm at a working distance |
| 5 | No staleness detection | M7 — drift monitor |

---

## Milestones

| M | Scope | State |
|---|---|---|
| **M1** | **Ingest** — images + target (checkerboard / ChArUco / circle grid), or existing calibration + detections. Pinhole/Brown-Conrady and fisheye/Kannala-Brandt. Optional robot poses for hand-eye. | in progress |
| **M2** | **Refit with instrumentation** — full parameter covariance, per-view residual distributions, per-corner residuals, condition number of the normal equations, parameter correlation matrix. | in progress |
| **M3** | **Out-of-sample error** — K-fold over views, in-sample and held-out RMS and the ratio between them. | **done** |
| **M4** | **Degeneracy and coverage diagnostics** — one named cause each for pose diversity, depth variation, frontoparallel dominance, image coverage, target scale, distortion model adequacy and per-view leverage. | **done** |
| M5 | Task-space translation (mm at working distance) | not started |
| M6 | Report generation, one command | not started |
| M7 | Staleness / drift detection | not started |

M1 through M4 are complete: 772 tests at 98% statement coverage, running both as
a pip install (three transitive dependencies) and as a PyInstaller binary with
no Python present.

Everything known to be wrong, missing, or resting on an unchecked assumption is
in [open-items.md](open-items.md) — nineteen items, ranked by how much each one
threatens the millimetre claim rather than by how hard it is to fix.

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


---

## M3 — Out-of-sample error · task list

- [x] K-fold assignment over views, shuffled by default and reproducible from a seed
- [x] Fold count chosen so every training set keeps enough views to fit
- [x] Held-out evaluation with each view's pose solved by PnP at the fold's intrinsics, so nothing leaks
- [x] Pooled out-of-sample RMS weighted by point count, not averaged over folds
- [x] In-sample, out-of-sample and the ratio between them
- [x] Per-fold identifiability, which gates the verdict
- [x] Across-fold parameter spread, reported beside the covariance's prediction
- [x] `--cross-validate` on `caltrust refit`, persisted in the fit bundle

### The blind spot this milestone found

**The out-of-sample ratio cannot see the focal-length/depth degeneracy.** On the
frontoparallel rig whose `fx` is wrong by over a thousand pixels, the ratio comes
out at **1.005**. A held-out view solves its own pose, so a proportionally wrong
depth cancels a proportionally wrong focal length and reprojection is perfect.

The plan called the ratio "the most useful thing in the report". It is useful,
and it is not sufficient: any degeneracy a free pose can absorb is invisible to
it. Two things now sit beside it. Per-fold identifiability, which gates the
verdict so the report never calls such a ratio honest. And the across-fold
parameter spread, which is model-free — it resamples the actual views and assumes
nothing about the noise — and which reads 1.15 px on the healthy rig against
326 px on the degenerate one.

That spread is also the only cross-check in the package that does not inherit the
i.i.d. Gaussian noise assumption, which is
[open item 1](open-items.md). It is not unbiased, because folds overlap, so it is
reported as a relative indicator.

## M4 — Diagnostics · task list

- [x] Pose diversity — max pairwise angle between board normals, plus the orientation tensor for "are the normals confined to a plane"
- [x] Depth variation — max-over-min working distance, with the focal-distance correlation from the covariance
- [x] Frontoparallel dominance — fraction of views within 15 degrees of the image plane, linked to the undetermined parameters it causes
- [x] Image coverage — grid occupancy, border-ring occupancy, and radial reach
- [x] Target scale — median nearest-neighbour corner spacing, against the residual sigma
- [x] Distortion model adequacy — omnibus and trend tests on the radial residual profile
- [x] Outlier views — per-view leverage as its share of the intrinsic information, paired with residual
- [x] Aggregation, severity ranking, `caltrust diagnose`, exit code 3 for a critical finding

### Three things M4 had to get right that were not obvious

**Leverage needed a definition.** "Per-view leverage on the objective" is only
well posed once you pick a quantity. The Schur complement is a sum over views,
`S = sum_i S_i`, so `trace(S_i S^-1)` sums to the parameter count and
`trace(S_i S^-1) / p` is that view's share of everything the capture knows about
the intrinsics. Shares sum to one and are directly readable. This needed the
per-view `u_view` blocks added to `NormalEquations`.

**The residual test had to cluster by view.** The obvious test — z-score each
radial bin's mean against `sigma / sqrt(n)` — assumes corners are independent.
They are not: corners in one view share that view's pose, so a small pose error
moves all of them together. Pooling corners reported +6.3 sigma of "structure" on
a capture whose focal length was recovered to 0.6 sigma. Treating the **view** as
the independent unit — one mean per view per band, standard error from the
scatter of those means — is cluster-robust, needs no noise assumption at all, and
brings the same capture to a plausible reading.

**A slope test alone misses truncation.** A truncated radial polynomial leaves a
residual that oscillates in sign, and a linear trend walks straight past it. The
primary criterion is now a chi-square over the bands (Wilson-Hilferty to a
z-score, so no special-function dependency); the slope only describes the shape.
Measured: correct models land between -0.6 and +0.4, a fisheye fitted as pinhole
lands at +5.2.

**And the advice had to stop quoting numbers the fit does not know.** The depth
diagnostic originally recommended "capture at 1030 mm and 2578 mm" on a rig whose
true distance was 800 mm — because the fitted depths scale with the wrong focal
length. The *ratio* survives that rescale, so the finding stands; the absolute
millimetres are now withheld whenever the fit is not identifiable.
