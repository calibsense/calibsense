# Open items

Everything known to be wrong, missing, or resting on an unchecked assumption.
Written down because a tool whose whole purpose is to expose unstated
assumptions has no business hiding its own.

Ordered by how much each one threatens the product's headline claim — a
defensible statement of measurement error in millimetres — not by how hard it is
to fix.

Last reviewed at the end of M7, with item 1 revisited after it was measured.

---

## Tier 1 — these threaten the millimetre claim

### 1. The noise model is now measured, and only one violation of it matters

This item used to say the noise model was asserted and never checked. It has
been measured, and the measurement changed the shape of the problem, so what
follows is the answer rather than the worry.

`sigma^2 = cost / (m - p)` estimates the noise *scale* from the data but asserts
its *shape*: one variance per coordinate, the same in x and y, uncorrelated
between points. Monte Carlo against known truth, 14 and 30 views, sigma = 0.25
px, injecting one violation at a time and comparing the predicted intrinsic
deviation against the empirical spread of repeated refits:

| Violation | predicted / empirical | 95% interval covers |
|---|---|---|
| i.i.d. isotropic (the assumption) | 1.01–1.08 | 94–97% |
| anisotropic, 3:1 along the edge direction | 0.98–1.13 | 94–97% |
| heteroscedastic across views, scale `exp(N(0,0.6))` | 1.04–1.17 | 95–97% |
| heavy tails, 5% of corners 5x worse | 0.97–1.04 | 93–96% |
| per-view radial wobble | 0.96–1.07 | 94–96% |
| **spatially correlated field, 60 px length** | **0.32–0.51** | **47%** |
| **spatially correlated field, 200 px length** | **0.12–0.39** | **15%** |

So the three violations this item used to name are all harmless. Several hundred
corners at varied orientations average anisotropy away, and `cost / dof` picks up
whatever average power heteroscedasticity leaves. The mechanism it named for
correlation was also wrong: `cornerSubPix` uses an 11 to 21 px window and corners
sit 60 px apart or more, so the windows never overlap.

**What does break the covariance is spatial correlation of the noise across the
frame, and it breaks it by a factor of eight.** The real sources are
field-varying defocus and astigmatism, illumination and vignetting gradients
pulling on the sub-pixel estimator, board non-flatness, motion blur and rolling
shutter — the last two being item 14, which this makes considerably more
important than "listed and absent".

The failure is silent in the worst possible way. A correlated field is partly
absorbable by the pose and distortion parameters, so it *lowers* the residual
while *raising* the estimator's real spread: at a 200 px correlation length
`sigma` fell from 0.25 px to 0.055 px while the spread of `fx` rose from 3.0 px
to 5.2 px. A beautiful RMS and an interval 7.7x too tight. Great RMS, wrong
answer, no warning — which is the headline failure this whole tool exists to
catch, and it was invisible.

**The old proposed fix was aimed at the wrong target.** Weights fix *efficiency*,
not *honesty*: under correlated noise `sigma^2 (J'WJ)^-1` is still wrong however
good the weights are, because the error lives in the off-diagonal of the noise
covariance and reweighting a diagonal cannot repair a correlation. Separately,
`refit.normal.assemble` takes one scalar per point, so it could not express
anisotropy even with item 5 unblocked. **This item is therefore not blocked by
items 5 or 6, and never was.**

*Done instead:* `refit.covariance.RobustCovariance`, a view-clustered sandwich
that assumes only that different views are independent — the same argument
`diagnose/model.py` already makes for the radial profile, applied to the
parameters. After eliminating each view's own pose, view `i` scores
`s_i = A_i'r_i - Y_i'(B_i'r_i)` and

    Cov = S^-1 (sum_i s_i s_i') S^-1 * G / (G - 1)

Nothing is assumed about the noise *within* a view. Measured:

| | classical | sandwich |
|---|---|---|
| i.i.d., 14 views | 0.98–1.06 | 0.67–0.96 |
| correlated 200 px, 14 views | **0.12–0.40** | 0.57–0.90 |
| i.i.d., 30 views | 0.97–1.08 | 0.83–0.99 |
| correlated 200 px, 30 views | **0.13–0.46** | 0.79–1.05 |

The ratio between the two estimates is reported as the `noise_model` finding,
which turns this item from an unfalsifiable caveat into a per-capture check that
runs on the user's own data. Agreement means the assumption held *here*, by
measurement. `tests/test_covariance.py` now injects a correlated field and pins
both halves: that the classical covariance fails coverage, and that the robust
one recovers it.

Three limitations, all real:

* The sandwich stays mildly optimistic at low view counts — the standard
  shrinkage of residuals at a fitted optimum, only partly undone by `G/(G-1)`.
  It is the right direction to be wrong in only because what it replaces was out
  by a factor of eight. A delete-one-view jackknife measured conservative
  instead (0.88–1.46) at the cost of `G` refits; worth adding if the optimism
  ever matters.
* It is **intrinsics only**. Each view's pose is estimated from that view's own
  residuals, so there is one cluster per pose and no between-cluster scatter to
  build a pose block from. See item 1b.
* It is blind to error that is *identical* across views. A board scale error
  moves every view coherently and contributes nothing to the between-view
  scatter, so it stays invisible here. That is item 2, and the two are
  complementary rather than alternatives.

M3's across-fold parameter spread remains a second model-free cross-check, still
not unbiased because the folds overlap.

### 1b. The noise-model inflation is reported but not propagated

The `noise_model` finding says every interval is, say, 3.2x too tight. It does
not widen them. Nothing downstream consumes the ratio: `task.sampling` still
factorises the classical joint covariance, so the task-space millimetres — the
product's headline number — carry the understatement the finding just named.

The reason it was left is that the honest scaling is not obvious. The sandwich
covers the intrinsics only, so inflating the joint covariance means either
scaling the pose and cross blocks by a factor measured from a different block,
or scaling the whole thing by one scalar. Both are defensible; neither is
derived. Item 3b's principle applies — inventing a number would be worse than
saying so — so for now the finding states the factor and tells the reader to
apply it.

*Fix:* the cleanest route is probably to sample from the robust intrinsic block
directly and keep the classical conditional structure for the poses, which is
consistent to first order and needs no invented scalar. Wants checking against
Monte Carlo in task space before it goes in.

### 2. Board geometry is treated as exact

`TargetSpec.object_points()` returns the nominal grid. A printed board's real
error — 50 to 200 micrometres of print scale and flatness on a typical target —
appears nowhere in the covariance. This is a *systematic* error on metric scale:
a 0.1% board scale error goes straight into `fx` and therefore straight into
millimetres. For a claim like "0.41 mm at 800 mm" it could plausibly dominate
everything currently modelled.

*Fix:* a board-scale nuisance parameter with a prior, at minimum. Self-calibration
of the board is the fuller answer and needs the native solver from item 6.

### 3. Extrinsic covariance lives in a chart, not the tangent space

`Cov([rx, ry, rz, tx, ty, tz])` is fine to report but only locally valid to
propagate. The rotation-vector parameterisation is a chart, and for large
rotations it is distorted.

Partly addressed since M4. The hand-eye solve updates both transforms by
right-multiplied increments, which is the well-behaved chart, and its
covariance is expressed in that frame. But `task.sampling.CovarianceSampler`
still perturbs the calibration's per-view poses *additively* in rotation-vector
coordinates, which is consistent with the linearisation the covariance came from
and therefore correct to first order, and wrong in the tail for a view at a
large rotation. A view at 150 degrees of board roll is where this would first
show up.

### 3b. Three inputs are treated as exact, and two of them usually dominate

Each is stated in the task description that relies on it, and each is a real
limitation rather than an oversight:

| Input | Where | Why it is not modelled |
|---|---|---|
| Stereo baseline | `StereoTriangulation` | caltrust does not do stereo calibration, so there is no covariance for it. For a real rig the baseline's own uncertainty usually dominates the triangulated depth. |
| Robot flange pose | `CameraToBase` | Robot repeatability is a specification of the arm. Inventing a number would be worse than saying so, but a cell integrator has that number and there is nowhere to put it. |
| Board geometry | everywhere | See item 2. |

The flange pose is the easiest to fix: accept a repeatability figure and sample
it. Doing so would also let the task report a three-way variance split rather
than the current two.

### 3c. The variance split lumps hand-eye in with the calibration

`TaskResult.variance_share` returns two numbers, and for `CameraToBase` the
first covers both the camera calibration and the hand-eye transform. For a robot
cell the interesting question is which of those two to spend money on, and the
report cannot currently answer it. A third run with only the hand-eye varying
would, at the cost of a fourth of the runtime.

---

### 3d. The hand-eye residual covariance is optimistic, and still reachable

Measured against known truth: the residual-based covariance gave a translation
deviation of 0.39 mm against an actual error of 1.29 mm, a factor of 3.1. The
cause is that it treats the target-in-camera poses as exact data when they come
from the calibration, and their errors are correlated across views because every
view shares the same intrinsics — a focal-length error tilts and scales all of
them coherently, so it does not average down.

`solve_hand_eye` now defaults to `monte_carlo=True`, which resamples whole
calibrations and lands at 1.22 mm. The residual path remains available for
speed and is labelled optimistic wherever it is reported, but nothing stops a
caller quoting it.

The same correlation problem almost certainly affects the *calibration*
covariance in the other direction, and that is item 1.

## Tier 2 — machinery that exists but cannot be reached

### 4. Outliers are detected and then ignored

M4 now names them, ranks them by leverage, and says which combine high leverage
with a high residual. Nothing down-weights or removes them. `sigma^2` is inflated
by a bad view, so *every* parameter's interval widens instead of the bad view
being excluded. There is no robust loss at all. The `outlier_views` finding says
this out loud, which is honest but not a fix.

### 5. Per-point weights are unreachable

`refit.normal.assemble()` accepts and correctly applies them, with tests.
`refit.engine.instrument()` never passes any, and OpenCV's optimiser cannot use
them regardless — so even wiring it up would weight the covariance while leaving
the fit unweighted.

This used to be listed as what blocks item 1. It is not: weights buy efficiency,
not honest intervals, and the measurement in item 1 shows the departures they
would correct are the ones that did not matter. They also cannot express
anisotropy as the signature stands — one scalar per point, repeated across x and
y — so an anisotropic weight needs a 2x2 per point and a signature change first.

### 6. Delegating the fit to OpenCV costs four things at once

No arbitrary parameter subsets (`cx` and `cy` together or neither, never one),
no custom or robust loss, no weights, and a forced float32 downcast on the
pinhole path. A native Levenberg-Marquardt over the Jacobians and Schur solver
already written would remove all four. **This is the highest-leverage remaining
work**: it unblocks items 4, 5, and the board-scale parameter in item 2. It is
no longer needed for item 1, which turned out not to require weights at all.

---

## Tier 3 — thresholds, and signals that do not reach the reader

### 7. Nine thresholds, calibrated to varying degrees

| Threshold | Value | Basis |
|---|---|---|
| `WEAK_DIRECTION_THRESHOLD` | 1e-6 | judgement; it is a **cliff** — it decides whether a user reads "unidentifiable" or "sd(fx) = 636 px" |
| `DEFAULT_RCOND` | 1e-12 | conventional |
| `OUTLIER_Z` | 3.5 | conventional for a MAD scale |
| `OPTIMUM_DECREMENT_TOLERANCE` | 1e-6 | judgement |
| `DIVERSITY_*_DEGREES` | 5 / 20 | **measured** against `examples/degeneracy_demo.py` |
| `DEPTH_*_RATIO` | 1.2 / 1.5 | 1.5 came from the plan; 1.2 is judgement |
| `FRONTOPARALLEL_DEGREES` | 15 | from the plan |
| `COVERAGE_*`, `REACH_*` | 0.35 / 0.65 / 0.5 | judgement, uncalibrated |
| `SPACING_*_PX` | 10 / 20 | reasoned from noise-over-spacing, not measured |
| `FLATNESS_*_Z`, `SLOPE_*_T`, `BIN_*_Z` | 3 / 5 | conventional z-score cuts |

The diversity thresholds are the only ones tied to a measured relationship
between the metric and actual parameter error. The rest are defensible but
unvalidated, and the coverage ones in particular fire CRITICAL on captures whose
focal length is recovered to 0.6 sigma.

### 8. Condition-number interpretation is asserted, not measured

The README says "roughly 1e3 is well posed; 1e8 and up is not". That has never
been measured against metric error. It should be a curve, produced the same way
the diversity thresholds were.

### 9. `model_ambiguous` is a dead signal

`ingest/readers/opencv_fs.py` correctly detects that a 4-coefficient OpenCV
FileStorage file with no declared model is genuinely ambiguous between
Brown-Conrady and Kannala-Brandt — and then nothing shows it to anyone. It is
computed, tested, and never surfaced in any report. This is exactly the silent
wrongness the tool exists to catch, sitting unreported in a metadata dictionary.

### 10. Two definitions of "working distance"

`InstrumentedFit.working_distances_mm()` measures to the board frame's origin
corner; `DiagnosticContext.board_distances_mm` measures to the centre of the
point pattern. For a 200 mm board these differ by over 100 mm. The diagnostics
use the correct one and the refit summary uses the other, so the two report
different depth ranges for the same capture. Unify in M5, when the single report
is assembled.

### 11. Checkerboard 180-degree ambiguity is unresolved

Harmless for intrinsics, because each view's pose absorbs the flip. **Not**
harmless for hand-eye, which is the stated reason M1 ingests robot poses. A
ChArUco target has no such ambiguity, which is the current recommendation, but
the checkerboard path is quietly unsafe for the one downstream use it was built
for.

### 12. ChArUco `legacy_pattern` cannot be auto-detected

A wrong setting produces a confidently wrong calibration rather than a detection
failure. Documented in the target docstring; undetected. Detecting it is
feasible — the two layouts place markers differently — and worth doing.

### 13. Circle-grid centroid bias is uncorrected

The centroid of a projected circle is not the projection of the circle's centre.
Under tilt this is a systematic sub-pixel bias that grows with obliquity, and it
biases exactly the tilted views that the pose-diversity diagnostic asks for.
Documented in the detector and deferred.

---

## Tier 4 — scope not yet reached

### 14. Motion blur and rolling shutter are not diagnosed

Listed as a cause in the original problem statement and absent from both the M4
task list and this implementation. Blur is detectable from the image gradient
distribution around detected corners; rolling shutter shows as a shear
correlated with target velocity, which needs timestamps.

### 15. Single camera only

No stereo, no multi-camera rigs. A Kalibr chain is read one camera at a time and
the extrinsics between cameras are ignored entirely.

### 16. Robot poses align by view id only

No timestamp interpolation or sync. Real robot logs are timestamped at a
different rate from the camera, and nearest-neighbour matching on ids assumes
someone already did that work.

### 17. Hand-eye calibration itself is not run

M1 asked for ingest and validation only, and that is what exists. `RobotPoses`
normalises to gripper-to-base and aligns to views, ready for
`cv2.calibrateHandEye`, but nothing calls it.

---

## Performance, and what it costs the report

### 20. The hand-eye Monte Carlo is slow because its Jacobian is numerical

`_jacobian` evaluates the residual set twenty-four times per Gauss-Newton step,
and the resampled covariance does three steps per sample. At 200 samples over 16
views that is about seven seconds. The analytic Jacobian needs the adjoint of
four composed transforms per view; it was left numerical on purpose, because the
analytic form is easy to get subtly wrong and the test suite could not have
caught a sign error in it as cleanly. Worth doing once there is a reference to
check it against.

### 21. `PlaneLocation` dominates a report's runtime

It solves a pose per sample, three times per sample for the variance
decomposition, so 1500 samples take about 1.3 seconds against 0.06 for
`LengthAtDepth`. A full report with three tasks is a few seconds; a report with
several plane tasks at high sample counts is not.

### 22. Reports are reproducible to about 1e-6 relative, not bit-exact

`cv2.calibrateCamera` reduces across threads and floating-point addition is not
associative, so eight identical refits produced eight different focal lengths
spanning 1.1e-12 px on a ten-thread machine. `--deterministic` pins OpenCV to
one thread and makes output byte-identical, at a cost in speed. The PDF footer
states the situation rather than claiming exact reproducibility.

## Known limitations of the PDF writer

### 23. Only CP1252 characters render

The base-14 fonts are used without embedding, which is what keeps the
dependency count at three. Anything outside CP1252 — a Greek letter in a camera
name, a CJK character in a plant name — becomes `?`. Fixing it means embedding a
TrueType font, which means either a data file in the package or a dependency,
and both were traded away deliberately.

### 24. The only chart is a horizontal bar

Enough for the variance split, which is the one place a picture helps. A radial
residual profile would be better shown as a plot than as the table it currently
is, and the writer has the primitives for it (rules and filled rectangles) but
no line-plot helper.

## Known limitations of the test infrastructure

### 18. The renderer introduces its own sub-pixel bias

`tests/rendering.py` warps a board texture and then resamples it to apply
distortion — two interpolation stages. On the rendered capture used in the
README, the distortion-model diagnostic reports a radial departure from flat at
z = +4.4, and that is very likely the renderer's bias rather than a real model
mismatch. It does not affect any assertion in the suite, because no test asserts
that diagnostic is clean on rendered images, but it means rendered images cannot
be used to validate sub-0.05 px effects.

*Fix:* render at higher resolution and downsample, or compute the distorted
sample positions in one composed map instead of two.

### 25. The forecast is not validated against an actually-captured improvement

`test_the_forecast_makes_a_degenerate_capture_identifiable` checks that the
forecast says the right *direction* and that the extended capture really is
identifiable. What is not checked is calibration: if the forecast says the
interval will fall to +/-0.35 mm, nobody has confirmed that capturing those
views really produces +/-0.35 mm. The check is possible — generate the extra
views from the *truth* camera rather than the fitted one, refit, and compare —
and it is the one claim in the report that is a prediction rather than a
measurement.

### 19. Cross-validation is not tested against a known out-of-sample truth

The M3 tests check internal consistency (folds partition, pooling is
point-weighted, the ratio rises with over-parameterisation) and the degeneracy
blind spot. What is missing is a case where the true out-of-sample error is known
independently, so the reported ratio can be checked for accuracy rather than
plausibility. A held-out set generated from the same truth camera but never
offered to any fold would do it.
