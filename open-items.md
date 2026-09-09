# Open items

Everything known to be wrong, missing, or resting on an unchecked assumption.
Written down because a tool whose whole purpose is to expose unstated
assumptions has no business hiding its own.

Ordered by how much each one threatens the product's headline claim — a
defensible statement of measurement error in millimetres — not by how hard it is
to fix.

Last reviewed at the end of M4.

---

## Tier 1 — these threaten the millimetre claim

### 1. The noise model is asserted, never measured

`sigma^2 = cost / (m - p)` assumes corner noise is independent, isotropic and
Gaussian per coordinate. Real corner noise is none of those. It is anisotropic
(elongated along the edge direction), correlated between neighbours (sub-pixel
windows overlap and share image gradients), and heteroscedastic (worse under
blur, high tilt and low contrast).

**And the Monte Carlo test that validates the covariance is circular on this
point.** `tests/test_covariance.py::test_covariance_is_calibrated_against_monte_carlo`
injects noise through `synthesise(noise_px=...)`, which is i.i.d. isotropic
Gaussian — exactly the model the covariance assumes. That test validates the
propagation algebra and says nothing about the assumption. Every interval
caltrust reports is conditional on a noise model nobody has checked against a
real camera.

M3 gives a partial answer that is worth noting: the across-fold parameter spread
resamples the actual views and assumes nothing about the noise. It is not an
unbiased estimate of the sampling deviation, because folds overlap, but it is a
genuine model-free cross-check and it is now reported beside the predicted one.

*Fix:* measure the corner covariance from repeated static captures, or estimate
per-point anisotropic weights from the image gradient structure tensor. Feed
them in as weights — see item 5, which currently blocks this.

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

`Cov([rx, ry, rz, tx, ty, tz])` is fine to report but wrong to propagate. The
rotation-vector parameterisation is a chart, and for large rotations it is
distorted, so pushing this covariance through a task-space Jacobian will be
subtly wrong. M4's task-space translation needs the pose covariance in the local
`se(3)` tangent space at the current pose, which means the right-Jacobian of the
exponential map. Not implemented.

---

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
the fit unweighted. This is what blocks item 1.

### 6. Delegating the fit to OpenCV costs four things at once

No arbitrary parameter subsets (`cx` and `cy` together or neither, never one),
no custom or robust loss, no weights, and a forced float32 downcast on the
pinhole path. A native Levenberg-Marquardt over the Jacobians and Schur solver
already written would remove all four. **This is the highest-leverage remaining
work**: it unblocks items 1, 4, 5, and the board-scale parameter in item 2.

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

### 19. Cross-validation is not tested against a known out-of-sample truth

The M3 tests check internal consistency (folds partition, pooling is
point-weighted, the ratio rises with over-parameterisation) and the degeneracy
blind spot. What is missing is a case where the true out-of-sample error is known
independently, so the reported ratio can be checked for accuracy rather than
plausibility. A held-out set generated from the same truth camera but never
offered to any fold would do it.
