# caltrust, end to end

What each API call does, in the order you would call them. Every number in the
examples is real output from `docs/../examples`, run against a 16-view capture of
a 9x6 checkerboard.

---

## The mental model

Four objects, produced in order. Each stage takes the previous one and adds
something the previous one could not know.

```
   images or a calibration file
              |
              |  session_from_images()          M1  ingest
              v
     CalibrationSession  ................  what was observed
              |
              |  instrument()                   M2  refit
              v
       InstrumentedFit  .................  what the parameters are, and how well known
              |
      +-------+-------+-------------+
      |               |             |
      |  cross_validate()   diagnose()      propagate()
      |       M3            M4               M5
      v               v             v
 CrossValidation   Diagnosis    TaskResult
  is the RMS       what is        how many
  honest?          wrong?         millimetres?
      |               |             |
      +-------+-------+-------------+
              |
              |  run_audit()  wraps all of the above   M7  report
              v
            Audit  ->  write_pdf() / write_json()
```

Hand-eye (M6) hangs off `InstrumentedFit` too, and feeds one of the M5 tasks.

**The one rule.** Read `fit.conditioning.identifiable` before you read any other
number. When it is `False`, every deviation, interval and ratio downstream is a
lower bound rather than an answer, because a pseudo-inverse assigns *zero*
variance to a direction the data does not constrain — so an undetermined
calibration reports false confidence, not a wide error bar. Every object in the
chain carries that flag forward (`sampler.bounded`, `validation.trustworthy`,
`audit.trustworthy`) and every printed statement says so.

---

## The short path

Five calls do everything.

```python
from caltrust import session_from_images, instrument, cross_validate, diagnose
from caltrust.cli.targets import resolve_target
from caltrust.task import LengthAtDepth, propagate

session    = session_from_images("captures/", resolve_target("checkerboard:9x6:25mm"))
fit        = instrument(session)
validation = cross_validate(session)
findings   = diagnose(fit, session.observations, validation)
error      = propagate(fit, LengthAtDepth(depth_mm=800.0, length_mm=100.0))
```

Or one call, if you want the report rather than the pieces:

```python
from caltrust.report import run_audit, write_pdf, write_json

audit = run_audit(session, tasks=[LengthAtDepth(800.0, 100.0)])
write_pdf(audit, "audit.pdf")
write_json(audit, "audit.json")
```

---

## M1 — ingest

Two entry points, one output. Nothing downstream knows which route was taken.

```python
session_from_images(
    images: str,                      # a folder, or one image
    target: TargetSpec,
    options: DetectorOptions = None,
    calibration: str = None,          # the calibration you already ship
    calibration_format: str = None,   # force a reader: opencv|ros|kalibr|native
    robot_poses: str = None,          # JSON or CSV, for hand-eye
    robot_convention: str = "gripper2base",
    robot_units: str = "mm",
    recursive: bool = True,
    progress: Callable = None,        # (index, total, view_id, error_or_none)
    metadata: Mapping = None,
) -> CalibrationSession

session_from_calibration(
    calibration: str,                 # the file to audit
    detections: str,                  # the corners behind it
    target: TargetSpec = None,        # if the detections file names none
    image_size: tuple = None,
    ...
) -> CalibrationSession
```

**Targets.** Build them directly or from shorthand:

```python
from caltrust import Checkerboard, CharucoBoard, CircleGrid
from caltrust.cli.targets import resolve_target

Checkerboard(columns=9, rows=6, square_size=25.0, units="mm")
CharucoBoard(squares_x=8, squares_y=11, square_size=20.0, marker_size=15.0)
CircleGrid(columns=4, rows=11, spacing=20.0, asymmetric=True)

resolve_target("checkerboard:9x6:25mm")             # shorthand, or a file path
resolve_target("charuco:8x11:20mm:15mm:DICT_5X5_1000")
resolve_target("circles:4x11:0.75in:asymmetric")
```

`DetectorOptions(refine=True, refine_window=5, min_points=8, fast=False)` tunes
detection. `min_points` is the floor for keeping a view; four is the algebraic
minimum for a pose and eight is the default because a view at the minimum
contributes noise rather than information.

**What you get back.**

```python
session.target.describe()          # '9x6 checkerboard, 25 mm squares'
session.observations.n_views       # 16
session.observations.total_points  # 864
session.observations.summary       # attempted, failures with reasons, detector
session.prior                      # the calibration you handed in, or None
session.has_hand_eye               # whether robot poses came along
session.summary_lines()            # a few readable lines
```

Persist it with `save_session(session, "s.npz")` / `load_session("s.npz")`. One
file, a JSON manifest inside, loaded with `allow_pickle=False` so opening a
result can never execute code from it.

**Lower-level pieces**, if you are assembling a session yourself:
`find_images`, `read_image`, `detect_in_images`, `detector_for`,
`read_calibration`, `read_detections`, `read_robot_poses`, and the matching
`write_*`.

---

## M2 — refit with instrumentation

```python
instrument(session, options: RefitOptions = None) -> InstrumentedFit
```

`RefitOptions` controls the fit:

| field | default | effect |
|---|---|---|
| `model` | `None` | `"pinhole"` or `"fisheye"`; `None` takes it from the session's calibration, else pinhole |
| `distortion_terms` | `5` | 4, 5, 8, 12 or 14. Passing 4 means five coefficients with `k3` held at zero |
| `fixed` | `()` | parameter names to hold; see the asymmetry note below |
| `tie_aspect` | `False` | hold `fy/fx` constant — a *proportional* constraint, not a fix |
| `refit` | `True` | `False` instruments the session's existing calibration in place |
| `use_prior_as_guess` | `True` | start the optimiser from the session's calibration |
| `rcond` | `1e-12` | relative eigenvalue cut on every inverse taken |
| `radial_bins` | `10` | bins in the radial residual profile |
| `check_condition` | `False` | ask OpenCV to reject ill-conditioned fisheye views |

> **One asymmetry to know.** `fixed=("k2",)` holds `k2` at its supplied value on
> the pinhole path and sets it to **zero** on the fisheye path. That is OpenCV's
> behaviour, not caltrust's. Either way the parameter leaves the covariance, so
> the reported uncertainty is right; only the resulting value differs.

### Reading the result

```python
fit.camera                       # PinholeBrownConrady or FisheyeKannalaBrandt
fit.poses                        # board-to-camera Pose per view
fit.rms                          # 0.1221 px  <- the number you already had
fit.summary_lines()              # the whole thing, readable

fit.conditioning.identifiable    # True   <- READ THIS FIRST
fit.conditioning.scaled_condition_number   # 9.0e+02, unit-free and meaningful
fit.conditioning.condition_number          # 1.2e+08, unit-dependent, ignore it
fit.conditioning.weak_directions           # what the capture cannot determine
fit.conditioning.participation             # per-parameter share of that subspace

fit.parameter_table()            # [(name, value, sd, weak_participation), ...]
fit.at_optimum                   # False when auditing someone else's fit
fit.relative_decrement           # how much a Newton step would still gain
```

The covariance is the point of M2, and it is kept factored so memory is linear
in the number of views rather than quadratic:

```python
cov = fit.covariance
cov.sigma                        # 0.149 px per coordinate
cov.intrinsic_std()              # sd of each free intrinsic
cov.intrinsic_correlation()      # the correlation matrix
cov.strongest_correlations(6)    # [('fx','fy', 0.99), ('k2','k3', -0.98), ...]

cov.correlation_with_poses("fx")[:, 5]     # corr(fx, tz) per view -> 0.928
cov.weak_directions()            # e.g. '-0.71*fx -0.71*fy' on a flat capture
cov.parameter_participation()    # 0.71 for fx and fy there
cov.is_identifiable()

cov.extrinsic_block(i)           # (6,6) for one view's pose
cov.cross_block(i)               # (p,6) intrinsics against that pose
cov.pose_cross_block(i, j)       # (6,6) between two views
cov.dense()                      # the whole matrix, if you want it
```

Residuals, which an RMS collapses away:

```python
fit.residuals.rms                # overall
fit.residuals.per_view           # rms, p95, max, signed bias, robust z, per view
fit.residuals.outlier_views()    # views whose error stands out
fit.residuals.worst_views(5)
fit.residuals.residuals          # every corner residual, (n, 2)
fit.residuals.radial             # binned by image radius, radial vs tangential
```

`save_fit` / `load_fit` persist it. The covariance is *not* stored — it is
rebuilt from the stored normal equations and `rcond`, so a saved covariance can
never disagree with the equations it came from.

---

## M3 — out-of-sample error

```python
cross_validate(
    session, options=None, folds=None, shuffle=True, seed=0
) -> CrossValidation
```

K-fold over views. Each fold fits on the rest and predicts the held-out views by
solving each one's pose with PnP at that fold's intrinsics — which is the
deployment-relevant question and also what stops a view leaking into its own
prediction. Pooled by point count, not averaged across folds.

```python
validation.in_sample_rms         # 0.1221 px
validation.out_of_sample_rms     # 0.1231 px
validation.ratio                 # 1.008
validation.trustworthy           # True  <- gates whether the ratio means anything
validation.degenerate_folds      # folds whose training set was undetermined

validation.spread_of("fx")       # 0.679  model-free, assumes nothing about noise
validation.predicted_of("fx")    # 0.751  what the covariance predicted
validation.spread_table()        # both, per parameter
validation.folds                 # per-fold camera, train/test ids, both RMS values
validation.worst_views(5)        # held-out views predicted worst
validation.summary_lines()
```

> **The ratio has a blind spot, and it is large.** Any degeneracy a held-out
> view's own free pose can absorb is invisible to it. On a frontoparallel capture
> whose `fx` is wrong by 3333 px, the ratio comes out at **1.005** — a
> proportionally wrong depth cancels a proportionally wrong focal length and
> reprojection is perfect.
>
> That is why `trustworthy` exists and why `spread_of()` sits beside the ratio.
> On that same capture the fold spread of `fx` reads **326 px** against 1.15 px
> on a healthy one. Check `trustworthy` first; if it is `False`, ignore the ratio
> and read the spread.

`fold_assignments`, `choose_folds` and `evaluate_held_out` are exposed if you
want to drive the folds yourself.

---

## M4 — diagnostics

```python
diagnose(fit, observations, cross_validation=None, diagnostics=None) -> Diagnosis
```

Eight checks, each returning one `Finding` whether or not it found anything.

| cause slug | measured as |
|---|---|
| `out_of_sample_error` | M3's ratio, gated on per-fold identifiability |
| `pose_diversity` | max pairwise angle between board normals, plus the orientation tensor |
| `frontoparallel_dominance` | fraction of views within 15 deg of the image plane |
| `depth_variation` | max-over-min working distance, with `corr(fx, tz)` |
| `image_coverage` | grid occupancy, border-ring occupancy, radial reach |
| `target_scale` | median nearest-neighbour corner spacing against the residual sigma |
| `distortion_model` | chi-square and trend tests on the radial residual profile |
| `outlier_views` | each view's share of the intrinsic information, with its residual |

```python
findings.verdict()               # one line naming the critical causes
findings.severity                # Severity.OK | NOTE | WARNING | CRITICAL
findings.critical                # the ones that invalidate something
findings.warnings, findings.notes, findings.passing
findings.ranked()                # worst first
findings.summary_lines(include_ok=False)

f = findings.by_cause("depth_variation")
f.summary                        # what was measured, with the numbers in it
f.action                         # what to do about it
f.metrics                        # {'depth_ratio': 2.55, 'nearest_mm': 411.0, ...}
```

> **Tilt before depth.** The plan blamed the focal-length ambiguity on
> insufficient depth variation. Measured against known truth, that is not right:
> a flat board at three working distances leaves `fx` undetermined, because every
> added view brings its own free `tz` and absorbs a global rescale. Tilt is what
> makes the focal length identifiable; depth spread halves the deviation
> afterwards. The recommendation order follows the measurement, not the plan.

The primitives are exposed too, if you want to compute one yourself:
`orientation_tensor`, `normal_spread_degrees`, `occupancy`,
`nearest_neighbour_spacing`, `clustered_radial_profile`, `flatness_z`,
`radial_trend`, `view_influences`.

---

## M5 — task-space error

```python
propagate(
    fit, task, n_samples=2000, observation_noise_px=None, seed=0
) -> TaskResult
```

Monte Carlo. Whole parameter sets are drawn from the *joint* covariance, not one
deviation at a time — sampling `fx` and `tz` from their marginals independently
would give a spread far wider than reality, since the two correlate at 0.93 and
their errors partly cancel in most tasks.

`observation_noise_px` defaults to the fit's own residual sigma, which is the
corner noise caltrust measured on this camera. Pass `0.0` to isolate the
calibration.

### The four tasks

```python
from caltrust.task import LengthAtDepth, PlaneLocation, StereoTriangulation, CameraToBase

LengthAtDepth(depth_mm=800.0, length_mm=100.0,
              centre_mm=(0.0, 0.0), orientation_deg=0.0)
# -> length_mm.  The headline task. Relative length error equals relative focal
#    length error to first order, so it is sensitive to exactly what a flat
#    capture fails to determine.

PlaneLocation(depth_mm=800.0, tilt_deg=25.0, extent_mm=200.0, grid=(5, 5))
# -> distance_mm, tilt_deg.  These fail differently: a focal error moves the
#    plane without tilting it, a principal-point error tilts it without moving it.

StereoTriangulation(baseline_mm=200.0, depth_mm=800.0, lateral_mm=(0.0, 0.0))
# -> depth_mm, range_mm.  The baseline is treated as EXACT: caltrust did not
#    measure it, and for a real rig its own uncertainty usually dominates.

CameraToBase(hand_eye_result=result, depth_mm=800.0, flange=pose)
# -> x_mm, y_mm, z_mm, position_mm.  Needs M6. The flange pose is treated as
#    EXACT, because robot repeatability is a spec of the arm, not a measurement.
```

Each task is specified as a *scene*, not as observations. The scene is projected
once through the fitted camera to fix the pixels a sensor would have seen, then
measured back under every sampled calibration. Re-projecting the scene with each
sample would ask a different and much less useful question.

### Reading the result

```python
d = error.distribution("length_mm")
d.nominal                        # 100.0, the measurement at the fitted calibration
d.expected_error                 # 0.107 mm, mean absolute error
d.half_width()                   # 0.267 mm, the +/- figure
d.interval(0.95)                 # (-0.26, 0.27)
d.std, d.rms_error, d.bias
d.bounded                        # False -> everything above is a lower bound
d.statement()                    # the product sentence, ready to print

error.variance_share("length_mm")   # (0.35, 0.65)
error.parameter_source_label        # 'the calibration' or '...and hand-eye transform'
error.distribution("length_mm", "parameters")   # calibration varying only
error.distribution("length_mm", "noise")        # pixels varying only
error.summary_lines()
```

> **The variance split is the most useful part.** The calibration and the pixel
> noise are independent, so their variances add, and separating them answers the
> question that follows every error figure. At 35% calibration and 65% pixel
> noise, re-calibrating this camera can improve the answer by about a third and
> no more.

`CovarianceSampler(fit, view_indices, seed, hand_eye)` is exposed if you want the
parameter draws for something caltrust does not model. `sampler.bounded` is the
flag to check.

---

## M6 — hand-eye

```python
solve_hand_eye(
    fit, session, mounting="eye_in_hand",
    rcond=1e-12, monte_carlo=True, n_samples=200, seed=0
) -> HandEyeResult

diagnose_hand_eye(result, robots, boards) -> Diagnosis
```

`mounting` is `"eye_in_hand"` (camera on the moving flange) or `"eye_to_hand"`
(camera bolted to the cell, target on the flange). Both transforms come out,
because neither is a nuisance parameter in practice.

```python
result.camera                    # flange->camera, or base->camera
result.target                    # base->target, or flange->target
result.covariance                # (12,12), camera then target, rad and mm
result.camera_covariance         # (6,6)
result.translation_std_mm()      # per axis
result.rotation_std_deg()
result.identifiable              # do the robot motions determine all twelve?
result.covariance_method         # 'monte_carlo' or 'residual'
result.optimism_factor()         # how much wider than a residual-only estimate
result.summary_lines()
```

> **Two things worth knowing.**
>
> The residual-based covariance is **3.1x optimistic** — measured 0.39 mm
> predicted against a 1.29 mm actual error. It treats the target-in-camera poses
> as exact data when they come from the calibration, and their errors are
> correlated across views because every view shares the same intrinsics.
> `monte_carlo=True` resamples whole calibrations and lands at 1.22 mm. It costs
> about seven seconds; leave it on for any number you will quote.
>
> Hand-eye is determined by **rotation and nothing else**. A robot that only
> translates determines nothing; one that rotates about a single axis determines
> the camera rotation only up to a rotation about that axis. That is why the four
> sufficiency checks exist, and every degenerate pose set pairs a small predicted
> deviation with a large real error:
>
> | pose set | true error | predicted sd | caught by |
> |---|---|---|---|
> | varied axes | 2.2 mm | 1.5 mm | (clean) |
> | mispaired robot poses | 367 mm | 1.9 mm | `hand_eye_pairing` |
> | one rotation axis | 88 mm | 1.5 mm | `hand_eye_rotation_axes` |
> | tiny wrist rotations | — | — | the solve refuses outright |

The four causes are `hand_eye_pairing`, `hand_eye_rotation_axes`,
`hand_eye_rotation_magnitude` and `hand_eye_conditioning`. `closed_form` and
`relative_motions` are exposed for inspection.

---

## M7 — the report

```python
run_audit(
    session, options=None, tasks=None, metadata=None,
    folds=None, n_samples=2000, mounting=None, seed=0, forecast_samples=800
) -> Audit
```

One call for refit, cross-validation, diagnosis, propagation, the forecast, and
hand-eye when `mounting` is given and the session carries robot poses. Cross-
validation is skipped without failing when there are too few views to hold any
back — the `out_of_sample_error` finding then reports its own absence.

With no `tasks`, it measures a 100 mm feature at the capture's own median working
distance, rounded to 50 mm.

```python
audit.headline()                 # the paragraph, as sentences
audit.trustworthy
audit.severity
audit.dominant_cause()           # the worst finding that names a cause, not a symptom
audit.tasks, audit.validation, audit.diagnosis, audit.fit
audit.prediction                 # the forecast, or None
audit.hand_eye, audit.hand_eye_diagnosis
audit.to_dict()                  # everything, JSON-ready
```

`ReportMetadata(title, camera_name, prepared_by, contact, open_question)` frames
the document. `contact` and `open_question` have defaults that are meant to be
replaced.

### The two outputs

```python
write_pdf(audit, "audit.pdf")    # the version a person signs
write_json(audit, "audit.json")  # every number, for a dashboard or a CI check
render_text(audit)               # for a terminal
render_pdf(audit) -> bytes       # if you want to stream it
render_json(audit) -> str
```

Both come from the same `Audit`, so the JSON and the PDF can never disagree.

### The forecast

```python
recommend(diagnosis, distances_mm) -> Recommendation | None
forecast(session, fit, task, current, recommendation, ...) -> Forecast

prediction.statement()           # 'Adding 6 views tilted about 40 degrees ...'
prediction.improvement           # how many times narrower
prediction.worthwhile
prediction.forecast_identifiable
```

> Uncertainty depends on the geometry of a capture and on the corner noise, not
> on the true parameter values — so synthesising the recommended views from the
> calibration already in hand and refitting gives a sound prediction of the
> interval. It does **not** predict correctness: the synthetic views agree with
> the current fit by construction, so the forecast says how much tighter the
> interval would get, not that the answer would be right.

---

## The CLI, and its Python equivalent

| command | equivalent |
|---|---|
| `caltrust ingest images --images D --target T -o s.npz` | `save_session(session_from_images(D, T), "s.npz")` |
| `caltrust ingest calibration --calibration C --detections D -o s.npz` | `session_from_calibration(C, D)` |
| `caltrust refit s.npz` | `instrument(load_session("s.npz"))` |
| `caltrust refit s.npz --cross-validate` | `+ cross_validate(session)` |
| `caltrust diagnose s.npz` | `+ diagnose(fit, obs, validation)` |
| `caltrust report s.npz --pdf p --json j` | `run_audit(...)` then `write_pdf` / `write_json` |
| `caltrust show s.npz` | `render_session` / `render_fit` |
| `caltrust formats` | `registered_targets`, `reader_names`, ... |

`caltrust diagnose` and `caltrust report` exit **3** on a critical finding, so a
CI step can gate on calibration quality without parsing the report.

`--deterministic` pins OpenCV to one thread. Without it, output differs in the
last few bits between runs: `cv2.calibrateCamera` reduces across threads and
floating-point addition is not associative — eight identical refits spanned
1.1e-12 px on a ten-thread machine.

Task shorthand for `--task`, repeatable:

```
length:800mm:100mm      plane:800mm:25deg
stereo:800mm:200mm      base:800mm
```

---

## Reading order, and the traps

1. **`fit.conditioning.identifiable`.** Nothing else means anything until you
   have read this. A pseudo-inverse gives an unconstrained direction zero
   variance, so a degenerate fit reports `sd(fx) = 0.019 px` while `fx` is wrong
   by 718 px.
2. **`validation.trustworthy`, then `validation.ratio`.** The ratio reads 1.005
   on a capture whose focal length is wrong by 3333 px.
3. **`findings.ranked()[0]`.** The cause, not the symptom.
4. **`error.distribution(...).statement()`** and **`error.variance_share(...)`.**
   The millimetres, and whether re-calibrating would help.
5. **`corr(fx, tz)` is not a degeneracy detector on its own.** It reads 0.93 on a
   *good* capture and near zero on a fully degenerate one, because a cut
   direction carries no variance to correlate. Check identifiability first.

Every result object has `to_dict()`; most have `summary_lines()`. Nothing needs
reaching into internals to get at a number.

For the generated reference of every module, function and class:

```
make docs        # pdoc over the docstrings -> docs/api/, 72 pages
```

Known gaps, unchecked assumptions and things measured to be wrong live in
[open-items.md](../open-items.md).

---

caltrust is licensed **AGPL-3.0-only**. Section 13 reaches network use: modify it
and let people interact with the result over a network, and you owe them the
source. See [LICENSE](../LICENSE) and the licence notes in
[README.md](../README.md).
