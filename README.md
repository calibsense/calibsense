# caltrust

Metric trust for camera calibration. It computes what a reprojection RMS cannot:
the full parameter covariance, which parameter combinations your capture does
not determine at all, where the residuals are structured rather than random, and
how well posed the estimation problem actually was.

Status: **M1 to M7 complete** — ingest, instrumented refit, out-of-sample error,
named-cause diagnostics, task-space millimetres, hand-eye with covariance, and
the report. Staleness detection is not built yet. See [tracker.md](tracker.md)
for the plan and [open-items.md](open-items.md) for the twenty-eight things known
to be wrong, missing, or resting on an unchecked assumption.

---

## The problem, in one table

Six synthetic rigs. One true camera, `fx = 900 px`. The same 0.2 px corner noise,
the same 18 views, the same checkerboard. Run `python examples/degeneracy_demo.py`:

| rig | reprojection RMS | fx error | sd(fx) | identifiable |
|---|---|---|---|---|
| flat, 1 depth | 0.2753 px | **+3333.5 px** | 707.7 | no |
| flat, 3 depths | 0.2761 px | **-139.1 px** | 0.336 | no |
| 5 deg tilt, 1 depth | 0.2754 px | +41.2 px | 28.5 | yes |
| 20 deg tilt, 1 depth | 0.2757 px | +2.3 px | 3.93 | yes |
| 40 deg tilt, 1 depth | 0.2761 px | +0.3 px | 2.24 | yes |
| 40 deg tilt, 3 depths | 0.2759 px | +1.3 px | 1.19 | yes |

The RMS column varies by 0.0008 px. The focal length varies by 3333 px. An
engineer looking at the first row sees 0.275 px and ships it.

Two things in that table are worth stating plainly, because both contradict the
usual advice:

**Tilt is what makes the focal length identifiable, not depth variation.** A
frontoparallel planar target gives one homography per view, and since every view
carries its own free translation, rescaling all focal lengths and all depths
together reproduces the images exactly. Adding a fourth working distance does not
help, because it arrives with its own free translation. Only perspective
foreshortening inside a single view pins the focal length. Depth variation is a
real second-order gain — sd(fx) drops from 2.24 to 1.19 — but only once tilt is
already there.

**Read `identifiable` before you read the standard deviation.** Row two reports
`sd(fx) = 0.336 px` while being wrong by 139 px. That is not a bug. A
pseudo-inverse assigns *zero* variance to a direction the data does not
constrain, so a rank-deficient fit reports false confidence rather than a large
error bar. This is why caltrust computes identifiability from the null space of
the Jacobi-scaled Schur complement and puts it above every other number in the
report.

---

## Install

    pip install caltrust

Three runtime dependencies: `numpy`, `opencv-contrib-python`, `pyyaml`. The list
is short on purpose, because the second shipping target is a single binary:

    make binary        # -> dist/caltrust, about 63 MB, no Python needed

---

## Two commands

Detect a target across a folder of images, optionally alongside the calibration
you are already shipping:

    caltrust ingest images \
        --images captures/ \
        --target checkerboard:9x6:25mm \
        --calibration shipped.yml \
        -o session.npz

Then refit, keeping everything the standard call throws away:

    caltrust refit session.npz -o fit.npz

    caltrust instrumented refit
    ===========================

      model        pinhole_brown_conrady (refitted)
      views        16, 864 points
      RMS          0.1221 px in-sample  (calibration file reported 0.2412 px)
      sigma        0.0891 px per coordinate, 1623 degrees of freedom
      depth range  411 to 1051 mm
      conditioning scaled condition number 1.522e+03, rank 9/9

    parameters
      name        value    std dev  relative  weak
      ----  -----------  ---------  --------  ----
        fx      899.564     0.7506     0.08%  0.00
        fy        904.6     0.7892     0.09%  0.00
        cx      640.703     0.8313     0.13%  0.00
        cy      359.927     0.7885     0.22%  0.00
        k1    -0.146432   0.008783     6.00%  0.00
        ...

    strongest intrinsic correlations
         pair  correlation
      -------  -----------
      fx ~ fy       +0.989
      k2 ~ k3       -0.981
      k1 ~ k2       -0.965

    focal length against working distance
      corr(fx, tz) mean +0.928, range +0.897 to +0.957 over 16 views

Those numbers come from images rendered through a camera with `fx = 900`. The
recovered `899.564 +/- 0.751` is 0.6 sigma from truth.

Add `--json` for a machine-readable version, `-v` for every view and the full
correlation matrix, and `--no-refit` to instrument a calibration you already have
without re-estimating it — which reports, among other things, that it is not the
optimum of its own detections:

    NOT AT OPTIMUM  a Newton step would cut the cost by 34.03%
    outlier views 4: view015 (0.51 px), view006 (0.50 px), view012 (0.44 px)

Then ask what is wrong with the capture:

    caltrust diagnose session.npz

    caltrust diagnosis
    ==================

      1 critical finding(s) — image coverage — each described below with what it invalidates

      [CRITICAL] Image coverage
                   corners occupy 33% of an 8x6 grid, 8% of the border ring, and reach
                   41% of the way to the frame corner; the distortion coefficients are
                   extrapolating over most of the frame
                   -> Move the board into the corners and edges of the image, not just
                      the middle. Distortion grows with radius, so the coefficients are
                      only measured over the radii you actually visit.

      clean: Out-of-sample error, Pose diversity, Frontoparallel dominance,
             Depth variation, Target scale, Outlier and high-leverage views

`diagnose` refits, cross-validates and runs every check. It exits 3 on a critical
finding, so a CI step can gate on calibration quality without parsing the report.

And finally, the whole thing in one command:

    caltrust report session.npz \
        --task length:800mm:100mm \
        --pdf audit.pdf --json audit.json \
        --camera-name "line-3 inspection" --contact metrology@example.com

    A 100 mm feature measured at 800 mm has an expected error of 0.107 mm, with a
    95% interval of +/-0.272 mm.

    The reported 0.1221 px reprojection error is an honest error estimate:
    held-out views reprojected to 0.1231 px, a ratio of 1.01x.

    The dominant cause is image coverage: corners occupy 33% of an 8x6 grid, 8%
    of the border ring, and reach 41% of the way to the frame corner; the
    distortion coefficients are extrapolating over most of the frame.

That is the paragraph from the problem statement, produced from measured data.
The JSON carries every number behind it; the PDF is the version a quality
manager can sign, leading with that statement and ending with an open question
and a contact address.

`caltrust show`, `caltrust formats` and `--help` cover the rest.

---

## What M1 ingests

| | |
|---|---|
| Targets | checkerboard, ChArUco, circle grid (symmetric and asymmetric) |
| Camera models | pinhole with Brown-Conrady (4, 5, 8, 12 or 14 coefficients), fisheye with Kannala-Brandt |
| Sources | a folder of images, or an existing calibration plus the detections behind it |
| Calibration formats | OpenCV FileStorage (`.yml`/`.yaml`/`.xml`), ROS `camera_info`, Kalibr camchain, caltrust JSON |
| Detections | caltrust JSON, or an NPZ of `image_points` |
| Robot poses | JSON or CSV, matrix / quaternion / rotation-vector form, either hand-eye direction, any length unit |

Point ids are per view, not a fixed grid, so a ChArUco board showing half of
itself contributes correctly labelled points and nothing downstream branches on
target type. A session is one `.npz` file with a JSON manifest inside it, loaded
with `allow_pickle=False`, so opening a result can never execute code from it.

Target shorthand avoids a config file for the common case:

    checkerboard:9x6:25mm
    charuco:8x11:20mm:15mm:DICT_5X5_1000
    circles:4x11:0.75in:asymmetric

## What M2 records

Everything `cv2.calibrateCamera` computes internally and then discards:

- **Full parameter covariance**, not marginal standard deviations. Kept in
  factored form — `sigma^2 S^-1` for the intrinsics plus per-view factors — so
  memory is linear in the number of views and any block is reconstructed on
  demand. `calibrateCameraExtended` gives you a column of numbers; the
  off-diagonals are where `corr(fx, tz) = 0.93` lives.
- **Per-view residual distributions**: RMS, p95, max, signed bias per axis, and a
  median-absolute-deviation z-score that flags outlier views.
- **Per-corner residuals**, kept in full, plus a profile binned by image radius
  that separates the radial from the tangential component. A signed radial mean
  that grows with radius is the signature of a distortion model that does not
  match the lens.
- **Condition number of the normal equations**, both raw and Jacobi-scaled. Only
  the scaled one is meaningful: a focal length in pixels and a distortion
  coefficient near zero are not comparable until the units are removed. Roughly
  1e3 is well posed; 1e8 and up is not.
- **Parameter correlation matrix**, and the parameter pairs it ranks worst.
- **Identifiability**: the weak directions of the scaled Schur complement, each
  described by the parameters that dominate it, and a per-parameter share of the
  weak subspace.
- **Whether the parameters sit at the optimum of their own objective**, measured
  as the relative cost reduction a full Newton step would predict. Always true
  after a refit; frequently false when auditing someone else's calibration.

---

## What M3 measures

K-fold over views: fit on the rest, predict the held-out ones, report in-sample
RMS, out-of-sample RMS and the ratio. Each held-out view solves its own pose by
PnP at the fold's intrinsics, which is the deployment-relevant question and also
what stops the view leaking into its own prediction.

**The ratio has a blind spot, and finding it changed the design.** On the
frontoparallel rig from the table above — the one whose `fx` is wrong by 3333 px
— the ratio comes out at **1.005**. A held-out view solves its own pose, so a
proportionally wrong depth cancels a proportionally wrong focal length and
reprojection is perfect. Any degeneracy a free pose can absorb is invisible to
reprojection-based cross-validation.

So two things sit beside it. Per-fold identifiability gates the verdict, and the
report never calls such a ratio honest. And the **across-fold parameter spread**
is reported next to the covariance's prediction:

| rig | out-of-sample ratio | fold spread of fx | covariance prediction |
|---|---|---|---|
| diverse | 1.01x | 1.15 px | 1.80 px |
| frontoparallel | 1.00x | **326 px** | 636 px |

That spread is the only cross-check in the package that assumes nothing about
the noise — it resamples the actual views. It is not unbiased, because folds
share most of their training data, so it is labelled a relative indicator.

## What M4 diagnoses

One named cause per check, each with the numbers behind it and what to do:

| Cause | Measured as |
|---|---|
| Pose diversity | max pairwise angle between board normals, plus the orientation tensor for whether the normals are confined to a plane |
| Depth variation | max-over-min working distance, with the focal-distance correlation from the covariance |
| Frontoparallel dominance | fraction of views within 15 degrees of the image plane, linked to the parameters it leaves undetermined |
| Image coverage | grid occupancy, border-ring occupancy, and radial reach |
| Target scale | median nearest-neighbour corner spacing, against the residual sigma |
| Distortion model adequacy | chi-square and trend tests on the radial residual profile |
| Outlier views | each view's share of the intrinsic information, paired with its residual |
| Out-of-sample error | M3's ratio, gated on per-fold identifiability |
| Noise model | classical intrinsic deviations against a view-clustered sandwich |

Three of these needed a real decision rather than a formula.

**Leverage needed a definition.** The Schur complement is a sum over views,
`S = sum_i S_i`, so `trace(S_i S^-1)` sums to the parameter count and
`trace(S_i S^-1) / p` is that view's share of everything the capture knows about
the intrinsics. Shares sum to one and read directly as percentages. A view that
combines a large share with a large residual is the dangerous case, because the
fit is being pulled by the frame it depends on most.

**The residual test had to cluster by view.** Z-scoring each radial bin against
`sigma / sqrt(n)` assumes corners are independent, and they are not — corners in
one view share that view's pose, so a small pose error moves all of them
together. That naive version reported +6.3 sigma of "structure" on a capture
whose focal length was recovered to 0.6 sigma. Treating the *view* as the
independent unit is cluster-robust and needs no noise assumption:

| case | reprojection RMS | flatness z | verdict |
|---|---|---|---|
| pinhole data, pinhole model | 0.2768 px | −0.16 | ok |
| pinhole data, k3 dropped | 0.2770 px | +0.44 | ok |
| fisheye data, fisheye model | 0.2767 px | −0.61 | ok |
| fisheye data, pinhole model | 0.2832 px | **+5.22** | critical |

The RMS moves by 2%. The residual structure is unmistakable.

**The same clustering argument fixes the covariance.** Every deviation caltrust
prints rests on `sigma^2 = cost / (m - p)`, which measures the noise scale but
asserts its shape. Injecting one violation at a time showed that most of the
shape does not matter: anisotropy along the edge direction, noise several times
worse in some views than others, and a few per cent of badly mis-detected
corners each leave the predicted deviation within about 15% of the truth.

Spatial correlation across the frame is the exception, and it is severe. A
correlated field is partly absorbable by the pose and distortion parameters, so
it *lowers* the residual while *raising* the estimator's real spread:

| noise | reported `sigma` | predicted sd(fx) | actual sd(fx) |
|---|---|---|---|
| independent, 0.25 px | 0.253 px | 3.14 px | 3.03 px |
| correlated, 60 px length | 0.162 px | 1.95 px | **5.69 px** |
| correlated, 200 px length | 0.055 px | 0.67 px | **5.24 px** |

The RMS gets *better* as the answer gets worse, which is why no residual
statistic can catch this. So the parameter covariance is computed a second way,
treating the view as the independent unit exactly as the radial profile does:

    Cov = S^-1 (sum_i s_i s_i') S^-1 * G / (G - 1)

where `s_i` is what view `i` contributes to the intrinsic estimate after its own
pose has absorbed what it can. Nothing is assumed about the noise within a view.
The ratio between the two estimates is the `noise_model` finding: near one means
the assumption held *for this capture*, and 6.5x means every interval in the
report is that much too tight. Weights would not have fixed this — the error is
in the off-diagonal of the noise covariance, and reweighting a diagonal cannot
repair a correlation.

**A slope test alone misses truncation.** A truncated radial polynomial leaves a
residual that oscillates in sign, so the primary criterion is a chi-square over
the bands (Wilson-Hilferty to a z-score, no special-function dependency) and the
slope only describes the shape.

## What M5 propagates

Monte Carlo from the covariance into task units. Whole parameter sets are drawn
from the joint distribution rather than one deviation at a time, because the
trade-offs between parameters are what determine the answer — sampling `fx` and
`tz` independently from their marginals would give a spread far wider than
reality, since the two are correlated at 0.93 and their errors partly cancel.

Four tasks: a length at a known depth, a plane's distance and orientation, a
stereo triangulation, and a point carried into the robot base frame through a
hand-eye transform.

**The variance split was not in the plan and is the most useful part.** The
calibration and the pixel noise at measurement time are independent, so their
variances add, and separating them answers the question that follows every error
figure:

    a 100 mm feature measured at 800 mm has an expected error of 0.253 mm,
    with a 95% interval of +/-0.622 mm
        of that, 38% comes from the calibration and 62% from 0.203 px of
        pixel noise

Re-calibrating that camera can improve the answer by at most a third. On a
degenerate capture the same task reports 111 mm, flagged as a lower bound
because an unconstrained direction carries no variance to sample along.

## What M6 solves

Eye-in-hand and eye-to-hand, with a covariance over all twelve parameters and
pose-set sufficiency diagnostics. The closed-form initialisation is implemented
here rather than delegated, because `cv2.calibrateHandEye` is absent from
OpenCV 5's Python bindings while its `CALIB_HAND_EYE_*` flags are still
exported.

Two things measurement caught that reasoning would not have.

**The obvious covariance is 3.1x optimistic.** A residual-based estimate gave a
translation deviation of 0.39 mm against an actual error of 1.29 mm. It treats
the target-in-camera poses as exact data when they come from the calibration,
and their errors are *correlated across views* because every view shares the
same intrinsics — a focal-length error tilts and scales all of them coherently,
so it does not average down. Resampling whole calibrations gives 1.22 mm, and
that is the default.

**Hand-eye is determined by rotation and by nothing else**, which is why the
sufficiency checks matter. Every degenerate pose set produces a small predicted
deviation next to a large actual error:

| pose set | true error | predicted sd | caught by |
|---|---|---|---|
| varied axes | 2.2 mm | 1.5 mm | (clean) |
| mispaired robot poses | **367 mm** | 1.9 mm | pose pairing |
| one rotation axis | **88 mm** | 1.5 mm | rotation axes, conditioning |
| tiny wrist rotations | — | — | the solve refuses outright |

The pairing check is the one worth having: conjugate rotations have equal
angles, so a robot pose shuffled against its image is detectable, and no amount
of optimisation would have revealed it.

## What M7 writes

One run, two outputs. The JSON has every number; the PDF has the subset a person
has to sign. The PDF writer is about 350 lines of this package rather than a
dependency, because three runtime dependencies is what makes the single binary
possible and a PDF of text, rules and bars needs nothing that is not already
here — the format is plain bytes, the base-14 fonts need no embedding, and
`zlib` is in the standard library.

The last sentence of the report is a forecast:

    Adding 6 views tilted about 40 degrees in varied directions would make the
    calibration identifiable, bringing the interval to approximately +/-0.98 mm
    from a figure that is currently only a lower bound.

That is answerable because uncertainty depends on the geometry of a capture and
on the corner noise, not on the true parameter values — so the recommended views
are synthesised from the calibration already in hand, refitted, and
re-propagated. It predicts uncertainty, not correctness, and says so.

## Verification

Every numeric claim has a test that could fail.

| Claim | How it is checked | Result |
|---|---|---|
| The analytic Jacobians are correct | central differences, 6 cameras x 3 poses, both models | agreement to 1e-9 relative |
| Marginal standard deviations are right | vs `cv2.calibrateCameraExtended` | agree to 6 significant figures |
| The Schur complement is exact | vs a dense `sigma^2 (J'J)^-1` | 6e-10 relative |
| The covariance is *calibrated* | 250 Monte Carlo refits vs the predicted covariance | predicted/empirical sd within 2-4%; mean Mahalanobis 9.37 against an expected 9 |
| The assumed noise *shape* mostly does not matter | Monte Carlo injecting anisotropy, per-view heteroscedasticity and heavy tails | predicted/empirical sd stays within 0.96-1.17 for all three |
| Correlated corner noise breaks it, badly | Monte Carlo with a 200 px correlated field | predicted/empirical sd 0.12-0.39, and `sigma` *falls* from 0.25 to 0.055 px while the true spread rises |
| The view-clustered covariance recovers it | same rig, sandwich vs classical | 0.12 -> 0.75 of the true spread at 14 views, 0.13 -> 0.84 at 30 |
| Predicted correlations are right | same Monte Carlo, correlation matrices | max difference 0.12 |
| Degeneracy is detected | frontoparallel rig | `identifiable = False`, weak direction `-0.71*fx -0.71*fy` |
| Rodrigues conversion is right | vs `cv2.Rodrigues`, 60k rotations including the pi singularity | 4e-8 worst round-trip |
| Detectors find real patterns | rendered images, all three targets, both camera models | 0.09-0.4 px mean localisation error |
| The frozen binary works | `dist/caltrust` in a stripped environment | full pipeline, `fx = 899.564 +/- 0.751` |
| Cross-validation folds are sound | partition, point-weighted pooling, no train/test overlap | exact |
| The ratio's blind spot is pinned | frontoparallel rig, asserted to stay under 1.1x | 1.005x, test would fail if it silently started working |
| Fold spread catches what the ratio misses | same two rigs | 280x apart |
| Each diagnostic is specific | six single-fault rigs x each cause | every fault fires, every unrelated cause stays quiet |
| Clustering by view deflates the z-score | naive per-corner vs cluster-robust on a healthy rig | asserted lower, measured 6.3 -> 1.6 |
| The omnibus test catches oscillation | synthetic profile orthogonal to constant and linear | slope t = 0, flatness z > 5 |
| Sampling reproduces the joint distribution | 4000 draws vs the predicted deviations and correlations | sd within 12%, correlation within 0.08 |
| Backprojection is exact | project then normalise, over the whole frame | 3e-13 mm at 800 mm, both models |
| Each task measures its own scene | noiseless round trip, four tasks | exact to 1e-6 |
| The variance split adds up | combined vs parameters + noise | within 20% |
| Hand-eye recovers known truth | both mountings, derived robot poses | under 6 mm and 1 deg |
| The resampled hand-eye covariance covers the error | truth vs predicted deviation | 1.29 mm actual, 1.22 mm predicted |
| Every degenerate pose set is caught | four hand-eye rigs | all flagged or refused |
| The PDF is a valid PDF | streams decompressed and searched; Quartz rasterises it | headline and contact present |
| Helvetica metrics are right | hand-checked against the AFM table | exact |

    943 passed, 11 skipped in 183s       # make test
    TOTAL  5191 statements, 153 missed, 97%

The skips are OpenCV-4-only flag-namespace checks. `make test` runs everything;
`make fast` skips the Monte Carlo, image-rendering and propagation tests.

Reports are reproducible to about 1e-6 relative rather than bit-exact, because
`cv2.calibrateCamera` reduces across threads and floating-point addition is not
associative — eight identical refits spanned 1.1e-12 px on a ten-thread machine.
`--deterministic` pins OpenCV to one thread and makes output byte-identical.

---

## Using it as a library

    from caltrust import load_session
    from caltrust.report import run_audit, write_json, write_pdf
    from caltrust.task import LengthAtDepth

    session = load_session("session.npz")
    audit = run_audit(session, tasks=[LengthAtDepth(depth_mm=800.0, length_mm=100.0)])

    print(audit.trustworthy)                    # read this first
    for sentence in audit.headline():
        print(sentence)
    write_pdf(audit, "audit.pdf")
    write_json(audit, "audit.json")

Or a piece at a time:

    from caltrust import cross_validate, diagnose, instrument, session_from_images
    from caltrust.cli.targets import resolve_target
    from caltrust.handeye import diagnose_hand_eye, solve_hand_eye
    from caltrust.task import LengthAtDepth, propagate

    session = session_from_images("captures/", resolve_target("checkerboard:9x6:25mm"))
    fit = instrument(session)
    validation = cross_validate(session)
    findings = diagnose(fit, session.observations, validation)
    error = propagate(fit, LengthAtDepth(depth_mm=800.0, length_mm=100.0))

    print(fit.conditioning.identifiable)                       # read this first
    print(validation.ratio, validation.spread_of("fx"))
    print(findings.verdict())
    print(error.distribution("length_mm").statement())
    print(error.variance_share("length_mm"))                   # calibration vs pixels

    print(fit.covariance.correlation_with_poses("fx")[:, 5])   # corr(fx, tz) per view
    print(fit.covariance.dense())                              # if you want it all

`caltrust.synthetic` generates captures with known truth, which is how the tests
work and how M3's "what if you added six views at 400 mm" will be answered.

API reference: `make docs` runs `pdoc` over the docstrings and writes a page per
module to `docs/api`. There is no hand-written duplicate of anything a docstring says.

---

## Design notes

**Layering.** `caltrust.core` holds the data — cameras, targets, detections,
poses, sessions — and does not import OpenCV. Projection, differentiation and
fitting live in `caltrust.refit`; detection and file reading in
`caltrust.ingest`. The dependency arrows all point at `core`, which is why the
core is testable without a single image.

**Registries are literal dictionaries** populated by static imports, never a
filesystem scan, and the package ships no data files. That is what makes the
frozen binary work without `sys._MEIPASS` handling, and a test asserts both
properties so they cannot quietly regress.

**Nothing calls `numpy.linalg.inv`.** Every inverse goes through an
eigendecomposition that reports its rank, condition numbers, and the directions
the data leaves unconstrained. A near-singular calibration is the finding, not an
inconvenience.

**OpenCV version differences are handled by name, not by value.** OpenCV 4 keeps
the fisheye flags in `cv2.fisheye` with their own bit numbering and defines
same-named constants in `cv2` with *different* values; OpenCV 5 merged them onto
the pinhole numbering. Reading `cv2.CALIB_FIX_K1` and handing it to
`cv2.fisheye.calibrate` therefore fixes the wrong coefficient on OpenCV 4 and the
right one on OpenCV 5. Every flag resolves through `caltrust.refit.cv_compat`,
and the test suite checks the resolved values against the fisheye namespace
wherever it exists.

Three OpenCV behaviours worth knowing, all found while building this and all
documented where they bite:

- `cv2.fisheye.calibrate` estimates its own starting intrinsics when not given
  any, and on ordinary captures that estimate *fails outright* — it raises
  `fabs(norm_u1) > 0` from `InitExtrinsics` rather than returning a poor answer.
  caltrust supplies its own ladder of focal-length guesses starting at
  `width / pi` and reports which rung converged.
- A `CALIB_FIX_Kn` flag holds the supplied value on the pinhole path and *zeroes*
  the coefficient on the fisheye path. Either way the parameter leaves the
  covariance, so the reported uncertainty is right; only the resulting value
  differs.
- `cv2.calibrateHandEye` is gone from OpenCV 5's Python bindings, while its
  `CALIB_HAND_EYE_*` flags are still exported. The closed form is implemented in
  this package instead, and cross-checked against OpenCV wherever OpenCV still
  has it.

---

## Licence

**AGPL-3.0-only.** See [LICENSE](LICENSE) for the full text; every source file
carries an `SPDX-License-Identifier: AGPL-3.0-only` header.

Two consequences worth stating plainly rather than leaving to be discovered.

**Section 13 reaches network use.** If you modify caltrust and let people
interact with the modified version over a network — an upload-your-calibration
web service, a report generator behind an internal API — you have to offer those
users the corresponding source. Running it as a CLI or importing it as a library
in software you do not distribute triggers nothing.

**Industrial legal departments frequently decline AGPL outright**, and this
tool's natural customer is a robot-cell integrator or a quality department. That
is a commercial constraint rather than a technical one, and the usual answer is
dual licensing: AGPL-3.0-only for the open version alongside a commercial licence
sold separately. That stays available here because the copyright is held by a
single author; it stops being simple the moment outside contributions land
without a contributor agreement.

The three runtime dependencies are compatible with AGPL-3.0: `numpy` is
BSD-3-Clause, `pyyaml` is MIT, and OpenCV is Apache-2.0 with LGPL-2.1 components
in the distributed wheels. All are one-way compatible into an AGPL work.

Note that a PyInstaller binary of an AGPL program is a distribution of the
program, so `dist/caltrust` has to be accompanied by the corresponding source or
a written offer for it.
