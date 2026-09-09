# caltrust

Metric trust for camera calibration. It computes what a reprojection RMS cannot:
the full parameter covariance, which parameter combinations your capture does
not determine at all, where the residuals are structured rather than random, and
how well posed the estimation problem actually was.

Status: **M1 (ingest) and M2 (refit with instrumentation) are complete.** M3 to
M6 — out-of-sample error, cause diagnosis, task-space millimetres, and staleness
detection — are not built yet. See [tracker.md](tracker.md).

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

## Verification

Every numeric claim has a test that could fail.

| Claim | How it is checked | Result |
|---|---|---|
| The analytic Jacobians are correct | central differences, 6 cameras x 3 poses, both models | agreement to 1e-9 relative |
| Marginal standard deviations are right | vs `cv2.calibrateCameraExtended` | agree to 6 significant figures |
| The Schur complement is exact | vs a dense `sigma^2 (J'J)^-1` | 6e-10 relative |
| The covariance is *calibrated* | 250 Monte Carlo refits vs the predicted covariance | predicted/empirical sd within 2-4%; mean Mahalanobis 9.37 against an expected 9 |
| Predicted correlations are right | same Monte Carlo, correlation matrices | max difference 0.12 |
| Degeneracy is detected | frontoparallel rig | `identifiable = False`, weak direction `-0.71*fx -0.71*fy` |
| Rodrigues conversion is right | vs `cv2.Rodrigues`, 60k rotations including the pi singularity | 4e-8 worst round-trip |
| Detectors find real patterns | rendered images, all three targets, both camera models | 0.09-0.4 px mean localisation error |
| The frozen binary works | `dist/caltrust` in a stripped environment | full pipeline, `fx = 899.564 +/- 0.751` |

    651 passed, 10 skipped in 84s        # make test
    571 passed, 80 deselected in 3s      # make fast
    TOTAL  2940 statements, 63 missed, 98%

The ten skips are OpenCV-4-only flag-namespace checks. `make test` runs
everything; `make fast` skips the Monte Carlo and image-rendering tests.

---

## Using it as a library

    from caltrust import instrument, session_from_images
    from caltrust.cli.targets import resolve_target

    session = session_from_images("captures/", resolve_target("checkerboard:9x6:25mm"))
    fit = instrument(session)

    print("\n".join(fit.summary_lines()))
    print(fit.conditioning.identifiable)
    print(fit.covariance.correlation_with_poses("fx")[:, 5])   # corr(fx, tz) per view
    print(fit.covariance.dense())                              # if you want it all

`caltrust.synthetic` generates captures with known truth, which is how the tests
work and how M3's "what if you added six views at 400 mm" will be answered.

API reference: `make docs` runs `pdoc` over the docstrings and writes 46 pages to
`docs/api`. There is no hand-written duplicate of anything a docstring says.

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

Two OpenCV behaviours worth knowing, both found while building this and both
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

---

## Licence

Apache-2.0.
