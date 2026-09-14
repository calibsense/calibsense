# calibsense

**Your camera calibration reports a reprojection error of 0.2 px. That number
does not tell you whether you can measure a part to half a millimetre.**

calibsense computes what the reprojection RMS cannot: the full parameter
covariance, the parameter combinations your capture does not determine at all,
whether the residuals are structured rather than random, and what that means in
the units you care about.

> A 100 mm feature measured at 800 mm has an expected error of 0.199 mm, with a
> 95% interval of ±0.485 mm. Of that, 40% comes from the calibration and 60%
> from 0.119 px of pixel noise at measurement time.

It names what is wrong with the capture and what to do about it, and refuses to
print a figure that cannot be trusted as though it could.

## The problem

Six synthetic rigs, one true camera at `fx = 900 px`, the same 0.2 px corner
noise, the same 18 views. Reproduce with `python examples/degeneracy_demo.py`:

| rig | reprojection RMS | fx error | sd(fx) | identifiable |
|---|---|---|---|---|
| flat, 1 depth | 0.2753 px | **+3334.2 px** | 708.0 | no |
| flat, 3 depths | 0.2761 px | **-139.1 px** | 0.336 | no |
| 5 deg tilt, 1 depth | 0.2754 px | +41.2 px | 28.5 | yes |
| 20 deg tilt, 1 depth | 0.2757 px | +2.3 px | 3.93 | yes |
| 40 deg tilt, 3 depths | 0.2759 px | +1.3 px | 1.19 | yes |

Every row fits its own images to within a thousandth of a pixel of every other
row, while the focal length ranges from correct to wrong by 3334 px. An engineer
looking at the first row sees 0.275 px and ships it.

**Tilt is what makes the focal length identifiable, not depth variation**: on a
frontoparallel target a global rescale of all focal lengths and all depths
reproduces the images exactly, and a fourth working distance arrives with its
own free translation, so only foreshortening within a single view pins `fx`.

**Read `identifiable` before the standard deviation**: row two reports
`sd(fx) = 0.336 px` while being wrong by 139 px, because a pseudo-inverse
assigns *zero* variance to a direction the data does not constrain. A
rank-deficient fit reports false confidence, not a wide error bar.

## Install

```sh
pip install calibsense
```

The current release is `0.1.0a1`, a pre-release. It is the complete tool and its
test suite passes, but treat the API as unsettled until `0.1.0`.

Three runtime dependencies: `numpy`, `opencv-contrib-python`, `pyyaml`. Python
3.9 or newer. `make binary` produces `dist/calibsense`, a 64 MB single file that
needs no Python present.

## Usage

### 1. Detect the target

```sh
calibsense ingest images --images captures/ \
    --target checkerboard:9x6:25mm -o session.npz
```

Failures are named, because a detector that silently drops a third of your
capture is the first thing you want to know:

```
  [   2/20] ... view001  no 9x6 checkerboard found
  ...
  target       9x6 checkerboard, 25 mm squares
  views        14
  points       756 total, 54-54 per view (median 54)
  detection    14/20 images (70%) via checkerboard
```

Target shorthand avoids a config file: `checkerboard:9x6:25mm`,
`charuco:8x11:20mm:15mm:DICT_5X5_1000`, `circles:4x11:0.75in:asymmetric`.

Add `--calibration shipped.yml` to audit one you already ship;
`calibsense refit session.npz --no-refit` then instruments it in place and
reports whether it is even the optimum of its own detections.

### 2. Ask what the capture supports

```sh
calibsense diagnose session.npz
```

```
  1 warning(s): image coverage

  [WARNING ] Image coverage
               corners occupy 40% of an 8x6 grid, 12% of the border ring, and
               reach 56% of the way to the frame corner; the periphery is thin
               -> Add views with the board against the frame edges and corners.

  clean: Out-of-sample error, Pose diversity, Frontoparallel dominance, Depth
         variation, Target scale, Distortion model adequacy, Noise model
```

**It exits 3 on a critical finding**, so CI can gate on calibration quality
without parsing anything.

### 3. Get the answer in millimetres

```sh
calibsense report session.npz --task length:800mm:100mm --task plane:800mm \
    --json audit.json --pdf audit.pdf --contact metrology@example.com
```

```
A 100 mm feature measured at 800 mm has an expected error of 0.199 mm, with a
95% interval of +/-0.485 mm.

The reported 0.1633 px reprojection error is an honest error estimate: held-out
views reprojected to 0.168 px, a ratio of 1.03x.

The dominant cause is image coverage: corners occupy 40% of an 8x6 grid, 12% of
the border ring, and reach 56% of the way to the frame corner.
```

That capture was rendered through a camera with `fx = 700.0`; the fit recovered
`700.855 ± 1.121`, 0.76 sigma from truth, with all nine intrinsics inside their
reported intervals.

The JSON carries every number; the PDF is the two-page version a quality manager
can sign. Both lead with the variance split, the line most people act on:
calibration error and measurement-time pixel noise are independent, so their
variances add, and splitting them says re-calibrating this camera can improve
the answer by at most 40%. The rest is photons.

Four tasks are available as `--task`: `length:DEPTH:SIZE` for measuring a
feature, `plane:DEPTH[:TILT]` for locating a plane, `stereo:DEPTH:BASELINE` for
triangulated depth, and `base:DEPTH` for a point in the robot's base frame.

When a figure cannot be read at face value the report opens with a banner and
the JSON carries `"trustworthy": false` with the reason, for example `THE MODEL
DOES NOT DESCRIBE THIS CAPTURE - THE INTERVALS BELOW ARE WIDENED UP TO 3.0X`.
Pass
`--deterministic` before the subcommand for reproducible output; without it
results move in the last few bits, because `cv2.calibrateCamera` reduces across
threads.

## Measuring the noise instead of inferring it

Everything above infers the noise scale from fit residuals, which assumes the
model is right. To take the model out of the loop, bolt the camera and target
down, capture thirty or more frames, and measure how far the corners wander:

```sh
calibsense noise-floor --images static/ --target checkerboard:9x6:25mm
```

```
frames       40 static frames, 54 corners
scale        0.0116 px total, 0.0111 px irreducible (9% absorbable by pose)
correlation  length 0.0 px, 0.00 of the 22 px corner spacing (independent)
shape        anisotropy 1.60 against a 1.21 floor from 40 frames
drift        0.0052 px largest whole-frame shift
```

The correlation length decides whether the rest of the report can be trusted.
`σ² = cost / (m - p)` measures the noise scale but asserts its shape, and
breaking that assumption one way at a time showed most of the shape does not
matter: anisotropy, per-view scale variation and a few per cent of mis-detected
corners each leave the predicted deviation within about 15% of the truth.
Spatial correlation is the exception, because a correlated field is partly
absorbable by the pose and distortion parameters:

| injected noise | reported σ̂ | predicted sd(fx) | true sd(fx) |
|---|---|---|---|
| independent, 0.25 px | 0.253 px | 3.14 px | 3.03 px |
| correlated, 60 px length | 0.162 px | 1.95 px | **5.69 px** |
| correlated, 200 px length | 0.055 px | 0.67 px | **5.24 px** |

The RMS gets *better* as the answer gets worse, which is why no residual
statistic can catch it, and why the covariance is also computed a second way
with the view as the independent unit. Where that second estimate is wider, it
is what the task-space millimetres are sampled from — so the intervals carry the
inflation rather than a footnote asking you to apply it. Measured against
repeated refits under a correlated field, the classical propagation lands at
0.12 of the true spread in millimetres and the widened one at 1.14.

That second estimate is a leave-one-view-out jackknife rather than a plain
sandwich, because residuals evaluated where the fit put them are smaller than
the errors that produced them, and the plain version inherits the shrinkage —
it measured 0.73 of the truth where the corrected one measures 0.95. The
correction errs wide on purpose, so **a capture whose noise model holds pays
about 23% for the insurance**; that is the deliberate trade, and
[limitations.md](limitations.md) states it rather than burying it.
`--no-widen` gives the unwidened figure for when you have independent reason to
trust the noise model. Feed the measured value back with `--noise-px 0.011` to
move the pixel-noise half.

The same comparison is an information-matrix test, so it fires on *any*
mismatch between the model and the data, not only on correlated noise. The one
worth knowing about is a constraint your camera does not satisfy: `--tie-aspect`
on a camera whose pixels are half a per cent off square raises a CRITICAL on
otherwise clean data. When the fit was constrained the report names that first,
because refitting without it is free and rules it out.

## Getting the intrinsics back out

Every command that fits also hands the fit over. `refit`, `diagnose` and
`report` take `--calibration-out`, which writes the intrinsics with the standard
deviation of each free parameter beside it:

```sh
calibsense report session.npz --task length:800mm:100mm \
    --pdf audit.pdf --calibration-out camera.yml
```

The extension picks the format — `.json` for calibsense's own, `.xml` or `.yml`
for OpenCV FileStorage — and `--calibration-out-format ros` writes ROS
`camera_info` instead. Every format names the distortion model explicitly, so a
four-coefficient file written by calibsense is never ambiguous between
Brown-Conrady and Kannala-Brandt the way OpenCV's own sample output is. A
calibration the capture does not fully determine travels with a warning saying
its deviations are lower bounds.

## What it accepts

| | |
|---|---|
| Targets | checkerboard, ChArUco, circle grid (symmetric and asymmetric) |
| Camera models | pinhole with Brown-Conrady (4, 5, 8, 12 or 14 coefficients), fisheye with Kannala-Brandt |
| Sources | a folder of images, or an existing calibration plus its detections |
| Calibration formats in | OpenCV FileStorage, ROS `camera_info`, Kalibr camchain, calibsense JSON |
| Calibration formats out | OpenCV FileStorage, ROS `camera_info`, calibsense JSON |
| Robot poses | JSON or CSV; matrix, quaternion or rotation-vector form; either hand-eye direction; any length unit |

Point ids are per view rather than a fixed grid, so a ChArUco board showing half
of itself still contributes labelled points. A session is one `.npz` loaded with
`allow_pickle=False`, so opening a result cannot execute code.

## What it computes

Everything `cv2.calibrateCamera` works out internally and then throws away: the
full parameter covariance in factored form, rather than the column of marginal
deviations `calibrateCameraExtended` returns, because the off-diagonals are
where `corr(fx, tz) = 0.93` lives; identifiability from the null space of the
Jacobi-scaled Schur complement; a second view-clustered covariance and the ratio
between the two; per-view and per-corner residuals with a radial profile that
separates radial from tangential; conditioning, raw and Jacobi-scaled; held-out
error from K-fold over views; and whether the parameters sit at the optimum of
their own objective, which is frequently false when auditing someone else's
calibration.

The held-out ratio has a blind spot: on the frontoparallel rig above it reads
**1.005** despite `fx` being wrong by 3334 px, because a held-out view solves
its own pose and a wrong depth cancels a wrong focal length. So per-fold
identifiability gates the verdict, and the across-fold parameter spread is
reported beside the covariance's prediction.

## Hand-eye

Eye-in-hand and eye-to-hand, with a covariance over all twelve parameters and
pose-set sufficiency checks. The closed form is implemented here rather than
delegated, because `cv2.calibrateHandEye` is absent from OpenCV 5's Python
bindings while its `CALIB_HAND_EYE_*` flags are still exported.

Two things measurement caught. The residual-based covariance is 3.1x optimistic,
giving 0.39 mm against an actual error of 1.29 mm, because it treats the
target-in-camera poses as exact when they come from the calibration and their
errors are correlated across views; resampling whole calibrations gives 1.22 mm
and is the default. And hand-eye is determined by rotation and nothing else, so
a degenerate pose set produces a small predicted deviation beside a large actual
error: one rotation axis gives 88 mm of error against 1.5 mm predicted, and a
shuffled robot pose gives 367 mm. Both are caught, the latter because conjugate
rotations have equal angles, which no amount of optimisation would have shown.

## Python API

```python
from calibsense import load_session
from calibsense.report import run_audit, write_json, write_pdf
from calibsense.task import LengthAtDepth

session = load_session("session.npz")
audit = run_audit(session, tasks=[LengthAtDepth(depth_mm=800.0, length_mm=100.0)])

print(audit.trustworthy)            # read this first
print(audit.caveats())              # and this, when it is False
print(audit.headline())             # the paragraph the report leads with

write_pdf(audit, "audit.pdf")
write_json(audit, "audit.json")
```

Or a piece at a time with `instrument`, `cross_validate`, `diagnose` and
`propagate` from the top-level package. An `InstrumentedFit` exposes
`conditioning.identifiable` (read it first), `covariance.intrinsic_std()`, the
view-clustered `covariance.robust.std()`, the ratio between them as
`covariance.worst_robust_inflation()` (compare it against
`diagnose.noise.inflation_floor(G)`, not against one — the estimate errs wide by
design), and
`covariance.correlation_with_poses("fx")` for `corr(fx, tz)` per view.

`calibsense.synthetic` generates captures with known truth, which is how the
test suite works. `make docs` runs `pdoc` over the docstrings into `docs/api`.

## Scope

Two bounds worth knowing, because neither is going away soon.

**One camera.** No stereo or multi-camera calibration; a Kalibr chain is read
one camera at a time and the extrinsics between cameras are ignored. The
`stereo` task takes your baseline as exact, and on a real rig the baseline's own
uncertainty usually dominates the triangulated depth.

**Board geometry is treated as exact.** A printed target's 50 to 200 µm of print
scale and flatness error appears nowhere in the covariance, and no check in the
package can see it: scaling the board rescales every pose by the same factor, so
the focal length does not move and the reprojection error does not change at
all. That also means the intrinsics and the `length` and `plane` tasks are
unaffected by it. **Hand-eye is not.** The robot's poses are in true millimetres
and the board-derived poses are in board millimetres, so the translation absorbs
the disagreement at about seven times the board error — at 0.1%, that is 0.7 mm
against a reported random error of 0.25 mm on the rig in the test suite. The
hand-eye report measures that sensitivity for your own rig and says what it
costs. Pass `--board-tolerance 0.001` if you know your target's pitch
uncertainty and it goes into the covariance rather than beside it; without it
the board is treated as exact, because a number invented here would be worse
than a stated assumption. Order a target with a calibration certificate if the
millimetres matter.

## Author

Written by **Abhishek Gola** ([abhishek-gola](https://github.com/abhishek-gola)).
Copyright (C) 2026 Abhishek Gola. Bugs, questions and licensing enquiries via
[the issue tracker](https://github.com/calibsense/calibsense/issues).

## Licence

**AGPL-3.0-only.** See [LICENSE](LICENSE); every source file carries an
`SPDX-License-Identifier` header.

Section 13 reaches network use: if you modify calibsense and let people interact
with the modified version over a network, you must offer those users the
corresponding source. Running it as a CLI, or importing it as a library in
software you do not distribute, triggers nothing. A PyInstaller binary is a
distribution of the program, so `dist/calibsense` must ship with the
corresponding source or a written offer for it.

If AGPL does not suit your organisation, contact the author; the copyright is
held by a single author, so other arrangements are possible.
