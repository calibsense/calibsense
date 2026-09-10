# calibsense

**Your camera calibration reports a reprojection error of 0.2 px. That number
does not tell you whether you can measure a part to half a millimetre.**

calibsense answers the question the reprojection RMS cannot. Given a folder of
calibration images — or a calibration you are already shipping plus the
detections behind it — it computes the full parameter covariance, finds the
parameter combinations your capture does not determine at all, tests whether the
residuals are structured rather than random, and propagates the result into the
units you actually care about:

> A 100 mm feature measured at 800 mm has an expected error of 0.199 mm, with a
> 95% interval of ±0.485 mm. Of that, 40% comes from the calibration and 60%
> from 0.119 px of pixel noise at measurement time.

It also tells you what is wrong with the capture, by name, with the number that
says so and what to do about it. And when a figure cannot be trusted, it refuses
to print it as though it could.

---

## The problem, in one table

Six synthetic rigs, one true camera at `fx = 900 px`, the same 0.2 px corner
noise, the same 18 views, the same board. Run it yourself with
`python examples/degeneracy_demo.py`:

| rig | reprojection RMS | fx error | sd(fx) | identifiable |
|---|---|---|---|---|
| flat, 1 depth | 0.2753 px | **+3334.2 px** | 708.0 | no |
| flat, 3 depths | 0.2761 px | **−139.1 px** | 0.336 | no |
| 5° tilt, 1 depth | 0.2754 px | +41.2 px | 28.5 | yes |
| 20° tilt, 1 depth | 0.2757 px | +2.3 px | 3.93 | yes |
| 40° tilt, 1 depth | 0.2761 px | +0.3 px | 2.24 | yes |
| 40° tilt, 3 depths | 0.2759 px | +1.3 px | 1.19 | yes |

Every row fits its own images to within a thousandth of a pixel of every other
row. The focal length ranges from correct to wrong by more than three thousand
pixels. An engineer looking at the first row sees 0.275 px and ships it.

Two results in that table contradict the usual advice, and both are worth
knowing before your next capture session.

**Tilt is what makes the focal length identifiable, not depth variation.** A
frontoparallel planar target gives one homography per view, and since every view
carries its own free translation, rescaling all focal lengths and all depths
together reproduces the images exactly. A fourth working distance does not help,
because it arrives with its own free translation. Only perspective foreshortening
*within* a single view pins the focal length. Depth spread is a real
second-order gain — sd(fx) falls from 2.24 to 1.19 — but only once tilt exists.

**Read `identifiable` before you read the standard deviation.** Row two reports
`sd(fx) = 0.336 px` while being wrong by 139 px. That is not a bug: a
pseudo-inverse assigns *zero* variance to a direction the data does not
constrain, so a rank-deficient fit reports false confidence rather than a wide
error bar. calibsense computes identifiability from the null space of the
Jacobi-scaled Schur complement and puts it above every other number it prints.

---

## Install

From a checkout:

```sh
pip install .
```

Three runtime dependencies — `numpy`, `opencv-contrib-python`, `pyyaml`. The
list is short on purpose, because the second shipping target is a single binary
that needs no Python present:

```sh
make binary        # -> dist/calibsense, 64 MB
```

Requires Python 3.9 or newer.

---

## Usage

### 1. Detect the target across your images

```sh
calibsense ingest images \
    --images captures/ \
    --target checkerboard:9x6:25mm \
    -o session.npz
```

Every image is reported, including the ones that failed, because a detector that
silently drops a third of your capture is the first thing you want to know
about:

```
  [   1/20] ok  view000
  [   2/20] ... view001  no 9x6 checkerboard found
  [   3/20] ok  view002
  ...
calibsense session
===============

  target       9x6 checkerboard, 25 mm squares
  image size   1280 x 720
  views        14
  points       756 total, 54-54 per view (median 54)
  detection    14/20 images (70%) via checkerboard
```

Target shorthand avoids a config file for the common case:

```
checkerboard:9x6:25mm
charuco:8x11:20mm:15mm:DICT_5X5_1000
circles:4x11:0.75in:asymmetric
```

Auditing a calibration you already ship? Point at it too:

```sh
calibsense ingest images --images captures/ --target checkerboard:9x6:25mm \
    --calibration shipped.yml -o session.npz
```

Then `calibsense refit session.npz --no-refit` instruments *that* calibration in
place rather than re-estimating it — which reports, among other things, whether
it is even the optimum of its own detections.

### 2. Ask what the capture supports

```sh
calibsense diagnose session.npz
```

```
calibsense diagnosis
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
         Depth variation, Target scale, Noise model, Outlier and high-leverage views
```

`diagnose` refits, cross-validates and runs every check. **It exits 3 on a
critical finding**, so a CI step can gate on calibration quality without parsing
anything.

### 3. Get the answer in millimetres

```sh
calibsense report session.npz \
    --task length:800mm:100mm --task plane:800mm \
    --json audit.json --pdf audit.pdf \
    --camera-name "line-3 inspection" --contact metrology@example.com
```

```
Camera calibration measurement audit
====================================

A 100 mm feature measured at 800 mm has an expected error of 0.199 mm, with a
95% interval of +/-0.485 mm.

The reported 0.1633 px reprojection error is an honest error estimate: held-out
views reprojected to 0.168 px, a ratio of 1.03x.

The dominant cause is image coverage: corners occupy 40% of an 8x6 grid, 12% of
the border ring, and reach 56% of the way to the frame corner; the periphery is
thin.
```

That capture was rendered through a camera with `fx = 700.0`. The fit recovered
`700.855 ± 1.121`, which is 0.76 sigma from truth, and all nine intrinsics land
inside their own reported interval.

The **JSON** carries every number behind the report. The **PDF** is the two-page
version a quality manager can sign: it leads with the sentence above, then the
task-space error table, the variance split, the full parameter table, the
findings, and a forecast of what more views would buy.

`calibsense noise-floor`, `calibsense refit`, `calibsense show`, `calibsense formats`
and `--help` cover the rest. For reproducible output, pass `--deterministic` before the subcommand —
`calibsense --deterministic report ...` — at some cost in speed.

---

## What the report contains

Abbreviated from a real run:

```
TASK-SPACE ERROR
quantity        nominal   expected error   95% low   95% high        bias
length_mm           100            0.199    -0.484    +0.485   +0.000601

where the error comes from
length_mm   ####################----------------------------   40% / 60%
Dark is the calibration; light is 0.119 px of pixel noise at measurement time.
Only the dark part can be improved by re-calibrating.

THE CALIBRATION
Model                    pinhole_brown_conrady, 5 distortion terms
Views / points           14 / 756
In-sample RMS            0.1633 px
Residual sigma           0.1192 px per coordinate
Identifiable             yes
Scaled condition number  1.174e+03 (rank 9/9)
Out-of-sample RMS        0.1680 px over 5 folds, ratio 1.03x

parameter        value      std dev     relative     weak
fx             700.855        1.121        0.16%     0.00
fy              703.95        1.132        0.16%     0.00
cx             639.047        1.482        0.23%     0.00
cy             357.777        1.464        0.41%     0.00
k1           -0.305716     0.003258        1.07%     0.00
...

WHAT MORE VIEWS WOULD BUY
Adding 6 views with the board pushed into the frame edges and corners would
reduce the interval to approximately +/-0.446 mm, 1.1x narrower than today.
```

The **variance split** is the line most people act on. Calibration error and
pixel noise at measurement time are independent, so their variances add, and
separating them answers the question that follows every error figure: on this
capture, re-calibrating can improve the answer by at most 40%. The rest is
photons.

The last line predicts uncertainty, not correctness — the recommended views are
synthesised from the calibration already in hand, so they agree with it by
construction. The report says so wherever it appears.

### When a number cannot be trusted, it says so first

```
THE CORNER NOISE IS CORRELATED - EVERY FIGURE BELOW IS TOO TIGHT BY ABOUT 3.0X
```

That banner opens the report, and the JSON carries `"trustworthy": false` with
the reason, so an automated consumer cannot miss it. Two conditions raise it: a
calibration that leaves a parameter direction undetermined, and corner noise
correlated enough that the covariance is measurably too tight.

---

## What it accepts

| | |
|---|---|
| Targets | checkerboard, ChArUco, circle grid (symmetric and asymmetric) |
| Camera models | pinhole with Brown–Conrady (4, 5, 8, 12 or 14 coefficients), fisheye with Kannala–Brandt |
| Sources | a folder of images, or an existing calibration plus its detections |
| Calibration formats | OpenCV FileStorage (`.yml`/`.yaml`/`.xml`), ROS `camera_info`, Kalibr camchain, calibsense JSON |
| Detections | calibsense JSON, or an NPZ of `image_points` |
| Robot poses | JSON or CSV; matrix, quaternion or rotation-vector form; either hand-eye direction; any length unit |

Point ids are per view rather than a fixed grid, so a ChArUco board showing half
of itself contributes correctly labelled points and nothing downstream branches
on target type. A session is one `.npz` with a JSON manifest inside it, loaded
with `allow_pickle=False`, so opening a result can never execute code from it.

---

## What it computes

Everything `cv2.calibrateCamera` works out internally and then throws away.

**Full parameter covariance**, not marginal standard deviations. Held in
factored form — `σ²S⁻¹` for the intrinsics plus per-view factors — so memory is
linear in the number of views and any block is rebuilt on demand.
`calibrateCameraExtended` hands you a column of numbers; the off-diagonals are
where `corr(fx, tz) = 0.93` lives, and that correlation is the whole story of why
focal length and working distance trade off against each other.

**Identifiability.** The weak directions of the scaled Schur complement, each
described by the parameters that dominate it, plus a per-parameter share of the
weak subspace. This outranks every other number.

**Two independent covariance estimates.** The classical one assumes corner noise
is independent, isotropic and homoscedastic. A second, view-clustered estimate
assumes only that different views are independent. The ratio between them is
reported as a finding — see *Noise model* below.

**Residual structure.** Per-view RMS, p95, max, signed bias per axis, and a
MAD-based z-score flagging outlier views. Per-corner residuals kept in full,
plus a profile binned by image radius separating radial from tangential. A signed
radial mean that grows with radius is the signature of a distortion model that
does not match the lens.

**Conditioning.** Condition number of the normal equations, raw and
Jacobi-scaled. Only the scaled one means anything, since a focal length in pixels
and a distortion coefficient near zero are not comparable until the units come
out. Roughly 1e3 is well posed; 1e8 and up is not.

**Out-of-sample error.** K-fold over views: fit on the rest, predict the held-out
ones, report both RMS values and the ratio. Each held-out view solves its own
pose by PnP at the fold's intrinsics, which is the deployment-relevant question
and also what stops a view leaking into its own prediction.

That ratio has a blind spot, and it is reported rather than hidden. On the
frontoparallel rig above — the one whose `fx` is wrong by 3334 px — the ratio
comes out at **1.005**, because a held-out view solves its own pose and a
proportionally wrong depth cancels a proportionally wrong focal length. Any
degeneracy a free pose can absorb is invisible to reprojection-based
cross-validation. So per-fold identifiability gates the verdict, and the
across-fold parameter spread is reported beside the covariance's prediction:

| rig | out-of-sample ratio | fold spread of fx | covariance prediction |
|---|---|---|---|
| diverse | 1.01x | 1.15 px | 1.80 px |
| frontoparallel | 1.00x | **326 px** | 636 px |

**Whether the parameters sit at the optimum of their own objective**, measured as
the relative cost reduction a full Newton step would predict. Always true after a
refit; frequently false when auditing someone else's calibration, which the
report states outright:

```
NOT AT OPTIMUM  a Newton step would cut the cost by 34.03%
```

---

## The diagnostics

One named cause per check, each with the number behind it and what to do:

| Cause | Measured as |
|---|---|
| Pose diversity | max pairwise angle between board normals, plus the orientation tensor for whether the normals are confined to a plane |
| Depth variation | max-over-min working distance, with the focal–distance correlation from the covariance |
| Frontoparallel dominance | fraction of views within 15° of the image plane, linked to the parameters it leaves undetermined |
| Image coverage | grid occupancy, border-ring occupancy, radial reach |
| Target scale | median nearest-neighbour corner spacing, against the residual sigma |
| Distortion model adequacy | chi-square and trend tests on the radial residual profile |
| Noise model | classical intrinsic deviations against a view-clustered sandwich |
| Outlier and high-leverage views | each view's share of the intrinsic information, paired with its residual |
| Out-of-sample error | the cross-validated ratio, gated on per-fold identifiability |

Three of these needed a real decision rather than a formula, and the reasoning
affects how you read them.

**Leverage needed a definition.** The Schur complement is a sum over views,
`S = Σᵢ Sᵢ`, so `trace(Sᵢ S⁻¹)` sums to the parameter count and
`trace(Sᵢ S⁻¹) / p` is that view's share of everything the capture knows about
the intrinsics. Shares sum to one and read directly as percentages. A view
combining a large share with a large residual is the dangerous case, because the
fit is being pulled by the frame it depends on most.

**The residual test clusters by view.** Z-scoring each radial bin against `σ/√n`
assumes corners are independent, and they are not — corners in one view share
that view's pose, so a small pose error moves all of them together. The naive
version reported +6.3 sigma of "structure" on a capture whose focal length was
recovered to 0.6 sigma. Treating the *view* as the independent unit needs no
noise assumption at all:

| case | reprojection RMS | flatness z | verdict |
|---|---|---|---|
| pinhole data, pinhole model | 0.2768 px | −0.16 | ok |
| pinhole data, k3 dropped | 0.2770 px | +0.44 | ok |
| fisheye data, fisheye model | 0.2767 px | −0.61 | ok |
| fisheye data, pinhole model | 0.2832 px | **+5.22** | critical |

The RMS moves by 2%. The residual structure is unmistakable.

**The same clustering argument protects the covariance.** Every deviation
calibsense prints rests on `σ² = cost / (m − p)`, which measures the noise scale
but *asserts* its shape. Breaking that assumption one way at a time showed most
of the shape does not matter: anisotropy along the edge direction, noise several
times worse in some views than others, and a few per cent of badly mis-detected
corners each leave the predicted deviation within about 15% of the truth.

Spatial correlation across the frame is the exception, and it is severe — because
a correlated field is partly absorbable by the pose and distortion parameters, so
it *lowers* the residual while *raising* the estimator's real spread:

| injected noise | reported σ̂ | predicted sd(fx) | true sd(fx) |
|---|---|---|---|
| independent, 0.25 px | 0.253 px | 3.14 px | 3.03 px |
| correlated, 60 px length | 0.162 px | 1.95 px | **5.69 px** |
| correlated, 200 px length | 0.055 px | 0.67 px | **5.24 px** |

The RMS gets *better* as the answer gets worse, which is why no residual
statistic can catch it. Field-varying defocus, an illumination or vignetting
gradient pulling on the sub-pixel refinement, a board that is not flat, motion
blur and a rolling shutter all produce exactly this. So the covariance is
computed a second way, with the view as the independent unit:

```
Cov = S⁻¹ (Σᵢ sᵢsᵢᵀ) S⁻¹ · G/(G−1)
```

where `sᵢ` is what view `i` contributes to the intrinsic estimate after its own
pose has absorbed what it can. Nothing is assumed about the noise *within* a
view. Near one, the assumption held for your capture and the intervals are
defensible by measurement rather than by assertion. At 3x, they are a third of
what the data supports, and the report says which parameter and by how much.

### Or measure the noise instead of inferring it

The sandwich estimator detects the problem without naming it. To see the noise
itself, take the model out of the loop entirely: point a fixed camera at a fixed
target, capture thirty or more frames without touching either, and look at how
far the detected corners wander.

```
calibsense noise-floor --images static/ --target checkerboard:9x6:25mm
```

```
frames       40 static frames, 54 corners
scale        0.0270 px total, 0.0262 px irreducible (6% absorbable by pose)
correlation  length 0.0 px, 0.00 of the 31 px corner spacing (independent)
shape        anisotropy 1.68 against a 1.21 floor from 40 frames, 65% of long
             axes within 30 deg of the board edge (33% by chance)
drift        0.0090 px largest whole-frame shift
use          sigma = 0.0262 px when propagating, in place of one inferred from
             residuals
```

The correlation length is the number that decides whether the rest of the report
can be trusted, and it is measured here rather than inferred. Independent noise
reads `0.0`; a field that moves neighbouring corners together reads at the scale
of that field.

It is judged **in corner spacings, not pixels**, because pixels do not transfer
between rigs: injecting one field into a 9x6 board at 700 mm gives a 51 px
correlation length and into a 21x14 board 79 px, for the same fault. What decides
whether the independence assumption fails is whether *adjacent* corners share
noise, and the denser board correctly reads as worse — more of the independent
observations the covariance is counting are not independent. Above one spacing
the robust covariance is the one to read; above three, the cause needs finding
before any interval is quotable. Drift is separated from a genuinely correlated field, because
drift is perfectly correlated across the whole board and disappears once a
per-frame affine is removed — the same thing a free per-view pose does during a
fit. A correlation length longer than the board cannot be measured, so both
profiles stop at the widest corner separation rather than extrapolating.

The anisotropy is quoted against its own floor, because a sample covariance from
few frames is elongated by chance: ten frames put the expected ratio at 1.53, so
a fixed threshold would call every clean short capture directional. The floor is
exact rather than tabulated — with `t = (λ₁−λ₂)/(λ₁+λ₂)`, `t²` follows
`Beta(1, (n−2)/2)`, verified against simulation to three decimals from ten
frames to a thousand. The run above found a real directional detector on
rendered images, at 1.68 against a 1.21 floor with two thirds of the long axes
lying on the board edge — the expected signature of anything that finds a corner
by locating two edges, and something the tool discovered rather than was told.

The measured sigma then feeds back in, so it changes the answer rather than just
sitting in a report:

```
calibsense report session.npz --pdf report.pdf --noise-px 0.026
```

That moves the pixel-noise half of the task-space error and leaves the
calibration half untouched. A capture whose corners move more than 5 px is
refused with an explanation instead of a number — that is motion, not detector
noise.

---

## Task-space error

Monte Carlo from the covariance into the units of the job. Whole parameter sets
are drawn from the joint distribution rather than one deviation at a time,
because the trade-offs between parameters are what determine the answer —
sampling `fx` and `tz` independently from their marginals would give a spread far
wider than reality, since the two correlate at 0.93 and their errors partly
cancel.

Four tasks, each available from the CLI as `--task`:

| Shorthand | Question it answers |
|---|---|
| `length:800mm:100mm` | how accurately can I measure a 100 mm feature at 800 mm? |
| `plane:800mm:15deg` | how accurately do I locate a plane's distance and tilt? |
| `stereo:800mm:120mm` | what is the depth error of a stereo triangulation? |
| `base:800mm` | where is a point in the robot's base frame? |

On a degenerate capture the same task reports its figure as a lower bound rather
than an answer, because an unconstrained direction carries no variance to sample
along.

---

## Hand-eye

Eye-in-hand and eye-to-hand, with a covariance over all twelve parameters and
pose-set sufficiency diagnostics. The closed-form initialisation is implemented
here rather than delegated, because `cv2.calibrateHandEye` is absent from
OpenCV 5's Python bindings while its `CALIB_HAND_EYE_*` flags are still exported.

Two things worth knowing before you trust a hand-eye number.

**The obvious covariance is 3.1x optimistic.** A residual-based estimate gave a
translation deviation of 0.39 mm against an actual error of 1.29 mm. It treats
the target-in-camera poses as exact data when they come from the calibration, and
their errors are *correlated across views* because every view shares the same
intrinsics — a focal-length error tilts and scales all of them coherently, so it
does not average down. Resampling whole calibrations gives 1.22 mm, and that is
the default.

**Hand-eye is determined by rotation and by nothing else**, which is why the
sufficiency checks matter. Every degenerate pose set produces a small predicted
deviation beside a large actual error:

| pose set | true error | predicted sd | caught by |
|---|---|---|---|
| varied axes | 2.2 mm | 1.5 mm | (clean) |
| mispaired robot poses | **367 mm** | 1.9 mm | pose pairing |
| one rotation axis | **88 mm** | 1.5 mm | rotation axes, conditioning |
| tiny wrist rotations | — | — | the solve refuses outright |

The pairing check is the one worth having: conjugate rotations have equal angles,
so a robot pose shuffled against its image is detectable, and no amount of
optimisation would have revealed it.

---

## Python API

The whole audit in one call:

```python
from calibsense import load_session
from calibsense.report import run_audit, write_json, write_pdf
from calibsense.task import LengthAtDepth

session = load_session("session.npz")
audit = run_audit(session, tasks=[LengthAtDepth(depth_mm=800.0, length_mm=100.0)])

print(audit.trustworthy)          # read this first
for reason in audit.caveats():    # and this, when it is False
    print(reason)
for sentence in audit.headline():
    print(sentence)

write_pdf(audit, "audit.pdf")
write_json(audit, "audit.json")
```

Or a piece at a time:

```python
from calibsense import cross_validate, diagnose, instrument, session_from_images
from calibsense.cli.targets import resolve_target
from calibsense.handeye import diagnose_hand_eye, solve_hand_eye
from calibsense.task import LengthAtDepth, propagate

session = session_from_images("captures/", resolve_target("checkerboard:9x6:25mm"))
fit = instrument(session)
validation = cross_validate(session)
findings = diagnose(fit, session.observations, validation)
error = propagate(fit, LengthAtDepth(depth_mm=800.0, length_mm=100.0))

print(fit.conditioning.identifiable)                      # read this first
print(validation.ratio, validation.spread_of("fx"))
print(findings.verdict())
print(error.distribution("length_mm").statement())
print(error.variance_share("length_mm"))                  # calibration vs pixels

print(fit.covariance.intrinsic_std())                     # the usual deviations
print(fit.covariance.robust.std())                        # assuming only view independence
print(fit.covariance.worst_robust_inflation())            # the ratio between them
print(fit.covariance.correlation_with_poses("fx")[:, 5])  # corr(fx, tz) per view
print(fit.covariance.dense())                             # if you want it all
```

`calibsense.synthetic` generates captures with known truth, which is how the test
suite works and the easiest way to explore what a capture geometry would buy you.

API reference: `make docs` runs `pdoc` over the docstrings into `docs/api`. There
is no hand-written duplicate of anything a docstring says.

---

## Limitations

Stated here rather than left to be discovered.

**One camera.** No stereo or multi-camera calibration. A Kalibr chain is read one
camera at a time and the extrinsics between cameras are ignored. The `stereo`
task takes your baseline as exact, and for a real rig the baseline's own
uncertainty usually dominates the triangulated depth.

**Board geometry is treated as exact.** A printed target's real error — 50 to
200 µm of print scale and flatness on a typical board — appears nowhere in the
covariance. This is a *systematic* on metric scale: a 0.1% board scale error goes
straight into `fx` and therefore straight into millimetres, and it could
plausibly dominate everything that is modelled. Order a target with a calibration
certificate if the millimetres matter.

**The noise-model inflation is reported, not applied.** When the `noise_model`
finding says your intervals are 3x too tight, that includes the task-space
millimetres. Multiply them yourself; the report does not.

**Robot repeatability is treated as exact.** The arm's own repeatability is a
specification you have, and there is currently nowhere to put it.

**Outliers are named but not down-weighted.** There is no robust loss, so a bad
view inflates `σ²` and therefore widens *every* parameter's interval instead of
being excluded. The finding says so.

**The fit is delegated to OpenCV**, which constrains what can be asked for: `cx`
and `cy` together or neither, never one; no custom or robust loss; no per-point
weights.

**A checkerboard's 180° ambiguity is unresolved.** Harmless for intrinsics, since
each view's pose absorbs the flip — but *not* harmless for hand-eye. Use a ChArUco
target if you are solving hand-eye.

**ChArUco `legacy_pattern` cannot be auto-detected.** A wrong setting produces a
confidently wrong calibration rather than a detection failure.

**Circle-grid centroid bias is uncorrected.** The centroid of a projected circle
is not the projection of the circle's centre, and under tilt that is a systematic
sub-pixel bias which grows with obliquity — biasing exactly the tilted views the
pose-diversity diagnostic asks for.

**Motion blur and rolling shutter are not diagnosed.** Both produce the
frame-correlated error the `noise_model` check will flag, but neither is named
directly.

**Robot poses align by view id only.** No timestamp interpolation or sync; real
robot logs are timestamped at a different rate from the camera.

**Some thresholds are judgement rather than measurement.** The pose-diversity cuts
are calibrated against rigs with known truth, and the noise-model cuts against
Monte Carlo. The coverage and reach thresholds are reasoned but unvalidated, and
the coverage one in particular will fire on captures whose focal length is
recovered to well under a sigma.

**No drift or staleness detection.** Whether a calibration is still valid three
months later is not something this answers yet.

**The PDF renders CP1252 only.** The base-14 fonts are used without embedding,
which is what keeps the dependency count at three; a Greek letter in a camera
name becomes `?`.

---

## How the claims are checked

Every numeric claim in this README has a test behind it that could fail.

| Claim | How it is checked | Result |
|---|---|---|
| The analytic Jacobians are correct | central differences, 6 cameras × 3 poses, both models | 1e-9 relative |
| Marginal standard deviations are right | vs `cv2.calibrateCameraExtended` | agree to 6 significant figures |
| The Schur complement is exact | vs a dense `σ²(JᵀJ)⁻¹` | 6e-10 relative |
| The covariance is *calibrated* | 250 Monte Carlo refits vs the prediction | predicted/empirical sd within 2–4%; mean Mahalanobis 9.37 against an expected 9 |
| The assumed noise *shape* mostly does not matter | Monte Carlo injecting anisotropy, per-view heteroscedasticity and heavy tails | predicted/empirical sd stays within 0.96–1.17 |
| Correlated corner noise breaks it | Monte Carlo, 200 px correlated field | predicted/empirical sd 0.12–0.39, while σ̂ *falls* from 0.25 to 0.055 px |
| The view-clustered covariance recovers it | same rig | 0.12 → 0.75 of the true spread at 14 views, 0.13 → 0.84 at 30 |
| Degeneracy is detected | frontoparallel rig | `identifiable = False`, weak direction `−0.71·fx −0.71·fy` |
| The cross-validation blind spot is pinned | frontoparallel rig, asserted to stay under 1.1x | 1.005x — the test fails if it silently starts working |
| Each diagnostic is specific | single-fault rigs × every cause | every fault fires, every unrelated cause stays quiet |
| Sampling reproduces the joint distribution | 4000 draws vs predicted deviations and correlations | sd within 12%, correlation within 0.08 |
| The variance split adds up | combined vs parameters + noise | within 20% |
| Hand-eye recovers known truth | both mountings, derived robot poses | under 6 mm and 1° |
| The resampled hand-eye covariance covers the error | truth vs predicted deviation | 1.29 mm actual, 1.22 mm predicted |
| Detectors find real patterns | rendered images, all three targets, both models | 0.09–0.4 px mean localisation error |
| The frozen binary works | `dist/calibsense` in a stripped environment | full pipeline, `fx = 899.564 ± 0.751` |

```
1038 passed, 11 skipped in 228s      # make test
TOTAL  5561 statements, 159 missed, 97%
```

Ten skips are OpenCV-4-only flag-namespace checks; the eleventh is a hand-eye
rig whose two variants happened to select the same views. `make test` runs everything;
`make fast` skips the Monte Carlo, image-rendering and propagation tests.

Reports are reproducible to about 1e-6 relative rather than bit-exact, because
`cv2.calibrateCamera` reduces across threads and floating-point addition is not
associative — eight identical refits spanned 1.1e-12 px on a ten-thread machine.
`--deterministic` pins OpenCV to one thread; output is then identical apart from
the two wall-clock timestamps.

---

## Design notes

**Layering.** `calibsense.core` holds the data — cameras, targets, detections,
poses, sessions — and does not import OpenCV. Projection, differentiation and
fitting live in `calibsense.refit`; detection and file reading in `calibsense.ingest`.
The dependency arrows all point at `core`, which is why the core is testable
without a single image.

**Registries are literal dictionaries** populated by static imports, never a
filesystem scan, and the package ships no data files. That is what makes the
frozen binary work without `sys._MEIPASS` handling, and a test asserts both
properties so they cannot quietly regress.

**Nothing calls `numpy.linalg.inv`.** Every inverse goes through an
eigendecomposition that reports its rank, its condition numbers and the
directions the data leaves unconstrained. A near-singular calibration is the
finding, not an inconvenience.

**OpenCV version differences are handled by name, not by value.** OpenCV 4 keeps
the fisheye flags in `cv2.fisheye` with their own bit numbering and defines
same-named constants in `cv2` with *different* values; OpenCV 5 merged them onto
the pinhole numbering. Reading `cv2.CALIB_FIX_K1` and handing it to
`cv2.fisheye.calibrate` therefore fixes the wrong coefficient on OpenCV 4 and the
right one on OpenCV 5. Every flag resolves through `calibsense.refit.cv_compat`.

Three OpenCV behaviours worth knowing, all found while building this:

- `cv2.fisheye.calibrate` estimates its own starting intrinsics when not given
  any, and on ordinary captures that estimate *fails outright* — it raises
  `fabs(norm_u1) > 0` from `InitExtrinsics` rather than returning a poor answer.
  calibsense supplies its own ladder of focal-length guesses starting at `width/π`
  and reports which rung converged.
- A `CALIB_FIX_Kn` flag holds the supplied value on the pinhole path and *zeroes*
  the coefficient on the fisheye path. Either way the parameter leaves the
  covariance, so the reported uncertainty is right; only the resulting value
  differs.
- `cv2.calibrateHandEye` is gone from OpenCV 5's Python bindings while its
  `CALIB_HAND_EYE_*` flags are still exported.

---

## Licence

**AGPL-3.0-only.** See [LICENSE](LICENSE) for the full text; every source file
carries an `SPDX-License-Identifier: AGPL-3.0-only` header.

One consequence worth stating plainly rather than leaving to be discovered.
**Section 13 reaches network use:** if you modify calibsense and let people
interact with the modified version over a network — an upload-your-calibration
web service, a report generator behind an internal API — you have to offer those
users the corresponding source. Running it as a CLI, or importing it as a library
in software you do not distribute, triggers nothing.

The three runtime dependencies are compatible: `numpy` is BSD-3-Clause, `pyyaml`
is MIT, and OpenCV is Apache-2.0 with LGPL-2.1 components in the distributed
wheels. All are one-way compatible into an AGPL work.

A PyInstaller binary of an AGPL program is a distribution of the program, so
`dist/calibsense` has to be accompanied by the corresponding source or a written
offer for it.

If AGPL does not suit your organisation, contact the author — the copyright is
held by a single author, so other arrangements are possible.
