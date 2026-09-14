# calibsense - measurement uncertainty for camera calibration.
# Copyright (C) 2026 Abhishek Gola
#
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License, version 3, as published by
# the Free Software Foundation. This program is distributed WITHOUT ANY WARRANTY;
# without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the LICENSE file, or <https://www.gnu.org/licenses/>.

"""M5 — sampling parameter sets from a covariance."""

from __future__ import annotations

import numpy as np
import pytest

from calibsense.errors import ValidationError
from calibsense.task.sampling import (
    CovarianceSampler,
    ParameterSample,
    factorise,
    joint_covariance,
    with_intrinsic_block,
)

from . import rigs


def test_factorise_reproduces_a_full_rank_covariance():
    rng = np.random.default_rng(0)
    root = rng.normal(size=(6, 6))
    covariance = root @ root.T + np.eye(6)
    factor, rank = factorise(covariance)
    assert rank == 6
    assert np.allclose(factor @ factor.T, covariance)


def test_factorise_handles_a_rank_deficient_covariance():
    """A calibration that is not identifiable has no Cholesky factor."""
    rng = np.random.default_rng(1)
    direction = rng.normal(size=5)
    direction /= np.linalg.norm(direction)
    projector = np.eye(5) - np.outer(direction, direction)
    root = rng.normal(size=(5, 5))
    covariance = projector @ (root @ root.T) @ projector
    factor, rank = factorise(covariance)
    assert rank == 4
    assert np.allclose(factor @ factor.T, covariance, atol=1e-8)
    # The cut direction gets no variance, which is the problem the flag exists for.
    assert abs(float(direction @ factor @ factor.T @ direction)) < 1e-8


def test_joint_covariance_assembles_the_blocks():
    context = rigs.healthy()
    covariance = context.fit.covariance
    joint, names = joint_covariance(covariance, (0, 2))
    p = covariance.n_intrinsic
    assert joint.shape == (p + 12, p + 12)
    assert len(names) == p + 12
    assert names[:p] == covariance.intrinsic_names
    assert names[p] == "view0.rx"
    assert names[p + 6] == "view2.rx"
    assert np.allclose(joint[:p, :p], covariance.intrinsic)
    assert np.allclose(joint[:p, p : p + 6], covariance.cross_block(0))
    assert np.allclose(joint[p : p + 6, p : p + 6], covariance.extrinsic_block(0))
    assert np.allclose(joint, joint.T)


def test_joint_covariance_with_no_views_is_the_intrinsic_block():
    covariance = rigs.healthy().fit.covariance
    joint, names = joint_covariance(covariance)
    assert np.allclose(joint, covariance.intrinsic)
    assert names == covariance.intrinsic_names


@pytest.mark.parametrize("views", [(0, 0), (99,), (-1,)])
def test_joint_covariance_rejects_bad_view_indices(views):
    with pytest.raises(ValidationError):
        joint_covariance(rigs.healthy().fit.covariance, views)


def test_samples_reproduce_the_marginal_deviations():
    fit = rigs.healthy().fit
    sampler = CovarianceSampler(fit, seed=1)
    draws = sampler.draw(4000)
    focal = np.array([d.camera.fx for d in draws])
    predicted = fit.covariance.intrinsic_std()[0]
    assert focal.std() == pytest.approx(predicted, rel=0.12)


def test_samples_reproduce_the_correlations():
    """Sampling marginals independently would miss the point of a covariance."""
    fit = rigs.healthy().fit
    sampler = CovarianceSampler(fit, view_indices=(0,), seed=2)
    draws = sampler.draw(4000)
    focal = np.array([d.camera.fx for d in draws])
    depth = np.array([d.poses[0].translation[2] for d in draws])
    sampled = float(np.corrcoef(focal, depth)[0, 1])
    predicted = float(fit.covariance.correlation_with_poses("fx")[0, 5])
    assert sampled == pytest.approx(predicted, abs=0.08)


def test_a_well_posed_fit_is_bounded():
    sampler = CovarianceSampler(rigs.healthy().fit, view_indices=(0, 1), seed=0)
    assert sampler.bounded
    assert sampler.rank == sampler.n_free


def test_a_degenerate_fit_is_not_bounded():
    sampler = CovarianceSampler(rigs.frontoparallel().fit, view_indices=(0, 1), seed=0)
    assert not sampler.bounded
    assert sampler.rank < sampler.n_free


def test_the_nominal_sample_is_the_fit_itself():
    fit = rigs.healthy().fit
    nominal = CovarianceSampler(fit, view_indices=(1,), seed=0).nominal()
    assert nominal.camera is fit.camera
    assert nominal.poses[0] is fit.poses[1]
    assert nominal.hand_eye is None


def test_samples_are_deterministic_for_a_seed():
    fit = rigs.healthy().fit
    first = CovarianceSampler(fit, seed=5).draw(20)
    second = CovarianceSampler(fit, seed=5).draw(20)
    assert [d.camera.fx for d in first] == [d.camera.fx for d in second]


def test_fixed_parameters_never_move():
    from calibsense.core.session import CalibrationRecord, CalibrationSession
    from calibsense.refit import RefitOptions, instrument

    capture = rigs.healthy()
    session = CalibrationSession(
        observations=capture.observations,
        prior=CalibrationRecord(rigs.WIDE_PINHOLE, rigs.IMAGE_SIZE, "a.yml"),
    )
    fit = instrument(session, RefitOptions(fixed=("cx", "cy")))
    draws = CovarianceSampler(fit, seed=0).draw(200)
    assert len({round(d.camera.cx, 9) for d in draws}) == 1
    assert len({round(d.camera.fx, 9) for d in draws}) > 1


def test_a_tied_aspect_ratio_moves_both_focal_lengths_together():
    from calibsense.core.session import CalibrationRecord, CalibrationSession
    from calibsense.refit import RefitOptions, instrument

    capture = rigs.healthy()
    session = CalibrationSession(
        observations=capture.observations,
        prior=CalibrationRecord(rigs.WIDE_PINHOLE, rigs.IMAGE_SIZE, "a.yml"),
    )
    fit = instrument(session, RefitOptions(tie_aspect=True))
    draws = CovarianceSampler(fit, seed=0).draw(300)
    ratios = np.array([d.camera.fy / d.camera.fx for d in draws])
    assert ratios.std() < 1e-9


def test_draw_rejects_a_non_positive_count():
    with pytest.raises(ValidationError, match="at least one sample"):
        CovarianceSampler(rigs.healthy().fit, seed=0).draw(0)


def test_marginal_std_matches_the_covariance_diagonal():
    """With the widening off, the sampler is the covariance and nothing else."""
    fit = rigs.healthy().fit
    sampler = CovarianceSampler(fit, view_indices=(0,), seed=0, widen=False)
    assert sampler.inflation == 1.0
    assert np.allclose(
        sampler.marginal_std()[: fit.covariance.n_intrinsic],
        fit.covariance.intrinsic_std(),
    )


def test_hand_eye_is_sampled_when_supplied():
    from calibsense.handeye import solve_hand_eye
    from calibsense.refit import instrument

    session = rigs.hand_eye_session("eye_in_hand")
    fit = instrument(session)
    result = solve_hand_eye(fit, session, "eye_in_hand", monte_carlo=False)
    sampler = CovarianceSampler(fit, seed=0, hand_eye=result)
    draws = sampler.draw(200)
    assert all(d.hand_eye is not None for d in draws)
    offsets = np.array([d.hand_eye.translation for d in draws])
    assert offsets.std(axis=0).max() > 0
    assert sampler.nominal().hand_eye is result.camera


# --------------------------------------------------------------------------
# widening the intrinsic block when the noise model does not hold
# --------------------------------------------------------------------------

def test_widening_costs_a_premium_and_not_a_multiple_when_the_noise_model_holds():
    """The check that only ever fires is not a check.

    The leverage correction in the view-clustered estimate is deliberately
    conservative, so a capture whose noise model holds perfectly still pays
    something here — measured at about a fifth, against the factor of eight the
    correction exists to catch. What must not happen is the premium turning into
    a multiple.
    """
    fit = rigs.healthy().fit
    classical = CovarianceSampler(fit, view_indices=(0, 1), seed=0, widen=False)
    widened = CovarianceSampler(fit, view_indices=(0, 1), seed=0, widen=True)
    assert widened.inflation < 1.4
    ratio = widened.marginal_std() / classical.marginal_std()
    assert ratio.max() < 1.4


def test_widening_never_narrows_any_direction():
    """Clamping at one is what makes this safe to leave on by default.

    The sandwich is mildly optimistic at low view counts, so substituting it
    wholesale would shrink the intervals on a clean short capture on the
    strength of a noisy estimate. Widening only where there is evidence to
    widen cannot make a report overconfident.
    """
    for fit in (rigs.healthy().fit, rigs.with_correlated_noise().fit):
        classical, _ = joint_covariance(fit.covariance, (0, 1))
        widened_block, factor = fit.covariance.widened_intrinsic()
        rebuilt = with_intrinsic_block(
            classical, fit.covariance.n_intrinsic, widened_block
        )
        difference = np.linalg.eigvalsh(rebuilt - classical)
        scale = np.trace(classical) / classical.shape[0]
        assert difference.min() > -1e-8 * scale, factor


def test_widening_is_the_identity_when_the_block_is_unchanged():
    """`with_intrinsic_block` must reproduce the joint it was given."""
    fit = rigs.healthy().fit
    classical, _ = joint_covariance(fit.covariance, (0, 2))
    rebuilt = with_intrinsic_block(
        classical, fit.covariance.n_intrinsic, fit.covariance.intrinsic
    )
    assert np.allclose(rebuilt, classical, atol=1e-9 * np.abs(classical).max())


def test_the_rebuilt_joint_stays_a_covariance():
    """A joint whose blocks disagree is not sampleable, however honest it is."""
    fit = rigs.with_correlated_noise().fit
    classical, _ = joint_covariance(fit.covariance, (0, 1, 2))
    block, _ = fit.covariance.widened_intrinsic()
    rebuilt = with_intrinsic_block(classical, fit.covariance.n_intrinsic, block)
    assert np.allclose(rebuilt, rebuilt.T)
    smallest = np.linalg.eigvalsh(rebuilt).min()
    assert smallest > -1e-8 * np.trace(rebuilt) / rebuilt.shape[0]


def test_a_replacement_of_the_wrong_shape_is_refused():
    fit = rigs.healthy().fit
    classical, _ = joint_covariance(fit.covariance, (0,))
    with pytest.raises(ValidationError, match="intrinsic block is"):
        with_intrinsic_block(classical, fit.covariance.n_intrinsic, np.eye(2))


def test_correlated_noise_widens_the_sampled_intrinsics_severalfold():
    """The whole point of item 1b: the finding's factor now reaches the sampler."""
    fit = rigs.with_correlated_noise(length_px=200.0).fit
    classical = CovarianceSampler(fit, view_indices=(0,), seed=0, widen=False)
    widened = CovarianceSampler(fit, view_indices=(0,), seed=0, widen=True)
    assert widened.inflation > 3.0
    assert classical.inflation == 1.0
    p = fit.covariance.n_intrinsic
    ratio = widened.marginal_std()[:p] / classical.marginal_std()[:p]
    assert ratio.min() > 2.0


def test_too_few_views_leaves_the_covariance_alone():
    """Below ten clusters the sandwich is noise, so it must not touch anything."""
    from calibsense.core.session import CalibrationSession
    from calibsense.refit import RefitOptions, instrument
    from calibsense.synthetic import diverse_poses, synthesise

    capture = synthesise(
        rigs.WIDE_PINHOLE, rigs.BOARD, diverse_poses(rigs.BOARD, 6, seed=3),
        rigs.IMAGE_SIZE, noise_px=0.2, seed=5,
    )
    fit = instrument(CalibrationSession(observations=capture.observations), RefitOptions())
    assert not fit.covariance.robust.usable
    sampler = CovarianceSampler(fit, seed=0, widen=True)
    assert sampler.inflation == 1.0
