"""The isotonic projection of survival curves (X14 E9, E9b, M10, M13).

``isotonic=True`` projects each regimen's and cause's levels onto a non-decreasing curve by
unit-weight isotonic regression over the reported horizons (Westling, van der Laan and Carone
2020, Section 2.1).  The influence curves, and so the standard errors, are the fit's own.  Each
curve's band limits are projected the same way, and a contrast keeps its band.

The witness sample is ``make_point_survival(n=400, seed=36, hazard_scale={3: 0.05})`` with
main-term GLM learners: its iterated-mean curves decrease by about 3e-4, so the projection
moves them.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly.datasets import make_point_survival
from cleverly.inference.multiplier import SimultaneousBands
from cleverly.longitudinal import LTMLE
from cleverly.longitudinal import estimator as estimator_module
from cleverly.longitudinal.hazard import project_curves

COLUMNS = {
    "outcome": [f"Y{k}" for k in range(1, 6)],
    "treatment": "A",
    "baseline": ["W1", "W2"],
    "censoring": [f"C{k}" for k in range(1, 6)],
}
REGIMENS = {"a1": 1, "a0": 0}


def _learners() -> dict[str, Any]:
    return {
        "outcome_learner": LogisticRegression(max_iter=1000),
        "pseudo_learner": LinearRegression(),
        "treatment_learner": LogisticRegression(max_iter=1000),
        "censoring_learner": LogisticRegression(max_iter=1000),
    }


def _frame(seed: int) -> Any:
    frame, _ = make_point_survival(n=400, seed=seed, layout="wide", hazard_scale={3: 0.05})
    return frame


def _fit(seed: int, **settings: Any) -> Any:
    learners = _learners()
    if settings.get("survival_model") == "hazard":
        learners.pop("pseudo_learner")
    return LTMLE(REGIMENS, n_folds=1, random_state=0, **learners, **settings).fit(
        _frame(seed), **COLUMNS
    )


@pytest.fixture(scope="module")
def pair() -> tuple[Any, Any]:
    return _fit(36), _fit(36, isotonic=True)


def _curve(result: Any, label: str) -> np.ndarray:
    return np.array([result[f"risk_regimen[{label} @ t={t}]"].psi for t in range(1, 6)])


def _pava(values: np.ndarray) -> np.ndarray:
    """Pool adjacent violators with unit weights, written out."""
    blocks: list[list[float]] = []
    for value in values:
        blocks.append([float(value), 1.0])
        while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0]:
            mean, size = blocks.pop()
            previous = blocks[-1]
            previous[0] = (previous[0] * previous[1] + mean * size) / (previous[1] + size)
            previous[1] += size
    return np.concatenate([[mean] * int(size) for mean, size in blocks])


def _lmtp_isotonic(survival: np.ndarray) -> np.ndarray:
    """``lmtp``'s ``isotonic_projection``: ``gpava`` with unit weights on ``1 - estimate``.

    Transcribed over all grid times.  ``gpava`` fits a non-decreasing step function, here to
    the risk, and the projected survival is one minus that fit.
    """
    return 1.0 - _pava(1.0 - survival)


# ------------------------------------------------------------------------------- E9


def test_e9_the_witness_curve_decreases_and_its_projection_does_not(
    pair: tuple[Any, Any],
) -> None:
    plain, projected = pair
    assert min(np.diff(_curve(plain, "a1")).min(), np.diff(_curve(plain, "a0")).min()) < -1e-6
    for label in REGIMENS:
        assert np.all(np.diff(_curve(projected, label)) >= 0.0)
    assert projected.isotonic_moves


def test_e9_the_projection_is_pava_and_lmtps_transcription(pair: tuple[Any, Any]) -> None:
    plain, projected = pair
    for label in REGIMENS:
        before = _curve(plain, label)
        np.testing.assert_allclose(_curve(projected, label), _pava(before), rtol=0, atol=1e-15)
        np.testing.assert_allclose(project_curves(before[None, :])[0], _pava(before), atol=1e-15)
        np.testing.assert_allclose(
            1.0 - _lmtp_isotonic(1.0 - before), _curve(projected, label), atol=1e-15
        )


def _identical_influence(plain: Any, projected: Any) -> bool:
    return all(
        np.array_equal(plain[name].influence_curve, projected[name].influence_curve)
        and plain[name].plugin_std_error == projected[name].plugin_std_error
        for name in plain.estimates
    )


def test_e9_the_influence_curves_and_standard_errors_do_not_move(pair: tuple[Any, Any]) -> None:
    assert _identical_influence(*pair)


def test_e9_an_unmoved_level_is_bit_identical(pair: tuple[Any, Any]) -> None:
    plain, projected = pair
    moved = {(label, horizon) for label, _, horizon, _, _ in projected.isotonic_moves}
    for label in REGIMENS:
        for t in range(1, 6):
            name = f"risk_regimen[{label} @ t={t}]"
            if (label, t) not in moved:
                assert projected[name].psi == plain[name].psi
                assert projected[name].plugin_interval == plain[name].plugin_interval


def test_e9_a_contrast_is_the_difference_of_projected_levels(pair: tuple[Any, Any]) -> None:
    _, projected = pair
    for t in range(1, 6):
        # The reference regimen is the first, a1, so the contrast is a0 minus a1.
        contrast = projected[f"ate_regimen[a0 vs a1 @ t={t}]"].psi
        levels = (
            projected[f"risk_regimen[a0 @ t={t}]"].psi - projected[f"risk_regimen[a1 @ t={t}]"].psi
        )
        assert contrast == pytest.approx(levels, abs=1e-15)


def test_e9_curve_band_limits_are_projected_and_contrast_bands_kept(
    pair: tuple[Any, Any],
) -> None:
    plain, projected = pair
    assert plain.simultaneous.critical_value == projected.simultaneous.critical_value
    bands = plain.simultaneous.bands
    for label in REGIMENS:
        # A parameter without inference is left out of the band, and so of its projection.
        names = [f"risk_regimen[{label} @ t={t}]" for t in range(1, 6)]
        names = [name for name in names if name in bands]
        for side in (0, 1):
            before = np.array([plain.simultaneous.bands[name][side] for name in names])
            after = np.array([projected.simultaneous.bands[name][side] for name in names])
            np.testing.assert_allclose(after, _pava(before), atol=1e-15)
    contrasts = [name for name in bands if name.startswith("ate_regimen")]
    assert contrasts
    for name in contrasts:
        assert projected.simultaneous.bands[name] == bands[name]


def test_e9_a_monotone_curve_is_left_bit_for_bit() -> None:
    """Seed 0 has no violation, and the projection is then the identity."""
    plain, projected = _fit(0), _fit(0, isotonic=True)
    assert projected.isotonic_moves == ()
    for name in plain.estimates:
        assert projected[name].psi == plain[name].psi
        assert projected[name].plugin_interval == plain[name].plugin_interval


@pytest.mark.parametrize(
    "settings",
    [
        {"survival_model": "hazard", "targeting": "one_step"},
        {"horizons": [3]},
    ],
    ids=["one-step hazard fit", "one horizon"],
)
def test_e9_the_projection_is_the_identity(settings: dict[str, Any]) -> None:
    """A one-step curve is monotone by construction, and one horizon has nothing to order."""
    plain, projected = _fit(36, **settings), _fit(36, isotonic=True, **settings)
    assert projected.isotonic_moves == ()
    for name in plain.estimates:
        assert projected[name].psi == plain[name].psi


# ------------------------------------------------------------------------------ M10


def test_m10_an_influence_curve_moved_with_its_level_fails_e9(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = estimator_module._project_fits

    def leaking(fits: Any) -> Any:
        projected, moves = original(fits)
        from dataclasses import replace

        for key, fit in projected.items():
            shift = fit.psi_scaled - fits[key].psi_scaled
            if shift:
                projected[key] = replace(
                    fit, influence_curve_scaled=fit.influence_curve_scaled - shift
                )
        return projected, moves

    monkeypatch.setattr(estimator_module, "_project_fits", leaking)
    assert not _identical_influence(_fit(36), _fit(36, isotonic=True))


# ------------------------------------------------------------------------------ E9b


def _bands(limits: dict[str, tuple[float, float]]) -> SimultaneousBands:
    return SimultaneousBands(
        critical_value=2.5, bands=limits, alpha=0.05, n_replicates=1000, kind="gaussian"
    )


def _band_violations(project: Any, draws: int = 400) -> int:
    """Count draws where the projected band breaks Theorem 1 (iv) or (v).

    Each draw takes a monotone truth, an estimate around it, and a band of random half
    widths.  When the unprojected band covers the truth, the projected band must cover it;
    it must contain the projected estimate; and it must be no wider on average or at its
    maximum.
    """
    rng = np.random.default_rng(20261012)
    violations = 0
    horizons = 6
    index = {f"risk_regimen[a1 @ t={t}]": ("a1", None, t) for t in range(1, horizons + 1)}
    for _ in range(draws):
        truth = np.cumsum(rng.uniform(0.0, 0.05, horizons))
        width = rng.uniform(0.005, 0.05, horizons)
        estimate = truth + rng.normal(0.0, 1.0, horizons) * width / 2.0
        lower, upper = estimate - width, estimate + width
        names = list(index)
        bands = _bands({name: (lower[i], upper[i]) for i, name in enumerate(names)})
        projected = project(bands, index)
        low = np.array([projected.bands[name][0] for name in names])
        high = np.array([projected.bands[name][1] for name in names])
        centre = _pava(estimate)
        covered = np.all((lower <= truth) & (truth <= upper))
        if (
            (covered and not np.all((low <= truth) & (truth <= high)))
            or not np.all((low <= centre + 1e-15) & (centre <= high + 1e-15))
            or np.mean(high - low) > np.mean(upper - lower) + 1e-15
            or np.max(high - low) > np.max(upper - lower) + 1e-15
        ):
            violations += 1
    return violations


def test_e9b_the_projected_band_keeps_coverage_and_is_no_wider() -> None:
    assert _band_violations(estimator_module._project_bands) == 0


def test_m13_a_band_centred_on_the_projected_estimate_fails_e9b() -> None:
    def centred(bands: SimultaneousBands, index: Any) -> SimultaneousBands:
        names = sorted(index, key=lambda name: index[name][2])
        lower = np.array([bands.bands[name][0] for name in names])
        upper = np.array([bands.bands[name][1] for name in names])
        centre = project_curves(((lower + upper) / 2.0)[None, :])[0]
        half = (upper - lower) / 2.0
        return _bands(
            {name: (centre[i] - half[i], centre[i] + half[i]) for i, name in enumerate(names)}
        )

    assert _band_violations(centred) > 0
