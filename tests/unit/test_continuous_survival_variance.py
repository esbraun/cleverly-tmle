"""The switched variance of the continuous-time TMLE (X6 part (a), plan rev 5, 4.1: T-a1 to T-a14).

The exact law is the two-cause ``PointLaw`` of ``tests/unit/test_continuous_survival.py``, whose
sample is the law.  At its true nuisances the predictable-variation covariance must equal the
population covariance of the influence curves: its diagonal is each target's ``P D^2`` and its
off-diagonal the cross moments.  The law's hazard jumps are large, so the squared term of the
multinomial variance, the survival just before each time and the cross-target block are each far
from zero there, and each mutation below moves the identity by far more than its tolerance.
"""

from __future__ import annotations

import warnings
from dataclasses import replace
from functools import cache
from typing import Any

import numpy as np
import pytest

from cleverly.exceptions import CapabilityError
from cleverly.inference import make_estimate
from cleverly.inference.cluster import influence_variance
from cleverly.inference.multiplier import simultaneous_bands
from cleverly.inference.results import ratio_contrast, smooth_contrast
from cleverly.inference.switched import SwitchedVariance, switched_record
from cleverly.survival import ContinuousSurvivalTMLE, CoxHazard, _targeting
from cleverly.survival import result as result_module
from cleverly.survival._targeting import (
    Problem,
    evaluate,
    predictable_covariance,
    restricted_weights,
)
from cleverly.survival.result import continuous_survival_support
from tests.unit.test_continuous_survival import _arrays, _problem, _sample

#: Relative tolerance of the exact identities: rounding only.
EXACT = 1e-10
#: A mutation must move some entry by more than this, relative to the largest entry.
MUTATION = 1e-6
TAU = 2.5


@cache
def _exact() -> tuple[Problem, np.ndarray]:
    """The exact-law problem with the three point targets and one restricted row per cause."""
    problem, dlam = _problem()
    grid = np.unique(_arrays()[2])
    rows = np.vstack([problem.target_weights, restricted_weights(grid, TAU)[None] / TAU])
    return replace(problem, target_weights=rows), dlam


def _moments(problem: Problem, dlam: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The predictable-variation matrix and the influence curves' population second moments."""
    evaluation = evaluate(problem, dlam)
    curves = evaluation.influence
    return predictable_covariance(problem, evaluation, dlam), curves.T @ curves / curves.shape[0]


def _relative_gap(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.max(np.abs(left - right)) / np.max(np.abs(right)))


# ------------------------------------------------------------------ T-a1, T-a12


def test_the_predictable_variance_is_the_population_second_moment_on_the_exact_law() -> None:
    """T-a1 and T-a12: every level, restricted row, difference and cross moment."""
    problem, dlam = _exact()
    pv, moments = _moments(problem, dlam)
    assert _relative_gap(pv, moments) < EXACT
    # The witnesses that the identity is not vacuous: the cross-target block and the cross-arm
    # block are both far from zero (the plug-in part is shared across arms).
    P = pv.shape[0]
    assert np.max(np.abs(pv - np.diag(np.diag(pv)))) > 0.1 * np.max(np.diag(pv))
    assert np.max(np.abs(pv[: P // 2, P // 2 :])) > 1e-3
    # The difference of the two arms' risks of cause 1 at the middle time adds both arms'
    # martingale terms (T-a5's identity), and a log ratio and an RMST read the cross block.
    evaluation = evaluate(problem, dlam)
    Q = problem.n_rows
    p0, p1 = 1, 2 * Q + 1
    for gradient in _gradients(evaluation.psi, p0, p1, Q):
        exact = float(np.mean((evaluation.influence @ gradient) ** 2))
        assert gradient @ pv @ gradient == pytest.approx(exact, rel=EXACT)


def _gradients(psi: np.ndarray, p0: int, p1: int, Q: int) -> list[np.ndarray]:
    """The difference, the log ratio and the RMST of arm 0 (both causes' restricted rows)."""
    P = psi.size
    difference = np.zeros(P)
    difference[[p1, p0]] = (1.0, -1.0)
    log_ratio = np.zeros(P)
    log_ratio[[p1, p0]] = (1.0 / psi[p1], -1.0 / psi[p0])
    rmst = np.zeros(P)
    rmst[[Q - 1, 2 * Q - 1]] = -TAU
    return [difference, log_ratio, rmst]


def test_zeroing_the_cross_target_block_breaks_the_ratio_and_the_rmst() -> None:
    """T-a12's mutation: the off-diagonal entries carry the log ratio and the RMST."""
    problem, dlam = _exact()
    evaluation = evaluate(problem, dlam)
    pv = np.diag(np.diag(predictable_covariance(problem, evaluation, dlam)))
    Q = problem.n_rows
    _, log_ratio, rmst = _gradients(evaluation.psi, 1, 2 * Q + 1, Q)
    for gradient in (log_ratio, rmst):
        exact = float(np.mean((evaluation.influence @ gradient) ** 2))
        assert abs(gradient @ pv @ gradient - exact) > MUTATION * exact


# ---------------------------------------------------------------- T-a2, T-a3, T-a5


def test_dropping_the_squared_multinomial_term_breaks_the_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """T-a2: the term -(sum_l c_l dLambda_l)^2 is large on the exact law's jumps."""

    def no_square(clever: np.ndarray, dlam: np.ndarray, rho: np.ndarray) -> np.ndarray:
        P = clever.shape[1]
        out = np.zeros((P, P))
        for l in range(clever.shape[0]):
            flat = (clever[l] * np.sqrt(rho * dlam[l])[None]).reshape(P, -1)
            out += flat @ flat.T
        return out

    monkeypatch.setattr(_targeting, "_multinomial_moment", no_square)
    assert _relative_gap(*_moments(*_exact())) > MUTATION


def test_the_survival_after_the_time_in_place_of_before_breaks_the_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """T-a3: S(u) in place of S(u-) in the at-risk probability."""
    monkeypatch.setattr(_targeting, "_before", lambda survival: survival)
    assert _relative_gap(*_moments(*_exact())) > MUTATION


def test_one_arms_martingale_term_alone_breaks_the_difference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """T-a5: the risk difference adds both arms' martingale terms."""
    problem, dlam = _exact()
    evaluation = evaluate(problem, dlam)
    original = _targeting._martingale_block

    def treated_only(*args: Any) -> np.ndarray:
        block = original(*args)
        return block if args[-1] == 1 else np.zeros_like(block)

    monkeypatch.setattr(_targeting, "_martingale_block", treated_only)
    pv = predictable_covariance(problem, evaluation, dlam)
    Q = problem.n_rows
    difference = _gradients(evaluation.psi, 1, 2 * Q + 1, Q)[0]
    exact = float(np.mean((evaluation.influence @ difference) ** 2))
    assert abs(difference @ pv @ difference - exact) > MUTATION * exact


# ------------------------------------------------------------------------- T-a4


def test_a_binding_floor_enters_squared() -> None:
    """T-a4: with the positivity floor binding, pi S_c w^2 is not w.

    The floored weight keeps the martingale structure, so the identity holds with the floored
    weight entering as ``pi S_c w^2``; reading it as ``w`` (the form that holds only where the
    floor does not bind) breaks it.
    """
    problem, dlam = _exact()
    product = 1.0 / problem.weight
    floor = float(np.quantile(product, 0.4))
    binding = product < floor
    assert binding.mean() > 0.2  # the witness: the floor binds on a large share of cells
    floored = replace(problem, weight=1.0 / np.maximum(product, floor))
    evaluation = evaluate(floored, dlam)
    curves = evaluation.influence
    moments = curves.T @ curves / curves.shape[0]
    reach = np.minimum(product / floor, 1.0)
    assert _relative_gap(predictable_covariance(floored, evaluation, dlam, reach), moments) < EXACT
    unfloored = predictable_covariance(floored, evaluation, dlam, None)
    assert _relative_gap(unfloored, moments) > MUTATION


# ------------------------------------------------------------------------- T-a6


def test_the_cross_term_is_zero_on_the_law_and_small_on_a_realized_sample() -> None:
    """T-a6: E[(F - psi) M_a] is omitted because it is zero at the truth."""
    problem, dlam = _exact()
    evaluation = evaluate(problem, dlam)
    rows = problem.target_weights.T
    J, Q = problem.n_causes, problem.n_rows

    def cross(problem: Problem, dlam: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        evaluation = evaluate(problem, dlam)
        plugins = np.zeros_like(evaluation.influence)
        for a in (0, 1):
            for j in range(J):
                block = slice((a * J + j) * Q, (a * J + j + 1) * Q)
                plugins[:, block] = evaluation.incidence[a, j] @ rows
        centred = plugins - plugins.mean(axis=0)
        product = centred * (evaluation.influence - centred)
        return product.mean(axis=0), product.std(axis=0, ddof=1) / np.sqrt(product.shape[0])

    exact, _ = cross(problem, dlam)
    assert np.max(np.abs(exact)) < 1e-13
    # A realized sample of 20,000 units drawn from the law (its rows uniformly), read at the
    # law's own nuisances.
    index = np.random.default_rng(20261114).integers(0, problem.n, 20_000)
    drawn = Problem(
        problem.arm[index],
        problem.at_risk[index],
        problem.events[:, index],
        problem.weight[:, index],
        problem.target_weights,
    )
    mean, se = cross(drawn, dlam[:, :, index])
    assert np.all(np.abs(mean) < 3.0 * se + 1e-15)
    assert np.max(np.abs(mean)) > 0.0
    assert evaluation.psi.size == mean.size


# ------------------------------------------------------------------------- T-a7


def _sparse_pair(ratio: float) -> tuple[dict[str, Any], dict[str, Any]]:
    """Two correlated sparse curves, as switched and as influence-curve estimates."""
    rng = np.random.default_rng(3)
    n = 400
    base = np.zeros((n, 2))
    hit = rng.choice(n, 12, replace=False)
    base[hit] = rng.normal(size=(12, 2)) * np.array([1.0, 0.7]) + np.array([3.0, 2.0])
    base[:, 1] += 0.5 * base[:, 0]
    base -= base.mean(axis=0)
    switched, plain = {}, {}
    for k, name in enumerate(("a", "b")):
        curve = base[:, k]
        pv = ratio**2 * influence_variance(curve, None)
        switched[name] = make_estimate(
            name,
            0.1,
            curve,
            n=n,
            rule_widened=False,
            covariance_rule="switched",
            predictable_variance=pv,
        )
        plain[name] = make_estimate(name, 0.1, curve, n=n, rule_widened=False)
    return switched, plain


def test_the_band_reads_the_curves_correlation_and_the_reported_widths() -> None:
    """T-a7: a band of switched estimates is the influence-curve band widened by the SE ratio."""
    switched, plain = _sparse_pair(1.2)
    assert all(e.variance_source == "predictable_variation" for e in switched.values())
    wide = simultaneous_bands(switched, random_state=0, band_scale="identity")
    narrow = simultaneous_bands(plain, random_state=0, band_scale="identity")
    assert wide.critical_value == narrow.critical_value
    for name in switched:
        ratio = (wide.bands[name][1] - wide.bands[name][0]) / (
            narrow.bands[name][1] - narrow.bands[name][0]
        )
        assert ratio == pytest.approx(1.2, rel=1e-12)


# ------------------------------------------------------------------- fits


def _engine(**extra: Any) -> ContinuousSurvivalTMLE:
    return ContinuousSurvivalTMLE(
        [0.5, 1.0],
        restricted_horizons=(1.0,),
        cross_fit=False,
        hazard_library=[CoxHazard("main")],
        censoring_library=[CoxHazard("main")],
        random_state=0,
        **extra,
    )


@cache
def _fits(seed: int = 0, n: int = 300) -> tuple[Any, Any]:
    frame = _sample(n, seed=seed)
    columns = {"time": "time", "event": "event", "treatment": "Z", "baseline": ["X3"]}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return (
            _engine().fit(frame, **columns),
            _engine(variance="influence_curve").fit(frame, **columns),
        )


# ------------------------------------------------------------------------- T-a8


def test_the_influence_curve_variance_is_the_shipped_interval() -> None:
    """T-a8: ``variance="influence_curve"`` is the shipped estimate, bit for bit.

    The variance does not enter the walk, so both fits hold the same values and curves; the
    influence-curve fit builds each estimate exactly as before part (a), from its curve alone.
    """
    switched, plain = _fits()
    assert plain.predictable is None and switched.predictable is not None
    assert list(plain) == list(switched)
    for name, estimate in plain.items():
        other = switched[name]
        assert estimate.psi == other.psi
        assert np.array_equal(estimate.influence_curve, other.influence_curve)
        assert estimate.switched is None and estimate.covariance_rule == "centered"
        assert estimate.variance_source == "influence_curve"
        assert estimate.variance == influence_variance(estimate.influence_curve, None)
        assert other.switched is not None
        assert other.switched.influence_variance == estimate.variance
        rebuilt = replace(
            other, covariance_rule="centered", variance=estimate.variance, switched=None
        )
        assert rebuilt.ci == estimate.ci
    assert all("variance away from the default" not in e.support.reason for e in switched.values())
    assert all("variance" in e.support.reason for e in plain.values())


# ------------------------------------------------------------------------- T-a9


@pytest.mark.parametrize("scale", ["level", "difference"])
def test_the_relative_sd_floor_labels_risks_and_differences(
    monkeypatch: pytest.MonkeyPatch, scale: str
) -> None:
    """T-a9: just below and at the floor supported, just above experimental with the reason."""
    monkeypatch.setattr(result_module, "MEASURED_REL_SD", 0.2)

    def label(value: float, *, scale: str = scale, share: float = 0.6, **extra: Any) -> Any:
        return continuous_survival_support(
            scale=scale,
            event_share=share,
            n=1_000,
            causes=2,
            cross_fitted=True,
            default_nuisances=True,
            settings=result_module.MEASURED_DEFAULTS,
            variance_rel_sd=value,
            **extra,
        )

    assert label(0.199).label == "supported"
    assert label(0.2).label == "supported"
    above = label(0.201)
    assert above.label == "experimental"
    assert "0.201, above 0.200" in above.reason
    assert "continuous-survival-variance" in above.reason
    assert label(float("nan")).label == "experimental"  # an undefined spread is not below it
    # Ratios and restricted means keep their own labels (review 4, m9).
    assert "sparse" not in label(0.5, scale="ratio").reason
    assert "sparse" not in label(0.5, restricted=True, restricted_horizons=1).reason
    # Below the share floor the early floor applies, and it is pending.
    assert "sparse" not in label(0.5, share=0.3).reason
    monkeypatch.setattr(result_module, "MEASURED_REL_SD_EARLY", 0.3)
    assert "sparse" in label(0.5, share=0.3).reason


def test_the_pending_floor_labels_nothing_and_the_engine_passes_each_curves_spread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """T-a9 on a fit: the engine hands each estimate's own relative SD to the floor."""
    assert result_module.MEASURED_REL_SD is None  # pending continuous-survival-variance
    floor = 0.3
    monkeypatch.setattr(result_module, "MEASURED_REL_SD", floor)
    monkeypatch.setattr(result_module, "MEASURED_REL_SD_EARLY", floor)
    frame = _sample(300, seed=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        refit = _engine().fit(frame, time="time", event="event", treatment="Z", baseline=["X3"])
    seen = {"above": 0, "below": 0}
    for name, estimate in refit.items():
        sparse = "sparse" in estimate.support.reason
        risk = name.startswith(("cif_regimen", "ate_regimen"))
        relative_sd = estimate.switched.relative_sd
        assert sparse == (risk and relative_sd > floor), name
        if risk:
            seen["above" if relative_sd > floor else "below"] += 1
    assert seen["above"] and seen["below"]  # both sides of the floor are exercised


# ------------------------------------------------------------------------ T-a10


def _check_switch(record_of: Any) -> None:
    """S equals max(IC, PV) above the threshold and IC below, with PV above IC on both."""
    rng = np.random.default_rng(5)
    dense = rng.normal(size=4_000)
    sparse = np.zeros(4_000)
    sparse[:15] = rng.normal(5.0, 1.0, size=15)
    sparse -= sparse.mean()
    for curve, expect_switch in ((sparse, True), (dense, False)):
        ic = influence_variance(curve, None)
        record = record_of(curve, ic, 1.5 * ic)
        assert (record.relative_sd > 0.125) == expect_switch  # the witness of each side
        assert record.variance == pytest.approx(1.5 * ic if expect_switch else ic)


def test_the_switch_reads_the_relative_sd_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    """T-a10, with both mutations: a switch always on and always off each fail one side."""
    _check_switch(switched_record)
    for always in (True, False):
        with monkeypatch.context() as patch:
            patch.setattr(SwitchedVariance, "switched", property(lambda self, v=always: v))
            with pytest.raises(AssertionError):
                _check_switch(switched_record)


def test_a_fit_reports_both_sources() -> None:
    """The engine's switch fires on one estimate and not another of the same fit."""
    switched, _ = _fits(seed=1)
    sources = {estimate.variance_source for estimate in switched.values()}
    assert sources == {"influence_curve", "predictable_variation"}
    for estimate in switched.values():
        record = estimate.switched
        assert estimate.variance == record.variance
        assert estimate.std_error == pytest.approx(np.sqrt(record.variance), rel=1e-15)
        row = estimate.to_dict()
        assert row["variance_source"] == estimate.variance_source
        assert row["variance_rel_sd"] == record.relative_sd


# ------------------------------------------------------------------------ T-a11


def test_the_switched_standard_error_is_never_below_the_influence_curves() -> None:
    """T-a11: over 50 seeded small fits, S's SE >= IC's for every target."""
    for seed in range(50):
        frame = _sample(300, seed=100 + seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = _engine().fit(frame, time="time", event="event", treatment="Z", baseline=["X3"])
        for estimate in fit.values():
            assert estimate.variance >= estimate.switched.influence_variance


# ------------------------------------------------------------------------ T-a13


def _rd_names(fit: Any) -> tuple[str, str, str, str]:
    return (
        "cif_regimen[1, 1 @ t=1]",
        "cif_regimen[0, 1 @ t=1]",
        "ate_regimen[1 vs 0, 1 @ t=1]",
        "rr_regimen[1 vs 0, 1 @ t=1]",
    )


def _check_derived(fit: Any) -> None:
    treated, control, difference, ratio = _rd_names(fit)
    derived = fit.contrast(lambda x: x[0] - x[1], [treated, control], scale="difference")
    stored = fit[difference]
    assert derived.psi == pytest.approx(stored.psi, rel=1e-12)
    assert derived.std_error == pytest.approx(stored.std_error, rel=1e-8)
    assert derived.ci == pytest.approx(stored.ci, rel=1e-8)
    assert derived.variance_source == stored.variance_source
    rr = fit.ratio(treated, control)
    assert rr.std_error == pytest.approx(fit[ratio].std_error, rel=1e-8)
    assert rr.ci == pytest.approx(fit[ratio].ci, rel=1e-8)
    forward = fit.rmtl("1", 1.0, "1", versus="0")
    reverse = fit.rmtl("0", 1.0, "1", versus="1")
    assert reverse.psi == -forward.psi
    assert reverse.variance == forward.variance
    assert reverse.variance_source == forward.variance_source
    rmst = fit.rmst("0", 1.0, versus="1")
    assert rmst.variance == fit.rmst("1", 1.0, versus="0").variance


def test_contrast_ratio_and_a_reversed_difference_report_the_stored_switched_variance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """T-a13, with the IC-only mutation of the result methods."""
    fit, _ = _fits()
    _, _, difference, ratio = _rd_names(fit)
    record = fit[difference].switched
    # The witness: the difference switched, with its predictable variation 10% above the IC's.
    assert fit[difference].variance_source == "predictable_variation"
    assert record.predictable_variance >= 1.1 * record.influence_variance
    assert fit[ratio].variance_source == "predictable_variation"
    _check_derived(fit)
    monkeypatch.setattr(
        type(fit),
        "_derive",
        lambda self, names, run: run(
            {key: result_module._influence_view(self.estimates[key]) for key in names}
        ),
    )
    with pytest.raises(AssertionError):
        _check_derived(fit)


# ------------------------------------------------------------------------ T-a14


def test_a_generic_contrast_of_switched_estimates_refuses() -> None:
    """T-a14: without the fit's covariance a contrast would report the IC variance."""
    fit, _ = _fits()
    treated, control, _, _ = _rd_names(fit)
    estimates = dict(fit.estimates)
    with pytest.raises(CapabilityError, match=r"ContinuousSurvivalResult\.contrast"):
        smooth_contrast(estimates, lambda x: x[0] - x[1], [treated, control], n=fit.n)
    with pytest.raises(CapabilityError, match=r"ContinuousSurvivalResult\.contrast"):
        ratio_contrast(estimates, treated, control, n=fit.n, name="rr")
