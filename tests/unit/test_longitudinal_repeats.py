"""Repeated cross-fitting of a longitudinal fit: the median over complete fold draws.

``repeats=R`` on :class:`LTMLE`, :class:`SDR` and :class:`IPSI` runs the whole single-draw fit
at ``R`` independent fold assignments and combines the draws by the rule of Chernozhukov et
al. (2018, Definition 3.3 and equation 3.14): the median point on each estimate's inference
scale, and the variance ``median_r(var_r + (t_r - t_med)^2)``.  What is checked here:

* ``repeats=1`` equals the golden record captured before the refactor (4.1);
* every draw is a complete single-draw fit at its own split and seed (4.1);
* every combination rule, each with a nonzero witness that a wrong rule would miss (4.2);
* every refusal, and a registry that maps each longitudinal surface to its witness (4.4).

Every fixture fits saturated cell means, so a draw differs from another draw only through its
fold assignment.  Floats are compared at ``rtol=1e-12`` with the named floor :data:`ATOL`;
seeds, digests and text are compared by equality.
"""

from __future__ import annotations

import dataclasses
import importlib
import inspect
import typing
import warnings
from dataclasses import replace
from functools import cache
from typing import Any

import numpy as np
import pandas as pd
import pytest

import cleverly.longitudinal.estimator as estimator_module
from cleverly import CausalStudy, LongitudinalTreatment, RegimeContrast
from cleverly._refusal import RefusalReason
from cleverly.assessment import ASSESSMENT_CAPABILITIES, SENSITIVITY_ROUTES
from cleverly.estimators.serialize import dumps, loads
from cleverly.exceptions import CapabilityError, MethodConfigurationError
from cleverly.inference.cluster import influence_variance
from cleverly.inference.delta import Transform
from cleverly.inference.influence import _median_adjusted_variance, make_estimate
from cleverly.inference.multiplier import median_bands, simultaneous_bands
from cleverly.inference.results import estimate_covariance, median_covariance, smooth_contrast
from cleverly.learners.crossfit import CrossFitPlan, check_integrity, random_partition
from cleverly.longitudinal import IPSI, LTMLE, SDR, LongitudinalData, LongitudinalResult
from cleverly.provenance import fold_fingerprint
from cleverly.study import LongPanel
from tests.unit import _longitudinal_repeats_support as support
from tests.unit._longitudinal_repeats_golden import GOLDEN

#: The relative tolerance of every float comparison.
RTOL = 1e-12
#: The absolute floor of every float comparison, for values at or near zero.
ATOL = 1e-13

LEVEL_A = "ey_regimen[treat_if_l2]"
LEVEL_B = "ey_regimen[always]"
REF = "ey_regimen[never]"
PAIRED = "ate_regimen[treat_if_l2 vs never]"


def _close(left: Any, right: Any) -> None:
    np.testing.assert_allclose(left, right, rtol=RTOL, atol=ATOL)


def _quiet(build: Any, *args: Any, **kwargs: Any) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return build(*args, **kwargs)


@cache
def _rows(design: str) -> pd.DataFrame:
    return support.DESIGNS[design][0]()


@cache
def _fit(design: str, **overrides: Any) -> Any:
    """One fit of a support design, cached: every test reads it and none mutates it."""
    return _quiet(support.DESIGNS[design][1], _rows(design), **overrides)


def _draw_psi(result: Any, name: str) -> np.ndarray:
    return np.array([draw.estimates[name].psi for draw in result.repeats])


def _central(values: np.ndarray) -> float:
    """The median by hand: the middle value, or the mean of the two middle values."""
    ordered = np.sort(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[middle])
    return float((ordered[middle - 1] + ordered[middle]) / 2.0)


# ================================================================ 4.1 identity and draws


@pytest.mark.parametrize("design", sorted(support.GOLDEN))
def test_one_repeat_is_bit_for_bit_today(design: str) -> None:
    """``repeats=1`` reproduces the record captured at ``300df99`` before the refactor."""
    result, warned = support.fit_recording(design)
    record = GOLDEN[design]
    numbers = support.numbers(result)
    assert sorted(numbers) == sorted(record["numbers"])
    for key, value in record["numbers"].items():
        np.testing.assert_allclose(numbers[key], value, rtol=RTOL, atol=ATOL, err_msg=key)
    assert support.texts(result, warned) == record["texts"]
    assert result.n_repeats == 1


def _assert_draw_equals_fit(draw: Any, single: Any) -> None:
    np.testing.assert_array_equal(draw.folds.assignment, single.folds.assignment)
    assert sorted(draw.estimates) == sorted(single.estimates)
    for name, estimate in single.estimates.items():
        _close(draw.estimates[name].psi, estimate.psi)
        _close(draw.estimates[name].variance, estimate.variance)
        _close(draw.estimates[name].influence_curve, estimate.influence_curve)
        assert draw.estimates[name].inference == estimate.inference
    _assert_mechanism_equal(draw.mechanism, single.mechanism)
    assert len(draw.fold_mechanisms) == len(single.fold_mechanisms)
    for left, right in zip(draw.fold_mechanisms, single.fold_mechanisms, strict=True):
        _assert_mechanism_equal(left.mechanism, right.mechanism)
    assert sorted(draw.fits) == sorted(single.fits)
    for key, fit in single.fits.items():
        _close(draw.fits[key].psi_scaled, fit.psi_scaled)
        _close(draw.fits[key].influence_curve_scaled, fit.influence_curve_scaled)
    assert len(draw.msm_fits) == len(single.msm_fits)


def _assert_mechanism_equal(left: Any, right: Any) -> None:
    for part in ("treatment", "censoring"):
        for mine, theirs in zip(getattr(left, part), getattr(right, part), strict=True):
            assert sorted(mine) == sorted(theirs)
            for label in theirs:
                _close(mine[label], theirs[label])
    for part in ("treatment_observed", "censoring_observed"):
        for mine, theirs in zip(getattr(left, part), getattr(right, part), strict=True):
            _close(mine, theirs)


@pytest.mark.parametrize("design", sorted(support.DESIGNS))
def test_each_draw_reproduces_a_single_draw_fit(design: str) -> None:
    """Draw ``r`` of a ``repeats=3`` fit is the ``repeats=1`` fit at ``random_state=seed_r``.

    An integer ``random_state`` reaches the learners at one draw too, so both sides hand the
    learners ``seed_r``.  The ``settings`` design moves every inference and bookkeeping
    keyword off its default, and each other design covers the keywords of its data.
    """
    result = _fit(design, repeats=3)
    assert result.n_repeats == 3
    for draw in result.repeats:
        single = _fit(design, random_state=draw.seed)
        _assert_draw_equals_fit(draw, single)
        assert single.scaler == result.scaler
        assert single.fitted_method == result.fitted_method


def test_every_draw_gets_its_own_split_and_seed() -> None:
    result = _fit("ltmle", repeats=3)
    seeds = [draw.seed for draw in result.repeats]
    planned = CrossFitPlan(n_folds=5, learner_folds=2, random_state=11, repeats=3).seeds()
    assert seeds == list(planned)
    assert len(set(seeds)) == 3
    assignments = [tuple(draw.folds.assignment) for draw in result.repeats]
    assert len(set(assignments)) == 3
    for draw in result.repeats:
        expected = random_partition(result.n, 5, cluster=None, seed=draw.seed)
        np.testing.assert_array_equal(draw.folds.assignment, expected.assignment)


def _record_learner_seeds(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    seen: list[Any] = []
    original = estimator_module.resolve_learner

    def recording(*args: Any, **kwargs: Any) -> Any:
        seen.append(kwargs.get("random_state"))
        return original(*args, **kwargs)

    monkeypatch.setattr(estimator_module, "resolve_learner", recording)
    return seen


def test_none_seed_draws_fresh_concrete_seeds(monkeypatch: pytest.MonkeyPatch) -> None:
    """Under ``random_state=None`` every draw gets a recorded concrete seed (review m11)."""
    seen = _record_learner_seeds(monkeypatch)
    rows = _rows("ltmle")
    result = _quiet(support.ltmle, rows, repeats=3, random_state=None)
    seeds = [draw.seed for draw in result.repeats]
    assert all(isinstance(seed, int) for seed in seeds)
    assert len(set(seeds)) == 3
    for draw in result.repeats:
        expected = random_partition(result.n, 5, cluster=None, seed=draw.seed)
        np.testing.assert_array_equal(draw.folds.assignment, expected.assignment)
    # Four resolved slots per draw: outcome, pseudo, treatment, censoring.
    assert seen == [seed for seed in seeds for _ in range(4)]
    # One multiplier seed for the fit, used by every draw's critical value.
    seed = result.config.multiplier_seed
    assert isinstance(seed, int)
    assert f"multiplier seed: {seed} (drawn, random_state=None)" in result.config.describe(
        contrast=True
    )
    rebuilt = median_bands(
        [draw.estimates for draw in result.repeats],
        {name: result[name] for name in result.simultaneous.bands},
        alpha=result.simultaneous.alpha,
        n_replicates=result.simultaneous.n_replicates,
        kind=result.simultaneous.kind,
        random_state=seed,
    )
    assert rebuilt.draw_critical_values == result.simultaneous.draw_critical_values
    # At one draw the learners keep ``None``, unchanged.
    seen.clear()
    single = _quiet(support.ltmle, rows, random_state=None)
    assert seen == [None] * 4
    assert single.config.multiplier_seed is None


@pytest.mark.parametrize("design", ["ltmle", "sdr", "ipsi"])
def test_every_draw_resolves_learners_at_its_seed(
    design: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The four learner slots, the pseudo learner among them, see ``seed_r`` in draw ``r``."""
    seen = _record_learner_seeds(monkeypatch)
    result = _quiet(support.DESIGNS[design][1], _rows(design), repeats=3)
    seeds = [draw.seed for draw in result.repeats]
    assert seen == [seed for seed in seeds for _ in range(4)]
    assert 11 not in seeds


def test_all_draws_preflight_before_any_learner(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two rows keep ``A1 = 1``: draws 1 and 2 hold them apart, draw 3 puts them in one fold.

    Both rows follow ``always`` to an observed outcome, so draws 1 and 2 pass every preflight,
    and draw 3 leaves one training complement with no first-node arm 1.
    """
    rows = _rows("ltmle").copy()
    kept = np.flatnonzero(
        (rows["A1"].to_numpy() == 1) & (rows["A2"].to_numpy() == 1) & rows["Y"].notna().to_numpy()
    )[:2]
    rows.loc[(rows["A1"] == 1) & ~rows.index.isin(rows.index[kept]), "A1"] = 0
    assert int((rows["A1"] == 1).sum()) == 2
    seeds = CrossFitPlan(n_folds=5, learner_folds=2, random_state=1, repeats=3).seeds()
    together = [
        bool(len(set(random_partition(len(rows), 5, seed=seed).assignment[kept])) == 1)
        for seed in seeds
    ]
    assert together == [False, False, True]

    def never(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("a mechanism was fitted before every draw was preflighted")

    monkeypatch.setattr(estimator_module, "fit_mechanism", never)
    with pytest.raises(Exception, match=r"^draw 3 of 3: ") as raised:
        _quiet(support.ltmle, rows, repeats=3, random_state=1)
    assert not isinstance(raised.value, AssertionError)
    assert "treatment node 'A1' is missing level(s) [1.0] in training fold" in str(raised.value)


def test_grouped_draws_keep_clusters_whole() -> None:
    result = _fit("clustered", repeats=3)
    cluster = result.data.cluster
    assert cluster is not None
    assignments = []
    for draw in result.repeats:
        check_integrity(draw.folds, cluster=cluster)
        for code in np.unique(cluster):
            assert len(set(draw.folds.assignment[cluster == code])) == 1
        assignments.append(tuple(draw.folds.assignment))
    assert len(set(assignments)) == 3


def test_provenance_covers_every_draw() -> None:
    result = _fit("ltmle", repeats=3)
    assert result.provenance.fold_fingerprint == result.split_plan.fingerprint
    assert result.provenance.fold_fingerprint == fold_fingerprint(
        draw.folds.assignment for draw in result.repeats
    )
    changed = [draw.folds.assignment for draw in result.repeats]
    changed[2] = changed[2][::-1]
    assert fold_fingerprint(changed) != result.provenance.fold_fingerprint
    assert result.split_plan.n_repeats == 3
    single = _fit("ltmle")
    assert single.provenance.fold_fingerprint == single.split_plan.fingerprint


def test_config_records_repeats_and_the_multiplier_seed() -> None:
    result = _fit("ltmle", repeats=3)
    assert result.config.repeats == 3
    assert result.config.multiplier_seed == 11
    assert "cross-fitting: 5 fold(s), median over 3 draws" in result.config.describe(contrast=True)
    single = _fit("ltmle")
    assert single.config.repeats == 1
    assert not any("median over" in line for line in single.config.describe(contrast=True))


def test_read_through_fields_are_draw_zero() -> None:
    result = _fit("sdr", repeats=3)
    first = result.repeats[0]
    for field in (
        "fits",
        "mechanism",
        "folds",
        "msm_fits",
        "fold_mechanisms",
        "replay_recipe",
        "isotonic_moves",
        "initial_hazards",
    ):
        assert getattr(result, field) is getattr(first, field), field


def test_converged_reads_every_draw() -> None:
    """``max_iter=1`` at ``tol=1e-6``: draw 0 converges and draw 2 does not."""
    result = _fit("ltmle", repeats=3, max_iter=1, tol=1e-6, random_state=3)
    per_draw = [all(fit.converged for fit in draw.fits.values()) for draw in result.repeats]
    assert per_draw == [True, True, False]
    assert not result.converged
    # The stagewise report's rows are draw 1's, and its per-draw field reads every draw.
    stagewise = result.diagnostics.support()
    assert all(stagewise.to_frame()["converged"])
    assert stagewise.draw_converged == (True, True, False)
    assert "in at least one draw" in result.summary()


# ================================================================ 4.2 aggregation


@pytest.mark.parametrize("design", ["ltmle", "sdr", "ipsi", "survival", "competing", "repeated"])
def test_point_is_the_median_of_draws(design: str) -> None:
    result = _fit(design, repeats=3)
    for name, estimate in result.estimates.items():
        values = np.array([draw.estimates[name].inference_value for draw in result.repeats])
        _close(estimate.inference_value, _central(values))


def _hand_variance(result: Any, name: str) -> float:
    parts = [draw.estimates[name] for draw in result.repeats]
    values = np.array([part.inference_value for part in parts])
    centre = np.median(values)
    return float(
        np.median(
            [part.variance + (v - centre) ** 2 for part, v in zip(parts, values, strict=True)]
        )
    )


def test_variance_is_median_within_plus_displacement() -> None:
    result = _quiet(support.ltmle, support.frame(n=500), repeats=3, n_folds=2)
    for name, estimate in result.estimates.items():
        _close(estimate.variance, _hand_variance(result, name))
        within = np.median([draw.estimates[name].variance for draw in result.repeats])
        assert estimate.variance >= within
    # The strict witness, on a declared name, after its preconditions.
    psi = _draw_psi(result, LEVEL_A)
    variances = np.array([draw.estimates[LEVEL_A].variance for draw in result.repeats])
    assert np.ptp(psi) >= 1e-4
    assert int(np.argsort(psi)[1]) != int(np.argsort(variances)[1])
    assert result[LEVEL_A].variance > np.median(variances) * (1 + 1e-9)


def test_contrast_is_median_of_paired_draw_contrasts() -> None:
    result = _fit("ltmle", repeats=3)
    per_draw = _draw_psi(result, PAIRED)
    _close(result[PAIRED].psi, np.median(per_draw))
    of_medians = np.median(_draw_psi(result, LEVEL_A)) - np.median(_draw_psi(result, REF))
    assert abs(result[PAIRED].psi - of_medians) > 1e-8


def test_contrast_variance_keeps_the_paired_covariance() -> None:
    result = _fit("ltmle", repeats=3)
    for draw in result.repeats:
        level, ref = draw.estimates[LEVEL_A], draw.estimates[REF]
        paired = influence_variance(level.influence_curve - ref.influence_curve)
        unpaired = influence_variance(level.influence_curve) + influence_variance(
            ref.influence_curve
        )
        assert (unpaired - paired) / paired > 0.25
        _close(draw.estimates[PAIRED].influence_curve, level.influence_curve - ref.influence_curve)
        _close(draw.estimates[PAIRED].variance, paired)


def test_ratio_is_per_draw_on_the_log_scale() -> None:
    """Even ``R``: the log-scale median and the ratio-scale median differ."""
    result = _fit("ltmle", repeats=4)
    ratios = _draw_psi(result, LEVEL_B) / _draw_psi(result, REF)
    ratio = result.ratio(LEVEL_B, REF)
    _close(ratio.psi, np.exp(_central(np.log(ratios))))
    assert abs(ratio.psi - _central(ratios)) > 1e-10


def test_transformed_contrast_combines_on_the_transform_scale() -> None:
    """``logit`` of the mean of two levels, at ``R = 4`` (review M1, review2 n8)."""
    result = _fit("ltmle", repeats=4)
    names = [LEVEL_B, REF]

    def mean(values: np.ndarray) -> float:
        return float((values[0] + values[1]) / 2.0)

    logit = Transform.logit()
    combined = result.contrast(mean, names, transform=logit, scale="level")
    natural = [
        smooth_contrast(draw.estimates, mean, names, n=result.n, scale="level")
        for draw in result.repeats
    ]
    values = np.array([part.psi for part in natural])
    scaled = np.log(values / (1 - values))
    slopes = 1.0 / (values * (1 - values))
    variances = np.array(
        [slope**2 * part.variance for slope, part in zip(slopes, natural, strict=True)]
    )
    centre = _central(scaled)
    se = np.sqrt(np.median(variances + (scaled - centre) ** 2))
    assert combined.transform is not None and combined.transform.name == logit.name
    _close(combined.psi, 1.0 / (1.0 + np.exp(-centre)))
    _close(combined.std_error, se)
    z = 1.959963984540054
    _close(
        combined.ci, (1 / (1 + np.exp(-(centre - z * se))), 1 / (1 + np.exp(-(centre + z * se))))
    )
    plain = result.contrast(mean, names, scale="level")
    assert abs(combined.psi - plain.psi) > 1e-10
    assert abs(combined.std_error - plain.std_error) > 1e-10


def test_transform_refusal_names_the_draw() -> None:
    result = _fit("ltmle", repeats=3)
    values = _draw_psi(result, LEVEL_B)
    order = np.argsort(values)
    cut = float((values[order[0]] + values[order[1]]) / 2.0)
    low = int(order[0]) + 1
    with (
        warnings.catch_warnings(),
        pytest.raises(ValueError, match=f"^draw {low} of 3: the contrast value ") as raised,
    ):
        # ``log`` of a negative value warns before the refusal, as on one draw.
        warnings.simplefilter("ignore", RuntimeWarning)
        result.contrast(lambda v: v[0] - cut, [LEVEL_B], transform=Transform.log(), scale="level")
    assert "must be defined at every draw's value" in str(raised.value)


def _with_draw_level(result: Any, draw: int, name: str, value: float) -> Any:
    repeats = list(result.repeats)
    estimates = dict(repeats[draw].estimates)
    estimates[name] = replace(estimates[name], psi=value)
    repeats[draw] = replace(repeats[draw], estimates=estimates)
    return replace(result, repeats=tuple(repeats))


def test_sdr_ratio_checks_every_draw() -> None:
    """A positive median level does not license a draw whose level is not (review M1)."""
    result = _with_draw_level(_fit("sdr", repeats=3), 1, LEVEL_B, -0.01)
    assert np.median(_draw_psi(result, LEVEL_B)) > 0
    with pytest.raises(CapabilityError) as raised:
        result.ratio(LEVEL_B, REF)
    text = str(raised.value)
    assert f"{LEVEL_B} is -0.01 in draw 2 of 3, at or below 0." in text
    assert "a positive level in every draw, because a repeated fit combines each draw" in text


def test_rmst_and_contrast_apply_per_draw() -> None:
    """Even ``R``, so a functional of the medians differs from the median of the functional."""
    result = _fit("survival", repeats=4)
    index = result.parameter_index

    def risk(draw: Any, arm: str, horizon: int) -> float:
        name = next(k for k, v in index.items() if v == (arm, None, horizon))
        return float(draw.estimates[name].psi)

    per_draw = np.array(
        [sum(risk(d, "never", t) - risk(d, "always", t) for t in (1, 2)) for d in result.repeats]
    )
    rmst = result.rmst("always", 3, versus="never")
    _close(rmst.psi, np.median(per_draw))
    medians = {
        (arm, t): np.median([risk(d, arm, t) for d in result.repeats])
        for arm in ("always", "never")
        for t in (1, 2)
    }
    of_medians = sum(medians["never", t] - medians["always", t] for t in (1, 2))
    assert abs(rmst.psi - of_medians) > 1e-10
    rmtl = result.rmtl("always", 3, versus="never")
    _close(rmtl.psi, -rmst.psi)
    names = ["risk_regimen[always @ t=2]", "risk_regimen[never @ t=2]"]

    def product(values: np.ndarray) -> float:
        return float(values[0] * values[1])

    products = np.array(
        [d.estimates[names[0]].psi * d.estimates[names[1]].psi for d in result.repeats]
    )
    contrast = result.contrast(product, names)
    _close(contrast.psi, np.median(products))
    of_medians = np.median(_draw_psi(result, names[0])) * np.median(_draw_psi(result, names[1]))
    assert abs(contrast.psi - of_medians) > 1e-12


def test_incidence_total_applies_per_draw() -> None:
    result = _fit("competing", repeats=3)
    frame = result.incidence_total()
    assert len(frame) > 0
    displaced = 0
    index = result.parameter_index
    for row in frame.itertuples():
        names = [
            name
            for name, key in index.items()
            if key[0] == row.regimen and key[2] == int(row.time) and name.startswith("cif_")
        ]
        assert len(names) == 2
        totals = np.array([sum(d.estimates[n].psi for n in names) for d in result.repeats])
        _close(row.total, np.median(totals))
        # Each draw's total has the variance of its summed curve; the median rule adds each
        # draw's displacement from the median total (Chernozhukov 2018, equation 3.14).
        variances = [
            float(
                influence_variance(
                    np.sum([d.estimates[n].influence_curve for n in names], axis=0),
                    cluster=result.data.cluster,
                )
            )
            for d in result.repeats
        ]
        _close(row.std_err, np.sqrt(_median_adjusted_variance(totals, variances)))
        displaced += abs(row.std_err - np.sqrt(np.median(variances))) > 1e-9
    # The displacement term moves at least one row's standard error.
    assert displaced > 0


def test_msm_coefficients_median_coordinatewise() -> None:
    result = _fit("msm", repeats=3)
    assert len(result.estimates) == 2
    for name, estimate in result.estimates.items():
        _close(estimate.psi, np.median(_draw_psi(result, name)))
    coefficients = result.coefficients()
    assert len(coefficients) == 2


@pytest.mark.parametrize(("design", "repeats"), [("ltmle", 4), ("repeated", 3)])
def test_band_uses_split_adjusted_se_and_median_critical_value(design: str, repeats: int) -> None:
    """The band's critical value is the median over draws, and not draw 1's (review major 1).

    On the three-draw law-D fit draw 1's critical value is the median, so a regression to
    draw 1 would pass there.  Both fixtures here have a draw-1 value away from the median.
    """
    result = _fit(design, repeats=repeats)
    band = result.simultaneous
    draws = np.array(band.draw_critical_values)
    assert len(draws) == repeats and len(set(draws.tolist())) == repeats
    assert abs(band.critical_value - draws[0]) > 1e-6, "draw 1 holds the median"
    _close(band.critical_value, np.median(draws))
    for name, (low, high) in band.bands.items():
        estimate = result[name]
        _close((high - low) / 2.0, band.critical_value * estimate.std_error)
        _close((high + low) / 2.0, estimate.psi)
        marginal = estimate.ci
        assert low <= marginal[0] and marginal[1] <= high


def _study_result(repeats: int, random_state: int | None = 11) -> Any:
    rows = _rows("ltmle")
    study = CausalStudy(
        rows,
        design=LongitudinalTreatment(
            outcome="Y",
            treatment=("A1", "A2"),
            baseline=("W",),
            time_varying=((), ("L2",)),
            censoring=("C1", "C2"),
        ),
    )
    effect = study.identify(RegimeContrast(support.REGIMENS, reference=support.REFERENCE))
    return _quiet(
        effect.estimate,
        **support.cells(),
        n_folds=5,
        learner_folds=2,
        g_bounds=support.G_BOUNDS,
        random_state=random_state,
        repeats=repeats,
    )


def test_causal_study_band_under_repeats_uses_the_median_rule() -> None:
    """The narrowed band follows the repeat rule over the narrowed per-draw family (B1)."""
    result = _study_result(3)
    names = sorted(result.estimates)
    assert names and all(name.startswith("ate_") for name in names)
    for draw in result.repeats:
        assert sorted(draw.estimates) == names
    family = {name: result[name] for name in result.simultaneous.bands}
    expected = median_bands(
        [draw.estimates for draw in result.repeats],
        family,
        alpha=result.simultaneous.alpha,
        n_replicates=result.simultaneous.n_replicates,
        kind=result.simultaneous.kind,
        random_state=result.config.multiplier_seed,
    )
    combined = simultaneous_bands(
        family,
        alpha=result.simultaneous.alpha,
        n_replicates=result.simultaneous.n_replicates,
        kind=result.simultaneous.kind,
        random_state=result.config.multiplier_seed,
    )
    assert abs(expected.critical_value - combined.critical_value) > 1e-6
    assert result.simultaneous.critical_value == expected.critical_value
    assert result.simultaneous.draw_critical_values == expected.draw_critical_values
    assert result.simultaneous.bands == expected.bands
    assert result.identified_effect is not None
    assert result.method is not None
    assert set(result.parameter_keys) == set(names)
    assert result.provenance.fold_fingerprint == result.split_plan.fingerprint
    assert result.data.n == _rows("ltmle").shape[0]


def _candidates(result: Any, names: list[str]) -> list[np.ndarray]:
    values = np.array(
        [[draw.estimates[name].inference_value for name in names] for draw in result.repeats]
    )
    offsets = values - np.median(values, axis=0, keepdims=True)
    return [
        estimate_covariance(draw.estimates, names) + np.outer(offset, offset)
        for draw, offset in zip(result.repeats, offsets, strict=True)
    ]


COVARIANCE_NAMES = [LEVEL_A, LEVEL_B, REF, PAIRED]


def test_covariance_is_the_median_operator_norm_candidate() -> None:
    odd = _fit("ltmle", repeats=3)
    candidates = _candidates(odd, COVARIANCE_NAMES)
    norms = [np.linalg.eigvalsh(c)[-1] for c in candidates]
    chosen = candidates[int(np.argsort(norms, kind="stable")[1])]
    matrix = odd.covariance(COVARIANCE_NAMES)
    elementwise = np.median(np.stack(candidates), axis=0)
    assert np.max(np.abs(elementwise - chosen)) > 1e-9
    _close(matrix, chosen)
    assert np.linalg.eigvalsh(matrix)[0] >= -ATOL
    even = _fit("ltmle", repeats=4)
    candidates = _candidates(even, COVARIANCE_NAMES)
    order = np.argsort([np.linalg.eigvalsh(c)[-1] for c in candidates], kind="stable")
    _close(even.covariance(COVARIANCE_NAMES), (candidates[order[1]] + candidates[order[2]]) / 2)


@pytest.mark.parametrize("repeats", [3, 4])
def test_one_name_covariance_is_the_reported_variance(repeats: int) -> None:
    result = _fit("ltmle", repeats=repeats)
    ratio = result.ratio(LEVEL_B, REF)
    gaps = []
    for name in (LEVEL_A, PAIRED):
        matrix = result.covariance([name])
        assert matrix.shape == (1, 1)
        assert matrix[0, 0] == result[name].variance
        candidates = [float(c[0, 0]) for c in _candidates(result, [name])]
        gaps.append(_central(np.array(candidates)) != result[name].variance)
    # A ratio's one-name matrix is the variance on the log scale, by the same scalar rule.
    draws = [draw.estimates for draw in result.repeats]
    from cleverly.inference.results import ratio_contrast

    per_draw = [
        {"r": ratio_contrast(d, LEVEL_B, REF, kind="rr", complement=False, n=result.n, name="r")}
        for d in draws
    ]
    assert median_covariance(per_draw, ["r"])[0, 0] == ratio.variance
    candidates = [
        float(estimate_covariance(d, ["r"])[0, 0])
        + (d["r"].inference_value - np.median([p["r"].inference_value for p in per_draw])) ** 2
        for d in per_draw
    ]
    gaps.append(_central(np.array(candidates)) != ratio.variance)
    if not any(gaps):
        # The witness needs the matrix route and the scalar route to differ in a last bit.
        # That gap depends on the floating-point stack, and its absence is no defect.
        pytest.skip("no last-bit gap on this platform: the equality witnesses nothing here")


def test_covariance_can_differ_from_the_reported_diagonal() -> None:
    result = _fit("ltmle", repeats=3)
    matrix = result.covariance(COVARIANCE_NAMES)
    reported = np.array([result[name].variance for name in COVARIANCE_NAMES])
    assert np.max(np.abs(np.diag(matrix) - reported) / reported) > 1e-6
    first = result.covariance([REF, LEVEL_A])[0, 0]
    second = result.covariance([REF, LEVEL_B])[0, 0]
    assert abs(first - second) / first > 1e-6


def test_elementwise_median_can_be_indefinite() -> None:
    """Three draws with one point and covariances ``[[4,2],[2,1]]``, ``[[1,2],[2,4]]``, ``I``."""
    n = 400
    raw = np.random.default_rng(3).normal(size=(n, 2))
    base = [make_estimate(name, 0.5, raw[:, j], n=n, scale="level") for j, name in enumerate("ab")]
    whitening = np.linalg.inv(
        np.linalg.cholesky(estimate_covariance({e.name: e for e in base}, ["a", "b"]))
    ).T
    white = raw @ whitening
    draws = []
    targets = [np.array([[4.0, 2.0], [2.0, 1.0]]), np.array([[1.0, 2.0], [2.0, 4.0]]), np.eye(2)]
    for target in targets:
        root = np.linalg.cholesky(target + 1e-12 * np.eye(2))
        curves = white @ root.T
        draws.append(
            {
                name: make_estimate(name, 0.5, curves[:, j], n=n, scale="level")
                for j, name in enumerate("ab")
            }
        )
    candidates = [estimate_covariance(draw, ["a", "b"]) for draw in draws]
    for candidate, target in zip(candidates, targets, strict=True):
        _close(candidate, target)
    elementwise = np.median(np.stack(candidates), axis=0)
    assert np.linalg.eigvalsh(elementwise)[0] < -0.5
    matrix = median_covariance(draws, ["a", "b"])
    assert np.linalg.eigvalsh(matrix)[0] >= -ATOL


def _stamp_second_draw(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    original = estimator_module._estimates
    calls = {"n": 0}

    def stamped(*args: Any, **kwargs: Any) -> Any:
        reported = original(*args, **kwargs)
        calls["n"] += 1
        if calls["n"] != 2:
            return reported
        estimates = dict(reported.estimates)
        estimates[name] = replace(estimates[name], inference="constant_node_plugin")
        return replace(reported, estimates=estimates)

    monkeypatch.setattr(estimator_module, "_estimates", stamped)


def test_status_is_the_precedent_of_the_draws(monkeypatch: pytest.MonkeyPatch) -> None:
    """One draw's ``constant_node_plugin`` withholds inference from every combination."""
    _stamp_second_draw(monkeypatch, LEVEL_B)
    result = _quiet(support.ltmle, _rows("ltmle"), repeats=3)
    assert [draw.estimates[LEVEL_B].inference for draw in result.repeats] == [
        "influence_curve",
        "constant_node_plugin",
        "influence_curve",
    ]
    combined = result[LEVEL_B]
    assert combined.inference == "constant_node_plugin"
    assert not combined.supplies_inference
    assert result[REF].supplies_inference
    contrast = result.contrast(lambda v: v[0] - v[1], [LEVEL_B, REF])
    ratio = result.ratio(LEVEL_B, REF)
    for derived in (contrast, ratio):
        assert derived.inference == "constant_node_plugin"
        assert not derived.supplies_inference
    matrix = result.covariance([LEVEL_B, REF])
    assert matrix.shape == (2, 2) and np.all(np.isfinite(matrix))
    assert LEVEL_B not in result.simultaneous.bands


def test_median_of_monotone_curves_is_monotone() -> None:
    result = _fit("survival", repeats=3)
    index = result.parameter_index
    for arm in ("never", "always"):
        names = [n for h in (1, 2, 3) for n, k in index.items() if k == (arm, None, h)]
        for draw in result.repeats:
            assert np.all(np.diff([draw.estimates[n].psi for n in names]) >= 0)
        assert np.all(np.diff([result[n].psi for n in names]) >= 0)
    curve = result.curve()
    assert len(curve) > 0


def test_truncation_curve_follows_every_draw() -> None:
    """Each bound's row is the median of the three single-draw replays."""
    bounds = [0.05, 0.2]
    result = _fit("ltmle", repeats=3)
    curve = result.diagnostics.truncation_curve(bounds=bounds)
    singles = [
        _fit("ltmle", random_state=draw.seed).diagnostics.truncation_curve(bounds=bounds)
        for draw in result.repeats
    ]
    for column in ("psi", "truncated_score_cells", "evaluated_score_cells"):
        stacked = np.stack([single[column].to_numpy(dtype=float) for single in singles])
        expected = np.median(stacked, axis=0) if column == "psi" else stacked.sum(axis=0)
        _close(curve[column].to_numpy(dtype=float), expected)
    gap = np.abs(curve["psi"].to_numpy() - singles[0]["psi"].to_numpy())
    assert np.max(gap) > 1e-8


def test_score_rows_cover_every_draw() -> None:
    result = _fit("ltmle", repeats=3)
    single = _fit("ltmle")
    rows = result.diagnostics.score_equations().to_frame()
    alone = single.diagnostics.score_equations().to_frame()
    assert "draw" not in alone.columns
    assert rows["draw"].tolist() == [d for d in (1, 2, 3) for _ in range(len(alone))]
    for draw in (1, 2, 3):
        part = rows[rows["draw"] == draw].drop(columns="draw").reset_index(drop=True)
        assert part[["regimen", "horizon", "time"]].equals(alone[["regimen", "horizon", "time"]])
    msm = _fit("msm", repeats=3).diagnostics.score_equations().to_frame()
    assert set(msm["draw"]) == {1, 2, 3}


def test_repeat_spread_is_the_sd_of_draws() -> None:
    result = _fit("ltmle", repeats=3)
    spread = result.repeat_spread()
    for name in result.estimates:
        values = [draw.estimates[name].inference_value for draw in result.repeats]
        _close(spread[name], np.std(values, ddof=1))
    broken = _with_draw_level(result, 1, LEVEL_B, float("nan"))
    assert np.isnan(broken.repeat_spread()[LEVEL_B])
    with pytest.raises(ValueError):
        _fit("ltmle").repeat_spread()


def test_stagewise_and_nuisance_reports_describe_draw_one() -> None:
    result = _fit("ltmle", repeats=3)
    stagewise = result.diagnostics.support()
    assert (stagewise.n_repeats, stagewise.reported_repeat) == (3, 1)
    nuisance = result.diagnostics.nuisance_models()
    assert (nuisance.n_repeats, nuisance.reported_repeat) == (3, 1)
    assert len(nuisance.repeat_spread) == len(result.estimates)
    single = _fit("ltmle", random_state=result.repeats[0].seed)
    pd.testing.assert_frame_equal(stagewise.to_frame(), single.diagnostics.support().to_frame())


def test_validate_and_assess_run_on_a_repeated_fit() -> None:
    result = _quiet(support.ltmle, _rows("ltmle"), repeats=3)
    assert result.assessment_family == _fit("ltmle").assessment_family
    report = result.validate()
    assert any("draw 1 of 3" in str(item) for item in report.items)
    assessed = result.assess()
    assert assessed.validation.items == report.items
    assert result.replayability.refit_nuisances
    assert result.assessment_cache is not None


# ================================================================ 4.4 refusals and registry


@pytest.mark.parametrize("cls", [LTMLE, SDR, IPSI])
@pytest.mark.parametrize("value", [0, -1, 1.5, "3", None])
def test_repeats_must_be_a_positive_integer(cls: Any, value: Any) -> None:
    with pytest.raises(ValueError) as raised:
        cls({"always": 1}, repeats=value)
    assert str(raised.value) == (
        f"repeats must be an integer of at least 1; got {value!r}. It is how many independent "
        "draws of the cross-fitting split the fit combines by their median."
    )


@pytest.mark.parametrize("cls", [LTMLE, SDR, IPSI])
def test_repeats_need_more_than_one_fold(cls: Any) -> None:
    with pytest.raises(ValueError) as raised:
        cls({"always": 1}, repeats=2, n_folds=1)
    assert str(raised.value) == (
        "repeats above 1 needs n_folds of 2 or more. One fold fits every nuisance on every "
        "row, so there is no split to draw again. Set n_folds to 2 or more, or set repeats=1."
    )


@pytest.mark.parametrize(
    "setting",
    [
        {"survival_model": "hazard"},
        {"survival_model": "hazard", "targeting": "one_step"},
        {"survival_model": "hazard", "step_size": 0.01},
        {"isotonic": True},
    ],
    ids=["hazard", "targeting", "step_size", "isotonic"],
)
def test_repeats_refuse_the_hazard_and_isotonic_curve_options(setting: dict[str, Any]) -> None:
    with pytest.raises(MethodConfigurationError) as raised:
        LTMLE({"always": 1}, repeats=2, n_folds=2, **setting)
    assert raised.value.refusal.reason is RefusalReason.INFERENCE_GAP
    assert "repeats=1" in (raised.value.refusal.alternative or "")


@pytest.mark.parametrize("cls", [LTMLE, SDR, IPSI])
def test_repeats_refuse_the_bootstrap(cls: Any) -> None:
    with pytest.raises(ValueError) as raised:
        cls({"always": 1}, repeats=2, n_bootstrap=20)
    assert str(raised.value) == (
        "n_bootstrap is not combined with repeats above 1. No registered study measures a "
        "bootstrap of the median over fold draws, and each replicate would refit R complete "
        "draws. Fit with repeats=1 for the bootstrap, or with n_bootstrap=0 for the "
        "split-adjusted median interval."
    )
    assert _fit("ltmle", repeats=3).bootstrap is None


def test_a_per_draw_refusal_names_its_draw() -> None:
    """``scale="ratio"`` on a draw whose value is not positive keeps its type (review2 n4)."""
    result = _with_draw_level(_fit("sdr", repeats=3), 2, LEVEL_B, -0.01)
    with pytest.raises(ValueError) as raised:
        result.contrast(lambda v: v[0], [LEVEL_B], scale="ratio")
    assert str(raised.value) == (
        "draw 3 of 3: a ratio-scale contrast needs a positive value; got -0.01"
    )
    assert isinstance(raised.value.__cause__, ValueError)


def test_point_keywords_are_refused_on_a_repeated_fit() -> None:
    with pytest.raises(TypeError, match="covariates"):
        LTMLE({"always": 1}, repeats=3).fit(_rows("ltmle"), covariates=["W"], **support.COLUMNS)


def test_influence_curves_are_the_combined_curves() -> None:
    for repeats in (3, 4):
        result = _fit("ltmle", repeats=repeats)
        curves = result.influence_curves
        for name, estimate in result.estimates.items():
            assert curves[name] is estimate.influence_curve
        values = _draw_psi(result, LEVEL_A)
        order = np.argsort(values)
        if repeats == 3:
            expected = result.repeats[int(order[1])].estimates[LEVEL_A].influence_curve
        else:
            expected = (
                result.repeats[int(order[1])].estimates[LEVEL_A].influence_curve
                + result.repeats[int(order[2])].estimates[LEVEL_A].influence_curve
            ) / 2
        _close(curves[LEVEL_A], expected)


def test_readers_report_the_combined_estimates() -> None:
    result = _fit("ltmle", repeats=3)
    single = _fit("ltmle")
    assert result.n == single.n
    assert result.parameter_index == single.parameter_index
    assert list(result.keys()) == list(result.estimates)
    assert list(result.values()) == list(result.estimates.values())
    assert dict(result.items()) == result.estimates
    assert result.get(LEVEL_A) is result.estimates[LEVEL_A]
    assert result.psi(LEVEL_A) == result[LEVEL_A].psi
    with pytest.raises(ValueError, match="contains multiple parameters"):
        _ = result.estimate
    sole = _quiet(
        LTMLE(
            {"always": 1},
            **support.cells(),
            n_folds=5,
            g_bounds=support.G_BOUNDS,
            random_state=11,
            repeats=3,
        ).fit,
        _rows("ltmle"),
        **support.COLUMNS,
    )
    assert sole.estimate is sole.estimates[LEVEL_B]
    assert sole.psi() == sole[LEVEL_B].psi
    frame = result.to_frame()
    for name, estimate in result.estimates.items():
        assert frame.loc[frame["estimand"] == name, "psi"].item() == estimate.psi
    text = str(result.summary())
    assert "median over 3 draws" in text
    assert "leverage of draw 1 of 3" in text
    # The split-noise block names every parameter with its spread across the draws.
    assert "split noise -- spread across the 3 draws on the inference scale" in text
    spread = result.repeat_spread()
    for name in result.estimates:
        assert f"  {name} {spread[name]:.4g}  (" in text
    assert "split noise" not in str(single.summary())


def test_unavailable_capabilities_keep_their_reason() -> None:
    repeated = _fit("ltmle", repeats=3)
    single = _fit("ltmle")
    for facade in ("diagnostics", "sensitivity"):
        mine = {c.operation: c for c in getattr(repeated, facade).capabilities}
        theirs = {c.operation: c for c in getattr(single, facade).capabilities}
        assert sorted(mine) == sorted(theirs)
        for operation, capability in theirs.items():
            if facade == "sensitivity":
                # Every route maps to this test in the registry, so a route that turns
                # available needs its own repeat witness before it leaves this list.
                assert not mine[operation].available, operation
            if capability.available:
                continue
            assert mine[operation].status == capability.status
            assert mine[operation].reason == capability.reason
            # The refusal a call raises, whatever it is, is the one-draw refusal.
            raised = [_raised(getattr(fit, facade), operation) for fit in (repeated, single)]
            assert raised[0] == raised[1], operation


def _raised(facade: Any, operation: str) -> tuple[str, str]:
    try:
        getattr(facade, operation)()
    except Exception as error:
        return type(error).__name__, str(error)
    raise AssertionError(f"{operation} ran on a fit that declares it unavailable")


def test_serialization_round_trip_keeps_every_draw() -> None:
    result = _fit("ltmle", repeats=3)
    loaded = loads(dumps(result))
    assert isinstance(loaded, LongitudinalResult)
    assert loaded.n_repeats == 3
    assert [d.seed for d in loaded.repeats] == [d.seed for d in result.repeats]
    for name, estimate in result.estimates.items():
        assert loaded[name].psi == estimate.psi
        assert loaded[name].variance == estimate.variance
    assert loaded.provenance.fold_fingerprint == result.provenance.fold_fingerprint
    assert loaded.ratio(LEVEL_B, REF).psi == result.ratio(LEVEL_B, REF).psi


def test_long_panel_repeats_reproduce_the_wide_fit() -> None:
    from tests.unit.test_long_panel import WIDE_ROLES, build, learners, long_panel, wide_frame

    panel = long_panel(n=400, seed=2)
    regimens = {"never": ("none",) * 3, "always": ("offer",) * 3}
    settings = {**learners(), "repeats": 3}
    adapted = _quiet(LTMLE(regimens, reference="never", **settings).fit, build(panel))
    wide = _quiet(
        LTMLE(regimens, reference="never", **settings).fit,
        LongitudinalData.from_frame(wide_frame(panel), **WIDE_ROLES),
    )
    for left, right in zip(adapted.repeats, wide.repeats, strict=True):
        assert left.seed == right.seed
        np.testing.assert_array_equal(left.folds.assignment, right.folds.assignment)
        for name, estimate in right.estimates.items():
            _close(left.estimates[name].psi, estimate.psi)
            _close(left.estimates[name].influence_curve, estimate.influence_curve)
    pd.testing.assert_frame_equal(adapted.to_frame(), wide.to_frame(), rtol=RTOL)
    describe = adapted.config.describe(contrast=True)
    assert any(line.startswith("long panel: unit column 'customer'") for line in describe)
    assert "cross-fitting: 2 fold(s), median over 3 draws" in describe
    # The study design ``LongPanel`` reaches the same repeated engine.
    from tests.unit.test_long_panel import ROLES

    via_study = _quiet(
        CausalStudy(panel, design=LongPanel(**ROLES)).estimate,
        RegimeContrast(regimens, reference="never"),
        **learners(),
        repeats=3,
    )
    assert via_study.n_repeats == 3
    name = "ate_regimen[always vs never]"
    _close(via_study[name].psi, adapted[name].psi)


def test_time_to_event_study_repeats_reach_the_engine() -> None:
    from cleverly import TimeToEvent

    result = _fit("time_to_event", repeats=3)
    rows = _rows("time_to_event")
    study = CausalStudy(
        rows, design=TimeToEvent(time="time", event="event", treatment="A", baseline=["W"])
    )
    from tests.discrete_law_point_survival import CellMeans as SurvivalMeans
    from tests.discrete_law_point_survival import CellProbabilities

    via_study = _quiet(
        study.estimate,
        RegimeContrast({"a0": 0, "a1": 1}, reference="a0"),
        outcome_learner=SurvivalMeans(),
        pseudo_learner=SurvivalMeans(),
        treatment_learner=CellProbabilities(),
        censoring_learner=SurvivalMeans(),
        n_folds=2,
        learner_folds=2,
        g_bounds=support.G_BOUNDS,
        random_state=11,
        repeats=3,
    )
    assert via_study.n_repeats == 3
    for name, estimate in via_study.estimates.items():
        _close(estimate.psi, result[name].psi)


# ================================================================ review witnesses


#: Law D with a lower bound that clips the cumulative mechanism of draw 3 only: the largest
#: clipped share of a node is 0 in draws 1 and 2 and 0.0595 in draw 3, above the 5% line.
LATE_TRUNCATION = {"g_bounds": (0.02, 1.0), "random_state": 3}


def test_a_later_draw_truncation_reaches_the_warning_and_assess() -> None:
    """A breach in draw 3 alone warns at fit time and reaches ``assess`` (review minors 1, 2)."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = support.ltmle(_rows("ltmle"), repeats=3, **LATE_TRUNCATION)
    stagewise = result.diagnostics.support()
    shares = stagewise.draw_max_share_truncated
    assert shares[0] <= 0.05 and shares[1] <= 0.05 and shares[2] > 0.05
    # The rows are draw 1's, so a grade that read only the rows would see no breach.
    assert max(stagewise.to_frame()["share_truncated"]) <= 0.05
    for draw, share in zip(result.repeats, shares, strict=True):
        single = _fit("ltmle", g_bounds=LATE_TRUNCATION["g_bounds"], random_state=draw.seed)
        assert share == max(single.diagnostics.support().to_frame()["share_truncated"])
    truncation = [str(w.message) for w in caught if "truncat" in str(w.message)]
    assert truncation and all("in draw(s) 3 of 3" in text for text in truncation), truncation
    item = next(i for i in result.validate().items if i.name == "support")
    assert item.status.value == "warning"
    assert "exceeds 5% at one or more nodes in draw 3 of 3" in item.detail
    assert "the rows describe draw 1 of 3" in item.detail


def test_causal_study_band_without_a_seed_reads_the_recorded_multiplier_seed() -> None:
    """Under ``random_state=None`` the narrowed band reproduces from the recorded seed."""
    result = _study_result(3, random_state=None)
    seed = result.config.multiplier_seed
    assert isinstance(seed, int)
    family = {name: result[name] for name in result.simultaneous.bands}
    expected = median_bands(
        [draw.estimates for draw in result.repeats],
        family,
        alpha=result.simultaneous.alpha,
        n_replicates=result.simultaneous.n_replicates,
        kind=result.simultaneous.kind,
        random_state=seed,
    )
    other = median_bands(
        [draw.estimates for draw in result.repeats],
        family,
        alpha=result.simultaneous.alpha,
        n_replicates=result.simultaneous.n_replicates,
        kind=result.simultaneous.kind,
        random_state=seed + 1,
    )
    assert other.critical_value != expected.critical_value
    assert result.simultaneous.critical_value == expected.critical_value
    assert result.simultaneous.bands == expected.bands


def test_repeated_outcome_curve_is_the_median_of_draws() -> None:
    """X32a's repeated outcome under ``repeats=3``: each horizon combines by the median rule.

    Each draw is a single-draw fit (``test_each_draw_reproduces_a_single_draw_fit``, case
    ``repeated``).  Here every horizon's mean and paired contrast is the median of its draws,
    with the split-adjusted variance, and ``curve(scale="mean")`` reports those numbers.  The
    nonzero witness: at least one row's median is away from draw 1's value.
    """
    result = _fit("repeated", repeats=3)
    assert result.data.is_repeated and result.config.outcome_kind == "repeated"
    assert set(result.estimates) == set(_fit("repeated").estimates)
    curve = result.curve(scale="mean")
    assert len(curve) == len(result.estimates)
    away = 0
    for row in curve.itertuples():
        values = _draw_psi(result, row.estimand)
        _close(row.psi, np.median(values))
        _close(result[row.estimand].variance, _hand_variance(result, row.estimand))
        low, high = result[row.estimand].ci
        _close((row.ci_lower, row.ci_upper), (low, high))
        away += abs(row.psi - values[0]) > 1e-9
    assert away > 0
    # A contrast is formed inside each draw, then combined: the paired rule.
    contrast = "ate_regimen[always vs never @ t=2]"
    paired = np.array(
        [
            d.estimates["ey_regimen[always @ t=2]"].psi - d.estimates["ey_regimen[never @ t=2]"].psi
            for d in result.repeats
        ]
    )
    _close(result[contrast].psi, np.median(paired))
    assert result.covariance(sorted(result.estimates)).shape == (15, 15)
    with pytest.raises(ValueError, match="not a cumulative risk"):
        result.curve(scale="risk")


def test_sensitivity_benchmark_refuses_a_longitudinal_fit_by_capability() -> None:
    """``benchmark`` on a longitudinal fit raises the capability row's refusal.

    It raised ``AttributeError`` from the point-treatment arm table before the fix.
    """
    for result in (_fit("ltmle"), _fit("ltmle", repeats=3)):
        capability = next(c for c in result.sensitivity.capabilities if c.operation == "benchmark")
        assert not capability.available
        for arguments in ((), (["W"],)):
            with pytest.raises(CapabilityError, match="no longitudinal benchmarking derivation"):
                result.sensitivity.benchmark(*arguments)


# ------------------------------------------------------------------ the registry

THIS = "tests.unit.test_longitudinal_repeats"
EACH_DRAW = "test_each_draw_reproduces_a_single_draw_fit"


def _witness(function: str, case: str | None = None) -> tuple[str, str, str | None]:
    return (THIS, function, case)


#: Every keyword of the three constructors and their ``fit``.  A case names the design of
#: ``test_each_draw_reproduces_a_single_draw_fit`` that sets the keyword.
KEYWORDS: dict[str, tuple[str, str, str | None]] = {
    "regimens": _witness(EACH_DRAW, "ltmle"),
    "reference": _witness(EACH_DRAW, "ltmle"),
    "horizons": _witness(EACH_DRAW, "survival"),
    "msm": _witness(EACH_DRAW, "msm"),
    "outcome_learner": _witness(EACH_DRAW, "ltmle"),
    "pseudo_learner": _witness(EACH_DRAW, "sdr"),
    "treatment_learner": _witness(EACH_DRAW, "ltmle"),
    "censoring_learner": _witness(EACH_DRAW, "ltmle"),
    "n_folds": _witness(EACH_DRAW, "ltmle"),
    "learner_folds": _witness(EACH_DRAW, "ltmle"),
    "g_bounds": _witness(EACH_DRAW, "ltmle"),
    "q_bounds": _witness(EACH_DRAW, "clustered"),
    "alpha": _witness(EACH_DRAW, "settings"),
    "alpha_sig": _witness(EACH_DRAW, "settings"),
    "simultaneous": _witness(EACH_DRAW, "settings"),
    "n_multiplier": _witness(EACH_DRAW, "settings"),
    "multiplier_kind": _witness(EACH_DRAW, "settings"),
    "max_iter": _witness(EACH_DRAW, "settings"),
    "tol": _witness(EACH_DRAW, "settings"),
    "random_state": _witness("test_none_seed_draws_fresh_concrete_seeds"),
    "run_id": _witness(EACH_DRAW, "settings"),
    "n_jobs": _witness(EACH_DRAW, "settings"),
    "n_bootstrap": _witness("test_repeats_refuse_the_bootstrap"),
    "bootstrap_resampling": _witness(EACH_DRAW, "settings"),
    "density_bins": _witness(EACH_DRAW, "mtp"),
    "ratio": _witness(EACH_DRAW, "mtp"),
    "repeats": _witness("test_repeats_must_be_a_positive_integer"),
    "refused": _witness("test_point_keywords_are_refused_on_a_repeated_fit"),
    "data": _witness(EACH_DRAW, "ltmle"),
    "outcome": _witness(EACH_DRAW, "competing"),
    "treatment": _witness(EACH_DRAW, "ltmle"),
    "baseline": _witness(EACH_DRAW, "ltmle"),
    "time_varying": _witness(EACH_DRAW, "ltmle"),
    "censoring": _witness(EACH_DRAW, "ltmle"),
    "id": _witness(EACH_DRAW, "clustered"),
    "weights": _witness(EACH_DRAW, "weighted"),
    "weights_type": _witness(EACH_DRAW, "weighted"),
    "weights_estimated": _witness(EACH_DRAW, "weighted"),
    "family": _witness(EACH_DRAW, "settings"),
    "continuous_treatment": _witness(EACH_DRAW, "mtp"),
    "treatment_probabilities": _witness(EACH_DRAW, "known"),
    "censoring_probabilities": _witness(EACH_DRAW, "known"),
    "kwargs": _witness(EACH_DRAW, "ipsi"),
    "survival_model": _witness("test_repeats_refuse_the_hazard_and_isotonic_curve_options"),
    "targeting": _witness("test_repeats_refuse_the_hazard_and_isotonic_curve_options"),
    "step_size": _witness("test_repeats_refuse_the_hazard_and_isotonic_curve_options"),
    "isotonic": _witness("test_repeats_refuse_the_hazard_and_isotonic_curve_options"),
}

_READERS = _witness("test_readers_report_the_combined_estimates")
_READ_THROUGH = _witness("test_read_through_fields_are_draw_zero")
_STUDY = _witness("test_causal_study_band_under_repeats_uses_the_median_rule")
_ASSESS = _witness("test_validate_and_assess_run_on_a_repeated_fit")

#: The dataclass fields and public members of ``LongitudinalResult``.
RESULT_SURFACE: dict[str, tuple[str, str, str | None]] = {
    "estimates": _witness("test_point_is_the_median_of_draws"),
    "data": _witness("test_long_panel_repeats_reproduce_the_wide_fit"),
    "config": _witness("test_config_records_repeats_and_the_multiplier_seed"),
    "scaler": _witness(EACH_DRAW, "clustered"),
    "provenance": _witness("test_provenance_covers_every_draw"),
    "simultaneous": _witness("test_band_uses_split_adjusted_se_and_median_critical_value"),
    "parameter_index": _READERS,
    "msm": _witness("test_msm_coefficients_median_coordinatewise"),
    "identified_effect": _STUDY,
    "method": _STUDY,
    "parameter_keys": _STUDY,
    "fitted_method": _witness(EACH_DRAW, "sdr"),
    "assessment_cache": _ASSESS,
    "bootstrap": _witness("test_repeats_refuse_the_bootstrap"),
    "repeats": _witness(EACH_DRAW, "ltmle"),
    "assess": _ASSESS,
    "assessment_family": _ASSESS,
    "coefficients": _witness("test_msm_coefficients_median_coordinatewise"),
    "contrast": _witness("test_contrast_is_median_of_paired_draw_contrasts"),
    "converged": _witness("test_converged_reads_every_draw"),
    "covariance": _witness("test_covariance_is_the_median_operator_norm_candidate"),
    "curve": _witness("test_median_of_monotone_curves_is_monotone"),
    "diagnostics": _witness("test_score_rows_cover_every_draw"),
    "estimate": _READERS,
    "fits": _READ_THROUGH,
    "fold_mechanisms": _READ_THROUGH,
    "folds": _READ_THROUGH,
    "initial_hazards": _READ_THROUGH,
    "isotonic_moves": _READ_THROUGH,
    "get": _READERS,
    "incidence_total": _witness("test_incidence_total_applies_per_draw"),
    "inference_status": _witness("test_status_is_the_precedent_of_the_draws"),
    "influence_curves": _witness("test_influence_curves_are_the_combined_curves"),
    "items": _READERS,
    "keys": _READERS,
    "mechanism": _READ_THROUGH,
    "msm_fits": _READ_THROUGH,
    "n": _READERS,
    "n_repeats": _witness("test_provenance_covers_every_draw"),
    "psi": _READERS,
    "ratio": _witness("test_ratio_is_per_draw_on_the_log_scale"),
    "repeat_spread": _witness("test_repeat_spread_is_the_sd_of_draws"),
    "replay_recipe": _READ_THROUGH,
    "replayability": _ASSESS,
    "rmst": _witness("test_rmst_and_contrast_apply_per_draw"),
    "rmtl": _witness("test_rmst_and_contrast_apply_per_draw"),
    "save": _witness("test_serialization_round_trip_keeps_every_draw"),
    "sensitivity": _witness("test_unavailable_capabilities_keep_their_reason"),
    "split_plan": _witness("test_provenance_covers_every_draw"),
    "summary": _READERS,
    "to_frame": _READERS,
    "validate": _ASSESS,
    "values": _READERS,
}

_UNAVAILABLE = _witness("test_unavailable_capabilities_keep_their_reason")

#: Every longitudinal assessment capability and every sensitivity route.
CAPABILITIES: dict[str, tuple[str, str, str | None]] = {
    "support": _witness("test_stagewise_and_nuisance_reports_describe_draw_one"),
    "nuisance_models": _witness("test_stagewise_and_nuisance_reports_describe_draw_one"),
    "score_equations": _witness("test_score_rows_cover_every_draw"),
    "truncation_curve": _witness("test_truncation_curve_follows_every_draw"),
    "corrections": _UNAVAILABLE,
    "refute": _UNAVAILABLE,
    **dict.fromkeys(SENSITIVITY_ROUTES, _UNAVAILABLE),
}

#: The witness design of each outcome kind, by the ``outcome_kind`` a fit records.
_EACH_KIND = {
    "end_of_study": "ltmle",
    "survival": "survival",
    "competing": "competing",
    "repeated": "repeated",
}

#: Every outcome kind, every ``is_*`` declaration of ``LongitudinalData``, every public
#: constructor, and every longitudinal design of ``CausalStudy``.
DATA_DECLARATIONS: dict[str, tuple[str, str, str | None]] = {
    **{f"kind:{kind}": _witness(EACH_DRAW, case) for kind, case in _EACH_KIND.items()},
    "is_survival": _witness(EACH_DRAW, "survival"),
    "is_competing": _witness(EACH_DRAW, "competing"),
    "is_repeated": _witness("test_repeated_outcome_curve_is_the_median_of_draws"),
    "is_weighted": _witness(EACH_DRAW, "weighted"),
    "is_continuous_node": _witness(EACH_DRAW, "mtp"),
    "is_held_node": _witness(EACH_DRAW, "time_to_event"),
    "from_frame": _witness(EACH_DRAW, "ltmle"),
    "from_long_panel": _witness("test_long_panel_repeats_reproduce_the_wide_fit"),
    "from_time_to_event": _witness(EACH_DRAW, "time_to_event"),
    "LongitudinalTreatment": _STUDY,
    "LongPanel": _witness("test_long_panel_repeats_reproduce_the_wide_fit"),
    "TimeToEvent": _witness("test_time_to_event_study_repeats_reach_the_engine"),
}

#: Every field that ``CausalStudy`` changes on a repeated engine result.
STUDY_FIELDS: dict[str, tuple[str, str, str | None]] = {
    "data": _STUDY,
    "provenance": _STUDY,
    "estimates": _STUDY,
    "parameter_index": _STUDY,
    "simultaneous": _STUDY,
    "bootstrap": _witness("test_repeats_refuse_the_bootstrap"),
    "repeats": _STUDY,
    "identified_effect": _STUDY,
    "method": _STUDY,
    "parameter_keys": _STUDY,
    "assessment_cache": _ASSESS,
}


def _live_keywords() -> set[str]:
    names: set[str] = set()
    for cls in (LTMLE, SDR, IPSI):
        for method in (cls.__init__, cls.fit):
            names |= set(inspect.signature(method).parameters) - {"self"}
    return names


def _live_result_surface() -> set[str]:
    fields = {field.name for field in dataclasses.fields(LongitudinalResult)}
    return fields | {name for name in dir(LongitudinalResult) if not name.startswith("_")}


def _live_capabilities() -> set[str]:
    rows = {c.operation for c in ASSESSMENT_CAPABILITIES if c.result_family == "longitudinal"}
    return rows | set(SENSITIVITY_ROUTES)


def _live_data_declarations() -> set[str]:
    # Every kind the estimator declares, read from its type, so a new kind fails here until
    # it has a row.  Each row's design must be a fit of that kind.
    kinds = {f"kind:{kind}" for kind in typing.get_args(estimator_module.OutcomeKind)}
    for kind, case in _EACH_KIND.items():
        assert _fit(case, repeats=3).config.outcome_kind == kind, (kind, case)
    flags = {name for name in dir(LongitudinalData) if name.startswith("is_")}
    constructors = {
        name
        for name, value in vars(LongitudinalData).items()
        if isinstance(value, classmethod) and not name.startswith("_")
    }
    hints = typing.get_type_hints(CausalStudy.__init__)["design"]
    designs = {cls.__name__ for cls in typing.get_args(hints)} - {"PointTreatment"}
    return kinds | flags | constructors | designs


def _live_study_fields() -> set[str]:
    engine = _fit("ltmle", repeats=3)
    study = _study_result(3)
    changed = set()
    for field in dataclasses.fields(LongitudinalResult):
        mine, theirs = getattr(study, field.name), getattr(engine, field.name)
        try:
            same = mine is theirs or bool(mine == theirs)
        except Exception:
            same = False
        if not same:
            changed.add(field.name)
    return changed


TABLES = {
    "estimator keywords": (KEYWORDS, _live_keywords),
    "result surface": (RESULT_SURFACE, _live_result_surface),
    "assessment capability rows": (CAPABILITIES, _live_capabilities),
    "data declarations": (DATA_DECLARATIONS, _live_data_declarations),
    "CausalStudy post-processing": (STUDY_FIELDS, _live_study_fields),
}


@pytest.mark.parametrize("table", sorted(TABLES))
def test_every_longitudinal_surface_has_a_repeat_witness(table: str) -> None:
    """Each live surface maps to a test that runs on a repeated fit (review M3).

    A surface that merges after X46 fails here until a row names its per-draw witness, or the
    test of its refusal.
    """
    rows, live = TABLES[table]
    surface = live()
    missing = sorted(surface - set(rows))
    assert not missing, (
        f"{table}: {missing} have no repeat witness. Add a row that names a test that runs "
        "on a fit with repeats > 1, or the test of a refusal by name."
    )
    for name, (module_name, function, case) in rows.items():
        module = importlib.import_module(module_name)
        test = getattr(module, function, None)
        assert callable(test) and function.startswith("test_"), (table, name, function)
        if case is not None:
            assert case in support.DESIGNS, (table, name, case)
