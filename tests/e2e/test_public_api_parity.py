"""The typed public API is an exact adapter over every evidenced analytic target."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import sklearn.linear_model

from cleverly import (
    ATC,
    ATE,
    ATT,
    CausalStudy,
    CounterfactualMean,
    IncrementalEffect,
    IncrementalMean,
    LongitudinalTreatment,
    ModifiedTreatmentPolicy,
    ModifiedTreatmentPolicyEffect,
    MSMProjection,
    NaturalCourseMean,
    OddsRatio,
    PointTreatment,
    PopulationAttributableFraction,
    PopulationAttributableRisk,
    RegimeContrast,
    RegimeMean,
    RiskRatio,
    load,
)
from cleverly.datasets import (
    make_binary_outcome,
    make_longitudinal,
    make_longitudinal_competing,
    make_longitudinal_survival,
    make_shift_dose,
)
from cleverly.estimators import TMLE
from cleverly.interventions import Incremental, Shift, Static
from cleverly.longitudinal import LTMLE
from cleverly.msm import MSM
from tests.conftest import FAST_KWARGS

POINT_SETTINGS = {**FAST_KWARGS, "simultaneous": False}
LONG_SETTINGS: dict[str, Any] = {
    "outcome_learner": sklearn.linear_model.LinearRegression(),
    "pseudo_learner": sklearn.linear_model.LinearRegression(),
    "treatment_learner": sklearn.linear_model.LogisticRegression(max_iter=1000),
    "n_folds": 3,
    "learner_folds": 3,
    "random_state": 0,
    "simultaneous": False,
}
LONG_COLUMNS: dict[str, Any] = {
    "outcome": "Y",
    "treatment": ("A1", "A2"),
    "baseline": ("W1", "W2"),
    "time_varying": ((), ("L2",)),
    "censoring": ("C1", "C2"),
}


def longitudinal_msm_design(label: str, horizon: int, frame: Any) -> np.ndarray:
    duration = {"always": 2.0, "never": 0.0, "early": 1.0}[label]
    return np.column_stack([np.ones(len(frame)), np.full(len(frame), duration + 0.0 * horizon)])


def assert_identical(old: Any, new: Any, *, check_bands: bool = False) -> None:
    """Point value and every inference quantity are bit-for-bit unchanged.

    ``check_bands`` is separate because most of these tests hand the study a narrower family
    than the engine reports, and a joint band over a subfamily is *supposed* to differ from
    the engine's. Pass it wherever the study reports exactly what the engine did -- there the
    critical value has to match too, and the fact that nothing checked it is how a study-driven
    longitudinal fit came to draw half the multipliers the engine declares.
    """
    expected_estimates = old.estimates if hasattr(old, "estimates") else old
    assert list(new.estimates) == list(expected_estimates)
    for name, expected in expected_estimates.items():
        observed = new[name]
        assert observed.psi == expected.psi
        np.testing.assert_array_equal(observed.influence_curve, expected.influence_curve)
        assert observed.variance == expected.variance
        assert observed.std_error == expected.std_error
        assert observed.ci == expected.ci
        assert observed.pvalue == expected.pvalue
    if not check_bands:
        return
    expected_bands, observed_bands = old.simultaneous, new.simultaneous
    assert (expected_bands is None) == (observed_bands is None)
    if expected_bands is None:
        return
    assert observed_bands.critical_value == expected_bands.critical_value
    assert observed_bands.n_replicates == expected_bands.n_replicates
    assert observed_bands.kind == expected_bands.kind
    assert observed_bands.alpha == expected_bands.alpha
    assert observed_bands.bands == expected_bands.bands


@pytest.mark.parametrize(
    ("target", "estimand"),
    [
        ("ate", ATE()),
        ("att", ATT()),
        ("atc", ATC()),
        ("ey", CounterfactualMean()),
        ("ey1", CounterfactualMean(treatment=1)),
        ("ey0", CounterfactualMean(treatment=0)),
        ("ey_obs", NaturalCourseMean()),
        ("par", PopulationAttributableRisk()),
        ("paf", PopulationAttributableFraction()),
        ("rr", RiskRatio()),
        ("or", OddsRatio()),
    ],
)
def test_every_arm_target_is_bit_for_bit_unchanged(target: str, estimand: Any) -> None:
    frame, _ = make_binary_outcome(n=180, seed=21)
    columns = ("W1", "W2", "W3")
    old = (
        TMLE(estimands=(target,), **POINT_SETTINGS)
        .fit(frame, outcome="Y", treatment="A", covariates=columns)
        .single()
    )
    study = CausalStudy(
        frame,
        design=PointTreatment(outcome="Y", treatment="A", adjustment=columns),
    )
    new = study.estimate(estimand, **POINT_SETTINGS)
    assert_identical(old, new)


@pytest.mark.parametrize(
    ("target", "estimand"),
    [
        ("ey_regime", RegimeMean),
        ("ate_regime", RegimeContrast),
    ],
)
def test_point_regimen_targets_are_bit_for_bit_unchanged(target: str, estimand: Any) -> None:
    frame, _ = make_binary_outcome(n=180, seed=22)
    columns = ("W1", "W2", "W3")
    regimens = (Static(0, name="never"), Static(1, name="always"))
    old = (
        TMLE(
            estimands=(target,),
            interventions=regimens,
            reference="never",
            **POINT_SETTINGS,
        )
        .fit(frame, outcome="Y", treatment="A", covariates=columns)
        .single()
    )
    study = CausalStudy(
        frame,
        design=PointTreatment(outcome="Y", treatment="A", adjustment=columns),
    )
    new = study.estimate(estimand(regimens, reference="never"), **POINT_SETTINGS)
    assert_identical(old, new)


@pytest.mark.parametrize(
    ("target", "estimand"),
    [("ey_ipsi", IncrementalMean), ("ate_ipsi", IncrementalEffect)],
)
def test_incremental_targets_are_bit_for_bit_unchanged(target: str, estimand: Any) -> None:
    frame, _ = make_binary_outcome(n=180, seed=23)
    columns = ("W1", "W2", "W3")
    tilts = (Incremental(1.0), Incremental(2.0))
    old = (
        TMLE(estimands=(target,), incremental=tilts, **POINT_SETTINGS)
        .fit(frame, outcome="Y", treatment="A", covariates=columns)
        .single()
    )
    study = CausalStudy(
        frame,
        design=PointTreatment(outcome="Y", treatment="A", adjustment=columns),
    )
    new = study.estimate(estimand(tilts), **POINT_SETTINGS)
    assert_identical(old, new)


@pytest.mark.parametrize(
    ("target", "estimand"),
    [
        ("ey_shift", ModifiedTreatmentPolicy),
        ("ate_shift", ModifiedTreatmentPolicyEffect),
    ],
)
def test_shift_targets_are_bit_for_bit_unchanged(target: str, estimand: Any) -> None:
    frame, _ = make_shift_dose(n=180, seed=24)
    columns = ("W1", "W2", "W3")
    shifts = (Shift(0.0, cap=None), Shift(0.5, cap=5.0))
    old = (
        TMLE(estimands=(target,), shifts=shifts, **POINT_SETTINGS)
        .fit(
            frame,
            outcome="Y",
            treatment="A",
            covariates=columns,
            treatment_kind="continuous",
        )
        .single()
    )
    study = CausalStudy(
        frame,
        design=PointTreatment(
            outcome="Y",
            treatment="A",
            adjustment=columns,
            treatment_kind="continuous",
        ),
    )
    new = study.estimate(estimand(shifts), **POINT_SETTINGS)
    assert_identical(old, new)


def test_msm_target_is_bit_for_bit_unchanged() -> None:
    frame, _ = make_binary_outcome(n=180, seed=25)
    columns = ("W1", "W2", "W3")
    model = MSM.linear()
    old = (
        TMLE(estimands=("msm",), msm=model, **POINT_SETTINGS)
        .fit(frame, outcome="Y", treatment="A", covariates=columns)
        .single()
    )
    study = CausalStudy(
        frame,
        design=PointTreatment(outcome="Y", treatment="A", adjustment=columns),
    )
    new = study.estimate(MSMProjection(model), **POINT_SETTINGS)
    assert_identical(old, new)


@pytest.mark.parametrize(
    ("estimand", "prefixes"),
    [
        (RegimeMean({"always": 1, "never": 0}, reference="always"), ("ey_regimen[",)),
        (
            RegimeContrast({"always": 1, "never": 0}, reference="always"),
            ("ate_regimen[",),
        ),
    ],
)
def test_end_of_study_longitudinal_results_are_bit_for_bit_unchanged(
    estimand: Any, prefixes: tuple[str, ...]
) -> None:
    frame, _ = make_longitudinal(n=220, seed=26)
    regimens = {"always": 1, "never": 0}
    old = LTMLE(regimens, reference="always", **LONG_SETTINGS).fit(frame, **LONG_COLUMNS)
    study = CausalStudy(frame, design=LongitudinalTreatment(**LONG_COLUMNS))
    new = study.estimate(estimand, **LONG_SETTINGS)
    expected = {name: value for name, value in old.estimates.items() if name.startswith(prefixes)}
    assert_identical(expected, new)


def test_longitudinal_msm_is_bit_for_bit_unchanged(tmp_path: Any) -> None:
    frame, _ = make_longitudinal(n=220, seed=28)
    regimens = {"always": 1, "never": 0, "early": (1, 0)}
    model = MSM(design=longitudinal_msm_design, terms=("(intercept)", "duration"))
    settings = {**LONG_SETTINGS, "n_folds": 1}
    old = LTMLE(regimens, msm=model, **settings).fit(frame, **LONG_COLUMNS)
    study = CausalStudy(frame, design=LongitudinalTreatment(**LONG_COLUMNS))
    new = study.estimate(MSMProjection(model, regimens=regimens), **settings)
    assert_identical(old, new)
    assert_identical(new, load(new.save(tmp_path / "longitudinal-msm.joblib")))


BAND_POINT_SETTINGS = {**FAST_KWARGS, "simultaneous": True}
BAND_LONG_SETTINGS = {**LONG_SETTINGS, "simultaneous": True}


def test_an_unnarrowed_fit_reports_the_engines_own_bands() -> None:
    """Simultaneous inference is part of the parity claim, not an extra.

    Every other test here sets ``simultaneous=False``, so nothing compared a critical value,
    a draw count, or a band. That is the blind spot three separate defects lived in.
    """
    frame, _ = make_binary_outcome(n=180, seed=21)
    columns = ("W1", "W2", "W3")
    old = (
        TMLE(estimands=("ey",), **BAND_POINT_SETTINGS)
        .fit(frame, outcome="Y", treatment="A", covariates=columns)
        .single()
    )
    study = CausalStudy(
        frame, design=PointTreatment(outcome="Y", treatment="A", adjustment=columns)
    )
    new = study.estimate(CounterfactualMean(), **BAND_POINT_SETTINGS)
    assert new.simultaneous is not None
    assert_identical(old, new, check_bands=True)


def test_a_longitudinal_fit_draws_the_multipliers_its_engine_declares() -> None:
    """The study path drew 1000 where ``LTMLE`` declares 2000, so every band was narrower."""
    frame, _ = make_longitudinal(n=220, seed=28)
    regimens = {"always": 1, "never": 0, "early": (1, 0)}
    model = MSM(design=longitudinal_msm_design, terms=("(intercept)", "duration"))
    settings = {**BAND_LONG_SETTINGS, "n_folds": 1}
    old = LTMLE(regimens, msm=model, **settings).fit(frame, **LONG_COLUMNS)
    study = CausalStudy(frame, design=LongitudinalTreatment(**LONG_COLUMNS))
    new = study.estimate(MSMProjection(model, regimens=regimens), **settings)
    assert new.simultaneous is not None
    assert new.simultaneous.n_replicates == 2000
    assert_identical(old, new, check_bands=True)


def test_selecting_one_arm_leaves_no_band_from_the_family_it_left() -> None:
    """A one-parameter result cannot carry a three-parameter critical value.

    ``CounterfactualMean(treatment=...)`` narrows a multi-arm fit after the fact. Filtering
    ``estimates`` alone left ``simultaneous`` holding the full family's critical value and
    bands for two arms the result no longer contains -- which ``summary()`` then printed.
    Both engines report no bands below two parameters, and so does this.
    """
    frame, _ = make_binary_outcome(n=240, seed=31)
    rng = np.random.default_rng(3)
    arms = np.array(["low", "mid", "high"])[rng.integers(0, 3, size=len(frame))]
    study = CausalStudy(
        frame.assign(A=arms),
        design=PointTreatment(outcome="Y", treatment="A", adjustment=("W1", "W2", "W3")),
    )
    every = study.estimate(CounterfactualMean(), **BAND_POINT_SETTINGS)
    assert len(every.estimates) == 3
    assert set(every.simultaneous.bands) == set(every.estimates)

    one = study.estimate(CounterfactualMean(treatment="mid"), **BAND_POINT_SETTINGS)
    assert list(one.estimates) == ["ey[mid]"]
    assert one.simultaneous is None
    assert "simultaneous" not in one.summary()


def test_bootstrap_draws_do_not_outlive_their_parameter() -> None:
    """Per-parameter draws stay correct under narrowing, but must not outlive the parameter.

    Unlike a joint band, each estimand's resampling distribution is its own, so nothing here
    is a wrong number -- the result simply handed out draws keyed by arms it had dropped and
    could no longer index.
    """
    frame, _ = make_binary_outcome(n=200, seed=31)
    rng = np.random.default_rng(3)
    arms = np.array(["low", "mid", "high"])[rng.integers(0, 3, size=len(frame))]
    study = CausalStudy(
        frame.assign(A=arms),
        design=PointTreatment(outcome="Y", treatment="A", adjustment=("W1", "W2", "W3")),
    )
    settings = {**POINT_SETTINGS, "n_bootstrap": 20}
    every = study.estimate(CounterfactualMean(), **settings)
    one = study.estimate(CounterfactualMean(treatment="mid"), **settings)

    assert set(every.bootstrap.draws) == set(every.estimates)
    assert set(one.bootstrap.draws) == set(one.estimates) == {"ey[mid]"}
    # Still usable, and the same draws it would have had unnarrowed.
    assert one.bootstrap.summary("ey[mid]").n_replicates == 20
    np.testing.assert_array_equal(one.bootstrap.draws["ey[mid]"], every.bootstrap.draws["ey[mid]"])


def test_a_narrowed_longitudinal_family_gets_its_own_critical_value() -> None:
    """Bands are recomputed for the reported family, not inherited from a wider one.

    ``RegimeMean`` keeps the means and drops the contrasts, so the engine's five-parameter
    critical value quantified a family two of whose members are gone. A max-t quantile over a
    subfamily cannot exceed the one over the family containing it -- that inequality is the
    property being checked, so it survives any rewrite of how the bands are rebuilt.
    """
    frame, _ = make_longitudinal(n=260, seed=33)
    regimens = {"always": 1, "never": 0, "early": (1, 0)}
    engine = LTMLE(regimens, reference="never", **BAND_LONG_SETTINGS).fit(frame, **LONG_COLUMNS)
    study = CausalStudy(frame, design=LongitudinalTreatment(**LONG_COLUMNS))
    means = study.estimate(RegimeMean(regimens, reference="never"), **BAND_LONG_SETTINGS)

    assert len(means.estimates) == 3 and len(engine.estimates) == 5
    assert set(means.simultaneous.bands) == set(means.estimates)
    assert means.simultaneous.critical_value <= engine.simultaneous.critical_value
    assert means.simultaneous.critical_value > means.simultaneous.pointwise_critical_value
    for name, (low, high) in means.simultaneous.bands.items():
        pointwise_low, pointwise_high = means[name].ci
        assert low <= pointwise_low and high >= pointwise_high


@pytest.mark.parametrize(
    ("factory", "outcome", "estimand_type", "prefix"),
    [
        (make_longitudinal_survival, ("Y1", "Y2"), RegimeMean, "risk_regimen["),
        (make_longitudinal_survival, ("Y1", "Y2"), RegimeContrast, "ate_regimen["),
        (
            make_longitudinal_competing,
            {"relapse": ("R1", "R2"), "death": ("D1", "D2")},
            RegimeMean,
            "cif_regimen[",
        ),
        (
            make_longitudinal_competing,
            {"relapse": ("R1", "R2"), "death": ("D1", "D2")},
            RegimeContrast,
            "ate_regimen[",
        ),
    ],
)
def test_longitudinal_curves_are_bit_for_bit_unchanged(
    factory: Any, outcome: Any, estimand_type: Any, prefix: str, tmp_path: Any
) -> None:
    n = 600 if factory is make_longitudinal_competing else 220
    frame, _ = factory(n=n, seed=27)
    columns = {**LONG_COLUMNS, "outcome": outcome}
    regimens = {"always": 1, "never": 0}
    old = LTMLE(regimens, reference="never", horizons=(1, 2), **LONG_SETTINGS).fit(frame, **columns)
    study = CausalStudy(frame, design=LongitudinalTreatment(**columns))
    new = study.estimate(
        estimand_type(regimens, reference="never", horizons=(1, 2)), **LONG_SETTINGS
    )
    expected = {name: value for name, value in old.estimates.items() if name.startswith(prefix)}
    assert_identical(expected, new)
    assert_identical(new, load(new.save(tmp_path / "longitudinal-curve.joblib")))

    # The keys say what the engine's own index says, cell for cell.  Rebuilding the
    # regimen/cause/horizon grid beside the index and zipping it positionally agreed with it
    # by construction on the counts, so only a pairing could be wrong -- and a competing-risks
    # fit over two horizons is the smallest grid where swapping the cause and horizon loops
    # relabels every parameter while every length still matches.
    assert set(new.parameter_keys) == set(new.estimates)
    for alias, key in new.parameter_keys.items():
        label, cause, horizon = old.parameter_index[alias]
        assert (key.cause, key.horizon) == (cause, horizon)
        assert key.regimen == key.value
        if key.reference is None:
            assert label == key.regimen
        else:
            assert label == f"{key.regimen} vs {key.reference}"


def test_a_competing_risks_grid_is_labelled_by_the_index_not_by_position() -> None:
    """The grid this exists for: two regimens, two causes, two horizons, all distinct.

    ``parameter_index`` is composed where each name is built, so it is the authority. Reading
    it also removes the reference test that looked for ``"ate_"`` inside the alias -- routing
    on a display name is what the result contract forbids, and what an unlucky regimen label
    would have broken.
    """
    frame, _ = make_longitudinal_competing(n=600, seed=27)
    columns = {**LONG_COLUMNS, "outcome": {"relapse": ("R1", "R2"), "death": ("D1", "D2")}}
    regimens = {"always": 1, "never": 0}
    study = CausalStudy(frame, design=LongitudinalTreatment(**columns))
    result = study.estimate(
        RegimeMean(regimens, reference="never", horizons=(1, 2)), **LONG_SETTINGS
    )

    seen = {(key.regimen, key.cause, key.horizon) for key in result.parameter_keys.values()}
    assert seen == {
        (regimen, cause, horizon)
        for regimen in ("always", "never")
        for cause in ("relapse", "death")
        for horizon in (1, 2)
    }
    # A mean is not a contrast, so nothing here carries a reference.
    assert all(key.reference is None for key in result.parameter_keys.values())
