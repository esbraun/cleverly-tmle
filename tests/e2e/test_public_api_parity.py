"""The typed public API is an exact adapter over every evidenced analytic target."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

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
    "outcome_learner": "glm",
    "pseudo_learner": "glm",
    "treatment_learner": "glm",
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


def assert_identical(old: Any, new: Any) -> None:
    """Point value and every inference quantity are bit-for-bit unchanged."""
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
    old = LTMLE(regimens, msm=model, **LONG_SETTINGS).fit(frame, **LONG_COLUMNS)
    study = CausalStudy(frame, design=LongitudinalTreatment(**LONG_COLUMNS))
    new = study.estimate(MSMProjection(model, regimens=regimens), **LONG_SETTINGS)
    assert_identical(old, new)
    assert_identical(new, load(new.save(tmp_path / "longitudinal-msm.npz")))


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
    assert_identical(new, load(new.save(tmp_path / "longitudinal-curve.npz")))
